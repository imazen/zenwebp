//! Regression tests for the 2026-05-06 security audit.
//!
//! Covers:
//! * H1 — lossless transform sub-image allocations bypass `Limits::check_memory`
//! * H2 — `WebPDemuxer` slice-indexing on attacker-controlled chunk sizes
//!         (truncated ANMF / VP8 / VP8L)

use zenwebp::{DecodeError, Limits, WebPDecoder};
use zenwebp::mux::WebPDemuxer;

/// Helper: bit-stream writer (LSB-first, matching VP8L's bitstream layout).
struct BitWriter {
    buf: Vec<u8>,
    cur: u64,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur: 0,
            n: 0,
        }
    }
    fn write(&mut self, value: u64, bits: u32) {
        debug_assert!(bits <= 32);
        self.cur |= (value & ((1u64 << bits) - 1)) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.buf.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.n -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push((self.cur & 0xff) as u8);
        }
        self.buf
    }
}

/// Build a VP8L bitstream with attacker-controlled width/height that triggers
/// large predictor + color transform sub-image allocations before any of the
/// decoder's normal Huffman-section work runs.
///
/// Layout (LSB-first):
/// * 8-bit signature 0x2f
/// * 14-bit width - 1
/// * 14-bit height - 1
/// * 1-bit alpha-used
/// * 3-bit version (0)
/// * 1-bit "transform present" = 1
/// * 2-bit transform type = 0 (predictor)
/// * 3-bit size_bits - 2 (we use size_bits = 2 → encoded value 0)
/// * (decoder then allocates predictor sub-image, this is the gate we trip)
///
/// We don't need to provide a valid Huffman section because the predictor
/// sub-image allocation happens *before* decode_image_stream is called on it.
/// The test asserts that the limits check fires before any real decode work.
fn build_amplifying_vp8l(width: u32, height: u32) -> Vec<u8> {
    assert!(width >= 1 && width <= 16384);
    assert!(height >= 1 && height <= 16384);

    let mut w = BitWriter::new();
    w.write(0x2f, 8);
    w.write((width - 1) as u64, 14);
    w.write((height - 1) as u64, 14);
    w.write(0, 1); // alpha-used
    w.write(0, 3); // version
    w.write(1, 1); // transform present
    w.write(0, 2); // type = 0 (predictor)
    w.write(0, 3); // size_bits - 2 = 0 → size_bits = 2 (block_xsize = ceil(W/4))
    // Stop here. Anything we add after won't be read because the allocation
    // check should reject before decode_image_stream is reached.
    w.finish()
}

/// Wrap raw VP8L bitstream in a RIFF/WEBP/VP8L container.
fn wrap_vp8l(vp8l_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    let chunk_size = vp8l_payload.len() as u32;
    let chunk_size_padded = chunk_size + (chunk_size & 1);
    let riff_size = 4 + 8 + chunk_size_padded; // "WEBP" + chunk header + payload
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8L");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(vp8l_payload);
    if chunk_size & 1 == 1 {
        out.push(0); // RIFF padding byte
    }
    out
}

// ---------------------------------------------------------------------------
// H1 — lossless transform allocations gated by Limits
// ---------------------------------------------------------------------------

#[test]
fn h1_predictor_transform_allocation_rejected_when_over_memory_limit() {
    // 8192 x 8192 → predictor sub-image with size_bits=2 = ceil(8192/4)^2 * 4
    // = 2048^2 * 4 = 16 MB. We set max_memory to 4 MB so the predictor
    // allocation must be rejected before any real decode work.
    //
    // 8192*8192 = 67 MP which fits within the default 100 MP total-pixels
    // ceiling, so the constructor's dimension validation passes and we reach
    // the lossless transform allocation site that this test is targeting.
    let payload = build_amplifying_vp8l(8192, 8192);
    let webp = wrap_vp8l(&payload);

    let limits = Limits::default().max_memory(4 * 1024 * 1024);

    let mut decoder = WebPDecoder::new(&webp).expect("header parse ok");
    decoder.set_limits(limits);

    let buf_size = decoder
        .output_buffer_size()
        .expect("output_buffer_size computable");
    let mut out = vec![0u8; buf_size];

    let err = decoder
        .read_image(&mut out)
        .expect_err("decode must reject due to memory limit");
    assert!(
        matches!(err.error(), DecodeError::MemoryLimitExceeded),
        "expected MemoryLimitExceeded, got: {:?}",
        err.error()
    );
}

#[test]
fn h1_predictor_transform_allocation_allowed_when_under_memory_limit() {
    // Same shape, but with a memory limit that comfortably covers the
    // intermediate predictor sub-image. We don't expect a clean decode (the
    // bitstream is truncated after the transform header), but we *do* expect
    // the failure to come from bitstream-shape checks downstream of the
    // memory check — not from MemoryLimitExceeded.
    let payload = build_amplifying_vp8l(64, 64);
    let webp = wrap_vp8l(&payload);

    let limits = Limits::default().max_memory(64 * 1024 * 1024);

    let mut decoder = WebPDecoder::new(&webp).expect("header parse ok");
    decoder.set_limits(limits);

    let buf_size = decoder
        .output_buffer_size()
        .expect("output_buffer_size computable");
    let mut out = vec![0u8; buf_size];

    let err = decoder
        .read_image(&mut out)
        .expect_err("truncated bitstream must fail");
    assert!(
        !matches!(err.error(), DecodeError::MemoryLimitExceeded),
        "memory limit should not fire for 64x64 / 64MB budget, got: {:?}",
        err.error()
    );
}

// ---------------------------------------------------------------------------
// H2 — demuxer slice indexing on attacker-controlled chunk sizes
// ---------------------------------------------------------------------------

/// Build a VP8X WebP container whose ANMF chunk header declares a payload
/// size larger than the remaining file. Pre-fix this caused parse_anmf_frame
/// to compute `&self.data[start..start + huge_size]` and panic.
fn build_truncated_anmf_container() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    // RIFF size will be patched after construction.
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(b"WEBP");

    // VP8X chunk: 10-byte payload.
    out.extend_from_slice(b"VP8X");
    out.extend_from_slice(&10u32.to_le_bytes());
    out.push(0b0000_0010); // is_animated = 1
    out.extend_from_slice(&[0, 0, 0]); // reserved
    // canvas_width - 1 = 9 (24-bit LE)
    out.extend_from_slice(&[9, 0, 0]);
    // canvas_height - 1 = 9
    out.extend_from_slice(&[9, 0, 0]);

    // ANMF chunk header lying about its size — claims it has 0xFFFF_FF00 bytes
    // of payload (~4 GB). The actual file ends right here.
    out.extend_from_slice(b"ANMF");
    out.extend_from_slice(&0xFFFF_FF00u32.to_le_bytes());
    // No payload follows. Demuxer should not register a FrameRecord whose
    // declared range exceeds the remaining buffer.

    // Patch RIFF size = total - 8.
    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());

    out
}

#[test]
fn h2_truncated_anmf_does_not_panic() {
    let data = build_truncated_anmf_container();

    // Demuxer creation must not panic and must either reject the file or
    // produce a demuxer whose frame iteration does not panic on truncated
    // ANMF records.
    match WebPDemuxer::new(&data) {
        Ok(d) => {
            // num_frames must reflect only frames whose declared range fits.
            let n = d.num_frames();
            // Iterating must not panic.
            for f in d.frames() {
                // Just touch fields to force any lazy panics.
                let _ = (f.frame_num, f.bitstream.len(), f.alpha_data.map(|s| s.len()));
            }
            // For truncated input we expect no usable frames.
            assert_eq!(
                n, 0,
                "truncated ANMF must not produce a usable frame record"
            );
        }
        Err(_) => {
            // Returning an error is also acceptable behavior.
        }
    }
}

#[test]
fn h2_truncated_simple_vp8l_does_not_panic() {
    // Build a RIFF/WEBP/VP8L header whose declared chunk_size exceeds the
    // remaining file. parse_simple_lossless previously stored
    // single_bitstream_range = (start, start + chunk_size) without bounds-
    // checking, panicking later when single_frame() sliced &self.data[..].
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0u32.to_le_bytes()); // patched below
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8L");
    // Lie: claim 1 GB of payload.
    out.extend_from_slice(&(1024u32 * 1024 * 1024).to_le_bytes());
    // Provide just enough for the 5-byte VP8L header parse.
    out.push(0x2f);
    out.extend_from_slice(&[0u8; 4]);

    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());

    match WebPDemuxer::new(&out) {
        Ok(d) => {
            // Calling frame(1) must not panic.
            if let Some(frame) = d.frame(1) {
                let _ = frame.bitstream.len();
            }
        }
        Err(_) => {
            // Returning an error is also fine.
        }
    }
}

#[test]
fn h2_truncated_simple_vp8_does_not_panic() {
    // Build a RIFF/WEBP/VP8  header whose declared chunk_size exceeds the
    // remaining file. Mirrors the VP8L test above for parse_simple_lossy.
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8 ");
    out.extend_from_slice(&(1024u32 * 1024 * 1024).to_le_bytes());
    // Minimal VP8 keyframe header (10 bytes):
    // frame_tag (3) + magic (3) + width (2) + height (2)
    out.extend_from_slice(&[0, 0, 0]); // frame_tag, low bit 0 = keyframe
    out.extend_from_slice(&[0x9D, 0x01, 0x2A]); // magic
    out.extend_from_slice(&[16u8, 0]); // width = 16
    out.extend_from_slice(&[16u8, 0]); // height = 16

    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());

    match WebPDemuxer::new(&out) {
        Ok(d) => {
            if let Some(frame) = d.frame(1) {
                let _ = frame.bitstream.len();
            }
        }
        Err(_) => {}
    }
}
