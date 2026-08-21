#![no_main]

//! Container/mux fuzzing: exercises the demuxer and metadata entry points
//! that `WebPDecoder`-based targets never reach.
//!
//! `WebPDecoder` walks VP8X chunks itself and never constructs a
//! `WebPDemuxer`, so the public `mux::WebPDemuxer` / `metadata::{icc,exif,xmp}`
//! surface was entirely unfuzzed — which is how a 38-byte crafted VP8X+ANIM
//! file reached an out-of-bounds panic (fixed; seeded in fuzz/regression).
//! Every method here must return Ok/Err/None without panicking on arbitrary
//! bytes.

use libfuzzer_sys::fuzz_target;
use zenwebp::mux::WebPDemuxer;

fuzz_target!(|data: &[u8]| {
    // Metadata free functions parse the container independently.
    let _ = zenwebp::metadata::icc_profile(data);
    let _ = zenwebp::metadata::exif(data);
    let _ = zenwebp::metadata::xmp(data);

    if let Ok(demuxer) = WebPDemuxer::new(data) {
        let n = demuxer.num_frames();
        let _ = demuxer.frame_count();
        let _ = demuxer.icc_profile();
        let _ = demuxer.exif();
        let _ = demuxer.xmp();

        // Iterate frames (bounded: a lying num_frames must not drive an
        // unbounded loop — the iterator terminates on the real data).
        for _frame in demuxer.frames() {}

        // Random-access a few frame indices around the declared count,
        // including out-of-range ones.
        for i in [0u32, 1, n, n.wrapping_add(1)] {
            if let Some(frame) = demuxer.frame(i) {
                let _ = frame.bitstream.len();
            }
        }
    }
});
