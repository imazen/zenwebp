//! Structural inventory of a WebP (RIFF) file: every chunk, with what the
//! zencodec decode path does with it.
//!
//! The walk never decodes pixels. Dispositions follow
//! [`WebPDecoder`](crate::decoder::WebPDecoder)`::read_data`
//! (`decoder/api.rs`), which feeds `ImageInfo::from_webp`, the still decode and
//! the animation decoder behind the zencodec adapter. The other two container
//! walkers (`mux::WebPDemuxer`, `detect::probe`) disagree with it in places
//! (see the report in the PR); this module deliberately mirrors the decoder.
//!
//! Conventions:
//! - A chunk part spans its 8-byte header, payload and a **zero** pad byte. A
//!   non-zero pad byte is its own `Padding` part, so a pad byte that carries
//!   data is visible.
//! - Bytes past `8 + riff_size` are `Trailing`, except chunks the decoder
//!   actually reads there (its VP8X walk limit overruns the RIFF end by 10
//!   bytes, see `walk_extended`).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Range;

use zencodec::ImageFormat;
use zencodec::inventory::{
    Disposition, Inventory, InventoryError, MetadataKind, Part, PartId, PartKind, PartTag,
};

use crate::DecodeError;

/// How often (in chunks) the cancellation check runs.
const CHECK_EVERY: u32 = 64;

/// Walk `data` and describe every byte. `check` is polled periodically and may
/// abort the walk (cooperative cancellation).
pub(crate) fn walk(
    data: &[u8],
    check: &mut dyn FnMut() -> Result<(), DecodeError>,
) -> Result<Inventory, DecodeError> {
    let mut w = Walker {
        data,
        len: data.len() as u64,
        inv: Inventory::new(ImageFormat::WebP, data.len() as u64),
        check,
        ticks: 0,
    };
    w.run()?;
    Ok(w.inv)
}

fn parts_err(e: InventoryError) -> DecodeError {
    DecodeError::InvalidParameter(format!("inventory: {e}"))
}

/// A chunk header found in the byte stream (not yet classified).
#[derive(Clone, Debug)]
struct RawChunk {
    /// Offset of the 8-byte header.
    pos: u64,
    fourcc: [u8; 4],
    /// Declared payload size.
    size: u64,
    /// Payload range, clamped to the bound it was read under.
    payload: Range<u64>,
    /// The declared payload runs past the bound.
    truncated: bool,
    /// Offset of the pad byte when the size is odd and the byte is present.
    pad: Option<u64>,
    /// Where the next chunk starts, following the decoder's rounding.
    next: u64,
}

impl RawChunk {
    fn is_known(&self) -> bool {
        matches!(
            &self.fourcc,
            b"RIFF"
                | b"WEBP"
                | b"VP8 "
                | b"VP8L"
                | b"VP8X"
                | b"ANIM"
                | b"ANMF"
                | b"ALPH"
                | b"ICCP"
                | b"EXIF"
                | b"XMP "
        )
    }
}

struct Walker<'a, 'c> {
    data: &'a [u8],
    len: u64,
    inv: Inventory,
    check: &'c mut dyn FnMut() -> Result<(), DecodeError>,
    ticks: u32,
}

/// Printable-ASCII rendering of a FourCC (non-printable bytes as `\xNN`).
fn fourcc_label(f: &[u8; 4]) -> String {
    let mut s = String::new();
    for &b in f {
        if (0x20..0x7f).contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("\\x{b:02x}"));
        }
    }
    s
}

impl Walker<'_, '_> {
    fn tick(&mut self) -> Result<(), DecodeError> {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks.is_multiple_of(CHECK_EVERY) {
            (self.check)()?;
        }
        Ok(())
    }

    fn push(&mut self, parent: Option<PartId>, part: Part) -> Result<PartId, DecodeError> {
        self.inv.push(parent, part).map_err(parts_err)
    }

    fn u32_at(&self, pos: u64) -> Option<u32> {
        let p = usize::try_from(pos).ok()?;
        let b = self.data.get(p..p.checked_add(4)?)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a chunk header at `pos`. `region_end` bounds where a header may
    /// start (it needs 8 bytes before it); `bound` clamps the payload.
    fn read_chunk(&self, pos: u64, region_end: u64, bound: u64) -> Option<RawChunk> {
        if pos.checked_add(8)? > region_end.min(bound) {
            return None;
        }
        let p = usize::try_from(pos).ok()?;
        let fourcc = [
            self.data[p],
            self.data[p + 1],
            self.data[p + 2],
            self.data[p + 3],
        ];
        let size = u64::from(self.u32_at(pos + 4)?);
        let declared_end = pos + 8 + size;
        let payload_end = declared_end.min(bound);
        let truncated = declared_end > bound;
        let pad = (size & 1 == 1 && !truncated && payload_end < bound).then_some(payload_end);
        Some(RawChunk {
            pos,
            fourcc,
            size,
            payload: pos + 8..payload_end,
            truncated,
            pad,
            // The decoder steps by the size rounded up to even, pad present or not.
            next: pos + 8 + size + (size & 1),
        })
    }

    fn pad_is_zero(&self, pad: u64) -> bool {
        self.data.get(pad as usize).copied() == Some(0)
    }

    /// Record `c` as a chunk part under `parent`; a non-zero pad byte becomes a
    /// sibling `Padding` part. A truncated chunk is `Malformed` whatever
    /// `disposition` says. Returns the chunk's id.
    fn emit(
        &mut self,
        parent: Option<PartId>,
        c: &RawChunk,
        disposition: Disposition,
        detail: Option<String>,
        body: bool,
    ) -> Result<PartId, DecodeError> {
        let zero_pad = c.pad.filter(|&p| self.pad_is_zero(p));
        let end = zero_pad.map_or(c.payload.end, |p| p + 1);
        let (disposition, detail) = if c.truncated {
            (
                Disposition::Malformed,
                Some(format!(
                    "chunk declares {} payload bytes, only {} are present{}",
                    c.size,
                    c.payload.end - c.payload.start,
                    detail.map(|d| format!(" ({d})")).unwrap_or_default()
                )),
            )
        } else {
            (disposition, detail)
        };
        let mut part = Part::new(
            PartKind::Chunk,
            PartTag::FourCc(c.fourcc),
            c.pos..end,
            disposition,
        )
        .with_label(fourcc_label(&c.fourcc));
        if body && c.payload.start < c.payload.end {
            part = part.with_body(c.payload.clone());
        }
        if let Some(d) = detail {
            part = part.with_detail(d);
        }
        let id = self.push(parent, part)?;
        if let Some(p) = c.pad
            && zero_pad.is_none()
        {
            let b = self.data.get(p as usize).copied().unwrap_or(0);
            self.push(
                parent,
                Part::new(PartKind::Gap, PartTag::None, p..p + 1, Disposition::Padding)
                    .with_detail(format!("non-zero RIFF pad byte 0x{b:02x}")),
            )?;
        }
        Ok(id)
    }

    /// First printable run of the payload, for unknown chunks.
    fn unknown_detail(&self, c: &RawChunk) -> String {
        let s = c.payload.start as usize;
        let e = (c.payload.end as usize).min(s.saturating_add(32));
        let run: String = self
            .data
            .get(s..e)
            .unwrap_or(&[])
            .iter()
            .take_while(|b| (0x20..0x7f).contains(*b))
            .map(|&b| b as char)
            .collect();
        if run.is_empty() {
            "unrecognised chunk type".to_string()
        } else {
            format!("unrecognised chunk type; payload starts {run:?}")
        }
    }

    /// Classify a chunk the decoder does not read: `Skipped` when the type is
    /// recognised, `Unknown` otherwise.
    fn unread(
        &mut self,
        parent: Option<PartId>,
        c: &RawChunk,
        why: &str,
    ) -> Result<PartId, DecodeError> {
        if c.is_known() {
            self.emit(
                parent,
                c,
                Disposition::Skipped,
                Some(why.to_string()),
                false,
            )
        } else {
            let d = self.unknown_detail(c);
            self.emit(parent, c, Disposition::Unknown, Some(d), false)
        }
    }

    fn run(&mut self) -> Result<(), DecodeError> {
        let len = self.len;
        if len == 0 {
            return Ok(());
        }
        let d = self.data;
        let header_ok = len >= 12 && &d[0..4] == b"RIFF" && &d[8..12] == b"WEBP";
        if !header_ok {
            let end = len.min(12);
            self.push(
                None,
                Part::new(
                    PartKind::Header,
                    PartTag::FourCc(*b"RIFF"),
                    0..end,
                    Disposition::Malformed,
                )
                .with_label("RIFF")
                .with_detail("missing or truncated RIFF....WEBP signature"),
            )?;
            return self
                .inv
                .fill_gaps(None, Disposition::Malformed)
                .map_err(parts_err);
        }
        let riff_size = u64::from(self.u32_at(4).unwrap_or(0));
        let declared_end = 8 + riff_size;
        let riff_end = declared_end.min(len).max(12);
        let mut detail = format!("riff_size={riff_size}");
        if declared_end > len {
            detail.push_str(&format!(
                "; declares a file end at {declared_end}, file has {len} bytes"
            ));
        } else if declared_end < len {
            detail.push_str(&format!(
                "; {} bytes follow the declared RIFF end at {declared_end}",
                len - declared_end
            ));
        }
        self.push(
            None,
            Part::new(
                PartKind::Header,
                PartTag::FourCc(*b"RIFF"),
                0..12,
                Disposition::Structure,
            )
            .with_label("RIFF/WEBP")
            .with_detail(detail),
        )?;

        match self.read_chunk(12, len, len) {
            None => {
                if len > 12 {
                    self.push(
                        None,
                        Part::new(
                            PartKind::Gap,
                            PartTag::None,
                            12..len.min(riff_end.max(13)),
                            Disposition::Malformed,
                        )
                        .with_detail("truncated chunk header"),
                    )?;
                }
            }
            Some(first) => match &first.fourcc {
                b"VP8 " | b"VP8L" => self.walk_simple(first, riff_end)?,
                b"VP8X" => self.walk_extended(first, riff_size, riff_end)?,
                _ => self.walk_rejected(first, riff_end)?,
            },
        }
        self.inv
            .fill_gaps(None, Disposition::Trailing)
            .map_err(parts_err)
    }

    /// Push a `Malformed` gap for a header that does not fit before `end`.
    fn truncated_header(
        &mut self,
        parent: Option<PartId>,
        pos: u64,
        end: u64,
    ) -> Result<(), DecodeError> {
        let end = end.min(self.len);
        if pos < end {
            self.push(
                parent,
                Part::new(
                    PartKind::Gap,
                    PartTag::None,
                    pos..end,
                    Disposition::Malformed,
                )
                .with_detail("truncated chunk header"),
            )?;
        }
        Ok(())
    }

    /// Chunks inside the RIFF range, none of them read.
    fn walk_tail(&mut self, mut pos: u64, riff_end: u64, why: &str) -> Result<(), DecodeError> {
        while pos < riff_end {
            self.tick()?;
            match self.read_chunk(pos, riff_end, self.len) {
                Some(c) => {
                    self.unread(None, &c, why)?;
                    pos = c.next;
                }
                None => {
                    self.truncated_header(None, pos, riff_end)?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Simple format: the first chunk is the image; `WebPDecoder::read_data`
    /// never looks past it (`decoder/api.rs`, `WebPRiffChunk::VP8` / `VP8L` arms).
    fn walk_simple(&mut self, first: RawChunk, riff_end: u64) -> Result<(), DecodeError> {
        let kind = if &first.fourcc == b"VP8 " {
            "lossy"
        } else {
            "lossless"
        };
        self.emit(
            None,
            &first,
            Disposition::ImageData,
            Some(format!("{kind} image")),
            false,
        )?;
        self.walk_tail(
            first.next,
            riff_end,
            "simple-format file: the decoder reads nothing after the image chunk",
        )
    }

    /// First chunk is not VP8/VP8L/VP8X: the decoder returns
    /// `ChunkHeaderInvalid`. Report everything as unread.
    fn walk_rejected(&mut self, first: RawChunk, riff_end: u64) -> Result<(), DecodeError> {
        let why = "decoder rejects the file: first chunk is not VP8, VP8L or VP8X";
        self.unread(None, &first, why)?;
        self.walk_tail(first.next, riff_end, why)
    }

    /// Extended (VP8X) format.
    ///
    /// The decoder's loop is `while position < position0 + riff_size - 12`
    /// where `position0` is the offset after the VP8X chunk (`api.rs`,
    /// `WebPRiffChunk::VP8X` arm). With the usual 10-byte VP8X payload that
    /// limit is `riff_size + 18`, ten bytes past the RIFF end at `riff_size +
    /// 8`, so a chunk header that starts in those ten bytes is read and kept
    /// (suspected bug, confirmed by the `read_past_riff_end` test). The walk
    /// reproduces the limit.
    fn walk_extended(
        &mut self,
        vp8x: RawChunk,
        riff_size: u64,
        riff_end: u64,
    ) -> Result<(), DecodeError> {
        // --- the VP8X chunk itself -------------------------------------------------
        let payload_len = vp8x.payload.end - vp8x.payload.start;
        let flags =
            (!vp8x.truncated && payload_len >= 10).then(|| self.data[vp8x.payload.start as usize]);
        let (animated, alpha_flag) = match flags {
            Some(f) => (f & 0x02 != 0, f & 0x10 != 0),
            None => (false, false),
        };
        match flags {
            Some(f) => {
                let s = vp8x.payload.start as usize;
                let b = &self.data[s..s + 10];
                let w = 1 + (u32::from(b[4]) | u32::from(b[5]) << 8 | u32::from(b[6]) << 16);
                let h = 1 + (u32::from(b[7]) | u32::from(b[8]) << 8 | u32::from(b[9]) << 16);
                let id = self.emit(
                    None,
                    &vp8x,
                    Disposition::Structure,
                    Some(format!(
                        "flags=0x{f:02x} (icc={} alpha={} exif={} xmp={} anim={}), canvas {w}x{h}; \
                         reserved bits are ignored",
                        u8::from(f & 0x20 != 0),
                        u8::from(f & 0x10 != 0),
                        u8::from(f & 0x08 != 0),
                        u8::from(f & 0x04 != 0),
                        u8::from(f & 0x02 != 0),
                    )),
                    true,
                )?;
                let s64 = vp8x.payload.start;
                self.push(
                    Some(id),
                    Part::new(
                        PartKind::Field,
                        PartTag::Name("flags+canvas".into()),
                        s64..s64 + 10,
                        Disposition::Structure,
                    ),
                )?;
                self.inv
                    .fill_gaps(Some(id), Disposition::Unreferenced)
                    .map_err(parts_err)?;
            }
            None => {
                self.emit(
                    None,
                    &vp8x,
                    Disposition::Malformed,
                    Some("VP8X payload shorter than 10 bytes".to_string()),
                    false,
                )?;
            }
        }

        // --- collect the chunks the decoder's loop visits --------------------------
        let max_parts = u64::from(part_budget());
        let mut chunks: Vec<RawChunk> = Vec::new();
        let mut pos = vp8x.next;
        let limit = pos.saturating_add(riff_size.saturating_sub(12));
        while pos < limit {
            self.tick()?;
            let Some(c) = self.read_chunk(pos, self.len, self.len) else {
                break; // the decoder's header read hits end of data
            };
            pos = c.next;
            if chunks.len() as u64 >= max_parts {
                return Err(parts_err(InventoryError::TooManyParts {
                    max: max_parts as u32,
                }));
            }
            chunks.push(c);
        }

        // --- classify --------------------------------------------------------------
        let has_vp8l = chunks.iter().any(|c| &c.fourcc == b"VP8L");
        let first_anmf = chunks.iter().find(|c| &c.fourcc == b"ANMF").map(|c| c.pos);
        // Frames are read by following `next_frame_start += anmf_size + 8`
        // (unrounded, `read_frame`), starting at the first ANMF.
        let mut frame_cursor = first_anmf;
        let mut frame_chain_ok = true;
        let mut seen: Vec<[u8; 4]> = Vec::new();

        for c in &chunks {
            self.tick()?;
            let past_end = c.pos >= riff_end;
            let dup = seen.contains(&c.fourcc);
            let first = !dup && c.is_known();
            if c.is_known() {
                seen.push(c.fourcc);
            }
            let note = |s: &str| -> Option<String> {
                if past_end {
                    Some(format!("{s}; starts past the RIFF end"))
                } else {
                    Some(s.to_string())
                }
            };
            let dup_note = "duplicate chunk: the decoder keeps the first one";

            // Chunks that start past the declared RIFF end are recorded only when
            // the decoder keeps them; the rest stays `Trailing`.
            let keeps = match &c.fourcc {
                b"ICCP" | b"EXIF" | b"XMP " => first,
                b"VP8 " | b"VP8L" => first && !animated,
                b"ALPH" => first && !animated && alpha_flag && !has_vp8l,
                b"ANIM" | b"ANMF" => animated,
                _ => false,
            };
            if past_end && !keeps {
                continue;
            }

            match &c.fourcc {
                b"VP8X" => {
                    self.emit(
                        None,
                        c,
                        Disposition::Skipped,
                        note("second VP8X chunk"),
                        false,
                    )?;
                }
                b"VP8 " | b"VP8L" => {
                    let (d, n) = if animated {
                        (
                            Disposition::Skipped,
                            "animated file: frames are read from ANMF, not from top-level image chunks",
                        )
                    } else if dup {
                        (Disposition::Skipped, dup_note)
                    } else {
                        (Disposition::ImageData, "image")
                    };
                    self.emit(None, c, d, note(n), false)?;
                }
                b"ALPH" => {
                    let (d, n) = if animated {
                        (
                            Disposition::Skipped,
                            "animated file: alpha is read from the ANMF sub-chunk",
                        )
                    } else if dup {
                        (Disposition::Skipped, dup_note)
                    } else if !alpha_flag {
                        (
                            Disposition::Skipped,
                            "VP8X alpha flag is clear: ALPH is ignored",
                        )
                    } else if has_vp8l {
                        (
                            Disposition::Skipped,
                            "VP8L image carries its own alpha: ALPH is ignored",
                        )
                    } else {
                        (Disposition::ImageData, "alpha plane for the VP8 image")
                    };
                    self.emit(None, c, d, note(n), false)?;
                }
                b"ICCP" | b"EXIF" | b"XMP " => {
                    let kind = match &c.fourcc {
                        b"ICCP" => MetadataKind::Icc,
                        b"EXIF" => MetadataKind::Exif,
                        _ => MetadataKind::Xmp,
                    };
                    // `chunks.entry(chunk).or_insert(range)`: the first one wins and the
                    // VP8X flag is not consulted (`api.rs`, `icc_profile`/`exif_metadata`).
                    let (d, n) = if dup {
                        (Disposition::Skipped, dup_note)
                    } else {
                        (Disposition::Metadata(kind), "first occurrence")
                    };
                    self.emit(None, c, d, note(n), false)?;
                }
                b"ANIM" => {
                    if !animated {
                        self.emit(
                            None,
                            c,
                            Disposition::Skipped,
                            note("animation flag is clear"),
                            false,
                        )?;
                    } else if dup {
                        self.emit(None, c, Disposition::Skipped, note(dup_note), false)?;
                    } else {
                        self.emit_anim(c)?;
                    }
                }
                b"ANMF" => {
                    if !animated {
                        self.emit(
                            None,
                            c,
                            Disposition::Skipped,
                            note("animation flag is clear"),
                            false,
                        )?;
                    } else {
                        let reachable = frame_chain_ok && frame_cursor == Some(c.pos);
                        if !reachable {
                            frame_chain_ok = false;
                        }
                        // `read_frame`: next_frame_start += anmf_size + 8, no rounding.
                        frame_cursor = Some(c.pos + 8 + c.size);
                        self.emit_anmf(c, reachable)?;
                    }
                }
                _ => {
                    let d = self.unknown_detail(c);
                    self.emit(None, c, Disposition::Unknown, note(&d), false)?;
                }
            }
        }
        Ok(())
    }

    /// ANIM: background colour (native API only) + loop count (zencodec
    /// `ImageSequence::Animation::loop_count`, `codec.rs`
    /// `animation_frame_decoder`).
    fn emit_anim(&mut self, c: &RawChunk) -> Result<(), DecodeError> {
        // `read_chunk(ANIM, 6)` fails on a longer payload and `read_exact`
        // fails on a shorter one: only exactly 6 bytes decode.
        if c.truncated || c.size != 6 {
            self.emit(
                None,
                c,
                Disposition::Malformed,
                Some(format!(
                    "ANIM payload is {} bytes, the decoder needs exactly 6",
                    c.size
                )),
                false,
            )?;
            return Ok(());
        }
        let id = self.emit(None, c, Disposition::Structure, None, true)?;
        let s = c.payload.start;
        self.push(
            Some(id),
            Part::new(
                PartKind::Field,
                PartTag::Name("background_color".into()),
                s..s + 4,
                Disposition::Dropped,
            )
            .with_detail(
                "native WebPDecoder::background_color_hint only; the zencodec path does not report it",
            ),
        )?;
        self.push(
            Some(id),
            Part::new(
                PartKind::Field,
                PartTag::Name("loop_count".into()),
                s + 4..s + 6,
                Disposition::Metadata(MetadataKind::Animation),
            ),
        )?;
        Ok(())
    }

    /// ANMF: 16-byte frame header, then sub-chunks.
    ///
    /// `read_frame` reads x, y, width, height, duration and the blend/dispose
    /// flags, then one image chunk (`VP8`, `VP8L`, or `ALPH` followed by
    /// `VP8`). Later sub-chunks are never read.
    fn emit_anmf(&mut self, c: &RawChunk, reachable: bool) -> Result<(), DecodeError> {
        let payload_len = c.payload.end - c.payload.start;
        if c.truncated || payload_len < 16 {
            self.emit(
                None,
                c,
                Disposition::Malformed,
                Some("ANMF shorter than its 16-byte frame header".to_string()),
                false,
            )?;
            return Ok(());
        }
        let note = if reachable {
            None
        } else {
            Some(
                "unreachable: read_frame advances by the unrounded ANMF size and expects the \
                 next ANMF header there, so this frame and every later one fail to decode"
                    .to_string(),
            )
        };
        let id = self.emit(None, c, Disposition::Structure, note, true)?;
        let s = c.payload.start;
        let header_disp = if reachable {
            Disposition::Metadata(MetadataKind::Animation)
        } else {
            Disposition::Skipped
        };
        self.push(
            Some(id),
            Part::new(
                PartKind::Header,
                PartTag::Name("frame header".into()),
                s..s + 16,
                header_disp,
            )
            .with_detail("frame x/y/size, duration, blend and dispose flags"),
        )?;

        // Sub-chunks inside the body.
        let body_end = c.payload.end;
        let mut subs: Vec<RawChunk> = Vec::new();
        let mut pos = s + 16;
        while pos < body_end {
            self.tick()?;
            match self.read_chunk(pos, body_end, body_end) {
                Some(sc) => {
                    pos = sc.next;
                    subs.push(sc);
                    if subs.len() as u64 > u64::from(part_budget()) {
                        return Err(parts_err(InventoryError::TooManyParts {
                            max: part_budget(),
                        }));
                    }
                }
                None => {
                    self.truncated_header(Some(id), pos, body_end)?;
                    break;
                }
            }
        }
        for (i, sc) in subs.iter().enumerate() {
            let (d, n): (Disposition, String) = if !reachable {
                (Disposition::Skipped, "frame is not reachable".to_string())
            } else {
                match (i, &sc.fourcc) {
                    (0, b"VP8 " | b"VP8L") => (Disposition::ImageData, "frame image".to_string()),
                    (0, b"ALPH") => (Disposition::ImageData, "frame alpha plane".to_string()),
                    (1, b"VP8 ") if subs[0].fourcc == *b"ALPH" => {
                        (Disposition::ImageData, "frame image".to_string())
                    }
                    (0, _) => (
                        Disposition::Malformed,
                        "read_frame needs VP8, VP8L or ALPH as the first sub-chunk".to_string(),
                    ),
                    (1, _) if subs[0].fourcc == *b"ALPH" => (
                        Disposition::Malformed,
                        "read_frame needs VP8 after ALPH".to_string(),
                    ),
                    _ => (
                        Disposition::Skipped,
                        "read_frame reads only the frame's image chunk(s)".to_string(),
                    ),
                }
            };
            if d == Disposition::Skipped && !sc.is_known() {
                let det = self.unknown_detail(sc);
                self.emit(Some(id), sc, Disposition::Unknown, Some(det), false)?;
            } else {
                self.emit(Some(id), sc, d, Some(n), false)?;
            }
        }
        self.inv
            .fill_gaps(Some(id), Disposition::Unreferenced)
            .map_err(parts_err)
    }
}

/// Part cap used to bound the intermediate chunk lists.
pub(crate) fn part_budget() -> u32 {
    zencodec::inventory::DEFAULT_MAX_PARTS
}
