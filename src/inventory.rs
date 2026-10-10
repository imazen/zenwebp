//! Structural inventory of a WebP (RIFF) file: every chunk, with what the
//! zencodec decode path does with it.
//!
//! The walk never decodes pixels. Dispositions follow what each zencodec
//! entry point returns. Three readers disagree about the same VP8X file:
//! [`WebPDecoder`](crate::decoder::WebPDecoder)`::read_data` (`probe()`,
//! `ImageInfo`, animation, lossless pixels), `mux::WebPDemuxer` (the pixels of
//! a lossy still in `decode()`) and `detect::probe` (`source_encoding_details`).
//! `walk_extended` models all three and says in each part's detail which one
//! decides it.
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
use alloc::vec;
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
        self.emit_ex(parent, c, disposition, detail, body, false)
    }

    /// [`emit`](Self::emit), optionally keeping the given disposition for a
    /// chunk that runs past the end of the data (`detect::probe` clamps the
    /// payload and still delivers it).
    fn emit_ex(
        &mut self,
        parent: Option<PartId>,
        c: &RawChunk,
        disposition: Disposition,
        detail: Option<String>,
        body: bool,
        keep_truncated: bool,
    ) -> Result<PartId, DecodeError> {
        let zero_pad = c.pad.filter(|&p| self.pad_is_zero(p));
        let end = zero_pad.map_or(c.payload.end, |p| p + 1);
        let (disposition, detail) = if c.truncated && keep_truncated {
            (
                disposition,
                Some(format!(
                    "chunk declares {} payload bytes, only {} are present{}",
                    c.size,
                    c.payload.end - c.payload.start,
                    detail.map(|d| format!("; {d}")).unwrap_or_default()
                )),
            )
        } else if c.truncated {
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

    fn child(
        &mut self,
        parent: PartId,
        name: &'static str,
        range: Range<u64>,
        disposition: Disposition,
        detail: Option<String>,
    ) -> Result<PartId, DecodeError> {
        let mut p = Part::new(
            PartKind::Field,
            PartTag::Name(name.into()),
            range,
            disposition,
        );
        if let Some(d) = detail {
            p = p.with_detail(d);
        }
        self.push(Some(parent), p)
    }

    fn append_detail(&mut self, id: PartId, text: &str) {
        let new = match self.inv.get(id).and_then(|p| p.detail.clone()) {
            Some(old) => format!("{old}; {text}"),
            None => text.to_string(),
        };
        self.inv.set_detail(id, new);
    }

    /// Every chunk from `start` to the end of the data, stepping like the
    /// decoder's walks (size rounded up to even). Also returns where the walk
    /// stopped.
    fn collect_chunks(&mut self, start: u64) -> Result<(Vec<RawChunk>, u64), DecodeError> {
        let mut v: Vec<RawChunk> = Vec::new();
        let mut pos = start;
        while let Some(c) = self.read_chunk(pos, self.len, self.len) {
            self.tick()?;
            pos = c.next;
            if v.len() as u64 >= u64::from(part_budget()) {
                return Err(parts_err(InventoryError::TooManyParts {
                    max: part_budget(),
                }));
            }
            v.push(c);
        }
        Ok((v, pos))
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
        let hdr = self.push(
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
                    self.truncated_header(None, 12, riff_end.max(13))?;
                }
            }
            Some(first) => match &first.fourcc {
                b"VP8 " | b"VP8L" => self.walk_simple(first, riff_end)?,
                b"VP8X" => self.walk_extended(hdr, first, riff_size, riff_end)?,
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

    /// Whether `detect::probe` accepts a VP8 payload: it needs the 10-byte
    /// uncompressed header (keyframe bit and `9d 01 2a`). The bool-decoder
    /// header after it is not checked here.
    fn vp8_header_ok(&self, payload: &Range<u64>) -> bool {
        let s = payload.start as usize;
        let e = (payload.end as usize).min(self.data.len());
        let Some(d) = self.data.get(s..e) else {
            return false;
        };
        d.len() >= 10
            && (u32::from(d[0]) | u32::from(d[1]) << 8 | u32::from(d[2]) << 16) & 1 == 0
            && d[3..6] == [0x9d, 0x01, 0x2a]
    }

    /// What `detect::probe` makes of the file, in file order over the chunks
    /// `all` (the first chunk included): `None` when it returns an error (so
    /// zencodec attaches no `WebPProbe`), else the index of the last non-empty
    /// ICCP it keeps and of the chunk that owns the bitstream (first top-level
    /// VP8/VP8L, or the first sub-chunk of an ANMF that comes earlier while
    /// the animation flag is set), plus whether that owner is inside an ANMF.
    /// One computation serves both the ICC claim and the bitstream-owner detail.
    fn detect_model(&self, all: &[&RawChunk]) -> Option<DetectModel> {
        let mut animated = false;
        let mut owner: Option<(usize, bool)> = None;
        let mut icc: Option<usize> = None;
        for (i, c) in all.iter().enumerate() {
            match &c.fourcc {
                b"VP8 " if owner.is_none() => {
                    if !self.vp8_header_ok(&c.payload) {
                        return None;
                    }
                    owner = Some((i, false));
                }
                b"VP8L" if owner.is_none() => owner = Some((i, false)),
                b"VP8X" if c.payload.end - c.payload.start >= 10 => {
                    animated = self.data[c.payload.start as usize] & 0x02 != 0;
                }
                b"ANMF" if animated && owner.is_none() && c.payload.end - c.payload.start >= 24 => {
                    let sub = c.payload.start + 16;
                    if sub + 8 <= c.payload.end {
                        let f = &self.data[sub as usize..sub as usize + 4];
                        let size = u64::from(self.u32_at(sub + 4).unwrap_or(0));
                        let pe = (sub + 8 + size).min(c.payload.end);
                        if f == b"VP8 " {
                            if !self.vp8_header_ok(&(sub + 8..pe)) {
                                return None;
                            }
                            owner = Some((i, true));
                        } else if f == b"VP8L" {
                            owner = Some((i, true));
                        }
                    }
                }
                b"ICCP" if c.payload.end > c.payload.start => icc = Some(i),
                _ => {}
            }
        }
        owner.map(|(o, in_anmf)| DetectModel {
            icc,
            owner: o,
            owner_in_anmf: in_anmf,
        })
    }

    const TRUNCATED_NOTE: &'static str = "decode() decodes the bytes present when they suffice, \
        else it falls back to WebPDecoder and fails";

    const PROBE_ICC: &'static str = "reported through source_encoding_details() \
        (WebPProbe::icc_profile: detect::probe keeps the last ICCP and ignores the RIFF size), \
        not through ImageInfo";

    /// Chunks after the first one, none of them read by the decoder.
    /// `probe_first` is the image chunk when the file decodes: `detect::probe`
    /// then adds its own view (the last ICCP) if it succeeds on this file.
    fn walk_tail(
        &mut self,
        start: u64,
        riff_end: u64,
        why: &str,
        probe_first: Option<&RawChunk>,
    ) -> Result<(), DecodeError> {
        let (chunks, end_pos) = self.collect_chunks(start)?;
        let p_icc = probe_first.and_then(|f| {
            let all: Vec<&RawChunk> = core::iter::once(f).chain(chunks.iter()).collect();
            self.detect_model(&all)
                .and_then(|m| m.icc)
                .and_then(|i| i.checked_sub(1))
        });
        for (i, c) in chunks.iter().enumerate() {
            let exposed = Some(i) == p_icc;
            if c.pos >= riff_end && !exposed {
                continue;
            }
            if exposed {
                let past = if c.pos >= riff_end {
                    "; starts past the RIFF end"
                } else {
                    ""
                };
                self.emit_ex(
                    None,
                    c,
                    Disposition::Metadata(MetadataKind::Icc),
                    Some(format!("{}{past}", Self::PROBE_ICC)),
                    false,
                    true,
                )?;
            } else {
                self.unread(None, c, why)?;
            }
        }
        if end_pos < riff_end {
            self.truncated_header(None, end_pos, riff_end)?;
        }
        Ok(())
    }

    /// Why `read_data` rejects a simple-format file, if it does: the checks
    /// on the first chunk's header bytes (read from the file, not clamped to
    /// the chunk) and `Limits::default()`, called the way the decoder calls it.
    fn simple_reject(&self, first: &RawChunk) -> Option<String> {
        let limits = crate::decoder::Limits::default();
        if let Err(e) = limits.check_file_size(self.len) {
            return Some(e.error().to_string());
        }
        let d = self.data.get(first.payload.start as usize..)?;
        let (w, h) = if &first.fourcc == b"VP8 " {
            if d.len() < 10 {
                return Some("VP8 header is truncated".to_string());
            }
            let tag = u32::from(d[0]) | u32::from(d[1]) << 8 | u32::from(d[2]) << 16;
            if tag & 1 != 0 {
                return Some("VP8 frame is not a keyframe".to_string());
            }
            if d[3..6] != [0x9d, 0x01, 0x2a] {
                return Some("VP8 start code 9d 01 2a is missing".to_string());
            }
            let w = u32::from(u16::from_le_bytes([d[6], d[7]]) & 0x3FFF);
            let h = u32::from(u16::from_le_bytes([d[8], d[9]]) & 0x3FFF);
            if w == 0 || h == 0 {
                return Some("VP8 dimensions are zero".to_string());
            }
            (w, h)
        } else {
            if d.len() < 5 {
                return Some("VP8L header is truncated".to_string());
            }
            if d[0] != 0x2f {
                return Some("VP8L signature byte 0x2f is missing".to_string());
            }
            let header = u32::from_le_bytes([d[1], d[2], d[3], d[4]]);
            if header >> 29 != 0 {
                return Some("VP8L version is not 0".to_string());
            }
            ((header & 0x3FFF) + 1, ((header >> 14) & 0x3FFF) + 1)
        };
        limits
            .check_dimensions(w, h)
            .err()
            .map(|e| e.error().to_string())
    }

    /// Simple format: the first chunk is the image; `WebPDecoder::read_data`
    /// never looks past it (`decoder/api.rs`, `WebPRiffChunk::VP8` / `VP8L` arms).
    fn walk_simple(&mut self, first: RawChunk, riff_end: u64) -> Result<(), DecodeError> {
        let kind = if &first.fourcc == b"VP8 " {
            "lossy"
        } else {
            "lossless"
        };
        if let Some(reason) = self.simple_reject(&first) {
            let why = format!("decoder rejects this file: {reason}");
            self.emit(
                None,
                &first,
                Disposition::Malformed,
                Some(why.clone()),
                false,
            )?;
            return self.walk_tail(first.next, riff_end, &why, None);
        }
        // A truncated simple VP8 still decodes when the bytes present suffice
        // (`decode_lossy_internal` clamps); a truncated VP8L fails in `chunk_slice`.
        let keep = first.truncated && &first.fourcc == b"VP8 ";
        let note = if keep {
            format!("; {}", Self::TRUNCATED_NOTE)
        } else {
            String::new()
        };
        self.emit_ex(
            None,
            &first,
            Disposition::ImageData,
            Some(format!(
                "{kind} image; bytes after the bitstream's own end are not distinguished{note}"
            )),
            false,
            keep,
        )?;
        self.walk_tail(
            first.next,
            riff_end,
            "simple-format file: the decoder reads nothing after the image chunk",
            Some(&first),
        )
    }

    /// First chunk is not VP8/VP8L/VP8X: the decoder returns
    /// `ChunkHeaderInvalid`. Report everything as unread.
    fn walk_rejected(&mut self, first: RawChunk, riff_end: u64) -> Result<(), DecodeError> {
        let why = "decoder rejects the file: first chunk is not VP8, VP8L or VP8X";
        self.unread(None, &first, why)?;
        self.walk_tail(first.next, riff_end, why, None)
    }

    /// Extended (VP8X) format. Three walkers read a VP8X file, and the
    /// zencodec entry points use different ones:
    ///
    /// - `WebPDecoder::read_data` (`decoder/api.rs`) feeds `probe()`,
    ///   `ImageInfo` metadata, animation, and the pixels of lossless and
    ///   non-lossy-still files. Its chunk loop ends at `position + riff_size -
    ///   12`, ten bytes past the RIFF end for the usual 10-byte VP8X payload.
    /// - `WebPDemuxer` supplies the pixels of a non-animated lossy still for
    ///   `decode()`, `push_decoder` and `streaming_decoder`: it keeps the
    ///   *last* VP8/VP8L and the last ALPH inside the RIFF size.
    /// - `detect::probe` (`source_encoding_details()`) walks to the end of the
    ///   data and keeps the last ICCP.
    fn walk_extended(
        &mut self,
        hdr: PartId,
        vp8x: RawChunk,
        riff_size: u64,
        riff_end: u64,
    ) -> Result<(), DecodeError> {
        let len = self.len;
        let (chunks, end_pos) = self.collect_chunks(vp8x.next)?;
        let m = self.model(&vp8x, &chunks, riff_size);

        // --- the VP8X chunk ---------------------------------------------------
        let vp8x_id = self.emit_vp8x(&vp8x, &m)?;
        if let Some(reason) = &m.rejected {
            self.append_detail(hdr, &format!("decoder rejects this file: {reason}"));
        } else if m.d_viable
            && let Some(e) = &m.native_err
        {
            {
                let t = format!(
                    "probe() fails here (read_data: {e}); decode() uses WebPDemuxer when the \
                     bitstream decodes (else it falls back to WebPDecoder and fails), and its \
                     ImageInfo carries no ICC/EXIF/XMP and no alpha"
                );
                self.append_detail(hdr, &t);
                self.append_detail(vp8x_id, &t);
            }
        }

        // --- the chunks the walkers visit -------------------------------------
        let mut frame_cursor = m.first_anmf.map(|i| chunks[i].pos);
        let mut frame_chain_ok = true;
        for (i, c) in chunks.iter().enumerate() {
            self.tick()?;
            let p_icc = m.p_icc == Some(i);
            // Past the RIFF end only a chunk some walker keeps is a chunk; the rest
            // stays Trailing.
            let kept_past_end = p_icc || (c.is_known() && m.first(&c.fourcc) == Some(i));
            if c.pos >= riff_end && !kept_past_end {
                continue;
            }
            let past = if c.pos >= riff_end {
                "; starts past the RIFF end"
            } else {
                ""
            };
            if let Some(reason) = &m.rejected {
                let why = format!("decoder rejects this file: {reason}");
                self.unread(None, c, &why)?;
                continue;
            }
            let dup_note = "duplicate chunk";
            match &c.fourcc {
                b"VP8X" => {
                    self.emit(
                        None,
                        c,
                        Disposition::Skipped,
                        Some(format!(
                            "second VP8X chunk; detect::probe re-reads every VP8X, so \
                             WebPProbe width/height/has_alpha come from the last one{past}"
                        )),
                        false,
                    )?;
                }
                b"VP8 " | b"VP8L" => {
                    let owner = if m.p_bitstream == Some((i, false)) {
                        "; WebPProbe quality/lossless come from this chunk"
                    } else {
                        ""
                    };
                    let (d, n): (Disposition, String) = if m.animated {
                        (
                            Disposition::Skipped,
                            format!("animated file: frames are read from ANMF{owner}"),
                        )
                    } else if m.d_viable {
                        if m.d_last_img == Some(i) {
                            let w_first = m.first(&c.fourcc);
                            let extra = if w_first != Some(i) {
                                "; the native WebPDecoder would use the first one instead"
                            } else {
                                ""
                            };
                            (
                                Disposition::ImageData,
                                format!(
                                    "lossy still: decode() takes the last image chunk \
                                     (WebPDemuxer) when the bitstream decodes, else it falls \
                                     back to WebPDecoder (first chunks){extra}{owner}"
                                ),
                            )
                        } else {
                            (
                                Disposition::Skipped,
                                format!(
                                    "decode() takes the last image chunk (WebPDemuxer); \
                                     WebPDecoder would use the first one, or reject the file \
                                     when both VP8 and VP8L are present{owner}"
                                ),
                            )
                        }
                    } else if m.first(&c.fourcc) == Some(i) {
                        (Disposition::ImageData, "image".to_string())
                    } else {
                        (
                            Disposition::Skipped,
                            format!("{dup_note}: the decoder keeps the first{owner}"),
                        )
                    };
                    let n = if d == Disposition::ImageData {
                        format!("{n}; bytes after the bitstream's own end are not distinguished")
                    } else {
                        n
                    };
                    let keep = d == Disposition::ImageData && c.truncated && m.d_viable;
                    let n = if keep {
                        format!("{n}; {}", Self::TRUNCATED_NOTE)
                    } else {
                        n
                    };
                    self.emit_ex(None, c, d, Some(format!("{n}{past}")), false, keep)?;
                }
                b"ALPH" => {
                    let used = if m.animated {
                        None
                    } else if m.d_viable {
                        (m.d_last_alph == Some(i)).then_some(
                            if m.alpha_flag && m.native_err.is_none() {
                                Disposition::ImageData
                            } else {
                                Disposition::Dropped
                            },
                        )
                    } else if m.first(b"ALPH") == Some(i)
                        && m.alpha_flag
                        && !chunks
                            .iter()
                            .enumerate()
                            .any(|(j, x)| m.w_in[j] && &x.fourcc == b"VP8L")
                    {
                        Some(Disposition::ImageData)
                    } else {
                        None
                    };
                    match used {
                        Some(d) => {
                            let n = if d == Disposition::Dropped && !m.alpha_flag {
                                "decoded, then discarded: the VP8X alpha flag is clear"
                            } else if d == Disposition::Dropped {
                                "decoded, then discarded: ImageInfo::from_webp fails for this \
                                 file, so decode() outputs RGB"
                            } else {
                                "alpha plane for the VP8 image"
                            };
                            self.emit_alph(c, d, &format!("{n}{past}"), m.canvas)?;
                        }
                        None => {
                            let n = if m.animated {
                                "animated file: alpha is read from the ANMF sub-chunk"
                            } else if m.d_viable {
                                "decode() takes the last ALPH chunk (WebPDemuxer)"
                            } else if !m.alpha_flag {
                                "VP8X alpha flag is clear: ALPH is ignored"
                            } else {
                                "not used: the image is VP8L, or this is a duplicate ALPH"
                            };
                            self.emit(
                                None,
                                c,
                                Disposition::Skipped,
                                Some(format!("{n}{past}")),
                                false,
                            )?;
                        }
                    }
                }
                b"ICCP" | b"EXIF" | b"XMP " => {
                    let kind = match &c.fourcc {
                        b"ICCP" => MetadataKind::Icc,
                        b"EXIF" => MetadataKind::Exif,
                        _ => MetadataKind::Xmp,
                    };
                    // `chunks.entry(chunk).or_insert(range)`: the first one wins and the
                    // VP8X flags are not consulted (`icc_profile`/`exif_metadata`/
                    // `xmp_metadata` -> `ImageInfo::from_webp`).
                    let is_first = m.first(&c.fourcc) == Some(i);
                    let exposed =
                        p_icc && kind == MetadataKind::Icc && !(is_first && m.native_err.is_none());
                    let (d, n): (Disposition, String) = if is_first && m.native_err.is_none() {
                        (Disposition::Metadata(kind), "first occurrence".to_string())
                    } else if is_first {
                        (
                            Disposition::Dropped,
                            "ImageInfo::from_webp fails for this file (see the RIFF header \
                             detail), so decode() reports no metadata"
                                .to_string(),
                        )
                    } else {
                        (Disposition::Skipped, dup_note.to_string())
                    };
                    if exposed && !matches!(d, Disposition::Metadata(_)) {
                        let why = if is_first {
                            "ImageInfo::from_webp fails for this file".to_string()
                        } else {
                            "duplicate: ImageInfo keeps the first ICCP".to_string()
                        };
                        self.emit_ex(
                            None,
                            c,
                            Disposition::Metadata(MetadataKind::Icc),
                            Some(format!("{why}; {}{past}", Self::PROBE_ICC)),
                            false,
                            true,
                        )?;
                    } else {
                        let n = if exposed {
                            format!("{n}; also {}", Self::PROBE_ICC)
                        } else {
                            n
                        };
                        self.emit_ex(None, c, d, Some(format!("{n}{past}")), false, exposed)?;
                    }
                }
                b"ANIM" => {
                    if !m.animated {
                        self.emit(
                            None,
                            c,
                            Disposition::Skipped,
                            Some("animation flag is clear".to_string()),
                            false,
                        )?;
                    } else if m.first(b"ANIM") != Some(i) {
                        self.emit(
                            None,
                            c,
                            Disposition::Skipped,
                            Some(format!("{dup_note}: the decoder keeps the first")),
                            false,
                        )?;
                    } else {
                        self.emit_anim(c)?;
                    }
                }
                b"ANMF" => {
                    if !m.animated && m.first_anmf == Some(i) {
                        self.emit_anmf_still(c, &m.quirk)?;
                    } else if !m.animated {
                        self.emit(
                            None,
                            c,
                            Disposition::Skipped,
                            Some("animation flag is clear".to_string()),
                            false,
                        )?;
                    } else {
                        let reachable =
                            frame_chain_ok && frame_cursor == Some(c.pos) && c.size >= 32;
                        if !reachable {
                            frame_chain_ok = false;
                        }
                        // `read_frame`: next_frame_start += anmf_size + 8, no rounding.
                        frame_cursor = Some(c.pos + 8 + c.size);
                        let id = self.emit_anmf(c, reachable, m.first_anmf == Some(i), &m.quirk)?;
                        if m.p_bitstream == Some((i, true))
                            && let Some(id) = id
                        {
                            self.append_detail(
                                id,
                                "WebPProbe quality/lossless come from this frame's first sub-chunk",
                            );
                        }
                    }
                }
                _ => {
                    let d = self.unknown_detail(c);
                    self.emit(
                        None,
                        c,
                        Disposition::Unknown,
                        Some(format!("{d}{past}")),
                        false,
                    )?;
                }
            }
        }
        // A header cut short inside the RIFF range.
        if end_pos < riff_end && end_pos < len {
            self.truncated_header(None, end_pos, riff_end)?;
        }
        Ok(())
    }

    /// What each walker does with this VP8X file.
    fn model(&self, vp8x: &RawChunk, chunks: &[RawChunk], riff_size: u64) -> Model {
        let len = self.len;
        let s = vp8x.payload.start;
        // `read_extended_header` reads 10 bytes at the payload start whatever
        // the declared VP8X size is.
        let have10 = s + 10 <= len;
        let (flags, canvas) = if have10 {
            let b = &self.data[s as usize..s as usize + 10];
            let w = 1 + (u32::from(b[4]) | u32::from(b[5]) << 8 | u32::from(b[6]) << 16);
            let h = 1 + (u32::from(b[7]) | u32::from(b[8]) << 8 | u32::from(b[9]) << 16);
            (b[0], (w, h))
        } else {
            (0, (0, 0))
        };
        let animated = flags & 0x02 != 0;
        let alpha_flag = flags & 0x10 != 0;
        let exif_flag = flags & 0x08 != 0;
        let xmp_flag = flags & 0x04 != 0;

        // --- WebPDecoder::read_data ------------------------------------------------
        // The limits are the decoder's own (`WebPDecoder::new` uses
        // `Limits::default()`), called in the order `read_data` calls them.
        let limits = crate::decoder::Limits::default();
        let mut native_err: Option<String> = limits
            .check_file_size(len)
            .err()
            .map(|e| e.error().to_string());
        if native_err.is_none() && !have10 {
            native_err = Some("VP8X header is truncated".to_string());
        } else if native_err.is_none() && vp8x.next > len {
            native_err = Some("the VP8X chunk runs past the end of the file".to_string());
        } else if native_err.is_none() && canvas.0.checked_mul(canvas.1).is_none() {
            // `read_extended_header`: the pixel count must fit in u32.
            native_err = Some("canvas pixel count overflows u32 (ImageTooLarge)".to_string());
        } else if native_err.is_none()
            && let Err(e) = limits.check_dimensions(canvas.0, canvas.1)
        {
            native_err = Some(e.error().to_string());
        }
        let limit = vp8x.next.saturating_add(riff_size.saturating_sub(12));
        let mut w_in = vec![false; chunks.len()];
        let mut frames = 0usize;
        // An error before the loop (file size, canvas limits) leaves `w_in` as
        // the loop would have set it, so "first occurrence" stays meaningful.
        let pre_err = native_err.is_some();
        for (i, c) in chunks.iter().enumerate() {
            if c.pos >= limit || (!pre_err && native_err.is_some()) {
                break;
            }
            w_in[i] = true;
            if pre_err {
                continue;
            }
            if &c.fourcc == b"ANMF" {
                frames += 1;
                if let Err(e) = limits.check_frame_count(frames) {
                    native_err = Some(e.error().to_string());
                    break;
                }
                if c.size < 24 {
                    native_err = Some(format!("ANMF at {} is smaller than 24 bytes", c.pos));
                    break;
                }
            }
            if c.next > len {
                native_err = Some(format!(
                    "chunk {} at {} runs past the end of the file",
                    fourcc_label(&c.fourcc),
                    c.pos
                ));
                break;
            }
        }
        let first_of = |f: &[u8; 4]| (0..chunks.len()).find(|&i| w_in[i] && &chunks[i].fourcc == f);
        if native_err.is_none() {
            let has = |f: &[u8; 4]| first_of(f).is_some();
            if animated && (!has(b"ANIM") || !has(b"ANMF")) {
                native_err = Some("animation flag set but ANIM or ANMF is missing".to_string());
            } else if exif_flag && !has(b"EXIF") {
                native_err = Some("EXIF flag set but there is no EXIF chunk".to_string());
            } else if xmp_flag && !has(b"XMP ") {
                native_err = Some("XMP flag set but there is no XMP chunk".to_string());
            } else if !animated && has(b"VP8 ") == has(b"VP8L") {
                native_err = Some(
                    "a still needs exactly one of VP8 and VP8L (both or neither present)"
                        .to_string(),
                );
            } else if animated
                && let Some(i) = first_of(b"ANIM")
                && chunks[i].size != 6
            {
                native_err = Some(format!(
                    "ANIM payload is {} bytes, the decoder needs exactly 6",
                    chunks[i].size
                ));
            }
        }
        // The first ANMF's sub-chunk scan (`read_data`, "store the ALPH, VP8,
        // and VP8L chunks of the first frame"): it reads a header, then reads the
        // next one from the wrong place unless the first sub-chunk is empty, and
        // records whatever FourCCs it sees, ICCP/EXIF/XMP included.
        let first_anmf = first_of(b"ANMF");
        let mut quirk: Vec<([u8; 4], Range<u64>)> = Vec::new();
        if native_err.is_none()
            && let Some(ai) = first_anmf
        {
            let a = &chunks[ai];
            let range_end = a.pos + 8 + a.size;
            let mut position = a.pos + 8 + 16;
            let mut rp = position;
            let mut seen: Vec<[u8; 4]> = Vec::new();
            if position > len {
                native_err = Some("first ANMF scan seeks past the end of the file".to_string());
            } else {
                for _ in 0..2 {
                    let Some(h) = self.read_chunk(rp, len, len) else {
                        native_err =
                            Some("first ANMF sub-chunk scan runs off the end of the file".into());
                        break;
                    };
                    let size = h.size;
                    rp += 8;
                    let sub = position + 8..position + 8 + size;
                    if first_of(&h.fourcc).is_none() && !seen.contains(&h.fourcc) {
                        seen.push(h.fourcc);
                        if matches!(&h.fourcc, b"ICCP" | b"EXIF" | b"XMP ") && sub.end <= len {
                            quirk.push((h.fourcc, sub));
                        }
                    }
                    position += 8 + size + (size & 1);
                    if position + 8 > range_end {
                        break;
                    }
                }
            }
        }

        // --- WebPDemuxer (non-animated lossy stills) -------------------------------
        let max_pos = 8 + riff_size;
        let d_in: Vec<bool> = chunks.iter().map(|c| c.pos + 8 <= max_pos).collect();
        let last_of = |pred: &dyn Fn(&RawChunk) -> bool| {
            (0..chunks.len())
                .rev()
                .find(|&i| d_in[i] && pred(&chunks[i]))
        };
        let d_last_img = last_of(&|c| matches!(&c.fourcc, b"VP8 " | b"VP8L"));
        let d_last_alph = last_of(&|c| &c.fourcc == b"ALPH");
        // The demuxer clamps a chunk that runs past EOF; whether the bytes
        // present suffice is a bitstream question (see `TRUNCATED_NOTE`).
        let d_viable = !animated
            && vp8x.size >= 10
            && d_last_img.is_some_and(|i| &chunks[i].fourcc == b"VP8 ");

        // --- detect::probe ----------------------------------------------------------
        let all: Vec<&RawChunk> = core::iter::once(vp8x).chain(chunks.iter()).collect();
        let det = self.detect_model(&all);
        let p_icc = det
            .as_ref()
            .and_then(|d| d.icc)
            .and_then(|i| i.checked_sub(1));
        let p_bitstream = det
            .as_ref()
            .and_then(|d| d.owner.checked_sub(1).map(|o| (o, d.owner_in_anmf)));

        let rejected = if d_viable { None } else { native_err.clone() };
        let firsts: Vec<([u8; 4], Option<usize>)> = [
            b"VP8 ", b"VP8L", b"ALPH", b"ICCP", b"EXIF", b"XMP ", b"ANIM", b"ANMF",
        ]
        .iter()
        .map(|f| (**f, first_of(f)))
        .collect();
        Model {
            flags,
            animated,
            alpha_flag,
            canvas,
            w_in,
            native_err,
            rejected,
            d_viable,
            d_last_img,
            d_last_alph,
            first_anmf,
            quirk,
            p_icc,
            p_bitstream,
            firsts,
        }
    }

    /// The VP8X chunk: flags, the three reserved bytes (read and discarded) and
    /// the canvas size, with anything past the 10 bytes the decoder reads left
    /// `Unreferenced`.
    fn emit_vp8x(&mut self, vp8x: &RawChunk, m: &Model) -> Result<PartId, DecodeError> {
        let s = vp8x.payload.start;
        let payload_len = vp8x.payload.end - s;
        let f = m.flags;
        let note = format!(
            "flags=0x{f:02x} (icc={} alpha={} exif={} xmp={} anim={}), canvas {}x{}",
            u8::from(f & 0x20 != 0),
            u8::from(f & 0x10 != 0),
            u8::from(f & 0x08 != 0),
            u8::from(f & 0x04 != 0),
            u8::from(f & 0x02 != 0),
            m.canvas.0,
            m.canvas.1
        );
        if vp8x.truncated || s + 10 > self.len {
            return self.emit(
                None,
                vp8x,
                Disposition::Malformed,
                Some("VP8X header is truncated".to_string()),
                false,
            );
        }
        if payload_len < 10 {
            return self.emit(
                None,
                vp8x,
                Disposition::Structure,
                Some(format!(
                    "payload is {payload_len} bytes but the decoder reads 10 (flags and canvas \
                     size continue into the following bytes); {note}"
                )),
                false,
            );
        }
        let id = self.emit(None, vp8x, Disposition::Structure, Some(note), true)?;
        self.child(id, "flags", s..s + 1, Disposition::Structure, None)?;
        self.child(
            id,
            "reserved",
            s + 1..s + 4,
            Disposition::Dropped,
            Some("read into _reserved_bytes and discarded".to_string()),
        )?;
        self.child(id, "canvas", s + 4..s + 10, Disposition::Structure, None)?;
        self.inv
            .fill_gaps(Some(id), Disposition::Unreferenced)
            .map_err(parts_err)?;
        Ok(id)
    }

    /// ALPH: with compression 0 the plane is `width * height` bytes after a
    /// 1-byte header, and anything beyond it is never read.
    fn emit_alph(
        &mut self,
        c: &RawChunk,
        disposition: Disposition,
        detail: &str,
        canvas: (u32, u32),
    ) -> Result<(), DecodeError> {
        let s = c.payload.start;
        let plane = 1 + u64::from(canvas.0) * u64::from(canvas.1);
        let uncompressed =
            c.payload.end > s && self.data.get(s as usize).is_some_and(|b| b & 3 == 0);
        if uncompressed && !c.truncated && s + plane < c.payload.end {
            let id = self.emit(
                None,
                c,
                Disposition::Structure,
                Some(detail.to_string()),
                true,
            )?;
            self.child(id, "alpha plane", s..s + plane, disposition, None)?;
            self.inv
                .fill_gaps(Some(id), Disposition::Unreferenced)
                .map_err(parts_err)?;
        } else {
            let note = if c.truncated {
                format!("{detail}; {}", Self::TRUNCATED_NOTE)
            } else if uncompressed {
                format!("{detail}; uncompressed alpha plane with no bytes after it")
            } else {
                format!(
                    "{detail}; compressed alpha: bytes after the stream's own end are not \
                     distinguished"
                )
            };
            self.emit_ex(None, c, disposition, Some(note), false, true)?;
        }
        Ok(())
    }

    /// ANIM: background colour (native API only) + loop count (zencodec
    /// `ImageSequence::Animation::loop_count`, `codec.rs`
    /// `animation_frame_decoder`).
    fn emit_anim(&mut self, c: &RawChunk) -> Result<(), DecodeError> {
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
        self.child(
            id,
            "background_color",
            s..s + 4,
            Disposition::Dropped,
            Some(
                "native WebPDecoder::background_color_hint only; the zencodec path does not \
                 report it"
                    .to_string(),
            ),
        )?;
        self.child(
            id,
            "loop_count",
            s + 4..s + 6,
            Disposition::Metadata(MetadataKind::Animation),
            None,
        )?;
        Ok(())
    }

    /// ANMF: 16-byte frame header, then sub-chunks.
    ///
    /// `read_data` reads every frame's header (rectangle, duration, dispose
    /// flag), its first sub-chunk's header, and for VP8L five more bytes (the
    /// `alpha_is_used` bit); those feed `has_alpha`, the frame count and the
    /// loop duration, reachable or not. `read_frame` then decodes a frame:
    /// one image chunk (`VP8`, `VP8L`, or `ALPH` followed by whatever
    /// chunk, decoded as `VP8`). Later sub-chunks are never read.
    fn emit_anmf(
        &mut self,
        c: &RawChunk,
        reachable: bool,
        is_first: bool,
        quirk: &[([u8; 4], Range<u64>)],
    ) -> Result<Option<PartId>, DecodeError> {
        self.emit_anmf_ex(c, reachable, is_first, quirk, false)
    }

    /// An ANMF in a file whose animation flag is clear. `read_frame` never
    /// runs for it, but `read_data` still reads its header and runs the
    /// first-ANMF sub-chunk scan, whose metadata reaches `ImageInfo`.
    fn emit_anmf_still(
        &mut self,
        c: &RawChunk,
        quirk: &[([u8; 4], Range<u64>)],
    ) -> Result<(), DecodeError> {
        self.emit_anmf_ex(c, false, true, quirk, true).map(|_| ())
    }

    fn emit_anmf_ex(
        &mut self,
        c: &RawChunk,
        reachable: bool,
        is_first: bool,
        quirk: &[([u8; 4], Range<u64>)],
        still: bool,
    ) -> Result<Option<PartId>, DecodeError> {
        let payload_len = c.payload.end - c.payload.start;
        if c.truncated || payload_len < 16 {
            self.emit(
                None,
                c,
                Disposition::Malformed,
                Some("ANMF shorter than its 16-byte frame header".to_string()),
                false,
            )?;
            return Ok(None);
        }
        let note = if still {
            Some(
                "animation flag is clear: read_frame never runs; read_data still reads this \
                 frame's header and runs its first-ANMF sub-chunk scan"
                    .to_string(),
            )
        } else if reachable {
            None
        } else {
            Some(
                "unreachable: read_frame advances by the unrounded ANMF size and expects the \
                 next ANMF header there (and needs size >= 32), so this frame and every later \
                 one fail to decode; read_data still reads its header"
                    .to_string(),
            )
        };
        let container = if still {
            Disposition::Skipped
        } else {
            Disposition::Structure
        };
        let id = self.emit(None, c, container, note, true)?;
        let s = c.payload.start;
        let header_disp = if still {
            Disposition::Dropped
        } else {
            Disposition::Metadata(MetadataKind::Animation)
        };
        let header_detail = if still {
            "parsed by read_data, reaches no caller (animation flag is clear)"
        } else if reachable {
            "frame x/y/size, duration, blend and dispose flags"
        } else {
            "read_data reads this header (rectangle, duration, dispose bit feed has_alpha and the \
             loop duration) although read_frame cannot reach the frame"
        };
        self.push(
            Some(id),
            Part::new(
                PartKind::Header,
                PartTag::Name("frame header".into()),
                s..s + 16,
                header_disp,
            )
            .with_detail(header_detail),
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
        let mut placed = vec![false; quirk.len()];
        for (i, sc) in subs.iter().enumerate() {
            // The first ANMF's sub-chunk scan files the bytes it misreads as
            // ICCP/EXIF/XMP, which `ImageInfo` then reports as the file's
            // metadata. Its second header is read from inside the first
            // sub-chunk, so the range can be any chunk's payload.
            let mut hits: Vec<usize> = Vec::new();
            if is_first && !sc.truncated {
                for (qi, (_, r)) in quirk.iter().enumerate() {
                    let disjoint = |o: &Range<u64>| r.end <= o.start || r.start >= o.end;
                    if !placed[qi]
                        && r.start >= sc.payload.start
                        && r.end <= sc.payload.end
                        && hits.iter().all(|&h| disjoint(&quirk[h].1))
                    {
                        hits.push(qi);
                    }
                }
            }
            let kind_of = |f: &[u8; 4]| match f {
                b"ICCP" => MetadataKind::Icc,
                b"EXIF" => MetadataKind::Exif,
                _ => MetadataKind::Xmp,
            };
            if let Some(&qi) = hits.iter().find(|&&qi| quirk[qi].1 == sc.payload) {
                placed[qi] = true;
                let f = quirk[qi].0;
                let note = if f == sc.fourcc {
                    "read_data's first-ANMF sub-chunk scan files this under its type and \
                     ImageInfo reports it as the file's metadata (no type filter)"
                        .to_string()
                } else {
                    format!(
                        "read_data's first-ANMF scan misreads a header and files these bytes \
                         as {} although this chunk is {}; ImageInfo reports them as the \
                         file's metadata",
                        fourcc_label(&f),
                        fourcc_label(&sc.fourcc)
                    )
                };
                self.emit(
                    Some(id),
                    sc,
                    Disposition::Metadata(kind_of(&f)),
                    Some(note),
                    false,
                )?;
                continue;
            }
            hits.retain(|&qi| quirk[qi].1.start < quirk[qi].1.end);
            if hits.is_empty() && !reachable && !still {
                if i == 0 {
                    self.emit_unreachable_first_sub(id, sc, c.size)?;
                } else {
                    self.unread(Some(id), sc, "frame is not reachable")?;
                }
                continue;
            }
            let after_alph = i == 1 && subs[0].fourcc == *b"ALPH";
            let (mut d, mut n): (Disposition, String) = if still {
                (
                    Disposition::Skipped,
                    "animation flag is clear: read_frame never runs".to_string(),
                )
            } else if !reachable {
                (Disposition::Skipped, "frame is not reachable".to_string())
            } else {
                match (i, &sc.fourcc) {
                    (0, b"VP8 " | b"VP8L") => (
                        Disposition::ImageData,
                        "frame image; bytes after the bitstream's own end are not distinguished"
                            .to_string(),
                    ),
                    (0, b"ALPH") => (
                        Disposition::ImageData,
                        "frame alpha plane; bytes after the stream's own end are not distinguished"
                            .to_string(),
                    ),
                    _ if after_alph => (
                        Disposition::ImageData,
                        if &sc.fourcc == b"VP8 " {
                            "frame image; bytes after the bitstream's own end are not \
                             distinguished"
                                .to_string()
                        } else {
                            "read_frame decodes the chunk after ALPH as VP8 whatever its FourCC; \
                             bytes after the bitstream's own end are not distinguished"
                                .to_string()
                        },
                    ),
                    (0, _) => (
                        Disposition::Malformed,
                        "read_frame needs VP8, VP8L or ALPH as the first sub-chunk".to_string(),
                    ),
                    _ => (
                        Disposition::Skipped,
                        "read_frame reads only the frame's image chunk(s)".to_string(),
                    ),
                }
            };
            if d == Disposition::Skipped && !sc.is_known() {
                n = self.unknown_detail(sc);
                d = Disposition::Unknown;
            }
            if hits.is_empty() {
                self.emit(Some(id), sc, d, Some(n), false)?;
                continue;
            }
            // Misread ranges inside this chunk's payload.
            hits.sort_by_key(|&qi| quirk[qi].1.start);
            let sid = self.emit(Some(id), sc, d, Some(n), true)?;
            for &qi in &hits {
                placed[qi] = true;
                let (f, r) = &quirk[qi];
                self.child(
                    sid,
                    "first-ANMF scan range",
                    r.clone(),
                    Disposition::Metadata(kind_of(f)),
                    Some(format!(
                        "read_data's first-ANMF scan misreads a header inside this chunk's \
                         payload and files these bytes as {}; ImageInfo reports them as the \
                         file's metadata",
                        fourcc_label(f)
                    )),
                )?;
            }
            let gap = if d.is_consumed() {
                d
            } else {
                Disposition::Skipped
            };
            self.inv.fill_gaps(Some(sid), gap).map_err(parts_err)?;
        }
        for (qi, (f, r)) in quirk.iter().enumerate() {
            if is_first && !placed[qi] {
                self.append_detail(
                    id,
                    &format!(
                        "read_data's first-ANMF scan also files bytes {}..{} as {}, which \
                         ImageInfo reports as the file's metadata",
                        r.start,
                        r.end,
                        fourcc_label(f)
                    ),
                );
            }
        }
        self.inv
            .fill_gaps(Some(id), Disposition::Unreferenced)
            .map_err(parts_err)?;
        Ok(Some(id))
    }

    /// First sub-chunk of an unreachable frame: `read_data` reads its 8-byte
    /// header (and 5 payload bytes of a VP8L, when the ANMF is at least 29
    /// bytes); the rest of the payload is never read.
    fn emit_unreachable_first_sub(
        &mut self,
        parent: PartId,
        sc: &RawChunk,
        anmf_size: u64,
    ) -> Result<(), DecodeError> {
        let s = sc.payload.start;
        let plen = sc.payload.end - s;
        if sc.truncated || plen == 0 {
            self.emit(
                Some(parent),
                sc,
                Disposition::Structure,
                Some("header read by read_data; frame is not reachable".to_string()),
                false,
            )?;
            return Ok(());
        }
        let id = self.emit(
            Some(parent),
            sc,
            Disposition::Structure,
            Some(
                "header read by read_data; the payload is never read (frame is not reachable)"
                    .to_string(),
            ),
            true,
        )?;
        if &sc.fourcc == b"VP8L" && anmf_size >= 29 && plen >= 5 {
            self.child(
                id,
                "alpha_is_used header",
                s..s + 5,
                Disposition::Structure,
                Some("signature and size word; bit 28 sets the file's has_alpha".to_string()),
            )?;
        }
        self.inv
            .fill_gaps(Some(id), Disposition::Skipped)
            .map_err(parts_err)
    }
}

/// `detect::probe`'s outcome when it succeeds (see `detect_model`).
struct DetectModel {
    icc: Option<usize>,
    owner: usize,
    owner_in_anmf: bool,
}

/// What each walker does with a VP8X file (see `walk_extended`).
struct Model {
    flags: u8,
    animated: bool,
    alpha_flag: bool,
    canvas: (u32, u32),
    /// Chunks `read_data`'s loop reads (up to its failure point).
    w_in: Vec<bool>,
    /// Why `read_data` fails (so `probe()` and `ImageInfo` fail), if it does.
    native_err: Option<String>,
    /// The file is rejected by every entry point.
    rejected: Option<String>,
    /// `decode()` takes the pixels from `WebPDemuxer` (non-animated, last image chunk is VP8).
    d_viable: bool,
    d_last_img: Option<usize>,
    d_last_alph: Option<usize>,
    first_anmf: Option<usize>,
    /// ICCP/EXIF/XMP ranges filed by the first-ANMF sub-chunk scan.
    quirk: Vec<([u8; 4], Range<u64>)>,
    /// The ICCP `detect::probe` keeps.
    p_icc: Option<usize>,
    /// The chunk `detect::probe` takes its quality from (index, inside an ANMF).
    p_bitstream: Option<(usize, bool)>,
    firsts: Vec<([u8; 4], Option<usize>)>,
}

impl Model {
    /// First chunk of this type that `read_data` stored (`or_insert`).
    fn first(&self, f: &[u8; 4]) -> Option<usize> {
        self.firsts
            .iter()
            .find(|(g, _)| g == f)
            .and_then(|(_, i)| *i)
    }
}

/// Part cap used to bound the intermediate chunk lists.
pub(crate) fn part_budget() -> u32 {
    zencodec::inventory::DEFAULT_MAX_PARTS
}
