//! VP8 keyframe parser → structured frame with **quantized coefficient
//! levels** (not dequantized; no IDCT/WHT). This is the input the
//! coefficient editor and re-emitter operate on.
//!
//! Mirrors zenwebp's decoder parse order
//! (`src/decoder/vp8v2/{header,coefficients}.rs`) exactly, but stores the
//! raw signed levels each `read_coefficients` produces rather than
//! dequantizing + transforming. Keyframe-only; non-keyframe / unsupported
//! layouts return `None` (the strategy then falls back to `Reencode`).

use super::bool::BoolDecoder;
use super::tables::*;

/// Per-macroblock parsed data needed to re-emit the token stream.
#[derive(Clone)]
pub struct Mb {
    pub segment_id: u8,
    pub skip: bool,
    /// Luma mode: 0=DC,1=V,2=H,3=TM,4=B(per-subblock).
    pub luma_mode: i8,
    /// Sub-block modes (only meaningful when `luma_mode == B_PRED`).
    pub bpred: [i8; 16],
    pub chroma_mode: i8,
    /// 25 blocks × 16 coefficients: [Y2][16×Y][8×UV] in natural order.
    /// Index 0 = Y2 (present only when luma_mode != B). 1..=16 = Y.
    /// 17..=24 = U(4) then V(4). Each block is 16 quantized levels in
    /// natural (de-zigzagged) order, matching the decoder's `output[ZIGZAG[n]]`.
    pub coeffs: Box<[[i32; 16]; 25]>,
}

impl Mb {
    fn new() -> Self {
        Mb {
            segment_id: 0,
            skip: false,
            luma_mode: DC_PRED,
            bpred: [B_DC_PRED; 16],
            chroma_mode: DC_PRED,
            coeffs: Box::new([[0i32; 16]; 25]),
        }
    }
    pub fn has_y2(&self) -> bool {
        self.luma_mode != B_PRED
    }
}

/// Parsed VP8 keyframe.
pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub mb_cols: usize,
    pub mb_rows: usize,
    /// Raw bytes of the uncompressed 10-byte frame tag + start code + dims,
    /// reproduced verbatim on emit.
    pub keyframe_header: Vec<u8>,
    /// Verbatim copy of partition-0 (the first/control partition) bytes as
    /// they appeared after the 10-byte header. We re-emit the header bytes
    /// 1:1 for now (verbatim path); the editor rewrites partition 0 + token
    /// partitions from the parsed structures.
    pub part0: Vec<u8>,
    // Header fields needed to re-encode the bitstream.
    pub color_space: u32,
    pub clamp_type: u32,
    pub segmentation: Segmentation,
    pub filter_type: bool,
    pub filter_level: u32,
    pub sharpness: u32,
    pub lf_adjust: LoopFilterAdjust,
    pub num_partitions: usize,
    pub quant: QuantHeader,
    pub refresh_entropy_probs: u32,
    pub coeff_probs: TokenProbTables,
    pub mb_no_skip_coeff: u32,
    pub prob_skip_false: Option<u8>,
    pub mbs: Vec<Mb>,
}

#[derive(Clone, Default)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub update_data: bool,
    pub abs_values: bool, // !delta mode
    pub quantizer: [i8; 4],
    pub quantizer_present: [bool; 4],
    pub loopfilter: [i8; 4],
    pub loopfilter_present: [bool; 4],
    pub tree_probs: [u8; 3],
    pub tree_probs_present: [bool; 3],
}

#[derive(Clone, Default)]
pub struct LoopFilterAdjust {
    pub enabled: bool,
    pub update: bool,
    pub ref_delta: [Option<i32>; 4],
    pub mode_delta: [Option<i32>; 4],
}

#[derive(Clone, Default)]
pub struct QuantHeader {
    pub yac_abs: u32,
    pub ydc_delta: Option<i32>,
    pub y2dc_delta: Option<i32>,
    pub y2ac_delta: Option<i32>,
    pub uvdc_delta: Option<i32>,
    pub uvac_delta: Option<i32>,
}

/// Neighbor non-zero context (complexity), layout y2,y,y,y,y,u,u,v,v.
#[derive(Clone, Copy)]
struct Ctx {
    complexity: [u8; 9],
    bpred: [i8; 4],
}
impl Ctx {
    fn new() -> Self {
        Ctx {
            complexity: [0; 9],
            bpred: [B_DC_PRED; 4],
        }
    }
}

/// Read an optional signed value: flag, then `n`-bit magnitude + sign.
fn read_opt_signed(b: &mut BoolDecoder, n: u8) -> Option<i32> {
    if b.get_flag() {
        Some(b.get_signed_literal(n))
    } else {
        None
    }
}

/// Parse a VP8 keyframe bitstream (the contents of the `VP8 ` chunk).
/// Returns `None` for non-keyframes or layouts we don't handle.
pub fn parse_vp8_keyframe(vp8: &[u8]) -> Option<Frame> {
    if vp8.len() < 10 {
        return None;
    }
    let tag = u32::from(vp8[0]) | (u32::from(vp8[1]) << 8) | (u32::from(vp8[2]) << 16);
    let keyframe = (tag & 1) == 0;
    if !keyframe {
        return None;
    }
    let first_part_size = (tag >> 5) as usize;
    if vp8[3..6] != [0x9d, 0x01, 0x2a] {
        return None;
    }
    let w16 = u16::from(vp8[6]) | (u16::from(vp8[7]) << 8);
    let h16 = u16::from(vp8[8]) | (u16::from(vp8[9]) << 8);
    let width = w16 & 0x3fff;
    let height = h16 & 0x3fff;
    let mb_cols = (width as usize).div_ceil(16);
    let mb_rows = (height as usize).div_ceil(16);

    let header_end = 10usize;
    let part0_start = header_end;
    let part0_end = part0_start.checked_add(first_part_size)?;
    if part0_end > vp8.len() {
        return None;
    }
    let part0 = &vp8[part0_start..part0_end];
    let after_part0 = &vp8[part0_end..];

    let mut b = BoolDecoder::new(part0);

    let color_space = b.get_bit(128);
    let clamp_type = b.get_bit(128);

    // --- Segmentation ---
    let mut seg = Segmentation {
        enabled: b.get_flag(),
        ..Default::default()
    };
    if seg.enabled {
        seg.update_map = b.get_flag();
        seg.update_data = b.get_flag();
        if seg.update_data {
            seg.abs_values = b.get_flag();
            for i in 0..4 {
                if b.get_flag() {
                    seg.quantizer[i] = b.get_signed_literal(7) as i8;
                    seg.quantizer_present[i] = true;
                }
            }
            for i in 0..4 {
                if b.get_flag() {
                    seg.loopfilter[i] = b.get_signed_literal(6) as i8;
                    seg.loopfilter_present[i] = true;
                }
            }
        }
        if seg.update_map {
            for i in 0..3 {
                if b.get_flag() {
                    seg.tree_probs[i] = b.get_literal(8) as u8;
                    seg.tree_probs_present[i] = true;
                } else {
                    seg.tree_probs[i] = 255;
                }
            }
        }
    }

    // --- Filter header ---
    let filter_type = b.get_flag();
    let filter_level = b.get_literal(6);
    let sharpness = b.get_literal(3);
    let mut lf = LoopFilterAdjust {
        enabled: b.get_flag(),
        ..Default::default()
    };
    if lf.enabled {
        lf.update = b.get_flag();
        if lf.update {
            for i in 0..4 {
                lf.ref_delta[i] = read_opt_signed(&mut b, 6);
            }
            for i in 0..4 {
                lf.mode_delta[i] = read_opt_signed(&mut b, 6);
            }
        }
    }

    // --- Partitions ---
    let log2_parts = b.get_literal(2);
    let num_partitions = 1usize << log2_parts;
    let token_parts = split_token_partitions(after_part0, num_partitions)?;

    // --- Quantization indices ---
    let quant = QuantHeader {
        yac_abs: b.get_literal(7),
        ydc_delta: read_opt_signed(&mut b, 4),
        y2dc_delta: read_opt_signed(&mut b, 4),
        y2ac_delta: read_opt_signed(&mut b, 4),
        uvdc_delta: read_opt_signed(&mut b, 4),
        uvac_delta: read_opt_signed(&mut b, 4),
    };

    // --- Refresh entropy probs (keyframe: 1 bit) ---
    let refresh_entropy_probs = b.get_bit(128);

    // --- Token probability updates ---
    let mut coeff_probs = COEFF_PROBS;
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for t in 0..11 {
                    if b.get_bit(COEFF_UPDATE_PROBS[i][j][k][t]) != 0 {
                        coeff_probs[i][j][k][t] = b.get_literal(8) as u8;
                    }
                }
            }
        }
    }

    // --- mb_no_skip_coeff ---
    let mb_no_skip_coeff = b.get_bit(128);
    let prob_skip_false = if mb_no_skip_coeff == 1 {
        Some(b.get_literal(8) as u8)
    } else {
        None
    };

    // --- Per-MB parse ---
    let mut mbs = Vec::with_capacity(mb_cols * mb_rows);
    let mut top: Vec<Ctx> = vec![Ctx::new(); mb_cols];
    // One token reader per partition.
    let mut readers: Vec<BoolDecoder> = token_parts.iter().map(|p| BoolDecoder::new(p)).collect();

    for mb_row in 0..mb_rows {
        let mut left = Ctx::new();
        let part = mb_row % num_partitions;
        for top_ctx in top.iter_mut() {
            let mut mb = Mb::new();
            parse_mb_header(&mut b, &seg, &prob_skip_false, top_ctx, &mut left, &mut mb);
            if !mb.skip {
                parse_mb_coeffs(
                    &mut readers[part],
                    &coeff_probs,
                    top_ctx,
                    &mut left,
                    &mut mb,
                );
            } else {
                // Skipped MB: clear contexts (no Y2 for B mode stays as-is).
                clear_skip_context(top_ctx, &mut left, mb.has_y2());
            }
            mbs.push(mb);
        }
    }

    Some(Frame {
        width,
        height,
        mb_cols,
        mb_rows,
        keyframe_header: vp8[..10].to_vec(),
        part0: part0.to_vec(),
        color_space,
        clamp_type,
        segmentation: seg,
        filter_type,
        filter_level,
        sharpness,
        lf_adjust: lf,
        num_partitions,
        quant,
        refresh_entropy_probs,
        coeff_probs,
        mb_no_skip_coeff,
        prob_skip_false,
        mbs,
    })
}

fn split_token_partitions(data: &[u8], n: usize) -> Option<Vec<&[u8]>> {
    let mut parts = Vec::with_capacity(n);
    if n > 1 {
        let table = 3 * (n - 1);
        if table > data.len() {
            return None;
        }
        let (sizes, mut rest) = data.split_at(table);
        for s in sizes.chunks(3) {
            let size = (s[0] as usize) | ((s[1] as usize) << 8) | ((s[2] as usize) << 16);
            if size > rest.len() {
                return None;
            }
            let (p, r) = rest.split_at(size);
            parts.push(p);
            rest = r;
        }
        parts.push(rest);
    } else {
        parts.push(data);
    }
    Some(parts)
}

fn read_segment_id(b: &mut BoolDecoder, probs: &[u8; 3]) -> u8 {
    if b.get_bit(probs[0]) == 0 {
        if b.get_bit(probs[1]) == 0 { 0 } else { 1 }
    } else if b.get_bit(probs[2]) == 0 {
        2
    } else {
        3
    }
}

fn read_ymode(b: &mut BoolDecoder) -> i8 {
    // Tree [-B_PRED, 2, 4, 6, -DC, -V, -H, -TM], probs 145,156,163,128.
    if b.get_bit(145) == 0 {
        B_PRED
    } else if b.get_bit(156) == 0 {
        if b.get_bit(163) == 0 { DC_PRED } else { V_PRED }
    } else if b.get_bit(128) == 0 {
        H_PRED
    } else {
        TM_PRED
    }
}

fn read_uvmode(b: &mut BoolDecoder) -> i8 {
    if b.get_bit(142) == 0 {
        DC_PRED
    } else if b.get_bit(114) == 0 {
        V_PRED
    } else if b.get_bit(183) == 0 {
        H_PRED
    } else {
        TM_PRED
    }
}

fn read_bmode(b: &mut BoolDecoder, probs: &[u8; 9]) -> i8 {
    if b.get_bit(probs[0]) == 0 {
        B_DC_PRED
    } else if b.get_bit(probs[1]) == 0 {
        B_TM_PRED
    } else if b.get_bit(probs[2]) == 0 {
        B_VE_PRED
    } else if b.get_bit(probs[3]) == 0 {
        if b.get_bit(probs[4]) == 0 {
            B_HE_PRED
        } else if b.get_bit(probs[5]) == 0 {
            B_RD_PRED
        } else {
            B_VR_PRED
        }
    } else if b.get_bit(probs[6]) == 0 {
        B_LD_PRED
    } else if b.get_bit(probs[7]) == 0 {
        B_VL_PRED
    } else if b.get_bit(probs[8]) == 0 {
        B_HD_PRED
    } else {
        B_HU_PRED
    }
}

fn parse_mb_header(
    b: &mut BoolDecoder,
    seg: &Segmentation,
    prob_skip_false: &Option<u8>,
    top: &mut Ctx,
    left: &mut Ctx,
    mb: &mut Mb,
) {
    if seg.enabled && seg.update_map {
        mb.segment_id = read_segment_id(b, &seg.tree_probs);
    }
    mb.skip = if let Some(p) = prob_skip_false {
        b.get_bit(*p) != 0
    } else {
        false
    };
    mb.luma_mode = read_ymode(b);
    if mb.luma_mode == B_PRED {
        for y in 0..4 {
            for x in 0..4 {
                let tm = top.bpred[x];
                let lm = left.bpred[y];
                let bmode = read_bmode(b, &KEYFRAME_BPRED_MODE_PROBS[tm as usize][lm as usize]);
                mb.bpred[x + y * 4] = bmode;
                top.bpred[x] = bmode;
                left.bpred[y] = bmode;
            }
        }
    } else {
        let m = mb.luma_mode;
        for i in 0..4 {
            mb.bpred[12 + i] = m;
            left.bpred[i] = m;
        }
    }
    mb.chroma_mode = read_uvmode(b);
    top.bpred = [mb.bpred[12], mb.bpred[13], mb.bpred[14], mb.bpred[15]];
}

/// Read one block's 16 quantized levels (natural order). Returns whether
/// any non-zero coefficient was present (the next-block context). Mirrors
/// `read_coefficients` but stores raw signed levels (no dequant).
fn read_block_levels(
    r: &mut BoolDecoder,
    out: &mut [i32; 16],
    plane: &[[[u8; 11]; 3]; 8],
    first: usize,
    complexity: usize,
) -> bool {
    // Probabilities are band-indexed: band = COEFF_BANDS[position].
    let mut n = first;
    let mut prob = &plane[COEFF_BANDS[n] as usize][complexity];
    while n < 16 {
        if r.get_bit(prob[0]) == 0 {
            break;
        }
        while r.get_bit(prob[1]) == 0 {
            n += 1;
            if n >= 16 {
                return true;
            }
            prob = &plane[COEFF_BANDS[n] as usize][0];
        }
        let v: i32;
        let next_ctx: usize;
        if r.get_bit(prob[2]) == 0 {
            v = 1;
            next_ctx = 1;
        } else {
            if r.get_bit(prob[3]) == 0 {
                if r.get_bit(prob[4]) == 0 {
                    v = 2;
                } else {
                    v = 3 + r.get_bit(prob[5]) as i32;
                }
            } else {
                v = read_large_value(r, prob);
            }
            next_ctx = 2;
        }
        // Signed level, stored at the de-zigzagged position.
        let level = if r.get_flag() { -v } else { v };
        out[ZIGZAG[n] as usize] = level;
        n += 1;
        if n < 16 {
            prob = &plane[COEFF_BANDS[n] as usize][next_ctx];
        }
    }
    n > first
}

/// Extra-bit counts for DCT categories 3,4,5,6 (indexed cat = 0..3).
/// Matches zenwebp `CAT_LENGTHS`.
const CAT_LENGTHS: [usize; 4] = [3, 4, 5, 11];

/// Decode a "large value" (DCT categories 1-6), returning the magnitude.
/// Exact port of zenwebp's `get_large_value`.
fn read_large_value(r: &mut BoolDecoder, prob: &[u8; 11]) -> i32 {
    if r.get_bit(prob[6]) == 0 {
        if r.get_bit(prob[7]) == 0 {
            // Cat 1: 5 or 6
            5 + r.get_bit(159) as i32
        } else {
            // Cat 2: 7..10
            7 + 2 * r.get_bit(165) as i32 + r.get_bit(145) as i32
        }
    } else {
        let bit1 = r.get_bit(prob[8]) as usize;
        let bit0 = r.get_bit(prob[9 + bit1]) as usize;
        let cat = 2 * bit1 + bit0; // 0..=3 → categories 3..=6
        let cat_probs = &PROB_DCT_CAT[2 + cat];
        let cat_len = CAT_LENGTHS[cat];
        let mut extra = 0i32;
        for &p in cat_probs.iter().take(cat_len) {
            extra = extra + extra + r.get_bit(p) as i32;
        }
        3 + (8 << cat) + extra
    }
}

fn parse_mb_coeffs(
    r: &mut BoolDecoder,
    probs: &TokenProbTables,
    top: &mut Ctx,
    left: &mut Ctx,
    mb: &mut Mb,
) {
    const PLANE_Y1: usize = 0;
    const PLANE_Y2: usize = 1;
    const PLANE_CHROMA: usize = 2;
    const PLANE_Y0: usize = 3;

    let has_y2 = mb.has_y2();
    let (y_plane, first_y) = if has_y2 {
        let complexity = (top.complexity[0] + left.complexity[0]) as usize;
        let n = read_block_levels(r, &mut mb.coeffs[0], &probs[PLANE_Y2], 0, complexity);
        let ctx = u8::from(n);
        left.complexity[0] = ctx;
        top.complexity[0] = ctx;
        (PLANE_Y1, 1usize)
    } else {
        (PLANE_Y0, 0usize)
    };

    // 16 Y blocks → coeffs[1..=16].
    for y in 0..4 {
        let mut left_ctx = left.complexity[y + 1];
        for x in 0..4 {
            let i = x + y * 4;
            let complexity = (top.complexity[x + 1] + left_ctx) as usize;
            let n = read_block_levels(
                r,
                &mut mb.coeffs[1 + i],
                &probs[y_plane],
                first_y,
                complexity,
            );
            left_ctx = u8::from(n);
            top.complexity[x + 1] = u8::from(n);
        }
        left.complexity[y + 1] = left_ctx;
    }

    // 8 chroma blocks → coeffs[17..=24]. j=5 (U), j=7 (V).
    let chroma = &probs[PLANE_CHROMA];
    for (plane_idx, &j) in [5usize, 7usize].iter().enumerate() {
        for y in 0..2 {
            let mut left_ctx = left.complexity[y + j];
            for x in 0..2 {
                let block = 17 + plane_idx * 4 + x + y * 2;
                let complexity = (top.complexity[x + j] + left_ctx) as usize;
                let n = read_block_levels(r, &mut mb.coeffs[block], chroma, 0, complexity);
                left_ctx = u8::from(n);
                top.complexity[x + j] = u8::from(n);
            }
            left.complexity[y + j] = left_ctx;
        }
    }
}

fn clear_skip_context(top: &mut Ctx, left: &mut Ctx, has_y2: bool) {
    // libwebp: on skip, all complexity contexts go to 0 EXCEPT Y2 which is
    // preserved when the MB has no Y2 (B-pred), per ParseResiduals.
    for i in 1..9 {
        top.complexity[i] = 0;
        left.complexity[i] = 0;
    }
    if has_y2 {
        top.complexity[0] = 0;
        left.complexity[0] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: parse a zenwebp-encoded keyframe and sanity-check structure.
    /// (Correctness is proven by the round-trip test once emit lands.)
    #[test]
    fn parse_smoke() {
        use zenwebp::PixelLayout;
        use zenwebp::encoder::{EncodeRequest, LossyConfig};

        let (w, h) = (64u32, 48u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                rgba[i] = (x * 4) as u8;
                rgba[i + 1] = (y * 5) as u8;
                rgba[i + 2] = ((x + y) * 3) as u8;
                rgba[i + 3] = 255;
            }
        }
        let webp = EncodeRequest::lossy(
            &LossyConfig::new().with_quality(70.0).with_method(4),
            &rgba,
            PixelLayout::Rgba8,
            w,
            h,
        )
        .encode()
        .unwrap();

        // Extract the VP8 chunk payload from the RIFF container.
        let vp8 = extract_vp8_chunk(&webp).expect("vp8 chunk");
        let frame = parse_vp8_keyframe(vp8).expect("parse keyframe");
        assert_eq!(frame.mb_cols, 4); // 64/16
        assert_eq!(frame.mb_rows, 3); // 48/16
        assert_eq!(frame.mbs.len(), 12);
        // At q70 a gradient has some non-zero coefficients somewhere.
        let total_nz: usize = frame
            .mbs
            .iter()
            .flat_map(|m| m.coeffs.iter())
            .flat_map(|b| b.iter())
            .filter(|&&c| c != 0)
            .count();
        assert!(total_nz > 0, "expected some non-zero coefficients");
    }

    /// Minimal RIFF/WEBP simple-format VP8 chunk extractor for the test.
    fn extract_vp8_chunk(webp: &[u8]) -> Option<&[u8]> {
        if webp.len() < 20 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
            return None;
        }
        let mut pos = 12;
        while pos + 8 <= webp.len() {
            let fourcc = &webp[pos..pos + 4];
            let size =
                u32::from_le_bytes([webp[pos + 4], webp[pos + 5], webp[pos + 6], webp[pos + 7]])
                    as usize;
            let body = pos + 8;
            if fourcc == b"VP8 " {
                return webp.get(body..body + size);
            }
            pos = body + size + (size & 1);
        }
        None
    }
}
