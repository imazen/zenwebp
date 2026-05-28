//! Re-emit a parsed [`Frame`] back to a VP8 keyframe bitstream.
//!
//! Every symbol is written as the exact inverse of `parse.rs`, using the
//! same boolean-coder probabilities and the same neighbour contexts, so a
//! verbatim re-emit reconstructs the identical coefficients → the decoded
//! image is pixel-exact. (Byte-identity is not guaranteed — trailing
//! padding may differ — but the decoded pixels are.)

use super::bool::BoolEncoder;
use super::parse::{Frame, Mb, Segmentation};
use super::tables::*;

const CAT_LENGTHS: [usize; 4] = [3, 4, 5, 11];

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

fn put_opt_signed(e: &mut BoolEncoder, n: u8, v: Option<i32>) {
    match v {
        Some(x) => {
            e.put_flag(true);
            e.put_signed_literal(n, x);
        }
        None => e.put_flag(false),
    }
}

/// Encode one block's 16 quantized levels (natural order) — exact inverse
/// of `parse::read_block_levels`. Returns whether any non-zero coefficient
/// was emitted (the next-block context).
fn encode_block_levels(
    e: &mut BoolEncoder,
    levels: &[i32; 16],
    plane: &[[[u8; 11]; 3]; 8],
    first: usize,
    complexity: usize,
) -> bool {
    // Last non-zero position in scan (zigzag) order.
    let mut last: i32 = -1;
    for n in 0..16 {
        if levels[ZIGZAG[n] as usize] != 0 {
            last = n as i32;
        }
    }

    let mut n = first;
    let mut ctx = complexity;
    while n < 16 {
        let band = COEFF_BANDS[n] as usize;
        if (n as i32) > last {
            // EOB.
            e.put_bit(0, plane[band][ctx][0]);
            break;
        }
        // Not EOB.
        e.put_bit(1, plane[band][ctx][0]);
        // Zero run.
        while levels[ZIGZAG[n] as usize] == 0 {
            let band = COEFF_BANDS[n] as usize;
            e.put_bit(0, plane[band][ctx][1]);
            n += 1;
            ctx = 0;
        }
        // Non-zero value at n.
        let band = COEFF_BANDS[n] as usize;
        e.put_bit(1, plane[band][ctx][1]);
        let lvl = levels[ZIGZAG[n] as usize];
        let v = lvl.unsigned_abs() as i32;
        encode_value(e, v, &plane[band][ctx]);
        e.put_flag(lvl < 0);
        ctx = if v == 1 { 1 } else { 2 };
        n += 1;
    }
    n > first
}

/// Encode a coefficient magnitude `v >= 1` (inverse of the value tokens in
/// `read_block_levels`).
fn encode_value(e: &mut BoolEncoder, v: i32, prob: &[u8; 11]) {
    if v == 1 {
        e.put_bit(0, prob[2]);
        return;
    }
    e.put_bit(1, prob[2]);
    if v <= 4 {
        e.put_bit(0, prob[3]);
        if v == 2 {
            e.put_bit(0, prob[4]);
        } else {
            // 3 or 4
            e.put_bit(1, prob[4]);
            e.put_bit((v - 3) as u32, prob[5]);
        }
    } else {
        e.put_bit(1, prob[3]);
        encode_large_value(e, v, prob);
    }
}

/// Encode DCT categories 1-6 (inverse of `read_large_value`).
fn encode_large_value(e: &mut BoolEncoder, v: i32, prob: &[u8; 11]) {
    if v <= 6 {
        // Cat 1: 5 or 6.
        e.put_bit(0, prob[6]);
        e.put_bit(0, prob[7]);
        e.put_bit((v - 5) as u32, 159);
    } else if v <= 10 {
        // Cat 2: 7..10 = 7 + 2*b1 + b0.
        e.put_bit(0, prob[6]);
        e.put_bit(1, prob[7]);
        let d = (v - 7) as u32;
        e.put_bit((d >> 1) & 1, 165);
        e.put_bit(d & 1, 145);
    } else {
        // Cat 3-6.
        e.put_bit(1, prob[6]);
        let cat = if v < 19 {
            0
        } else if v < 35 {
            1
        } else if v < 67 {
            2
        } else {
            3
        };
        let bit1 = (cat >> 1) & 1;
        let bit0 = cat & 1;
        e.put_bit(bit1 as u32, prob[8]);
        e.put_bit(bit0 as u32, prob[9 + bit1]);
        let base = 3 + (8 << cat);
        let extra = (v - base) as u32;
        let len = CAT_LENGTHS[cat];
        let cat_probs = &PROB_DCT_CAT[2 + cat];
        for (i, &p) in cat_probs.iter().take(len).enumerate() {
            let bit = (extra >> (len - 1 - i)) & 1;
            e.put_bit(bit, p);
        }
    }
}

fn emit_segment_id(e: &mut BoolEncoder, id: u8, probs: &[u8; 3]) {
    match id {
        0 => {
            e.put_bit(0, probs[0]);
            e.put_bit(0, probs[1]);
        }
        1 => {
            e.put_bit(0, probs[0]);
            e.put_bit(1, probs[1]);
        }
        2 => {
            e.put_bit(1, probs[0]);
            e.put_bit(0, probs[2]);
        }
        _ => {
            e.put_bit(1, probs[0]);
            e.put_bit(1, probs[2]);
        }
    }
}

fn emit_ymode(e: &mut BoolEncoder, m: i8) {
    // Tree probs 145,156,163,128.
    if m == B_PRED {
        e.put_bit(0, 145);
    } else {
        e.put_bit(1, 145);
        match m {
            DC_PRED => {
                e.put_bit(0, 156);
                e.put_bit(0, 163);
            }
            V_PRED => {
                e.put_bit(0, 156);
                e.put_bit(1, 163);
            }
            H_PRED => {
                e.put_bit(1, 156);
                e.put_bit(0, 128);
            }
            _ => {
                // TM_PRED
                e.put_bit(1, 156);
                e.put_bit(1, 128);
            }
        }
    }
}

fn emit_uvmode(e: &mut BoolEncoder, m: i8) {
    match m {
        DC_PRED => e.put_bit(0, 142),
        V_PRED => {
            e.put_bit(1, 142);
            e.put_bit(0, 114);
        }
        H_PRED => {
            e.put_bit(1, 142);
            e.put_bit(1, 114);
            e.put_bit(0, 183);
        }
        _ => {
            e.put_bit(1, 142);
            e.put_bit(1, 114);
            e.put_bit(1, 183);
        }
    }
}

fn emit_bmode(e: &mut BoolEncoder, m: i8, probs: &[u8; 9]) {
    match m {
        B_DC_PRED => e.put_bit(0, probs[0]),
        B_TM_PRED => {
            e.put_bit(1, probs[0]);
            e.put_bit(0, probs[1]);
        }
        B_VE_PRED => {
            e.put_bit(1, probs[0]);
            e.put_bit(1, probs[1]);
            e.put_bit(0, probs[2]);
        }
        B_HE_PRED | B_RD_PRED | B_VR_PRED => {
            e.put_bit(1, probs[0]);
            e.put_bit(1, probs[1]);
            e.put_bit(1, probs[2]);
            e.put_bit(0, probs[3]);
            match m {
                B_HE_PRED => e.put_bit(0, probs[4]),
                B_RD_PRED => {
                    e.put_bit(1, probs[4]);
                    e.put_bit(0, probs[5]);
                }
                _ => {
                    // B_VR_PRED
                    e.put_bit(1, probs[4]);
                    e.put_bit(1, probs[5]);
                }
            }
        }
        _ => {
            // B_LD_PRED, B_VL_PRED, B_HD_PRED, B_HU_PRED
            e.put_bit(1, probs[0]);
            e.put_bit(1, probs[1]);
            e.put_bit(1, probs[2]);
            e.put_bit(1, probs[3]);
            match m {
                B_LD_PRED => e.put_bit(0, probs[6]),
                B_VL_PRED => {
                    e.put_bit(1, probs[6]);
                    e.put_bit(0, probs[7]);
                }
                B_HD_PRED => {
                    e.put_bit(1, probs[6]);
                    e.put_bit(1, probs[7]);
                    e.put_bit(0, probs[8]);
                }
                _ => {
                    // B_HU_PRED
                    e.put_bit(1, probs[6]);
                    e.put_bit(1, probs[7]);
                    e.put_bit(1, probs[8]);
                }
            }
        }
    }
}

fn emit_mb_header(
    e: &mut BoolEncoder,
    seg: &Segmentation,
    prob_skip: &Option<u8>,
    top: &mut Ctx,
    left: &mut Ctx,
    mb: &Mb,
) {
    if seg.enabled && seg.update_map {
        emit_segment_id(e, mb.segment_id, &seg.tree_probs);
    }
    if let Some(p) = prob_skip {
        e.put_bit(mb.skip as u32, *p);
    }
    emit_ymode(e, mb.luma_mode);
    if mb.luma_mode == B_PRED {
        for y in 0..4 {
            for x in 0..4 {
                let tm = top.bpred[x];
                let lm = left.bpred[y];
                let bmode = mb.bpred[x + y * 4];
                emit_bmode(
                    e,
                    bmode,
                    &KEYFRAME_BPRED_MODE_PROBS[tm as usize][lm as usize],
                );
                top.bpred[x] = bmode;
                left.bpred[y] = bmode;
            }
        }
    } else {
        // Non-B: left/top bpred context = B-mode equivalent (see parse).
        let m = super::parse::ymode_to_bcontext(mb.luma_mode);
        for i in 0..4 {
            left.bpred[i] = m;
        }
    }
    emit_uvmode(e, mb.chroma_mode);
    top.bpred = [mb.bpred[12], mb.bpred[13], mb.bpred[14], mb.bpred[15]];
}

fn emit_mb_coeffs(
    e: &mut BoolEncoder,
    probs: &TokenProbTables,
    top: &mut Ctx,
    left: &mut Ctx,
    mb: &Mb,
) {
    const PLANE_Y1: usize = 0;
    const PLANE_Y2: usize = 1;
    const PLANE_CHROMA: usize = 2;
    const PLANE_Y0: usize = 3;

    let (y_plane, first_y) = if mb.has_y2() {
        let complexity = (top.complexity[0] + left.complexity[0]) as usize;
        let n = encode_block_levels(e, &mb.coeffs[0], &probs[PLANE_Y2], 0, complexity);
        let ctx = u8::from(n);
        left.complexity[0] = ctx;
        top.complexity[0] = ctx;
        (PLANE_Y1, 1usize)
    } else {
        (PLANE_Y0, 0usize)
    };

    for y in 0..4 {
        let mut left_ctx = left.complexity[y + 1];
        for x in 0..4 {
            let i = x + y * 4;
            let complexity = (top.complexity[x + 1] + left_ctx) as usize;
            let n = encode_block_levels(e, &mb.coeffs[1 + i], &probs[y_plane], first_y, complexity);
            left_ctx = u8::from(n);
            top.complexity[x + 1] = u8::from(n);
        }
        left.complexity[y + 1] = left_ctx;
    }

    let chroma = &probs[PLANE_CHROMA];
    for (plane_idx, &j) in [5usize, 7usize].iter().enumerate() {
        for y in 0..2 {
            let mut left_ctx = left.complexity[y + j];
            for x in 0..2 {
                let block = 17 + plane_idx * 4 + x + y * 2;
                let complexity = (top.complexity[x + j] + left_ctx) as usize;
                let n = encode_block_levels(e, &mb.coeffs[block], chroma, 0, complexity);
                left_ctx = u8::from(n);
                top.complexity[x + j] = u8::from(n);
            }
            left.complexity[y + j] = left_ctx;
        }
    }
}

fn clear_skip_context(top: &mut Ctx, left: &mut Ctx, has_y2: bool) {
    for i in 1..9 {
        top.complexity[i] = 0;
        left.complexity[i] = 0;
    }
    if has_y2 {
        top.complexity[0] = 0;
        left.complexity[0] = 0;
    }
}

/// Re-emit the control partition (partition 0): header + per-MB modes.
fn emit_part0(frame: &Frame) -> Vec<u8> {
    let mut e = BoolEncoder::new();
    e.put_bit(frame.color_space, 128);
    e.put_bit(frame.clamp_type, 128);

    let seg = &frame.segmentation;
    e.put_flag(seg.enabled);
    if seg.enabled {
        e.put_flag(seg.update_map);
        e.put_flag(seg.update_data);
        if seg.update_data {
            e.put_flag(seg.abs_values);
            for i in 0..4 {
                put_opt_present(&mut e, seg.quantizer_present[i], 7, seg.quantizer[i] as i32);
            }
            for i in 0..4 {
                put_opt_present(
                    &mut e,
                    seg.loopfilter_present[i],
                    6,
                    seg.loopfilter[i] as i32,
                );
            }
        }
        if seg.update_map {
            for i in 0..3 {
                if seg.tree_probs_present[i] {
                    e.put_flag(true);
                    e.put_literal(8, seg.tree_probs[i] as u32);
                } else {
                    e.put_flag(false);
                }
            }
        }
    }

    e.put_flag(frame.filter_type);
    e.put_literal(6, frame.filter_level);
    e.put_literal(3, frame.sharpness);
    let lf = &frame.lf_adjust;
    e.put_flag(lf.enabled);
    if lf.enabled {
        e.put_flag(lf.update);
        if lf.update {
            for d in lf.ref_delta {
                put_opt_signed(&mut e, 6, d);
            }
            for d in lf.mode_delta {
                put_opt_signed(&mut e, 6, d);
            }
        }
    }

    let log2 = (frame.num_partitions.trailing_zeros()) as u8;
    e.put_literal(2, log2 as u32);

    let q = &frame.quant;
    e.put_literal(7, q.yac_abs);
    put_opt_signed(&mut e, 4, q.ydc_delta);
    put_opt_signed(&mut e, 4, q.y2dc_delta);
    put_opt_signed(&mut e, 4, q.y2ac_delta);
    put_opt_signed(&mut e, 4, q.uvdc_delta);
    put_opt_signed(&mut e, 4, q.uvac_delta);

    e.put_bit(frame.refresh_entropy_probs, 128);

    // Token prob updates: emit update flag + new prob where it differs from
    // the default, matching what parse read. (4-deep table walk; index form
    // is clearest.)
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for t in 0..11 {
                    if frame.coeff_probs[i][j][k][t] != COEFF_PROBS[i][j][k][t] {
                        e.put_bit(1, COEFF_UPDATE_PROBS[i][j][k][t]);
                        e.put_literal(8, frame.coeff_probs[i][j][k][t] as u32);
                    } else {
                        e.put_bit(0, COEFF_UPDATE_PROBS[i][j][k][t]);
                    }
                }
            }
        }
    }

    e.put_bit(frame.mb_no_skip_coeff, 128);
    if let Some(p) = frame.prob_skip_false {
        e.put_literal(8, p as u32);
    }

    // Per-MB mode data (segment, skip, ymode, bmodes, uvmode) lives in
    // partition 0 too, after the header — same encoder.
    let mut top = vec![Ctx::new(); frame.mb_cols];
    let mut idx = 0;
    for _ in 0..frame.mb_rows {
        let mut left = Ctx::new();
        for top_ctx in top.iter_mut() {
            emit_mb_header(
                &mut e,
                seg,
                &frame.prob_skip_false,
                top_ctx,
                &mut left,
                &frame.mbs[idx],
            );
            idx += 1;
        }
    }

    e.finish()
}

/// Emit the `num_partitions` token partitions (coefficients), distributing
/// MB rows round-robin by `row % num_partitions` exactly as parse read them.
fn emit_token_partitions(frame: &Frame) -> Vec<Vec<u8>> {
    let mut encoders: Vec<BoolEncoder> = (0..frame.num_partitions)
        .map(|_| BoolEncoder::new())
        .collect();
    let mut top = vec![Ctx::new(); frame.mb_cols];
    let mut idx = 0;
    for mb_row in 0..frame.mb_rows {
        let mut left = Ctx::new();
        let part = mb_row % frame.num_partitions;
        for top_ctx in top.iter_mut() {
            let mb = &frame.mbs[idx];
            idx += 1;
            if mb.skip {
                clear_skip_context(top_ctx, &mut left, mb.has_y2());
            } else {
                emit_mb_coeffs(
                    &mut encoders[part],
                    &frame.coeff_probs,
                    top_ctx,
                    &mut left,
                    mb,
                );
            }
        }
    }
    encoders.into_iter().map(|e| e.finish()).collect()
}

/// Re-emit the full VP8 keyframe bitstream (the `VP8 ` chunk payload).
pub fn emit_vp8_keyframe(frame: &Frame) -> Vec<u8> {
    let part0 = emit_part0(frame);
    let token_parts = emit_token_partitions(frame);

    // Frame tag (3 bytes): bit0 = keyframe(0), bits1-3 = version|show,
    // bits5+ = first_partition_size. Reuse the original version/show bits
    // from the parsed 10-byte header, but overwrite the size field.
    let orig_tag = u32::from(frame.keyframe_header[0])
        | (u32::from(frame.keyframe_header[1]) << 8)
        | (u32::from(frame.keyframe_header[2]) << 16);
    let low5 = orig_tag & 0x1f; // keyframe + version + show
    let tag = low5 | ((part0.len() as u32) << 5);

    let mut out = Vec::with_capacity(part0.len() + 64);
    out.push((tag & 0xff) as u8);
    out.push(((tag >> 8) & 0xff) as u8);
    out.push(((tag >> 16) & 0xff) as u8);
    // Start code + dimensions: bytes 3..10 of the original header verbatim.
    out.extend_from_slice(&frame.keyframe_header[3..10]);
    // Partition 0.
    out.extend_from_slice(&part0);
    // Token-partition size table (3 bytes each for the first n-1).
    if token_parts.len() > 1 {
        for p in &token_parts[..token_parts.len() - 1] {
            let s = p.len() as u32;
            out.push((s & 0xff) as u8);
            out.push(((s >> 8) & 0xff) as u8);
            out.push(((s >> 16) & 0xff) as u8);
        }
    }
    for p in &token_parts {
        out.extend_from_slice(p);
    }
    out
}

fn put_opt_present(e: &mut BoolEncoder, present: bool, n: u8, v: i32) {
    if present {
        e.put_flag(true);
        e.put_signed_literal(n, v);
    } else {
        e.put_flag(false);
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_vp8_keyframe;
    use super::*;

    /// Extract the `VP8 ` chunk payload from a simple-format WebP.
    fn extract_vp8(webp: &[u8]) -> Option<&[u8]> {
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

    /// Wrap a VP8 payload back into a minimal simple-format WebP container.
    fn wrap_vp8(vp8: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vp8.len() + 20);
        out.extend_from_slice(b"RIFF");
        let riff_size = 4 + 8 + vp8.len() + (vp8.len() & 1);
        out.extend_from_slice(&(riff_size as u32).to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(b"VP8 ");
        out.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
        out.extend_from_slice(vp8);
        if vp8.len() & 1 == 1 {
            out.push(0);
        }
        out
    }

    /// Debug: parse → emit → re-parse, compare structures (isolates an
    /// internal parse/emit asymmetry from a decoder-expectation mismatch).
    #[test]
    fn parse_emit_reparse_consistent() {
        use zenwebp::PixelLayout;
        use zenwebp::encoder::{EncodeRequest, LossyConfig};
        let (w, h) = (64u32, 48u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 13) & 0x0f;
                rgba[i] = ((x * 3) + n) as u8;
                rgba[i + 1] = ((y * 4) + n) as u8;
                rgba[i + 2] = (((x + y) * 2) + n) as u8;
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
        let vp8 = extract_vp8(&webp).unwrap();
        let f1 = parse_vp8_keyframe(vp8).unwrap();
        eprintln!(
            "orig vp8 len={}, num_parts={}, part0 orig size(tag)={}",
            vp8.len(),
            f1.num_partitions,
            {
                let t = u32::from(vp8[0]) | (u32::from(vp8[1]) << 8) | (u32::from(vp8[2]) << 16);
                t >> 5
            }
        );
        let reemit = emit_vp8_keyframe(&f1);
        let my_tag =
            u32::from(reemit[0]) | (u32::from(reemit[1]) << 8) | (u32::from(reemit[2]) << 16);
        eprintln!(
            "reemit len={}, my part0 size(tag)={}",
            reemit.len(),
            my_tag >> 5
        );
        // Try decoding via zenwebp and print the full error chain.
        let rewrapped = wrap_vp8(&reemit);
        match zenwebp::oneshot::decode_rgba(&rewrapped) {
            Ok(_) => eprintln!("zenwebp decode OK"),
            Err(e) => eprintln!("zenwebp decode ERR: {e:?}"),
        }
        // Byte-diff original part0 vs my emitted part0 (both start at offset 10).
        let orig_p0 =
            (u32::from(vp8[0]) | (u32::from(vp8[1]) << 8) | (u32::from(vp8[2]) << 16)) >> 5;
        let o = &vp8[10..10 + orig_p0 as usize];
        let m = &reemit[10..10 + (my_tag >> 5) as usize];
        let first = o.iter().zip(m.iter()).position(|(a, b)| a != b);
        eprintln!(
            "part0 byte diff at {first:?} (orig {} bytes, mine {} bytes)",
            o.len(),
            m.len()
        );
        if let Some(d) = first {
            let lo = d.saturating_sub(2);
            eprintln!("  orig[{lo}..]={:02x?}", &o[lo..(lo + 8).min(o.len())]);
            eprintln!("  mine[{lo}..]={:02x?}", &m[lo..(lo + 8).min(m.len())]);
        }
        let prob_updates = {
            let mut c = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                for j in 0..8 {
                    for k in 0..3 {
                        for t in 0..11 {
                            if f1.coeff_probs[i][j][k][t] != COEFF_PROBS[i][j][k][t] {
                                c += 1;
                            }
                        }
                    }
                }
            }
            c
        };
        eprintln!("prob_updates(differ from default)={prob_updates}");
        // Count B_PRED MBs (suspect path).
        let bpred_mbs = f1.mbs.iter().filter(|mb| mb.luma_mode == B_PRED).count();
        eprintln!("B_PRED mbs={bpred_mbs} / {}", f1.mbs.len());
        let f2 = parse_vp8_keyframe(&reemit).expect("re-parse my own emit");
        assert_eq!(f1.mbs.len(), f2.mbs.len(), "mb count");
        let mut mode_diff = 0;
        let mut coeff_diff = 0;
        let mut bpred_diff = 0;
        for (a, b) in f1.mbs.iter().zip(f2.mbs.iter()) {
            if a.luma_mode != b.luma_mode || a.chroma_mode != b.chroma_mode || a.skip != b.skip {
                mode_diff += 1;
            }
            if a.luma_mode == B_PRED && a.bpred != b.bpred {
                bpred_diff += 1;
            }
            if a.coeffs != b.coeffs {
                coeff_diff += 1;
            }
        }
        eprintln!("mode_diff={mode_diff} bpred_diff={bpred_diff} coeff_diff={coeff_diff}");
        assert_eq!(mode_diff, 0, "modes diverge on re-parse");
        assert_eq!(coeff_diff, 0, "coeffs diverge on re-parse");
    }

    /// THE validation gate: parse → re-emit verbatim → decode both → assert
    /// the decoded pixels are identical.
    #[test]
    fn verbatim_roundtrip_pixel_exact() {
        use zenwebp::PixelLayout;
        use zenwebp::encoder::{EncodeRequest, LossyConfig};
        use zenwebp::oneshot::decode_rgba;

        // Real lossy WebPs from the test corpus exercise true encoder
        // output (segmentation, prob updates, large coeffs, varied modes).
        for name in [
            "gallery2/1_webp_a",
            "gallery2/4_webp_a",
            "gallery1/2",
            "gallery1/4",
        ] {
            let path = format!("../tests/images/{name}.webp");
            let Ok(webp) = std::fs::read(&path) else {
                continue;
            };
            // Only lossy VP8 keyframes are in scope.
            let Some(vp8) = extract_vp8(&webp) else {
                continue;
            };
            let Some(frame) = parse_vp8_keyframe(vp8) else {
                continue;
            };
            let reemit = emit_vp8_keyframe(&frame);
            let rewrapped = wrap_vp8(&reemit);
            let (orig_px, _, _) = decode_rgba(&webp).expect("decode orig");
            match decode_rgba(&rewrapped) {
                Ok((re_px, _, _)) => {
                    // Compare RGB only: the transcoder operates on the VP8
                    // color plane; alpha (ALPH chunk) is preserved separately
                    // by the real strategy and is dropped by this test wrapper.
                    let diffs = orig_px
                        .chunks_exact(4)
                        .zip(re_px.chunks_exact(4))
                        .filter(|(a, b)| a[0..3] != b[0..3])
                        .count();
                    eprintln!(
                        "{name} parts={}: RGB-diff pixels={diffs}/{}",
                        frame.num_partitions,
                        orig_px.len() / 4
                    );
                    assert_eq!(diffs, 0, "real image {name} color not pixel-exact");
                }
                Err(e) => panic!("{name}: decode ERR {e:?}"),
            }
        }

        for (q, w, h) in [
            (70.0f32, 64u32, 48u32),
            (50.0, 96, 64),
            (90.0, 80, 80),
            (40.0, 320, 240),
            (95.0, 512, 384),
        ] {
            let mut rgba = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 4) as usize;
                    let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 13) & 0x0f;
                    rgba[i] = ((x * 3) + n) as u8;
                    rgba[i + 1] = ((y * 4) + n) as u8;
                    rgba[i + 2] = (((x + y) * 2) + n) as u8;
                    rgba[i + 3] = 255;
                }
            }
            let webp = EncodeRequest::lossy(
                &LossyConfig::new().with_quality(q).with_method(4),
                &rgba,
                PixelLayout::Rgba8,
                w,
                h,
            )
            .encode()
            .unwrap();
            let vp8 = extract_vp8(&webp).expect("vp8");
            let frame = parse_vp8_keyframe(vp8).expect("parse");
            let reemit = emit_vp8_keyframe(&frame);
            let rewrapped = wrap_vp8(&reemit);

            let frame_parts = frame.num_partitions;
            let (orig_px, _ow, _oh) = decode_rgba(&webp).expect("decode orig");
            match decode_rgba(&rewrapped) {
                Ok((re_px, _, _)) => {
                    let diffs = orig_px
                        .iter()
                        .zip(re_px.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!(
                        "q{q} {w}x{h} parts={frame_parts}: decode OK, pixel diffs={diffs}/{}",
                        orig_px.len()
                    );
                    assert_eq!(diffs, 0, "pixels differ q{q} {w}x{h}");
                }
                Err(e) => panic!("q{q} {w}x{h} parts={frame_parts}: decode ERR {e:?}"),
            }
        }
    }
}
