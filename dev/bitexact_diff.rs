//! Bit-exact parity diff harness for CostModel::StrictLibwebpParity (#38).
//!
//! Compares zenwebp (StrictLibwebpParity) against libwebp (webpx) at matched
//! settings, three levels deep: VP8 frame-header fields (segments, filter,
//! quant, proba-update counts), then the per-MB keyframe mode stream
//! (y/uv/b4 modes, segment ids, skip bits) via a self-contained RFC 6386
//! boolean decoder. Reports field diffs and mode-agreement percentages.
//!
//! Usage: move to examples/ or add an [[example]] entry, then:
//!   cargo run --release --example bitexact_diff [image.png]
//!
//! Findings 2026-07-14 (CID22 382297, q75): header fields align after the
//! profile-bit + absolute-segment-quantizer fixes; remaining divergence is
//! mode selection (y 89%, uv 81%, b4 12% agreement at m4) and proba-update
//! counts. libwebp quirk: at m0-m2 with one effective segment, StatLoop
//! bails early (OneStatPass returns size_p0==0) and ships DEFAULT probas,
//! costing it 25-35% size on those cells - do not "fix" zen to match
//! without gating on StrictLibwebpParity.
// The 4x8x3x11 walks over COEFF_UPDATE_PROBS mirror the RFC 6386 (13.4) header
// layout and libwebp's own loop nesting, so the indices ARE the meaning here —
// iterator adaptors would obscure what bit position is being parsed.
#![allow(clippy::needless_range_loop)]

use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

/// Same generator as `dev/byteparity_sweep.rs` — `synth:WxH:SEED` image specs
/// reproduce the sweep's synthetic cells exactly. (#38)
fn synth(w: u32, h: u32, seed: u32) -> (Vec<u8>, u32, u32) {
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 24) as u8 / 8;
            let r = ((x * 255 / w.max(1)) as u8).wrapping_add(n);
            let g = ((y * 255 / h.max(1)) as u8).wrapping_add(n);
            let b = (((x + y) * 255 / (w + h).max(1)) as u8).wrapping_add(n);
            px.extend_from_slice(&[r, g, b]);
        }
    }
    (px, w, h)
}

fn load(path: &str) -> (Vec<u8>, u32, u32) {
    if let Some(spec) = path.strip_prefix("synth:") {
        let (dims, seed) = spec.split_once(':').expect("synth:WxH:SEED");
        let (w, h) = dims.split_once('x').expect("synth:WxH:SEED");
        return synth(
            w.parse().unwrap(),
            h.parse().unwrap(),
            seed.parse().unwrap(),
        );
    }
    let file = std::fs::File::open(path).unwrap();
    let mut d = png::Decoder::new(std::io::BufReader::new(file));
    d.set_transformations(png::Transformations::normalize_to_color8());
    let mut r = d.read_info().unwrap();
    let mut buf = vec![0u8; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => unreachable!(),
    };
    (rgb, info.width, info.height)
}

fn vp8_payload(webp: &[u8]) -> &[u8] {
    assert_eq!(&webp[0..4], b"RIFF");
    assert_eq!(&webp[8..16], b"WEBPVP8 ");
    let sz = u32::from_le_bytes(webp[16..20].try_into().unwrap()) as usize;
    &webp[20..20 + sz]
}

/// RFC 6386 boolean decoder.
struct Bool<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    value: u32,
    bits: i32,
}

impl<'a> Bool<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut b = Bool {
            data,
            pos: 0,
            range: 255,
            value: 0,
            bits: -8,
        };
        b.value = (b.byte() as u32) << 8;
        b.value |= b.byte() as u32;
        b.bits = 0;
        b
    }
    fn byte(&mut self) -> u8 {
        let v = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        v
    }
    fn bit(&mut self, prob: u8) -> u32 {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let bigsplit = split << 8;
        let ret = if self.value >= bigsplit {
            self.range -= split;
            self.value -= bigsplit;
            1
        } else {
            self.range = split;
            0
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bits += 1;
            if self.bits == 8 {
                self.bits = 0;
                self.value |= self.byte() as u32;
            }
        }
        ret
    }
    fn literal(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit(128);
        }
        v
    }
    fn signed(&mut self, n: u32) -> i32 {
        let v = self.literal(n) as i32;
        if self.bit(128) == 1 { -v } else { v }
    }
    fn flagged_signed(&mut self, n: u32) -> i32 {
        if self.bit(128) == 1 {
            self.signed(n)
        } else {
            0
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Hdr {
    version: u8,
    part1_size: usize,
    color_space: u32,
    clamping: u32,
    segments_enabled: u32,
    seg_update_map: u32,
    seg_update_data: u32,
    seg_abs: u32,
    seg_q: [i32; 4],
    seg_lf: [i32; 4],
    seg_tree_probs: [u32; 3],
    filter_type: u32,
    filter_level: u32,
    sharpness: u32,
    lf_delta_enabled: u32,
    log2_parts: u32,
    q_yac: u32,
    q_ydc: i32,
    q_y2dc: i32,
    q_y2ac: i32,
    q_uvdc: i32,
    q_uvac: i32,
    refresh_entropy: u32,
    n_proba_updates: u32,
    use_skip: u32,
    skip_prob: u32,
    header_bits_after: usize, // bool pos after fixed header (rough anchor)
}

// Default coeff update probs from RFC 6386 (13.4) — needed to know update bit prob.
include!("coeff_update_probs.rs");

fn parse_hdr(p: &[u8]) -> Hdr {
    let tag = u32::from_le_bytes([p[0], p[1], p[2], 0]);
    let version = ((tag >> 1) & 7) as u8;
    let part1_size = (tag >> 5) as usize;
    let mut b = Bool::new(&p[10..10 + part1_size]);
    let color_space = b.bit(128);
    let clamping = b.bit(128);
    let segments_enabled = b.bit(128);
    let (mut seg_update_map, mut seg_update_data, mut seg_abs) = (0, 0, 0);
    let mut seg_q = [0i32; 4];
    let mut seg_lf = [0i32; 4];
    let mut seg_tree_probs = [255u32; 3];
    if segments_enabled == 1 {
        seg_update_map = b.bit(128);
        seg_update_data = b.bit(128);
        if seg_update_data == 1 {
            seg_abs = b.bit(128);
            for q in seg_q.iter_mut() {
                *q = b.flagged_signed(7);
            }
            for lf in seg_lf.iter_mut() {
                *lf = b.flagged_signed(6);
            }
        }
        if seg_update_map == 1 {
            for t in seg_tree_probs.iter_mut() {
                *t = if b.bit(128) == 1 { b.literal(8) } else { 255 };
            }
        }
    }
    let filter_type = b.bit(128);
    let filter_level = b.literal(6);
    let sharpness = b.literal(3);
    let lf_delta_enabled = b.bit(128);
    if lf_delta_enabled == 1 && b.bit(128) == 1 {
        for _ in 0..8 {
            b.flagged_signed(6);
        }
    }
    let log2_parts = b.literal(2);
    let q_yac = b.literal(7);
    let q_ydc = b.flagged_signed(4);
    let q_y2dc = b.flagged_signed(4);
    let q_y2ac = b.flagged_signed(4);
    let q_uvdc = b.flagged_signed(4);
    let q_uvac = b.flagged_signed(4);
    let refresh_entropy = b.bit(128);
    let mut n_proba_updates = 0;
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for l in 0..11 {
                    if b.bit(COEFF_UPDATE_PROBS[i][j][k][l]) == 1 {
                        b.literal(8);
                        n_proba_updates += 1;
                    }
                }
            }
        }
    }
    let use_skip = b.bit(128);
    let skip_prob = if use_skip == 1 { b.literal(8) } else { 0 };
    Hdr {
        version,
        part1_size,
        color_space,
        clamping,
        segments_enabled,
        seg_update_map,
        seg_update_data,
        seg_abs,
        seg_q,
        seg_lf,
        seg_tree_probs,
        filter_type,
        filter_level,
        sharpness,
        lf_delta_enabled,
        log2_parts,
        q_yac,
        q_ydc,
        q_y2dc,
        q_y2ac,
        q_uvdc,
        q_uvac,
        refresh_entropy,
        n_proba_updates,
        use_skip,
        skip_prob,
        header_bits_after: b.pos,
    }
}

pub const KEYFRAME_YMODE_TREE: [i8; 8] = [-B_PRED, 2, 4, 6, -DC_PRED, -V_PRED, -H_PRED, -TM_PRED];
pub const KEYFRAME_YMODE_PROBS: [u8; 4] = [145, 156, 163, 128];
pub const KEYFRAME_BPRED_MODE_TREE: [i8; 18] = [
    -B_DC_PRED, 2, -B_TM_PRED, 4, -B_VE_PRED, 6, 8, 12, -B_HE_PRED, 10, -B_RD_PRED, -B_VR_PRED,
    -B_LD_PRED, 14, -B_VL_PRED, 16, -B_HD_PRED, -B_HU_PRED,
];
pub const KEYFRAME_BPRED_MODE_PROBS: [[[u8; 9]; 10]; 10] = [
    [
        [231, 120, 48, 89, 115, 113, 120, 152, 112],
        [152, 179, 64, 126, 170, 118, 46, 70, 95],
        [175, 69, 143, 80, 85, 82, 72, 155, 103],
        [56, 58, 10, 171, 218, 189, 17, 13, 152],
        [144, 71, 10, 38, 171, 213, 144, 34, 26],
        [114, 26, 17, 163, 44, 195, 21, 10, 173],
        [121, 24, 80, 195, 26, 62, 44, 64, 85],
        [170, 46, 55, 19, 136, 160, 33, 206, 71],
        [63, 20, 8, 114, 114, 208, 12, 9, 226],
        [81, 40, 11, 96, 182, 84, 29, 16, 36],
    ],
    [
        [134, 183, 89, 137, 98, 101, 106, 165, 148],
        [72, 187, 100, 130, 157, 111, 32, 75, 80],
        [66, 102, 167, 99, 74, 62, 40, 234, 128],
        [41, 53, 9, 178, 241, 141, 26, 8, 107],
        [104, 79, 12, 27, 217, 255, 87, 17, 7],
        [74, 43, 26, 146, 73, 166, 49, 23, 157],
        [65, 38, 105, 160, 51, 52, 31, 115, 128],
        [87, 68, 71, 44, 114, 51, 15, 186, 23],
        [47, 41, 14, 110, 182, 183, 21, 17, 194],
        [66, 45, 25, 102, 197, 189, 23, 18, 22],
    ],
    [
        [88, 88, 147, 150, 42, 46, 45, 196, 205],
        [43, 97, 183, 117, 85, 38, 35, 179, 61],
        [39, 53, 200, 87, 26, 21, 43, 232, 171],
        [56, 34, 51, 104, 114, 102, 29, 93, 77],
        [107, 54, 32, 26, 51, 1, 81, 43, 31],
        [39, 28, 85, 171, 58, 165, 90, 98, 64],
        [34, 22, 116, 206, 23, 34, 43, 166, 73],
        [68, 25, 106, 22, 64, 171, 36, 225, 114],
        [34, 19, 21, 102, 132, 188, 16, 76, 124],
        [62, 18, 78, 95, 85, 57, 50, 48, 51],
    ],
    [
        [193, 101, 35, 159, 215, 111, 89, 46, 111],
        [60, 148, 31, 172, 219, 228, 21, 18, 111],
        [112, 113, 77, 85, 179, 255, 38, 120, 114],
        [40, 42, 1, 196, 245, 209, 10, 25, 109],
        [100, 80, 8, 43, 154, 1, 51, 26, 71],
        [88, 43, 29, 140, 166, 213, 37, 43, 154],
        [61, 63, 30, 155, 67, 45, 68, 1, 209],
        [142, 78, 78, 16, 255, 128, 34, 197, 171],
        [41, 40, 5, 102, 211, 183, 4, 1, 221],
        [51, 50, 17, 168, 209, 192, 23, 25, 82],
    ],
    [
        [125, 98, 42, 88, 104, 85, 117, 175, 82],
        [95, 84, 53, 89, 128, 100, 113, 101, 45],
        [75, 79, 123, 47, 51, 128, 81, 171, 1],
        [57, 17, 5, 71, 102, 57, 53, 41, 49],
        [115, 21, 2, 10, 102, 255, 166, 23, 6],
        [38, 33, 13, 121, 57, 73, 26, 1, 85],
        [41, 10, 67, 138, 77, 110, 90, 47, 114],
        [101, 29, 16, 10, 85, 128, 101, 196, 26],
        [57, 18, 10, 102, 102, 213, 34, 20, 43],
        [117, 20, 15, 36, 163, 128, 68, 1, 26],
    ],
    [
        [138, 31, 36, 171, 27, 166, 38, 44, 229],
        [67, 87, 58, 169, 82, 115, 26, 59, 179],
        [63, 59, 90, 180, 59, 166, 93, 73, 154],
        [40, 40, 21, 116, 143, 209, 34, 39, 175],
        [57, 46, 22, 24, 128, 1, 54, 17, 37],
        [47, 15, 16, 183, 34, 223, 49, 45, 183],
        [46, 17, 33, 183, 6, 98, 15, 32, 183],
        [65, 32, 73, 115, 28, 128, 23, 128, 205],
        [40, 3, 9, 115, 51, 192, 18, 6, 223],
        [87, 37, 9, 115, 59, 77, 64, 21, 47],
    ],
    [
        [104, 55, 44, 218, 9, 54, 53, 130, 226],
        [64, 90, 70, 205, 40, 41, 23, 26, 57],
        [54, 57, 112, 184, 5, 41, 38, 166, 213],
        [30, 34, 26, 133, 152, 116, 10, 32, 134],
        [75, 32, 12, 51, 192, 255, 160, 43, 51],
        [39, 19, 53, 221, 26, 114, 32, 73, 255],
        [31, 9, 65, 234, 2, 15, 1, 118, 73],
        [88, 31, 35, 67, 102, 85, 55, 186, 85],
        [56, 21, 23, 111, 59, 205, 45, 37, 192],
        [55, 38, 70, 124, 73, 102, 1, 34, 98],
    ],
    [
        [102, 61, 71, 37, 34, 53, 31, 243, 192],
        [69, 60, 71, 38, 73, 119, 28, 222, 37],
        [68, 45, 128, 34, 1, 47, 11, 245, 171],
        [62, 17, 19, 70, 146, 85, 55, 62, 70],
        [75, 15, 9, 9, 64, 255, 184, 119, 16],
        [37, 43, 37, 154, 100, 163, 85, 160, 1],
        [63, 9, 92, 136, 28, 64, 32, 201, 85],
        [86, 6, 28, 5, 64, 255, 25, 248, 1],
        [56, 8, 17, 132, 137, 255, 55, 116, 128],
        [58, 15, 20, 82, 135, 57, 26, 121, 40],
    ],
    [
        [164, 50, 31, 137, 154, 133, 25, 35, 218],
        [51, 103, 44, 131, 131, 123, 31, 6, 158],
        [86, 40, 64, 135, 148, 224, 45, 183, 128],
        [22, 26, 17, 131, 240, 154, 14, 1, 209],
        [83, 12, 13, 54, 192, 255, 68, 47, 28],
        [45, 16, 21, 91, 64, 222, 7, 1, 197],
        [56, 21, 39, 155, 60, 138, 23, 102, 213],
        [85, 26, 85, 85, 128, 128, 32, 146, 171],
        [18, 11, 7, 63, 144, 171, 4, 4, 246],
        [35, 27, 10, 146, 174, 171, 12, 26, 128],
    ],
    [
        [190, 80, 35, 99, 180, 80, 126, 54, 45],
        [85, 126, 47, 87, 176, 51, 41, 20, 32],
        [101, 75, 128, 139, 118, 146, 116, 128, 85],
        [56, 41, 15, 176, 236, 85, 37, 9, 62],
        [146, 36, 19, 30, 171, 255, 97, 27, 20],
        [71, 30, 17, 119, 118, 255, 17, 18, 138],
        [101, 38, 60, 138, 55, 70, 43, 26, 142],
        [138, 45, 61, 62, 219, 1, 81, 188, 64],
        [32, 41, 20, 117, 151, 142, 20, 21, 163],
        [112, 19, 12, 61, 195, 128, 48, 4, 24],
    ],
];
pub const KEYFRAME_UV_MODE_TREE: [i8; 6] = [-DC_PRED, 2, -V_PRED, 4, -H_PRED, -TM_PRED];
pub const KEYFRAME_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];
pub const DC_PRED: i8 = 0;
pub const V_PRED: i8 = 1;
pub const H_PRED: i8 = 2;
pub const TM_PRED: i8 = 3;
pub const B_PRED: i8 = 4;
pub const B_DC_PRED: i8 = 0;
pub const B_TM_PRED: i8 = 1;
pub const B_VE_PRED: i8 = 2;
pub const B_HE_PRED: i8 = 3;
pub const B_LD_PRED: i8 = 4;
pub const B_RD_PRED: i8 = 5;
pub const B_VR_PRED: i8 = 6;
pub const B_VL_PRED: i8 = 7;
pub const B_HD_PRED: i8 = 8;
pub const B_HU_PRED: i8 = 9;

fn diff_fields(z: &Hdr, l: &Hdr) -> Vec<String> {
    let mut out = Vec::new();
    macro_rules! cmp {
        ($f:ident) => {
            if z.$f != l.$f {
                out.push(format!("{}: zen={:?} lib={:?}", stringify!($f), z.$f, l.$f));
            }
        };
    }
    cmp!(version);
    cmp!(color_space);
    cmp!(clamping);
    cmp!(segments_enabled);
    cmp!(seg_update_map);
    cmp!(seg_update_data);
    cmp!(seg_abs);
    cmp!(seg_q);
    cmp!(seg_lf);
    cmp!(seg_tree_probs);
    cmp!(filter_type);
    cmp!(filter_level);
    cmp!(sharpness);
    cmp!(lf_delta_enabled);
    cmp!(log2_parts);
    cmp!(q_yac);
    cmp!(q_ydc);
    cmp!(q_y2dc);
    cmp!(q_y2ac);
    cmp!(q_uvdc);
    cmp!(q_uvac);
    cmp!(refresh_entropy);
    cmp!(n_proba_updates);
    cmp!(use_skip);
    cmp!(skip_prob);
    out
}

struct ModeStream {
    y_modes: Vec<i8>, // per MB: B_PRED(-?) or I16 mode constant
    uv_modes: Vec<i8>,
    b_modes: Vec<[i8; 16]>, // valid when y == B_PRED
    skips: Vec<u32>,
    segments: Vec<u32>,
}

fn tree_decode(b: &mut Bool, tree: &[i8], probs: &[u8]) -> i8 {
    let mut i = 0usize;
    loop {
        let bit = b.bit(probs[i >> 1]);
        let v = tree[i + bit as usize];
        if v <= 0 {
            return -v;
        }
        i = v as usize;
    }
}

fn parse_modes(p: &[u8], mb_w: usize, mb_h: usize) -> (Hdr, ModeStream) {
    let hdr = parse_hdr(p);
    let tag = u32::from_le_bytes([p[0], p[1], p[2], 0]);
    let part1_size = (tag >> 5) as usize;
    let mut b = Bool::new(&p[10..10 + part1_size]);
    // replay the fixed header bits to position the reader
    replay_fixed_header(&mut b, &hdr);

    let mut ms = ModeStream {
        y_modes: Vec::new(),
        uv_modes: Vec::new(),
        b_modes: Vec::new(),
        skips: Vec::new(),
        segments: Vec::new(),
    };
    // b-mode contexts: above row of 4-per-MB, left col of 4
    let mut above: Vec<i8> = vec![B_DC_PRED; mb_w * 4];
    for _mby in 0..mb_h {
        let mut left: [i8; 4] = [B_DC_PRED; 4];
        for mbx in 0..mb_w {
            if hdr.segments_enabled == 1 && hdr.seg_update_map == 1 {
                let probs: Vec<u8> = hdr.seg_tree_probs.iter().map(|&x| x as u8).collect();
                let seg = if b.bit(probs[0]) == 0 {
                    b.bit(probs[1])
                } else {
                    2 + b.bit(probs[2])
                };
                ms.segments.push(seg);
            }
            if hdr.use_skip == 1 {
                ms.skips.push(b.bit(hdr.skip_prob as u8));
            }
            let y = tree_decode(&mut b, &KEYFRAME_YMODE_TREE, &KEYFRAME_YMODE_PROBS);
            ms.y_modes.push(y);
            let mut bm = [B_DC_PRED; 16];
            if y == B_PRED {
                for sy in 0..4 {
                    for sx in 0..4 {
                        let a = if sy == 0 {
                            above[mbx * 4 + sx]
                        } else {
                            bm[(sy - 1) * 4 + sx]
                        };
                        let l = if sx == 0 {
                            left[sy]
                        } else {
                            bm[sy * 4 + sx - 1]
                        };
                        let m = tree_decode(
                            &mut b,
                            &KEYFRAME_BPRED_MODE_TREE,
                            &KEYFRAME_BPRED_MODE_PROBS[a as usize][l as usize],
                        );
                        bm[sy * 4 + sx] = m;
                    }
                }
            } else {
                // implied sub-modes for context
                let imp = match y {
                    x if x == DC_PRED => B_DC_PRED,
                    x if x == V_PRED => B_VE_PRED,
                    x if x == H_PRED => B_HE_PRED,
                    _ => B_TM_PRED,
                };
                bm = [imp; 16];
            }
            for sx in 0..4 {
                above[mbx * 4 + sx] = bm[12 + sx];
            }
            for sy in 0..4 {
                left[sy] = bm[sy * 4 + 3];
            }
            ms.b_modes.push(bm);
            let uv = tree_decode(&mut b, &KEYFRAME_UV_MODE_TREE, &KEYFRAME_UV_MODE_PROBS);
            ms.uv_modes.push(uv);
        }
    }
    (hdr, ms)
}

fn replay_fixed_header(b: &mut Bool, h: &Hdr) {
    b.bit(128); // color space
    b.bit(128); // clamping
    b.bit(128); // segments_enabled
    if h.segments_enabled == 1 {
        b.bit(128); // update_map
        b.bit(128); // update_data
        if h.seg_update_data == 1 {
            b.bit(128); // abs
            for _ in 0..4 {
                b.flagged_signed(7);
            }
            for _ in 0..4 {
                b.flagged_signed(6);
            }
        }
        if h.seg_update_map == 1 {
            for _ in 0..3 {
                if b.bit(128) == 1 {
                    b.literal(8);
                }
            }
        }
    }
    b.bit(128); // filter type
    b.literal(6);
    b.literal(3);
    if b.bit(128) == 1 && b.bit(128) == 1 {
        for _ in 0..8 {
            b.flagged_signed(6);
        }
    }
    b.literal(2); // parts
    b.literal(7); // q_yac
    for _ in 0..5 {
        b.flagged_signed(4);
    }
    b.bit(128); // refresh entropy
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for l in 0..11 {
                    if b.bit(COEFF_UPDATE_PROBS[i][j][k][l]) == 1 {
                        b.literal(8);
                    }
                }
            }
        }
    }
    if b.bit(128) == 1 {
        b.literal(8); // skip prob
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/home/lilith/.cache/codec-corpus/v1/CID22/CID22-512/validation/382297.png");
    let (rgb, w, h) = load(img_path);
    let (mb_w, mb_h) = (w.div_ceil(16) as usize, h.div_ceil(16) as usize);
    println!("image: {img_path} {w}x{h} ({mb_w}x{mb_h} MBs)");

    // The grid is overridable from argv so this can chase cells outside q75.
    // That matters: after the Cat5/Cat6 fix (44ae3a0) the sns=0 configs are
    // ~98% byte-identical and 201 of the 237 remaining failures live in the
    // SNS + filter + multi-segment cells, which this tool could not reach while
    // the grid was hardcoded to q75.
    //   argv: [image] [q] [m] [sns] [flt] [segs]
    // With no overrides it runs the original two-config q75 grid across m2-m6.
    let arg_u8 = |i: usize| -> Option<u8> { args.get(i).and_then(|s| s.parse().ok()) };
    let grid: Vec<(u8, u8, u8, u8)> = match (arg_u8(2), arg_u8(4), arg_u8(5), arg_u8(6)) {
        (Some(q), Some(sns), Some(flt), Some(segs)) => vec![(q, sns, flt, segs)],
        _ => vec![(75u8, 0u8, 0u8, 1u8), (75, 50, 60, 4)],
    };
    let methods: Vec<u8> = match arg_u8(3) {
        Some(m) => vec![m],
        None => vec![2u8, 3, 4, 5, 6],
    };

    for (q, sns, flt, segs) in grid {
        for m in methods.iter().copied() {
            let cfg = LossyConfig::new()
                .with_quality(q as f32)
                .with_method(m)
                .with_segments(segs)
                .with_sns_strength(sns)
                .with_filter_strength(flt)
                .with_filter_sharpness(0)
                .with_cost_model(CostModel::StrictLibwebpParity);
            let zen = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
                .encode()
                .unwrap();
            let lib = webpx::EncoderConfig::new()
                .quality(q as f32)
                .method(m)
                .segments(segs)
                .sns_strength(sns)
                .filter_strength(flt)
                .filter_sharpness(0)
                .encode_rgb(&rgb, w, h, webpx::Unstoppable)
                .unwrap();
            let (zh, zm) = parse_modes(vp8_payload(&zen), mb_w, mb_h);
            let (lh, lm) = parse_modes(vp8_payload(&lib), mb_w, mb_h);
            for d in diff_fields(&zh, &lh) {
                println!("   {d}");
            }
            let n = zm.y_modes.len();
            let y_same = zm
                .y_modes
                .iter()
                .zip(&lm.y_modes)
                .filter(|(a, b)| a == b)
                .count();
            let uv_same = zm
                .uv_modes
                .iter()
                .zip(&lm.uv_modes)
                .filter(|(a, b)| a == b)
                .count();
            let b_same = zm
                .b_modes
                .iter()
                .zip(&lm.b_modes)
                .zip(zm.y_modes.iter().zip(&lm.y_modes))
                .filter(|((_, _), (zy, ly))| **zy == B_PRED && **ly == B_PRED)
                .filter(|((zb, lb), _)| zb == lb)
                .count();
            let both_b = zm
                .y_modes
                .iter()
                .zip(&lm.y_modes)
                .filter(|(z, l)| **z == B_PRED && **l == B_PRED)
                .count();
            let zen_b = zm.y_modes.iter().filter(|&&y| y == B_PRED).count();
            let lib_b = lm.y_modes.iter().filter(|&&y| y == B_PRED).count();
            let seg_same = if zm.segments.len() == lm.segments.len() && !zm.segments.is_empty() {
                format!(
                    " seg_same={:.1}%",
                    100.0
                        * zm.segments
                            .iter()
                            .zip(&lm.segments)
                            .filter(|(a, b)| a == b)
                            .count() as f64
                        / n as f64
                )
            } else {
                String::new()
            };
            println!(
                "m{m} sns{sns} flt{flt} segs{segs}: y_same {:.1}%  uv_same {:.1}%  b4_same {}/{}  i4-count zen {} lib {}{}",
                100.0 * y_same as f64 / n as f64,
                100.0 * uv_same as f64 / n as f64,
                b_same,
                both_b,
                zen_b,
                lib_b,
                seg_same
            );
        }
    }
}
