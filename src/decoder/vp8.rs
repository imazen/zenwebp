//! An implementation of the VP8 Video Codec
//!
//! This module contains a partial implementation of the
//! VP8 video format as defined in RFC-6386.
//!
//! It decodes Keyframes only.
//! VP8 is the underpinning of the WebP image format
//!
//! # Related Links
//! * [rfc-6386](http://tools.ietf.org/html/rfc6386) - The VP8 Data Format and Decoding Guide
//! * [VP8.pdf](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37073.pdf) - An overview of of the VP8 format

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_else_if)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::default::Default;

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use archmage::SimdToken;

use super::api::{DecodeError, UpsamplingMethod};
use super::internal_error::InternalDecodeError;
use super::yuv;
use crate::common::prediction::*;
use crate::common::types::*;
use crate::slice_reader::SliceReader;

use super::bit_reader::{ActivePartitionReader, VP8HeaderBitReader, VP8Partitions};
use super::loop_filter_dispatch;
use crate::common::transform;
use loop_filter_dispatch::*;

/// Maximum stride supported for bounds-check-free loop filtering.
/// WebP max dimension is 16383, rounded up to MB boundary = 16384.
const MAX_FILTER_STRIDE: usize = 16384;

/// Padding added to pixel buffers for bounds-check-free loop filtering.
/// Allows fixed-size region extraction without per-access bounds checks.
/// Size: 7 * max_stride + 16 bytes (covers 8 rows for normal filter: p3-p0, q0-q3).
const FILTER_PADDING: usize = 7 * MAX_FILTER_STRIDE + 16;

// ============================================================================
// Diagnostic Types for I4 Encoding Efficiency Analysis
// ============================================================================

/// Raw quantized coefficient levels for a single 4x4 block (pre-dequantization).
/// Captures the exact values written to the bitstream for comparison with libwebp.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockDiagnostic {
    /// Raw quantized levels in zigzag order (before dequantization)
    pub levels: [i32; 16],
    /// Number of coefficients decoded (0 = all-zero block, position of last nonzero + 1)
    pub eob_position: u8,
}

/// Diagnostic capture for a single macroblock's encoded state.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct MacroblockDiagnostic {
    /// Luma prediction mode for this macroblock.
    pub luma_mode: LumaMode,
    /// Chroma prediction mode for this macroblock.
    pub chroma_mode: ChromaMode,
    /// Segment index (0-3).
    pub segment_id: u8,
    /// Whether all coefficients were skipped (zero block).
    pub coeffs_skipped: bool,
    /// I4 sub-block prediction modes (only valid when luma_mode == LumaMode::B)
    pub bpred_modes: [IntraMode; 16],
    /// Y2 (WHT) block coefficients (only used for non-I4 modes)
    pub y2_block: BlockDiagnostic,
    /// 16 Y blocks (4x4 each)
    pub y_blocks: [BlockDiagnostic; 16],
    /// 8 UV blocks (4 U + 4 V)
    pub uv_blocks: [BlockDiagnostic; 8],
}

/// Complete diagnostic capture for a decoded VP8 frame.
/// Allows comparison of intermediate encoding state between zenwebp and libwebp.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DiagnosticFrame {
    /// Width in macroblocks.
    pub mb_width: u16,
    /// Height in macroblocks.
    pub mb_height: u16,
    /// Per-segment quantizer values: (ydc, yac, y2dc, y2ac, uvdc, uvac)
    pub segments: [(i16, i16, i16, i16, i16, i16); 4],
    /// All macroblocks in raster order
    pub macroblocks: Vec<MacroblockDiagnostic>,
    /// Final token probability tables (for comparing probability updates)
    pub token_probs: Box<TokenProbTreeNodes>,
    /// Size of partition 0 (header + mode data)
    pub partition0_size: u32,
}

/// VP8 probability tree node for coefficient decoding.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct TreeNode {
    /// Left branch index or value.
    pub left: u8,
    /// Right branch index or value.
    pub right: u8,
    /// Probability for this branch.
    pub prob: Prob,
    /// Index in the tree.
    pub index: u8,
}

impl TreeNode {
    const UNINIT: TreeNode = TreeNode {
        left: 0,
        right: 0,
        prob: 0,
        index: 0,
    };

    const fn prepare_branch(t: i8) -> u8 {
        if t > 0 {
            (t as u8) / 2
        } else {
            let value = -t;
            0x80 | (value as u8)
        }
    }

    pub(crate) const fn value_from_branch(t: u8) -> i8 {
        (t & !0x80) as i8
    }
}

const fn tree_nodes_from<const N: usize, const M: usize>(
    tree: [i8; N],
    probs: [Prob; M],
) -> [TreeNode; M] {
    if N != 2 * M {
        panic!("invalid tree with probs");
    }
    let mut nodes = [TreeNode::UNINIT; M];
    let mut i = 0;
    while i < M {
        nodes[i].left = TreeNode::prepare_branch(tree[2 * i]);
        nodes[i].right = TreeNode::prepare_branch(tree[2 * i + 1]);
        nodes[i].prob = probs[i];
        nodes[i].index = i as u8;
        i += 1;
    }
    nodes
}

const SEGMENT_TREE_NODE_DEFAULTS: [TreeNode; 3] = tree_nodes_from(SEGMENT_ID_TREE, [255; 3]);

const KEYFRAME_YMODE_NODES: [TreeNode; 4] =
    tree_nodes_from(KEYFRAME_YMODE_TREE, KEYFRAME_YMODE_PROBS);

const KEYFRAME_BPRED_MODE_NODES: [[[TreeNode; 9]; 10]; 10] = {
    let mut output = [[[TreeNode::UNINIT; 9]; 10]; 10];
    let mut i = 0;
    while i < output.len() {
        let mut j = 0;
        while j < output[i].len() {
            output[i][j] =
                tree_nodes_from(KEYFRAME_BPRED_MODE_TREE, KEYFRAME_BPRED_MODE_PROBS[i][j]);
            j += 1;
        }
        i += 1;
    }
    output
};

const KEYFRAME_UV_MODE_NODES: [TreeNode; 3] =
    tree_nodes_from(KEYFRAME_UV_MODE_TREE, KEYFRAME_UV_MODE_PROBS);

type TokenProbTreeNodes = [[[[TreeNode; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];

/// Position-indexed probability table for faster coefficient reading.
/// Indexed by [plane][coeff_position][context] instead of [plane][band][context].
/// This eliminates the COEFF_BANDS lookup in the hot path.
/// Position 16 is a sentinel (copies band 7) for n+1 lookahead.
type TokenProbsByPosition = [[[[TreeNode; NUM_DCT_TOKENS - 1]; 3]; 17]; 4];

const COEFF_PROB_NODES: TokenProbTreeNodes = {
    let mut output = [[[[TreeNode::UNINIT; 11]; 3]; 8]; 4];
    let mut i = 0;
    while i < output.len() {
        let mut j = 0;
        while j < output[i].len() {
            let mut k = 0;
            while k < output[i][j].len() {
                output[i][j][k] = tree_nodes_from(DCT_TOKEN_TREE, COEFF_PROBS[i][j][k]);
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    output
};

#[derive(Default, Clone, Copy)]
struct MacroBlock {
    bpred: [IntraMode; 16],
    luma_mode: LumaMode,
    chroma_mode: ChromaMode,
    segmentid: u8,
    coeffs_skipped: bool,
    non_zero_dct: bool,
    /// True if any UV sub-block has non-zero AC coefficients.
    /// Used to suppress dithering on blocks with actual chroma detail.
    has_nonzero_uv_ac: bool,
    /// Per-block non-zero bitmap. Bit i set means block i has non-zero coefficients.
    /// Blocks 0-15 = Y, 16-19 = U, 20-23 = V.
    /// Used to skip IDCT on zero blocks (matches libwebp's non_zero_y/non_zero_uv).
    non_zero_blocks: u32,
}

/// Info required from a previously decoded macro block in future
/// For the top macroblocks this will be the bottom values, for the left macroblock the right values
#[derive(Default, Clone, Copy)]
struct PreviousMacroBlock {
    bpred: [IntraMode; 4],
    // complexity is laid out like: y2,y,y,y,y,u,u,v,v
    complexity: [u8; 9],
}

/// A Representation of the last decoded video frame
#[derive(Default, Debug, Clone)]
pub struct Frame {
    /// The width of the luma plane
    pub width: u16,

    /// The height of the luma plane
    pub height: u16,

    /// The luma plane of the frame
    pub ybuf: Vec<u8>,

    /// The blue plane of the frame
    pub ubuf: Vec<u8>,

    /// The red plane of the frame
    pub vbuf: Vec<u8>,

    pub(crate) version: u8,

    /// Indicates whether this frame is intended for display
    pub for_display: bool,

    // Section 9.2
    /// The pixel type of the frame as defined by Section 9.2
    /// of the VP8 Specification
    pub pixel_type: u8,

    // Section 9.4 and 15
    pub(crate) filter_type: bool, //if true uses simple filter // if false uses normal filter
    pub(crate) filter_level: u8,
    pub(crate) sharpness_level: u8,
}

impl Frame {
    const fn chroma_width(&self) -> u16 {
        self.width.div_ceil(2)
    }

    const fn buffer_width(&self) -> u16 {
        let difference = self.width % 16;
        if difference > 0 {
            self.width + (16 - difference % 16)
        } else {
            self.width
        }
    }

    /// Fills an rgb buffer from the YUV buffers
    pub(crate) fn fill_rgb(&self, buf: &mut [u8], upsampling_method: UpsamplingMethod) {
        const BPP: usize = 3;

        match upsampling_method {
            UpsamplingMethod::Bilinear => {
                yuv::fill_rgb_buffer_fancy::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.height),
                    usize::from(self.buffer_width()),
                );
            }
            UpsamplingMethod::Simple => {
                yuv::fill_rgb_buffer_simple::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.chroma_width()),
                    usize::from(self.buffer_width()),
                );
            }
        }
    }

    /// Fills an rgba buffer from the YUV buffers
    pub(crate) fn fill_rgba(&self, buf: &mut [u8], upsampling_method: UpsamplingMethod) {
        const BPP: usize = 4;

        match upsampling_method {
            UpsamplingMethod::Bilinear => {
                yuv::fill_rgb_buffer_fancy::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.height),
                    usize::from(self.buffer_width()),
                );
            }
            UpsamplingMethod::Simple => {
                yuv::fill_rgb_buffer_simple::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.chroma_width()),
                    usize::from(self.buffer_width()),
                );
            }
        }
    }
    /// Gets the buffer size
    #[must_use]
    pub fn get_buf_size(&self) -> usize {
        self.ybuf.len() * 3
    }
}

/// Decode large coefficient values (categories 3-6).
///
/// Separated from `read_coefficients` to keep the hot path small,
/// matching C's `GetLargeValue`/`GetCoeffsFast` split. Both functions
/// are `#[inline(never)]` so the CPU branch predictor tracks each
/// through a single set of addresses, reducing BTB aliasing.
#[inline(never)]
fn get_large_value(
    reader: &mut ActivePartitionReader<'_>,
    prob: &[TreeNode; NUM_DCT_TOKENS - 1],
) -> i32 {
    if reader.get_bit(prob[6].prob) == 0 {
        if reader.get_bit(prob[7].prob) == 0 {
            5 + reader.get_bit(159)
        } else {
            7 + 2 * reader.get_bit(165) + reader.get_bit(145)
        }
    } else {
        let bit1 = reader.get_bit(prob[8].prob);
        let bit0 = reader.get_bit(prob[9 + bit1 as usize].prob);
        let cat = (2 * bit1 + bit0) as usize;

        let cat_probs = &PROB_DCT_CAT[2 + cat];
        let cat_len = [3usize, 4, 5, 11][cat];
        let mut extra = 0i32;
        for i in 0..cat_len {
            extra = extra + extra + reader.get_bit(cat_probs[i]);
        }
        3 + (8 << cat) + extra
    }
}

/// Read and dequantize DCT coefficients for one 4x4 block.
///
/// Returns true if any coefficient beyond `first` was non-zero.
///
/// NOT inlined so the CPU branch predictor tracks all coefficient
/// decoding through a single set of branch addresses. When inlined 25x
/// (once per block in a macroblock), BTB aliasing causes ~2.3M extra
/// mispredicts per decode (measured via cachegrind). The function call
/// overhead is compensated by the mispredict savings.
///
/// Does NOT check for EOF — caller must check `reader.is_eof()` after
/// processing all blocks in the macroblock (matching libwebp's approach
/// where VP8DecodeMB checks EOF once per MB, not per block).
#[inline(never)]
fn read_coefficients(
    reader: &mut ActivePartitionReader<'_>,
    output: &mut [i32; 16],
    probs: &[[[TreeNode; NUM_DCT_TOKENS - 1]; 3]; 17],
    first: usize,
    complexity: usize,
    dq: [i32; 2], // [dc_quant, ac_quant] — indexed by (n > 0) like libwebp
) -> bool {
    debug_assert!(complexity <= 2);

    let mut n = first;
    let mut prob = &probs[n][complexity];

    while n < 16 {
        if reader.get_bit(prob[0].prob) == 0 {
            break;
        }

        while reader.get_bit(prob[1].prob) == 0 {
            n += 1;
            if n >= 16 {
                return true;
            }
            prob = &probs[n][0];
        }

        let v: i32;
        let next_ctx: usize;

        if reader.get_bit(prob[2].prob) == 0 {
            v = 1;
            next_ctx = 1;
        } else {
            if reader.get_bit(prob[3].prob) == 0 {
                if reader.get_bit(prob[4].prob) == 0 {
                    v = 2;
                } else {
                    v = 3 + reader.get_bit(prob[5].prob);
                }
            } else {
                v = get_large_value(reader, prob);
            }
            next_ctx = 2;
        }

        // Branchless sign reading (VP8GetSigned) + dequantize with array lookup
        output[ZIGZAG[n] as usize] = reader.get_signed(v) * dq[(n > 0) as usize];

        n += 1;
        if n < 16 {
            prob = &probs[n][next_ctx];
        }
    }

    n > first
}

/// VP8 Decoder
///
/// Only decodes keyframes
pub struct Vp8Decoder<'a> {
    r: SliceReader<'a>,
    b: VP8HeaderBitReader,

    mbwidth: u16,
    mbheight: u16,
    macroblocks: Vec<MacroBlock>,

    frame: Frame,

    segments_enabled: bool,
    segments_update_map: bool,
    segment: [Segment; MAX_SEGMENTS],

    loop_filter_adjustments_enabled: bool,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],

    partitions: VP8Partitions,
    num_partitions: u8,

    segment_tree_nodes: [TreeNode; 3],
    token_probs: Box<TokenProbTreeNodes>,
    /// Position-indexed probability table (populated from token_probs after parsing)
    token_probs_by_pos: Box<TokenProbsByPosition>,

    // Section 9.11
    prob_skip_false: Option<Prob>,

    top: Vec<PreviousMacroBlock>,
    left: PreviousMacroBlock,

    // The borders from the previous macroblock, used for predictions
    // See Section 12
    // Note that the left border contains the top left pixel
    top_border_y: Vec<u8>,
    left_border_y: Vec<u8>,

    top_border_u: Vec<u8>,
    left_border_u: Vec<u8>,

    top_border_v: Vec<u8>,
    left_border_v: Vec<u8>,

    // Row cache for better cache locality during loop filtering
    // Layout: [extra_rows for filter context][current row of macroblocks]
    // The extra_rows area holds the bottom rows from the previous macroblock row,
    // needed for filtering the top edge of the current row.
    cache_y: Vec<u8>,
    cache_u: Vec<u8>,
    cache_v: Vec<u8>,
    cache_y_stride: usize,  // mbwidth * 16
    cache_uv_stride: usize, // mbwidth * 8
    extra_y_rows: usize,    // 8 for normal filter, 2 for simple, 0 for none

    // Reusable coefficient buffer for macroblock decoding.
    // Initialized to zeros and maintained as zeros between macroblocks.
    // Each 16-element block is cleared after use in intra_predict_*.
    coeff_blocks: [i32; 384],

    // Reusable prediction workspaces — avoids re-zeroing 544+288+288=1120 bytes per macroblock.
    // update_border_* writes all border pixels; prediction functions write all interior pixels
    // in raster order before any reads, so zero-init is not needed between macroblocks.
    luma_ws: [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: [u8; CHROMA_BLOCK_SIZE],

    // Diagnostic capture (None for normal decoding, Some for diagnostic mode)
    diagnostic_capture: Option<Vec<MacroblockDiagnostic>>,
    current_mb_diag: Option<MacroblockDiagnostic>,

    // Partition 0 size (header + mode data) for diagnostic reporting
    first_partition_size: u32,

    // Cooperative cancellation token
    stop: Option<&'a dyn enough::Stop>,

    // Chroma dithering state
    dither_strength_pending: u8,
    dither_enabled: bool,
    dither_rg: super::dither::VP8Random,
    dither_amp: [i32; MAX_SEGMENTS],
    /// UV AC quantizer indices per segment (for dithering amplitude computation)
    uv_quant_indices: [i32; MAX_SEGMENTS],

    // Reusable filter parameter buffer — avoids per-row heap allocation.
    mb_filter_params: Vec<MbFilterParams>,
}

impl<'a> Vp8Decoder<'a> {
    /// Create a new decoder.
    /// The data must be a raw vp8 bitstream
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let r = SliceReader::new(data);
        let f = Frame::default();

        Self {
            r,
            b: VP8HeaderBitReader::new(),

            mbwidth: 0,
            mbheight: 0,
            macroblocks: Vec::new(),

            frame: f,
            segments_enabled: false,
            segments_update_map: false,
            segment: array::from_fn(|_| Segment::default()),

            loop_filter_adjustments_enabled: false,
            ref_delta: [0; 4],
            mode_delta: [0; 4],

            partitions: VP8Partitions::new(),

            num_partitions: 1,

            segment_tree_nodes: SEGMENT_TREE_NODE_DEFAULTS,
            token_probs: Box::new(COEFF_PROB_NODES),
            token_probs_by_pos: Box::new([[[[TreeNode::UNINIT; 11]; 3]; 17]; 4]),

            // Section 9.11
            prob_skip_false: None,

            top: Vec::new(),
            left: PreviousMacroBlock::default(),

            top_border_y: Vec::new(),
            left_border_y: Vec::new(),

            top_border_u: Vec::new(),
            left_border_u: Vec::new(),

            top_border_v: Vec::new(),
            left_border_v: Vec::new(),

            cache_y: Vec::new(),
            cache_u: Vec::new(),
            cache_v: Vec::new(),
            cache_y_stride: 0,
            cache_uv_stride: 0,
            extra_y_rows: 0,

            coeff_blocks: [0i32; 384],

            luma_ws: [0u8; LUMA_BLOCK_SIZE],
            chroma_u_ws: [0u8; CHROMA_BLOCK_SIZE],
            chroma_v_ws: [0u8; CHROMA_BLOCK_SIZE],

            diagnostic_capture: None,
            current_mb_diag: None,

            first_partition_size: 0,

            stop: None,

            dither_strength_pending: 0,
            dither_enabled: false,
            dither_rg: super::dither::VP8Random::new(),
            dither_amp: [0; MAX_SEGMENTS],
            uv_quant_indices: [0; MAX_SEGMENTS],

            mb_filter_params: Vec::new(),
        }
    }

    fn update_token_probabilities(&mut self) -> Result<(), DecodeError> {
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if self.b.read_bool(*prob) {
                            let v = self.b.read_literal(8);
                            self.token_probs[i][j][k][t].prob = v;
                        }
                    }
                }
            }
        }
        Ok(self.b.check(())?)
    }

    /// Populate the position-indexed probability table from token_probs.
    /// This eliminates the COEFF_BANDS lookup in the coefficient reading hot path.
    fn populate_probs_by_position(&mut self) {
        for plane in 0..4 {
            for pos in 0..17 {
                // Position 16 uses band 7 (sentinel for n+1 lookahead)
                let band = if pos < 16 {
                    COEFF_BANDS[pos] as usize
                } else {
                    7
                };
                for ctx in 0..3 {
                    self.token_probs_by_pos[plane][pos][ctx] = self.token_probs[plane][band][ctx];
                }
            }
        }
    }

    fn init_partitions(&mut self, n: usize) -> Result<(), DecodeError> {
        use byteorder_lite::{ByteOrder, LittleEndian};

        let mut all_data = Vec::new();
        let mut boundaries = Vec::with_capacity(n);

        if n > 1 {
            let mut sizes = vec![0; 3 * n - 3];
            self.r.read_exact(sizes.as_mut_slice())?;

            for s in sizes.chunks(3) {
                let size = LittleEndian::read_u24(s) as usize;

                let start = all_data.len();
                all_data.resize(start + size, 0);
                self.r.read_exact(&mut all_data[start..start + size])?;
                boundaries.push((start, size));
            }
        }

        // Last partition - read to end
        let start = all_data.len();
        self.r.read_to_end(&mut all_data)?;
        let size = all_data.len() - start;
        boundaries.push((start, size));

        self.partitions.init(all_data, &boundaries);

        Ok(())
    }

    fn read_quantization_indices(&mut self) -> Result<(), DecodeError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let yac_abs = self.b.read_literal(7);
        let ydc_delta = self.b.read_optional_signed_value(4);
        let y2dc_delta = self.b.read_optional_signed_value(4);
        let y2ac_delta = self.b.read_optional_signed_value(4);
        let uvdc_delta = self.b.read_optional_signed_value(4);
        let uvac_delta = self.b.read_optional_signed_value(4);

        let n = if self.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for i in 0usize..n {
            let base = i32::from(if self.segments_enabled {
                if self.segment[i].delta_values {
                    i16::from(self.segment[i].quantizer_level) + i16::from(yac_abs)
                } else {
                    i16::from(self.segment[i].quantizer_level)
                }
            } else {
                i16::from(yac_abs)
            });

            self.segment[i].ydc = dc_quant(base + ydc_delta);
            self.segment[i].yac = ac_quant(base);

            self.segment[i].y2dc = dc_quant(base + y2dc_delta) * 2;
            // The intermediate result (max`284*155`) can be larger than the `i16` range.
            self.segment[i].y2ac = (i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16;

            self.segment[i].uvdc = dc_quant(base + uvdc_delta);
            self.segment[i].uvac = ac_quant(base + uvac_delta);
            // Store UV AC quantizer index for dithering amplitude computation
            self.uv_quant_indices[i] = base + uvac_delta;

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        Ok(self.b.check(())?)
    }

    fn read_loop_filter_adjustments(&mut self) -> Result<(), DecodeError> {
        if self.b.read_flag() {
            for i in 0usize..4 {
                self.ref_delta[i] = self.b.read_optional_signed_value(6);
            }

            for i in 0usize..4 {
                self.mode_delta[i] = self.b.read_optional_signed_value(6);
            }
        }

        Ok(self.b.check(())?)
    }

    fn read_segment_updates(&mut self) -> Result<(), DecodeError> {
        // Section 9.3
        self.segments_update_map = self.b.read_flag();
        let update_segment_feature_data = self.b.read_flag();

        if update_segment_feature_data {
            let segment_feature_mode = self.b.read_flag();

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].quantizer_level = self.b.read_optional_signed_value(7) as i8;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].loopfilter_level = self.b.read_optional_signed_value(6) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0usize..3 {
                let update = self.b.read_flag();

                let prob = if update { self.b.read_literal(8) } else { 255 };
                self.segment_tree_nodes[i].prob = prob;
            }
        }

        Ok(self.b.check(())?)
    }

    fn read_frame_header(&mut self) -> Result<(), DecodeError> {
        let tag = self.r.read_u24_le()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodeError::UnsupportedFeature(
                "Non-keyframe frames".into(),
            ));
        }

        self.frame.version = ((tag >> 1) & 7) as u8;
        self.frame.for_display = (tag >> 4) & 1 != 0;

        let first_partition_size = tag >> 5;
        self.first_partition_size = first_partition_size;

        let mut tag = [0u8; 3];
        self.r.read_exact(&mut tag)?;

        if tag != [0x9d, 0x01, 0x2a] {
            return Err(DecodeError::Vp8MagicInvalid(tag));
        }

        let w = self.r.read_u16_le()?;
        let h = self.r.read_u16_le()?;

        self.frame.width = w & 0x3FFF;
        self.frame.height = h & 0x3FFF;

        self.mbwidth = self.frame.width.div_ceil(16);
        self.mbheight = self.frame.height.div_ceil(16);

        // defaults are intra mode DC and complexity 0
        self.top = vec![PreviousMacroBlock::default(); self.mbwidth.into()];
        self.left = PreviousMacroBlock::default();

        // Pre-allocate macroblocks to avoid repeated Vec reallocation in decode loop
        self.macroblocks =
            Vec::with_capacity(usize::from(self.mbwidth) * usize::from(self.mbheight));

        // Frame buffers don't need FILTER_PADDING — padding is only needed on
        // cache buffers where loop filtering operates. Frame buffers receive the
        // final output from output_row_from_cache via copy_from_slice within bounds.
        self.frame.ybuf = vec![
            0u8;
            usize::from(self.mbwidth) * 16 * usize::from(self.mbheight) * 16
        ];
        self.frame.ubuf = vec![
            0u8;
            usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8
        ];
        self.frame.vbuf = vec![
            0u8;
            usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8
        ];

        self.top_border_y = vec![127u8; self.frame.width as usize + 4 + 16];
        self.left_border_y = vec![129u8; 1 + 16];

        // 8 pixels per macroblock
        self.top_border_u = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_u = vec![129u8; 1 + 8];

        self.top_border_v = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_v = vec![129u8; 1 + 8];

        // Initialize row cache for better cache locality during loop filtering
        // We allocate with max extra_rows (8 for normal filter) - actual value set after reading filter_type
        self.cache_y_stride = usize::from(self.mbwidth) * 16;
        self.cache_uv_stride = usize::from(self.mbwidth) * 8;
        // extra_y_rows will be set properly after we read filter_type, for now use 0
        self.extra_y_rows = 0;

        let size = first_partition_size as usize;
        let mut data = vec![0u8; size];
        self.r.read_exact(&mut data)?;

        // initialise binary decoder
        self.b.init(data)?;

        let color_space = self.b.read_literal(1);
        self.frame.pixel_type = self.b.read_literal(1);

        if color_space != 0 {
            return Err(DecodeError::ColorSpaceInvalid(color_space));
        }

        self.segments_enabled = self.b.read_flag();
        if self.segments_enabled {
            self.read_segment_updates()?;
        }

        self.frame.filter_type = self.b.read_flag();
        self.frame.filter_level = self.b.read_literal(6);
        self.frame.sharpness_level = self.b.read_literal(3);

        self.loop_filter_adjustments_enabled = self.b.read_flag();
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments()?;
        }

        let num_partitions = 1 << self.b.read_literal(2) as usize;
        self.b.check(())?;

        // Now that we know filter_type, allocate the row cache
        // extra_rows: 8 for normal filter, 2 for simple filter, 0 for no filter
        self.extra_y_rows = if self.frame.filter_level == 0 {
            0
        } else if self.frame.filter_type {
            2 // simple filter
        } else {
            8 // normal filter
        };
        let extra_uv_rows = self.extra_y_rows / 2;

        // Cache layout: [extra_rows][16 rows for current macroblock row]
        // extra_rows holds bottom rows from previous MB row for filter context
        // FILTER_PADDING allows fixed-size region extraction for bounds-check-free filtering.
        // Uses MAX_FILTER_STRIDE to match compile-time constant region sizes in loop_filter_avx2.
        let cache_y_rows = self.extra_y_rows + 16;
        let cache_uv_rows = extra_uv_rows + 8;
        self.cache_y = vec![128u8; cache_y_rows * self.cache_y_stride + FILTER_PADDING];
        self.cache_u = vec![128u8; cache_uv_rows * self.cache_uv_stride + FILTER_PADDING];
        self.cache_v = vec![128u8; cache_uv_rows * self.cache_uv_stride + FILTER_PADDING];

        self.num_partitions = num_partitions as u8;
        self.init_partitions(num_partitions)?;

        self.read_quantization_indices()?;

        // Refresh entropy probs ?????
        let _ = self.b.read_literal(1);

        self.update_token_probabilities()?;
        self.populate_probs_by_position();

        let mb_no_skip_coeff = self.b.read_literal(1);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(self.b.read_literal(8))
        } else {
            None
        };
        self.b.check(())?;
        Ok(())
    }

    fn read_macroblock_header(&mut self, mbx: usize) -> Result<MacroBlock, InternalDecodeError> {
        let mut mb = MacroBlock::default();

        if self.segments_enabled && self.segments_update_map {
            mb.segmentid = self.b.read_with_tree(&self.segment_tree_nodes) as u8;
        };

        mb.coeffs_skipped = if let Some(prob) = self.prob_skip_false {
            self.b.read_bool(prob)
        } else {
            false
        };

        // intra prediction
        let luma = self.b.read_with_tree(&KEYFRAME_YMODE_NODES);
        mb.luma_mode =
            LumaMode::from_i8(luma).ok_or(InternalDecodeError::LumaPredictionModeInvalid)?;

        // Extract top[mbx] once to eliminate per-access bounds checks in the B-mode loop
        let top_mb = &mut self.top[mbx];
        match mb.luma_mode.into_intra() {
            // `LumaMode::B` - This is predicted individually
            None => {
                for y in 0usize..4 {
                    for x in 0usize..4 {
                        let top = top_mb.bpred[x];
                        let left = self.left.bpred[y];
                        let intra = self.b.read_with_tree(
                            &KEYFRAME_BPRED_MODE_NODES[top as usize][left as usize],
                        );
                        let bmode = IntraMode::from_i8(intra)
                            .ok_or(InternalDecodeError::IntraPredictionModeInvalid)?;
                        mb.bpred[x + y * 4] = bmode;

                        top_mb.bpred[x] = bmode;
                        self.left.bpred[y] = bmode;
                    }
                }
            }
            Some(mode) => {
                for i in 0usize..4 {
                    mb.bpred[12 + i] = mode;
                    self.left.bpred[i] = mode;
                }
            }
        }

        let chroma = self.b.read_with_tree(&KEYFRAME_UV_MODE_NODES);
        mb.chroma_mode =
            ChromaMode::from_i8(chroma).ok_or(InternalDecodeError::ChromaPredictionModeInvalid)?;

        // top should store the bottom of the current bpred, which is the final 4 values
        top_mb.bpred = mb.bpred[12..].try_into().unwrap();

        self.b.check(mb)
    }

    fn intra_predict_luma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        simd_token: SimdTokenType,
    ) {
        let stride = LUMA_STRIDE;
        let mw = self.mbwidth as usize;
        // Reuse persistent workspace — only borders are updated; interior is overwritten
        // by prediction + IDCT before any reads occur (raster-order processing invariant).
        let ws = &mut self.luma_ws;
        update_border_luma(ws, mbx, mby, mw, &self.top_border_y, &self.left_border_y);

        let nz = mb.non_zero_blocks;

        match mb.luma_mode {
            LumaMode::V => predict_vpred(ws, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(ws, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(ws, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(ws, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => {
                // B-mode: predict and add residue for each 4x4 sub-block
                for sby in 0usize..4 {
                    for sbx in 0usize..4 {
                        let i = sbx + sby * 4;
                        let y0 = sby * 4 + 1;
                        let x0 = sbx * 4 + 1;

                        match mb.bpred[i] {
                            IntraMode::TM => predict_tmpred(ws, 4, x0, y0, stride),
                            IntraMode::VE => predict_bvepred(ws, x0, y0, stride),
                            IntraMode::HE => predict_bhepred(ws, x0, y0, stride),
                            IntraMode::DC => predict_bdcpred(ws, x0, y0, stride),
                            IntraMode::LD => predict_bldpred(ws, x0, y0, stride),
                            IntraMode::RD => predict_brdpred(ws, x0, y0, stride),
                            IntraMode::VR => predict_bvrpred(ws, x0, y0, stride),
                            IntraMode::VL => predict_bvlpred(ws, x0, y0, stride),
                            IntraMode::HD => predict_bhdpred(ws, x0, y0, stride),
                            IntraMode::HU => predict_bhupred(ws, x0, y0, stride),
                        }

                        let rb: &mut [i32; 16] =
                            (&mut self.coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        // Skip IDCT for zero blocks (matches libwebp's DoTransform case 0)
                        if nz & (1u32 << i) != 0 {
                            if let Some(token) = simd_token {
                                idct_add_residue_and_clear_with_token(
                                    token, ws, rb, y0, x0, stride,
                                );
                            } else {
                                idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                            }
                        } else {
                            // Verify the block is truly zero (debug assertion)
                            debug_assert!(rb.iter().all(|&c| c == 0),
                                "Block {i} has non-zero coeffs but non_zero_blocks bit not set");
                        }
                    }
                }
            }
        }

        if mb.luma_mode != LumaMode::B {
            // Skip IDCT entirely if no Y blocks have non-zero coefficients
            // (matches libwebp's `if (bits != 0)` check in ReconstructRow)
            if nz & 0xFFFF != 0 {
                for y in 0usize..4 {
                    for x in 0usize..4 {
                        let i = x + y * 4;
                        let rb: &mut [i32; 16] =
                            (&mut self.coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        if nz & (1u32 << i) != 0 {
                            let y0 = 1 + y * 4;
                            let x0 = 1 + x * 4;

                            if let Some(token) = simd_token {
                                idct_add_residue_and_clear_with_token(
                                    token, ws, rb, y0, x0, stride,
                                );
                            } else {
                                idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                            }
                        } else {
                            debug_assert!(rb.iter().all(|&c| c == 0),
                                "Block {i} has non-zero coeffs but non_zero_blocks bit not set");
                        }
                    }
                }
            }
        }

        self.left_border_y[0] = ws[16];

        for (i, left) in self.left_border_y[1..][..16].iter_mut().enumerate() {
            *left = ws[(i + 1) * stride + 16];
        }

        self.top_border_y[mbx * 16..][..16].copy_from_slice(&ws[16 * stride + 1..][..16]);

        // Write to row cache instead of final buffer
        // Cache layout: [extra_y_rows rows][16 rows for current MB row]
        // Pre-slice the destination region once to eliminate per-iteration bounds checks.
        let cache_y_offset = self.extra_y_rows * self.cache_y_stride;
        let cache_y_stride = self.cache_y_stride;
        let region_start = cache_y_offset + mbx * 16;
        let region_len = 15 * cache_y_stride + 16;
        let cache_region = &mut self.cache_y[region_start..region_start + region_len];
        for y in 0usize..16 {
            let src_start = (1 + y) * stride + 1;
            cache_region[y * cache_y_stride..][..16].copy_from_slice(&ws[src_start..][..16]);
        }
    }

    fn intra_predict_chroma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        simd_token: SimdTokenType,
    ) {
        let stride = CHROMA_STRIDE;

        // Reuse persistent workspaces — avoids re-zeroing 288 bytes × 2 per macroblock.
        let uws = &mut self.chroma_u_ws;
        let vws = &mut self.chroma_v_ws;
        update_border_chroma(uws, mbx, mby, &self.top_border_u, &self.left_border_u);
        update_border_chroma(vws, mbx, mby, &self.top_border_v, &self.left_border_v);

        match mb.chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(uws, 8, stride, mby != 0, mbx != 0);
                predict_dcpred(vws, 8, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(uws, 8, 1, 1, stride);
                predict_vpred(vws, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(uws, 8, 1, 1, stride);
                predict_hpred(vws, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tmpred(uws, 8, 1, 1, stride);
                predict_tmpred(vws, 8, 1, 1, stride);
            }
        }

        let nz = mb.non_zero_blocks;
        // Skip all UV IDCT if no UV blocks have non-zero coefficients
        // (matches libwebp's DoUVTransform `if (bits & 0xff)` check)
        if nz & 0xFF_0000 != 0 {
            for y in 0usize..2 {
                for x in 0usize..2 {
                    let i = x + y * 2;
                    let u_idx = 16 + i; // U blocks at indices 16-19
                    let v_idx = 20 + i; // V blocks at indices 20-23

                    let y0 = 1 + y * 4;
                    let x0 = 1 + x * 4;

                    if nz & (1u32 << u_idx) != 0 {
                        let urb: &mut [i32; 16] = (&mut self.coeff_blocks[u_idx * 16..][..16])
                            .try_into()
                            .unwrap();
                        if let Some(token) = simd_token {
                            idct_add_residue_and_clear_with_token(
                                token, uws, urb, y0, x0, stride,
                            );
                        } else {
                            idct_add_residue_and_clear(uws, urb, y0, x0, stride);
                        }
                    }

                    if nz & (1u32 << v_idx) != 0 {
                        let vrb: &mut [i32; 16] = (&mut self.coeff_blocks[v_idx * 16..][..16])
                            .try_into()
                            .unwrap();
                        if let Some(token) = simd_token {
                            idct_add_residue_and_clear_with_token(
                                token, vws, vrb, y0, x0, stride,
                            );
                        } else {
                            idct_add_residue_and_clear(vws, vrb, y0, x0, stride);
                        }
                    }
                }
            }
        }

        set_chroma_border(&mut self.left_border_u, &mut self.top_border_u, uws, mbx);
        set_chroma_border(&mut self.left_border_v, &mut self.top_border_v, vws, mbx);

        // Write to row cache instead of final buffer
        // Pre-slice destination regions once to eliminate per-iteration bounds checks.
        let extra_uv_rows = self.extra_y_rows / 2;
        let cache_uv_stride = self.cache_uv_stride;
        let cache_uv_offset = extra_uv_rows * cache_uv_stride;
        let uv_region_start = cache_uv_offset + mbx * 8;
        let uv_region_len = 7 * cache_uv_stride + 8;
        let cache_u_region = &mut self.cache_u[uv_region_start..uv_region_start + uv_region_len];
        let cache_v_region = &mut self.cache_v[uv_region_start..uv_region_start + uv_region_len];
        for y in 0usize..8 {
            let ws_index = (1 + y) * stride + 1;
            cache_u_region[y * cache_uv_stride..][..8].copy_from_slice(&uws[ws_index..][..8]);
            cache_v_region[y * cache_uv_stride..][..8].copy_from_slice(&vws[ws_index..][..8]);
        }
    }

    /// Read DCT coefficients using libwebp's inline tree structure.
    /// This mirrors GetCoeffsFast from libwebp for maximum performance.
    /// Writes to self.coeff_blocks at the given block index.
    fn read_residual_data(
        &mut self,
        mb: &mut MacroBlock,
        mbx: usize,
        p: usize,
        _simd_token: SimdTokenType, // Kept for API consistency, IDCT deferred to prediction
    ) -> Result<(), InternalDecodeError> {
        // Uses self.coeff_blocks which is maintained as zeros between calls.
        // After each IDCT, the block is left with transformed data for intra_predict to use.
        // intra_predict_* is responsible for clearing blocks after use.
        let sindex = mb.segmentid as usize;

        // Extract quantizers as 2-element arrays [dc, ac] — indexed by (n > 0) like libwebp
        let y2_dq = [i32::from(self.segment[sindex].y2dc), i32::from(self.segment[sindex].y2ac)];
        let y_dq = [i32::from(self.segment[sindex].ydc), i32::from(self.segment[sindex].yac)];
        let uv_dq = [i32::from(self.segment[sindex].uvdc), i32::from(self.segment[sindex].uvac)];

        // Split borrows: create active reader from partition fields directly
        // This avoids the per-block PartitionReader creation overhead (~7.7M instructions/decode)
        let (start, len) = self.partitions.boundaries[p];
        let partition_data = &self.partitions.data[start..start + len];
        let partition_state = &mut self.partitions.states[p];
        let mut reader = ActivePartitionReader::new(partition_data, partition_state);

        // Get references to other fields we need
        let probs = &*self.token_probs_by_pos;
        let coeff_blocks = &mut self.coeff_blocks;
        let top = &mut self.top[mbx];
        let left = &mut self.left;

        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = top.complexity[0] + left.complexity[0];
            let mut block = [0i32; 16];
            let n = read_coefficients(
                &mut reader,
                &mut block,
                &probs[Plane::Y2 as usize],
                0, // first
                complexity as usize,
                y2_dq,
            );

            left.complexity[0] = if n { 1 } else { 0 };
            top.complexity[0] = if n { 1 } else { 0 };

            // Optimized WHT: if only DC is non-zero, broadcast simplified value
            // (matches libwebp's shortcut in ParseResiduals)
            let has_ac = block[1..].iter().any(|&c| c != 0);
            if has_ac {
                transform::iwht4x4(&mut block);
                for (k, &val) in block.iter().enumerate() {
                    coeff_blocks[16 * k] = val;
                    if val != 0 {
                        mb.non_zero_blocks |= 1u32 << k;
                    }
                }
            } else if block[0] != 0 {
                // DC-only: simplified WHT = (dc[0] + 3) >> 3, broadcast to all 16 blocks
                let dc0 = (block[0] + 3) >> 3;
                for k in 0..16 {
                    coeff_blocks[16 * k] = dc0;
                }
                // All 16 Y blocks have non-zero DC
                mb.non_zero_blocks |= 0xFFFF;
            }
            // else: all zeros, nothing to do

            plane = Plane::YCoeff1;
        }

        let first_y = if plane == Plane::YCoeff1 { 1 } else { 0 };

        for y in 0usize..4 {
            let mut left_ctx = left.complexity[y + 1];
            for x in 0usize..4 {
                let i = x + y * 4;
                let complexity = top.complexity[x + 1] + left_ctx;

                let block: &mut [i32; 16] =
                    (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                let n = read_coefficients(
                    &mut reader,
                    block,
                    &probs[plane as usize],
                    first_y,
                    complexity as usize,
                    y_dq,
                );

                // Track non-zero DCT but defer IDCT to fused function during prediction.
                // Also set per-block bitmap for IDCT skip optimization.
                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    mb.non_zero_blocks |= 1u32 << i;
                }

                left_ctx = if n { 1 } else { 0 };
                top.complexity[x + 1] = if n { 1 } else { 0 };
            }

            left.complexity[y + 1] = left_ctx;
        }

        // Chroma
        let chroma_probs = &probs[Plane::Chroma as usize];

        for &j in &[5usize, 7usize] {
            for y in 0usize..2 {
                let mut left_ctx = left.complexity[y + j];

                for x in 0usize..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let complexity = top.complexity[x + j] + left_ctx;

                    let block: &mut [i32; 16] =
                        (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                    let n = read_coefficients(
                        &mut reader,
                        block,
                        chroma_probs,
                        0, // first
                        complexity as usize,
                        uv_dq,
                    );

                    // Track non-zero DCT but defer IDCT to fused function during prediction.
                    // Also set per-block bitmap.
                    if block[0] != 0 || n {
                        mb.non_zero_dct = true;
                        mb.non_zero_blocks |= 1u32 << i;
                    }

                    // Check for non-zero UV AC coefficients (any coeff at index > 0).
                    // libwebp suppresses dithering on MBs with UV AC content
                    // (non_zero_uv & 0xaaaa test in ParseResiduals).
                    // n is true when at least one coefficient was decoded, but that
                    // includes DC-only. Check the block directly for AC content.
                    if !mb.has_nonzero_uv_ac {
                        for &coeff in block[1..].iter() {
                            if coeff != 0 {
                                mb.has_nonzero_uv_ac = true;
                                break;
                            }
                        }
                    }

                    left_ctx = if n { 1 } else { 0 };
                    top.complexity[x + j] = if n { 1 } else { 0 };
                }

                left.complexity[y + j] = left_ctx;
            }
        }

        // Single EOF check after all blocks in the MB — matches libwebp's
        // VP8DecodeMB which checks once per MB, not per block.
        if reader.is_eof() {
            return Err(InternalDecodeError::BitStreamError);
        }

        Ok(())
    }

    /// Filters a row of macroblocks in the cache
    /// This operates on cache_y/u/v which have stride cache_y_stride/cache_uv_stride
    fn filter_row_in_cache(&mut self, mby: usize, simd_token: SimdTokenType) {
        let mbwidth = self.mbwidth as usize;
        let cache_y_stride = self.cache_y_stride;
        let cache_uv_stride = self.cache_uv_stride;
        let extra_y_rows = self.extra_y_rows;

        // Precompute filter parameters for all macroblocks in this row.
        // Reuse persistent buffer to avoid per-row heap allocation.
        self.mb_filter_params.clear();
        if self.mb_filter_params.capacity() < mbwidth {
            self.mb_filter_params.reserve(mbwidth - self.mb_filter_params.capacity());
        }
        for mbx in 0..mbwidth {
            let mb = self.macroblocks[mby * mbwidth + mbx];
            let (filter_level, interior_limit, hev_threshold) =
                self.calculate_filter_parameters(&mb);

            let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
            let sub_bedge_limit = (filter_level * 2) + interior_limit;
            let do_subblock_filtering =
                mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped && mb.non_zero_dct);

            self.mb_filter_params.push(loop_filter_dispatch::MbFilterParams {
                filter_level,
                interior_limit,
                hev_threshold,
                mbedge_limit,
                sub_bedge_limit,
                do_subblock_filtering,
            });
        }

        // Take ownership of mb_params to allow split borrows with cache buffers
        let mb_params = core::mem::take(&mut self.mb_filter_params);

        // Use the single #[arcane] entry point when SIMD is available.
        // All #[rite] filter functions inline into this one target_feature region,
        // eliminating per-call dispatch overhead (~7.6M instructions saved).
        #[cfg(all(feature = "simd", any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        )))]
        if let Some(token) = simd_token {
            loop_filter_dispatch::filter_row_simd(
                token,
                &mut self.cache_y[..],
                &mut self.cache_u[..],
                &mut self.cache_v[..],
                cache_y_stride,
                cache_uv_stride,
                extra_y_rows,
                self.frame.filter_type,
                mby,
                &mb_params,
            );
            self.mb_filter_params = mb_params;
            return;
        }

        // Scalar fallback (no SIMD token available)
        let extra_uv_rows = extra_y_rows / 2;
        for mbx in 0..mbwidth {
            let p = &mb_params[mbx];
            if p.filter_level == 0 {
                continue;
            }

            let mbedge_limit = p.mbedge_limit;
            let sub_bedge_limit = p.sub_bedge_limit;
            let hev_threshold = p.hev_threshold;
            let interior_limit = p.interior_limit;
            let do_subblock_filtering = p.do_subblock_filtering;

            if mbx > 0 {
                if self.frame.filter_type {
                    simple_filter_horizontal_16_rows(
                        &mut self.cache_y[..], extra_y_rows, mbx * 16,
                        cache_y_stride, mbedge_limit,
                    );
                } else {
                    normal_filter_horizontal_mb_16_rows(
                        &mut self.cache_y[..], extra_y_rows, mbx * 16,
                        cache_y_stride, hev_threshold, interior_limit, mbedge_limit,
                    );
                    normal_filter_horizontal_uv_mb(
                        &mut self.cache_u[..], &mut self.cache_v[..],
                        extra_uv_rows, mbx * 8, cache_uv_stride,
                        hev_threshold, interior_limit, mbedge_limit,
                    );
                }
            }

            if do_subblock_filtering {
                if self.frame.filter_type {
                    for x in (4usize..16 - 1).step_by(4) {
                        simple_filter_horizontal_16_rows(
                            &mut self.cache_y[..], extra_y_rows, mbx * 16 + x,
                            cache_y_stride, sub_bedge_limit,
                        );
                    }
                } else {
                    for x in (4usize..16 - 3).step_by(4) {
                        normal_filter_horizontal_sub_16_rows(
                            &mut self.cache_y[..], extra_y_rows, mbx * 16 + x,
                            cache_y_stride, hev_threshold, interior_limit, sub_bedge_limit,
                        );
                    }
                    normal_filter_horizontal_uv_sub(
                        &mut self.cache_u[..], &mut self.cache_v[..],
                        extra_uv_rows, mbx * 8 + 4, cache_uv_stride,
                        hev_threshold, interior_limit, sub_bedge_limit,
                    );
                }
            }

            if mby > 0 {
                if self.frame.filter_type {
                    simple_filter_vertical_16_cols(
                        &mut self.cache_y[..], extra_y_rows, mbx * 16,
                        cache_y_stride, mbedge_limit,
                    );
                } else {
                    normal_filter_vertical_mb_16_cols(
                        &mut self.cache_y[..], extra_y_rows, mbx * 16,
                        cache_y_stride, hev_threshold, interior_limit, mbedge_limit,
                    );
                    normal_filter_vertical_uv_mb(
                        &mut self.cache_u[..], &mut self.cache_v[..],
                        extra_uv_rows, mbx * 8, cache_uv_stride,
                        hev_threshold, interior_limit, mbedge_limit,
                    );
                }
            }

            if do_subblock_filtering {
                if self.frame.filter_type {
                    for y in (4usize..16 - 1).step_by(4) {
                        simple_filter_vertical_16_cols(
                            &mut self.cache_y[..], extra_y_rows + y, mbx * 16,
                            cache_y_stride, sub_bedge_limit,
                        );
                    }
                } else {
                    for y in (4usize..16 - 3).step_by(4) {
                        normal_filter_vertical_sub_16_cols(
                            &mut self.cache_y[..], extra_y_rows + y, mbx * 16,
                            cache_y_stride, hev_threshold, interior_limit, sub_bedge_limit,
                        );
                    }
                    normal_filter_vertical_uv_sub(
                        &mut self.cache_u[..], &mut self.cache_v[..],
                        extra_uv_rows + 4, mbx * 8, cache_uv_stride,
                        hev_threshold, interior_limit, sub_bedge_limit,
                    );
                }
            }
        }

        self.mb_filter_params = mb_params;
    }

    /// Copy the filtered row from cache to final output buffers
    /// Uses delayed output: filter modifies pixels above and below the edge,
    /// so we delay outputting the bottom extra_rows until the next row is filtered.
    fn output_row_from_cache(&mut self, mby: usize) {
        let mbwidth = self.mbwidth as usize;
        let mbheight = self.mbheight as usize;
        let luma_w = mbwidth * 16;
        let chroma_w = mbwidth * 8;
        let extra_y_rows = self.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;
        let is_first_row = mby == 0;
        let is_last_row = mby == mbheight - 1;

        // Determine which rows to output:
        // - First row: output rows extra_y_rows to extra_y_rows + (16 - extra_y_rows) = rows up to 16
        //   but skip the bottom extra_y_rows (they'll be output with next row after filtering)
        // - Middle rows: output extra area (0 to extra_y_rows) + current row minus bottom extra_y_rows
        // - Last row: output extra area + full current row

        let (src_start_row, num_y_rows, dst_start_y_row) = if is_first_row {
            (extra_y_rows, 16 - extra_y_rows, 0usize)
        } else if is_last_row {
            (0, extra_y_rows + 16, mby * 16 - extra_y_rows)
        } else {
            (0, 16, mby * 16 - extra_y_rows)
        };

        // Copy luma rows from cache to frame buffer.
        // cache_y_stride == luma_w, so source rows are contiguous.
        // Pre-slice both source and destination to eliminate per-row bounds checks.
        {
            let src_start = src_start_row * self.cache_y_stride;
            let dst_start = dst_start_y_row * luma_w;
            let total = num_y_rows * luma_w;
            let src = &self.cache_y[src_start..src_start + total];
            let dst = &mut self.frame.ybuf[dst_start..dst_start + total];
            for y in 0..num_y_rows {
                let off = y * luma_w;
                dst[off..off + luma_w].copy_from_slice(&src[off..off + luma_w]);
            }
        }

        // Same logic for chroma but with half the rows
        let (src_start_row_uv, num_uv_rows, dst_start_uv_row) = if is_first_row {
            (extra_uv_rows, 8 - extra_uv_rows, 0usize)
        } else if is_last_row {
            (0, extra_uv_rows + 8, mby * 8 - extra_uv_rows)
        } else {
            (0, 8, mby * 8 - extra_uv_rows)
        };

        // Copy chroma rows from cache to frame buffer.
        // cache_uv_stride == chroma_w, so source rows are contiguous.
        // Pre-slice both source and destination to eliminate per-row bounds checks.
        {
            let src_start = src_start_row_uv * self.cache_uv_stride;
            let dst_start = dst_start_uv_row * chroma_w;
            let total = num_uv_rows * chroma_w;
            let u_src = &self.cache_u[src_start..src_start + total];
            let v_src = &self.cache_v[src_start..src_start + total];
            let u_dst = &mut self.frame.ubuf[dst_start..dst_start + total];
            let v_dst = &mut self.frame.vbuf[dst_start..dst_start + total];
            for y in 0..num_uv_rows {
                let off = y * chroma_w;
                u_dst[off..off + chroma_w].copy_from_slice(&u_src[off..off + chroma_w]);
                v_dst[off..off + chroma_w].copy_from_slice(&v_src[off..off + chroma_w]);
            }
        }
    }

    /// Copy bottom rows of current cache to extra area for next row's filtering
    fn rotate_extra_rows(&mut self) {
        let extra_y_rows = self.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;

        if extra_y_rows == 0 {
            return;
        }

        // Copy bottom extra_y_rows of current MB row to the extra area
        // Source: rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16) = rows 16..(extra_y_rows + 16)
        // Actually: the bottom extra_y_rows of the 16-row area
        // Which is rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16)
        // = rows 16..(16 + extra_y_rows)... wait that's wrong
        // The current row is at rows extra_y_rows..(extra_y_rows + 16)
        // The bottom extra_y_rows are at rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16)
        // = rows 16..(extra_y_rows + 16) -- no that's still wrong
        // Let me think again:
        // - Current row occupies rows extra_y_rows to extra_y_rows + 15 (16 rows)
        // - We want the bottom extra_y_rows of these, which are rows (extra_y_rows + 16 - extra_y_rows) to (extra_y_rows + 15)
        // - Wait, extra_y_rows + 16 - extra_y_rows = 16, so rows 16..16+extra_y_rows = rows 16..16+extra_y_rows
        // - Hmm, that's outside the current row area...

        // Let me reconsider. Current row (16 pixels):
        // - Starts at row index extra_y_rows
        // - Ends at row index extra_y_rows + 15
        // - Bottom extra_y_rows rows are at indices (extra_y_rows + 16 - extra_y_rows) to (extra_y_rows + 15)
        // - = indices 16 to (extra_y_rows + 15)... that's wrong

        // Actually: the 16-row area is at indices extra_y_rows..(extra_y_rows + 16)
        // The last extra_y_rows rows of this area are at indices:
        //   (extra_y_rows + 16 - extra_y_rows) .. (extra_y_rows + 16)
        //   = 16 .. (extra_y_rows + 16)
        // Wait, that gives 16..24 for extra_y_rows=8, which is 8 rows. That's correct!
        // But 16 > extra_y_rows when extra_y_rows = 8, so indices 16..24 are valid.

        // Destination: rows 0..extra_y_rows

        // For luma:
        let src_start_row = 16; // = extra_y_rows + 16 - extra_y_rows = 16
        let src_start = src_start_row * self.cache_y_stride;
        let copy_size = extra_y_rows * self.cache_y_stride;
        // Copy from src_start to 0
        self.cache_y
            .copy_within(src_start..src_start + copy_size, 0);

        // For chroma:
        let src_start_row_uv = 8; // = extra_uv_rows + 8 - extra_uv_rows = 8
        let src_start_uv = src_start_row_uv * self.cache_uv_stride;
        let copy_size_uv = extra_uv_rows * self.cache_uv_stride;
        self.cache_u
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
        self.cache_v
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
    }

    //return values are the filter level, interior limit and hev threshold
    fn calculate_filter_parameters(&self, macroblock: &MacroBlock) -> (u8, u8, u8) {
        let segment = &self.segment[macroblock.segmentid as usize];
        let mut filter_level = i32::from(self.frame.filter_level);

        // if frame level filter level is 0, we must skip loop filter
        if filter_level == 0 {
            return (0, 0, 0);
        }

        if self.segments_enabled {
            if segment.delta_values {
                filter_level += i32::from(segment.loopfilter_level);
            } else {
                filter_level = i32::from(segment.loopfilter_level);
            }
        }

        filter_level = filter_level.clamp(0, 63);

        if self.loop_filter_adjustments_enabled {
            filter_level += self.ref_delta[0];
            if macroblock.luma_mode == LumaMode::B {
                filter_level += self.mode_delta[0];
            }
        }

        let filter_level = filter_level.clamp(0, 63) as u8;

        //interior limit
        let mut interior_limit = filter_level;

        if self.frame.sharpness_level > 0 {
            interior_limit >>= if self.frame.sharpness_level > 4 { 2 } else { 1 };

            if interior_limit > 9 - self.frame.sharpness_level {
                interior_limit = 9 - self.frame.sharpness_level;
            }
        }

        if interior_limit == 0 {
            interior_limit = 1;
        }

        // high edge variance threshold
        let hev_threshold = if filter_level >= 40 {
            2
        } else if filter_level >= 15 {
            1
        } else {
            0
        };

        (filter_level, interior_limit, hev_threshold)
    }

    /// Decodes the current frame
    pub fn decode_frame(data: &'a [u8]) -> Result<Frame, DecodeError> {
        let decoder = Self::new(data);
        decoder.decode_frame_()
    }

    /// Decodes the current frame with cooperative cancellation support and dithering.
    pub fn decode_frame_with_stop(
        data: &'a [u8],
        stop: Option<&'a dyn enough::Stop>,
        dithering_strength: u8,
    ) -> Result<Frame, DecodeError> {
        let mut decoder = Self::new(data);
        decoder.stop = stop;
        decoder.dither_strength_pending = dithering_strength;
        decoder.decode_frame_()
    }

    /// Decodes the frame with diagnostic capture enabled.
    /// Returns both the decoded Frame and a DiagnosticFrame containing
    /// raw quantized coefficient levels, mode decisions, and probability tables.
    #[doc(hidden)]
    pub fn decode_diagnostic(data: &'a [u8]) -> Result<(Frame, DiagnosticFrame), DecodeError> {
        let mut decoder = Self::new(data);
        decoder.diagnostic_capture = Some(Vec::new());
        decoder.decode_frame_diagnostic()
    }

    fn decode_frame_diagnostic(mut self) -> Result<(Frame, DiagnosticFrame), DecodeError> {
        self.read_frame_header()?;

        // Summon SIMD token once for the entire decode
        let simd_token: SimdTokenType = {
            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
            {
                archmage::X64V3Token::summon()
            }
            #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
            {
                None
            }
        };

        // Capture segment quantizers
        let segments: [(i16, i16, i16, i16, i16, i16); 4] = core::array::from_fn(|i| {
            let s = &self.segment[i];
            (s.ydc, s.yac, s.y2dc, s.y2ac, s.uvdc, s.uvac)
        });

        // Run the main decode loop (this populates diagnostic_capture)
        self.decode_mb_rows_diagnostic(simd_token)
            .map_err(DecodeError::from)?;

        let diagnostic = DiagnosticFrame {
            mb_width: self.mbwidth,
            mb_height: self.mbheight,
            segments,
            macroblocks: self.diagnostic_capture.take().unwrap_or_default(),
            token_probs: self.token_probs.clone(),
            partition0_size: self.first_partition_size,
        };

        Ok((self.frame, diagnostic))
    }

    /// Diagnostic decode loop — same as `decode_mb_rows` but captures per-MB diagnostics.
    fn decode_mb_rows_diagnostic(
        &mut self,
        simd_token: SimdTokenType,
    ) -> Result<(), InternalDecodeError> {
        for mby in 0..self.mbheight as usize {
            let p = mby % self.num_partitions as usize;
            self.left = PreviousMacroBlock::default();

            for mbx in 0..self.mbwidth as usize {
                // Initialize diagnostic capture for this macroblock
                self.current_mb_diag = Some(MacroblockDiagnostic::default());

                let mut mb = self.read_macroblock_header(mbx)?;

                // Capture mode decisions
                if let Some(ref mut diag) = self.current_mb_diag {
                    diag.luma_mode = mb.luma_mode;
                    diag.chroma_mode = mb.chroma_mode;
                    diag.segment_id = mb.segmentid;
                    diag.coeffs_skipped = mb.coeffs_skipped;
                    diag.bpred_modes = mb.bpred;
                }

                if !mb.coeffs_skipped {
                    self.read_residual_data(&mut mb, mbx, p, simd_token)?;
                } else {
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        self.top[mbx].complexity[0] = 0;
                    }
                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        self.top[mbx].complexity[i] = 0;
                    }
                }

                self.intra_predict_luma(mbx, mby, &mb, simd_token);
                self.intra_predict_chroma(mbx, mby, &mb, simd_token);
                self.macroblocks.push(mb);

                // Finalize diagnostic capture
                if let Some(diag) = self.current_mb_diag.take()
                    && let Some(ref mut capture) = self.diagnostic_capture
                {
                    capture.push(diag);
                }
            }

            self.filter_row_in_cache(mby, simd_token);
            self.output_row_from_cache(mby);
            self.rotate_extra_rows();

            self.left_border_y.fill(129u8);
            self.left_border_u.fill(129u8);
            self.left_border_v.fill(129u8);
        }

        Ok(())
    }

    fn decode_frame_(mut self) -> Result<Frame, DecodeError> {
        self.read_frame_header()?;

        // Initialize chroma dithering after frame header (which populates segments)
        if self.dither_strength_pending > 0 {
            let (enabled, amps) = super::dither::init_dither_amplitudes(
                &self.uv_quant_indices,
                self.dither_strength_pending,
            );
            self.dither_enabled = enabled;
            self.dither_amp = amps;
        }

        // Summon SIMD token once for the entire decode, avoiding ~312K per-call atomic loads
        let simd_token: SimdTokenType = {
            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
            {
                archmage::X64V3Token::summon()
            }
            #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
            {
                None
            }
        };

        self.decode_mb_rows(simd_token)
            .map_err(DecodeError::from)?;

        Ok(self.frame)
    }

    /// Main decode loop — uses `InternalDecodeError` to avoid String drop
    /// overhead on every `?` operator in the hot path.
    fn decode_mb_rows(&mut self, simd_token: SimdTokenType) -> Result<(), InternalDecodeError> {
        for mby in 0..self.mbheight as usize {
            let p = mby % self.num_partitions as usize;
            self.left = PreviousMacroBlock::default();

            // Decode all macroblocks in this row (writes to cache)
            for mbx in 0..self.mbwidth as usize {
                let mut mb = self.read_macroblock_header(mbx)?;

                if !mb.coeffs_skipped {
                    // Decode coefficients into self.coeff_blocks
                    self.read_residual_data(&mut mb, mbx, p, simd_token)?;
                } else {
                    // self.coeff_blocks is already zeros (invariant maintained by intra_predict_*)
                    // Extract top[mbx] once to avoid repeated bounds checks
                    let top_mb = &mut self.top[mbx];
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        top_mb.complexity[0] = 0;
                    }

                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        top_mb.complexity[i] = 0;
                    }
                }

                // intra_predict_* reads from self.coeff_blocks and clears after use
                self.intra_predict_luma(mbx, mby, &mb, simd_token);
                self.intra_predict_chroma(mbx, mby, &mb, simd_token);

                self.macroblocks.push(mb);
            }

            // Row complete: filter in cache, dither chroma, output to final buffer
            self.filter_row_in_cache(mby, simd_token);

            // Apply chroma dithering after filtering, before output.
            // Per libwebp: only dither MBs with all-zero UV AC coefficients
            // (smooth/flat blocks where banding is visible).
            if self.dither_enabled {
                let extra_uv_rows = self.extra_y_rows / 2;
                let mbwidth = self.mbwidth as usize;
                let mb_row_start = mby * mbwidth;
                // Compute per-MB dither amplitude: segment amp if UV AC is zero,
                // 0 if UV has AC content or coefficients were skipped.
                let mb_dither: Vec<i32> = (0..mbwidth)
                    .map(|mbx| {
                        let mb = &self.macroblocks[mb_row_start + mbx];
                        if mb.coeffs_skipped || mb.has_nonzero_uv_ac {
                            0
                        } else {
                            self.dither_amp[mb.segmentid as usize]
                        }
                    })
                    .collect();
                super::dither::dither_row(
                    &mut self.dither_rg,
                    super::dither::DitherRowParams {
                        cache_u: &mut self.cache_u,
                        cache_v: &mut self.cache_v,
                        cache_uv_stride: self.cache_uv_stride,
                        extra_uv_rows,
                        mb_dither_amps: &mb_dither,
                    },
                );
            }

            self.output_row_from_cache(mby);
            self.rotate_extra_rows();

            // Check for cooperative cancellation between macroblock rows
            if let Some(stop) = self.stop {
                stop.check().map_err(InternalDecodeError::from)?;
            }

            self.left_border_y.fill(129u8);
            self.left_border_u.fill(129u8);
            self.left_border_v.fill(129u8);
        }

        Ok(())
    }
}

// set border
fn set_chroma_border(
    left_border: &mut [u8],
    top_border: &mut [u8],
    chroma_block: &[u8; CHROMA_BLOCK_SIZE],
    mbx: usize,
) {
    let stride = CHROMA_STRIDE;
    // top left is top right of previous chroma block
    left_border[0] = chroma_block[8];

    // left border — chroma_block is fixed-size so stride offsets are provably in-bounds
    for (i, left) in left_border[1..][..8].iter_mut().enumerate() {
        *left = chroma_block[(i + 1) * stride + 8];
    }

    // Pre-slice top_border to eliminate repeated mbx*8 bounds check
    let top_region = &mut top_border[mbx * 8..][..8];
    top_region.copy_from_slice(&chroma_block[8 * stride + 1..8 * stride + 9]);
}
