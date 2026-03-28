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

mod cache;
mod coefficients;
mod header;
mod predict;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::default::Default;

use archmage::SimdToken;

use super::api::{DecodeError, UpsamplingMethod};
use super::internal_error::InternalDecodeError;
use super::yuv;
use crate::common::prediction::*;
use crate::common::types::*;
use crate::slice_reader::SliceReader;

use super::bit_reader::{ActivePartitionReader, VP8HeaderBitReader, VP8Partitions};
use super::loop_filter;
use crate::common::transform;
use loop_filter::*;

/// Summon the best available SIMD token for the current platform.
/// Called once at decode start to avoid per-call atomic loads.
#[inline(always)]
fn summon_simd_token() -> SimdTokenType {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        archmage::X64V3Token::summon()
    }
    #[cfg(target_arch = "aarch64")]
    {
        archmage::NeonToken::summon()
    }
    #[cfg(target_arch = "wasm32")]
    {
        archmage::Wasm128Token::summon()
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        None
    }
}

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

/// Precomputed filter parameters for a (segment, is_b_mode) combination.
/// Computed once per frame; avoids re-deriving per macroblock.
#[derive(Clone, Copy, Default)]
struct PrecomputedFilterParams {
    filter_level: u8,
    interior_limit: u8,
    hev_threshold: u8,
    mbedge_limit: u8,
    sub_bedge_limit: u8,
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

    // Reusable dither amplitude buffer — avoids per-row Vec allocation.
    mb_dither_buf: Vec<i32>,

    // Precomputed filter parameters per segment+mode.
    // Indexed by [segment_id][is_b_mode], where is_b_mode = (luma_mode == LumaMode::B).
    // Computed once after reading frame header; eliminates per-MB calculation.
    precomputed_filter: [[PrecomputedFilterParams; 2]; MAX_SEGMENTS],
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
            mb_dither_buf: Vec::new(),

            precomputed_filter: [[PrecomputedFilterParams::default(); 2]; MAX_SEGMENTS],
        }
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
        let simd_token: SimdTokenType = summon_simd_token();

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
        let simd_token: SimdTokenType = summon_simd_token();

        self.decode_mb_rows(simd_token).map_err(DecodeError::from)?;

        Ok(self.frame)
    }

    /// Main decode loop — uses `InternalDecodeError` to avoid String drop
    /// overhead on every `?` operator in the hot path.
    fn decode_mb_rows(&mut self, simd_token: SimdTokenType) -> Result<(), InternalDecodeError> {
        let mbwidth = self.mbwidth as usize;
        let dither_enabled = self.dither_enabled;

        // Pre-size the filter params and dither buffers once.
        if self.mb_filter_params.capacity() < mbwidth {
            self.mb_filter_params
                .reserve(mbwidth - self.mb_filter_params.capacity());
        }
        if dither_enabled && self.mb_dither_buf.capacity() < mbwidth {
            self.mb_dither_buf
                .reserve(mbwidth - self.mb_dither_buf.capacity());
        }

        for mby in 0..self.mbheight as usize {
            let p = mby % self.num_partitions as usize;
            self.left = PreviousMacroBlock::default();

            // Clear per-row buffers (capacity already ensured above).
            self.mb_filter_params.clear();
            self.mb_dither_buf.clear();

            // Single-pass: decode each MB, then immediately compute its
            // filter params and dither amplitude — no second iteration needed.
            for mbx in 0..mbwidth {
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

                // Compute filter params from precomputed table (table lookup, no branches).
                let is_b = mb.luma_mode == LumaMode::B;
                let fp = &self.precomputed_filter[mb.segmentid as usize][is_b as usize];
                let do_subblock_filtering =
                    is_b || (!mb.coeffs_skipped && mb.non_zero_dct);
                self.mb_filter_params.push(loop_filter::MbFilterParams {
                    filter_level: fp.filter_level,
                    interior_limit: fp.interior_limit,
                    hev_threshold: fp.hev_threshold,
                    mbedge_limit: fp.mbedge_limit,
                    sub_bedge_limit: fp.sub_bedge_limit,
                    do_subblock_filtering,
                });

                // Compute dither amplitude inline.
                if dither_enabled {
                    let amp = if mb.coeffs_skipped || mb.has_nonzero_uv_ac {
                        0
                    } else {
                        self.dither_amp[mb.segmentid as usize]
                    };
                    self.mb_dither_buf.push(amp);
                }

                self.macroblocks.push(mb);
            }

            // Row complete: filter in cache using pre-populated mb_filter_params.
            self.filter_row_in_cache_precomputed(mby, simd_token);

            // Apply chroma dithering after filtering, before output.
            if dither_enabled {
                let extra_uv_rows = self.extra_y_rows / 2;
                let dither_buf = core::mem::take(&mut self.mb_dither_buf);
                super::dither::dither_row(
                    &mut self.dither_rg,
                    super::dither::DitherRowParams {
                        cache_u: &mut self.cache_u,
                        cache_v: &mut self.cache_v,
                        cache_uv_stride: self.cache_uv_stride,
                        extra_uv_rows,
                        mb_dither_amps: &dither_buf,
                    },
                );
                self.mb_dither_buf = dither_buf;
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
