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


use alloc::vec::Vec;
use core::array;
use core::default::Default;


use crate::common::types::*;



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

