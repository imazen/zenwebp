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

use byteorder_lite::{LittleEndian, ReadBytesExt};
use std::default::Default;
use std::io::Read;

use crate::decoder::{DecodingError, UpsamplingMethod};
use crate::vp8_common::*;
use crate::vp8_prediction::*;
use crate::yuv;

use super::vp8_arithmetic_decoder::ArithmeticDecoder;
use super::vp8_bit_reader::VP8Partitions;
use super::{loop_filter, transform};

/// Helper to apply simple horizontal filter to 16 rows with SIMD when available.
/// Filters the vertical edge at column x0, processing all 16 rows at once.
#[inline]
fn simple_filter_horizontal_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    edge_limit: u8,
) {
    #[cfg(all(feature = "unsafe-simd", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        // Use the new 16-pixel-at-once approach with transpose
        unsafe {
            crate::loop_filter_avx2::simple_h_filter16(buf, x0, y_start, stride, i32::from(edge_limit));
        }
        return;
    }

    // Scalar fallback
    for y in 0usize..16 {
        let y0 = y_start + y;
        loop_filter::simple_segment_horizontal(edge_limit, &mut buf[y0 * stride + x0 - 4..][..8]);
    }
}

/// Helper to apply simple vertical filter to 16 columns with SIMD when available.
/// Filters the horizontal edge at row y0, processing all 16 columns at once.
#[inline]
fn simple_filter_vertical_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    edge_limit: u8,
) {
    #[cfg(all(feature = "unsafe-simd", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        // Use the new 16-pixel-at-once approach
        let point = y0 * stride + x_start;
        unsafe {
            crate::loop_filter_avx2::simple_v_filter16(buf, point, stride, i32::from(edge_limit));
        }
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::simple_segment_vertical(edge_limit, buf, point, stride);
    }
}

/// Helper to apply normal vertical macroblock filter to 16 columns with SIMD when available.
/// Filters the horizontal edge at row y0, processing all 16 columns at once.
#[inline]
fn normal_filter_vertical_mb_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    #[cfg(all(feature = "unsafe-simd", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        let point = y0 * stride + x_start;
        unsafe {
            crate::loop_filter_avx2::normal_v_filter16_edge(
                buf, point, stride,
                i32::from(hev_threshold),
                i32::from(interior_limit),
                i32::from(edge_limit),
            );
        }
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Helper to apply normal vertical subblock filter to 16 columns with SIMD when available.
#[inline]
fn normal_filter_vertical_sub_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    #[cfg(all(feature = "unsafe-simd", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        let point = y0 * stride + x_start;
        unsafe {
            crate::loop_filter_avx2::normal_v_filter16_inner(
                buf, point, stride,
                i32::from(hev_threshold),
                i32::from(interior_limit),
                i32::from(edge_limit),
            );
        }
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TreeNode {
    pub left: u8,
    pub right: u8,
    pub prob: Prob,
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

/// VP8 Decoder
///
/// Only decodes keyframes
pub struct Vp8Decoder<R> {
    r: R,
    b: ArithmeticDecoder,

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
    cache_y_stride: usize,   // mbwidth * 16
    cache_uv_stride: usize,  // mbwidth * 8
    extra_y_rows: usize,     // 8 for normal filter, 2 for simple, 0 for none
}

impl<R: Read> Vp8Decoder<R> {
    /// Create a new decoder.
    /// The reader must present a raw vp8 bitstream to the decoder
    fn new(r: R) -> Self {
        let f = Frame::default();

        Self {
            r,
            b: ArithmeticDecoder::new(),

            mbwidth: 0,
            mbheight: 0,
            macroblocks: Vec::new(),

            frame: f,
            segments_enabled: false,
            segments_update_map: false,
            segment: std::array::from_fn(|_| Segment::default()),

            loop_filter_adjustments_enabled: false,
            ref_delta: [0; 4],
            mode_delta: [0; 4],

            partitions: VP8Partitions::new(),

            num_partitions: 1,

            segment_tree_nodes: SEGMENT_TREE_NODE_DEFAULTS,
            token_probs: Box::new(COEFF_PROB_NODES),

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
        }
    }

    fn update_token_probabilities(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if self.b.read_bool(*prob).or_accumulate(&mut res) {
                            let v = self.b.read_literal(8).or_accumulate(&mut res);
                            self.token_probs[i][j][k][t].prob = v;
                        }
                    }
                }
            }
        }
        self.b.check(res, ())
    }

    fn init_partitions(&mut self, n: usize) -> Result<(), DecodingError> {
        let mut all_data = Vec::new();
        let mut boundaries = Vec::with_capacity(n);

        if n > 1 {
            let mut sizes = vec![0; 3 * n - 3];
            self.r.read_exact(sizes.as_mut_slice())?;

            for s in sizes.chunks(3) {
                let size = { s }
                    .read_u24::<LittleEndian>()
                    .expect("Reading from &[u8] can't fail and the chunk is complete")
                    as usize;

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

    fn read_quantization_indices(&mut self) -> Result<(), DecodingError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let mut res = self.b.start_accumulated_result();

        let yac_abs = self.b.read_literal(7).or_accumulate(&mut res);
        let ydc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2dc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2ac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvdc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);

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

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        self.b.check(res, ())
    }

    fn read_loop_filter_adjustments(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        if self.b.read_flag().or_accumulate(&mut res) {
            for i in 0usize..4 {
                self.ref_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }

            for i in 0usize..4 {
                self.mode_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }
        }

        self.b.check(res, ())
    }

    fn read_segment_updates(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        // Section 9.3
        self.segments_update_map = self.b.read_flag().or_accumulate(&mut res);
        let update_segment_feature_data = self.b.read_flag().or_accumulate(&mut res);

        if update_segment_feature_data {
            let segment_feature_mode = self.b.read_flag().or_accumulate(&mut res);

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].quantizer_level =
                    self.b.read_optional_signed_value(7).or_accumulate(&mut res) as i8;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].loopfilter_level =
                    self.b.read_optional_signed_value(6).or_accumulate(&mut res) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0usize..3 {
                let update = self.b.read_flag().or_accumulate(&mut res);

                let prob = if update {
                    self.b.read_literal(8).or_accumulate(&mut res)
                } else {
                    255
                };
                self.segment_tree_nodes[i].prob = prob;
            }
        }

        self.b.check(res, ())
    }

    fn read_frame_header(&mut self) -> Result<(), DecodingError> {
        let tag = self.r.read_u24::<LittleEndian>()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodingError::UnsupportedFeature(
                "Non-keyframe frames".to_owned(),
            ));
        }

        self.frame.version = ((tag >> 1) & 7) as u8;
        self.frame.for_display = (tag >> 4) & 1 != 0;

        let first_partition_size = tag >> 5;

        let mut tag = [0u8; 3];
        self.r.read_exact(&mut tag)?;

        if tag != [0x9d, 0x01, 0x2a] {
            return Err(DecodingError::Vp8MagicInvalid(tag));
        }

        let w = self.r.read_u16::<LittleEndian>()?;
        let h = self.r.read_u16::<LittleEndian>()?;

        self.frame.width = w & 0x3FFF;
        self.frame.height = h & 0x3FFF;

        self.mbwidth = self.frame.width.div_ceil(16);
        self.mbheight = self.frame.height.div_ceil(16);

        // defaults are intra mode DC and complexity 0
        self.top = vec![PreviousMacroBlock::default(); self.mbwidth.into()];
        self.left = PreviousMacroBlock::default();

        self.frame.ybuf =
            vec![0u8; usize::from(self.mbwidth) * 16 * usize::from(self.mbheight) * 16];
        self.frame.ubuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];
        self.frame.vbuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];

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
        let mut buf = vec![[0; 4]; size.div_ceil(4)];
        let bytes: &mut [u8] = buf.as_mut_slice().as_flattened_mut();
        self.r.read_exact(&mut bytes[..size])?;

        // initialise binary decoder
        self.b.init(buf, size)?;

        let mut res = self.b.start_accumulated_result();
        let color_space = self.b.read_literal(1).or_accumulate(&mut res);
        self.frame.pixel_type = self.b.read_literal(1).or_accumulate(&mut res);

        if color_space != 0 {
            return Err(DecodingError::ColorSpaceInvalid(color_space));
        }

        self.segments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.segments_enabled {
            self.read_segment_updates()?;
        }

        self.frame.filter_type = self.b.read_flag().or_accumulate(&mut res);
        self.frame.filter_level = self.b.read_literal(6).or_accumulate(&mut res);
        self.frame.sharpness_level = self.b.read_literal(3).or_accumulate(&mut res);

        self.loop_filter_adjustments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments()?;
        }

        let num_partitions = 1 << self.b.read_literal(2).or_accumulate(&mut res) as usize;
        self.b.check(res, ())?;

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
        let cache_y_rows = self.extra_y_rows + 16;
        let cache_uv_rows = extra_uv_rows + 8;
        self.cache_y = vec![128u8; cache_y_rows * self.cache_y_stride];
        self.cache_u = vec![128u8; cache_uv_rows * self.cache_uv_stride];
        self.cache_v = vec![128u8; cache_uv_rows * self.cache_uv_stride];

        self.num_partitions = num_partitions as u8;
        self.init_partitions(num_partitions)?;

        self.read_quantization_indices()?;

        // Refresh entropy probs ?????
        let _ = self.b.read_literal(1);

        self.update_token_probabilities()?;

        let mut res = self.b.start_accumulated_result();
        let mb_no_skip_coeff = self.b.read_literal(1).or_accumulate(&mut res);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(self.b.read_literal(8).or_accumulate(&mut res))
        } else {
            None
        };
        self.b.check(res, ())?;
        Ok(())
    }

    fn read_macroblock_header(&mut self, mbx: usize) -> Result<MacroBlock, DecodingError> {
        let mut mb = MacroBlock::default();
        let mut res = self.b.start_accumulated_result();

        if self.segments_enabled && self.segments_update_map {
            mb.segmentid =
                (self.b.read_with_tree(&self.segment_tree_nodes)).or_accumulate(&mut res) as u8;
        };

        mb.coeffs_skipped = if let Some(prob) = self.prob_skip_false {
            self.b.read_bool(prob).or_accumulate(&mut res)
        } else {
            false
        };

        // intra prediction
        let luma = (self.b.read_with_tree(&KEYFRAME_YMODE_NODES)).or_accumulate(&mut res);
        mb.luma_mode =
            LumaMode::from_i8(luma).ok_or(DecodingError::LumaPredictionModeInvalid(luma))?;

        match mb.luma_mode.into_intra() {
            // `LumaMode::B` - This is predicted individually
            None => {
                for y in 0usize..4 {
                    for x in 0usize..4 {
                        let top = self.top[mbx].bpred[x];
                        let left = self.left.bpred[y];
                        let intra = self.b.read_with_tree(
                            &KEYFRAME_BPRED_MODE_NODES[top as usize][left as usize],
                        );
                        let intra = intra.or_accumulate(&mut res);
                        let bmode = IntraMode::from_i8(intra)
                            .ok_or(DecodingError::IntraPredictionModeInvalid(intra))?;
                        mb.bpred[x + y * 4] = bmode;

                        self.top[mbx].bpred[x] = bmode;
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

        let chroma = (self.b.read_with_tree(&KEYFRAME_UV_MODE_NODES)).or_accumulate(&mut res);
        mb.chroma_mode = ChromaMode::from_i8(chroma)
            .ok_or(DecodingError::ChromaPredictionModeInvalid(chroma))?;

        // top should store the bottom of the current bpred, which is the final 4 values
        self.top[mbx].bpred = mb.bpred[12..].try_into().unwrap();

        self.b.check(res, mb)
    }

    fn intra_predict_luma(&mut self, mbx: usize, mby: usize, mb: &MacroBlock, resdata: &[i32]) {
        let stride = LUMA_STRIDE;
        let mw = self.mbwidth as usize;
        let mut ws = create_border_luma(mbx, mby, mw, &self.top_border_y, &self.left_border_y);

        match mb.luma_mode {
            LumaMode::V => predict_vpred(&mut ws, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut ws, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut ws, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut ws, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => predict_4x4(&mut ws, stride, &mb.bpred, resdata),
        }

        if mb.luma_mode != LumaMode::B {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    // Create a reference to a [i32; 16] array for add_residue (slices of size 16 do not work).
                    let rb: &[i32; 16] = resdata[i * 16..][..16].try_into().unwrap();
                    let y0 = 1 + y * 4;
                    let x0 = 1 + x * 4;

                    add_residue(&mut ws, rb, y0, x0, stride);
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
        let cache_y_offset = self.extra_y_rows * self.cache_y_stride;
        for y in 0usize..16 {
            let dst_start = cache_y_offset + y * self.cache_y_stride + mbx * 16;
            let src_start = (1 + y) * stride + 1;
            self.cache_y[dst_start..][..16].copy_from_slice(&ws[src_start..][..16]);
        }
    }

    fn intra_predict_chroma(&mut self, mbx: usize, mby: usize, mb: &MacroBlock, resdata: &[i32]) {
        let stride = CHROMA_STRIDE;

        //8x8 with left top border of 1
        let mut uws = create_border_chroma(mbx, mby, &self.top_border_u, &self.left_border_u);
        let mut vws = create_border_chroma(mbx, mby, &self.top_border_v, &self.left_border_v);

        match mb.chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(&mut uws, 8, stride, mby != 0, mbx != 0);
                predict_dcpred(&mut vws, 8, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(&mut uws, 8, 1, 1, stride);
                predict_vpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(&mut uws, 8, 1, 1, stride);
                predict_hpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut uws, 8, 1, 1, stride);
                predict_tmpred(&mut vws, 8, 1, 1, stride);
            }
        }

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = resdata[16 * 16 + i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut uws, urb, y0, x0, stride);

                let vrb: &[i32; 16] = resdata[20 * 16 + i * 16..][..16].try_into().unwrap();

                add_residue(&mut vws, vrb, y0, x0, stride);
            }
        }

        set_chroma_border(&mut self.left_border_u, &mut self.top_border_u, &uws, mbx);
        set_chroma_border(&mut self.left_border_v, &mut self.top_border_v, &vws, mbx);

        // Write to row cache instead of final buffer
        let extra_uv_rows = self.extra_y_rows / 2;
        let cache_uv_offset = extra_uv_rows * self.cache_uv_stride;
        for y in 0usize..8 {
            let dst_start = cache_uv_offset + y * self.cache_uv_stride + mbx * 8;
            let ws_index = (1 + y) * stride + 1;
            self.cache_u[dst_start..][..8].copy_from_slice(&uws[ws_index..][..8]);
            self.cache_v[dst_start..][..8].copy_from_slice(&vws[ws_index..][..8]);
        }
    }

    /// Read DCT coefficients using libwebp's inline tree structure.
    /// This mirrors GetCoeffsFast from libwebp for maximum performance.
    fn read_coefficients(
        &mut self,
        block: &mut [i32; 16],
        p: usize,
        plane: Plane,
        complexity: usize,
        dcq: i16,
        acq: i16,
    ) -> Result<bool, DecodingError> {
        debug_assert!(complexity <= 2);

        let first = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];
        let mut reader = self.partitions.reader(p);

        // Probability indices in tree nodes match libwebp:
        // [0] = not EOB, [1] = not zero, [2] = not 1
        // [3] = >=5 vs 2/3/4, [4] = 3/4 vs 2, [5] = 4 vs 3
        // [6] = CAT3-6 vs CAT1/2, [7] = CAT2 vs CAT1
        // [8] = CAT5/6 vs CAT3/4, [9] = CAT4 vs CAT3, [10] = CAT6 vs CAT5

        let mut n = first;
        let mut band = COEFF_BANDS[n] as usize;
        let mut prob = &probs[band][complexity];

        while n < 16 {
            // Check for EOB (p[0])
            if reader.get_bit(prob[0].prob) == 0 {
                break;
            }

            // Skip zeros (p[1]) - libwebp style loop
            while reader.get_bit(prob[1].prob) == 0 {
                n += 1;
                if n >= 16 {
                    if reader.is_eof() {
                        return Err(DecodingError::BitStreamError);
                    }
                    return Ok(true);
                }
                band = COEFF_BANDS[n] as usize;
                prob = &probs[band][0]; // context 0 after zero
            }

            // Non-zero coefficient - get next context probabilities
            let next_band = if n + 1 < 16 { COEFF_BANDS[n + 1] as usize } else { 0 };
            let v: i32;
            let next_ctx: usize;

            // Check if value is 1 (p[2])
            if reader.get_bit(prob[2].prob) == 0 {
                v = 1;
                next_ctx = 1;
            } else {
                // Larger value - inline GetLargeValue
                // Check if 2/3/4 vs categories (p[3])
                if reader.get_bit(prob[3].prob) == 0 {
                    // Value is 2, 3, or 4
                    if reader.get_bit(prob[4].prob) == 0 {
                        v = 2;
                    } else {
                        v = 3 + reader.get_bit(prob[5].prob);
                    }
                } else {
                    // Category token (CAT1-CAT6)
                    if reader.get_bit(prob[6].prob) == 0 {
                        // CAT1 or CAT2
                        if reader.get_bit(prob[7].prob) == 0 {
                            // CAT1: base 5, 1 extra bit
                            v = 5 + reader.get_bit(159);
                        } else {
                            // CAT2: base 7, 2 extra bits
                            v = 7 + 2 * reader.get_bit(165) + reader.get_bit(145);
                        }
                    } else {
                        // CAT3-6: use p[8], p[9+bit1]
                        let bit1 = reader.get_bit(prob[8].prob);
                        let bit0 = reader.get_bit(prob[9 + bit1 as usize].prob);
                        let cat = (2 * bit1 + bit0) as usize;

                        // Read extra bits from category probability table
                        let cat_probs = &PROB_DCT_CAT[2 + cat]; // CAT3 is index 2
                        let mut extra = 0i32;
                        for &p in cat_probs.iter() {
                            if p == 0 {
                                break;
                            }
                            extra = extra + extra + reader.get_bit(p);
                        }
                        // CAT3: 11, CAT4: 19, CAT5: 35, CAT6: 67
                        v = 3 + (8 << cat) + extra;
                    }
                }
                next_ctx = 2;
            }

            // Read sign and apply
            let signed_v = if reader.get_bit(128) != 0 { -v } else { v };

            let zigzag = ZIGZAG[n] as usize;
            let q = if zigzag > 0 { acq } else { dcq };
            block[zigzag] = signed_v * i32::from(q);

            n += 1;
            if n < 16 {
                band = COEFF_BANDS[n] as usize;
                prob = &probs[band][next_ctx];
            }
        }

        if reader.is_eof() {
            return Err(DecodingError::BitStreamError);
        }
        Ok(n > first)
    }

    fn read_residual_data(
        &mut self,
        mb: &mut MacroBlock,
        mbx: usize,
        p: usize,
    ) -> Result<[i32; 384], DecodingError> {
        let sindex = mb.segmentid as usize;
        let mut blocks = [0i32; 384];
        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = self.top[mbx].complexity[0] + self.left.complexity[0];
            let mut block = [0i32; 16];
            let dcq = self.segment[sindex].y2dc;
            let acq = self.segment[sindex].y2ac;
            let n = self.read_coefficients(&mut block, p, plane, complexity as usize, dcq, acq)?;

            self.left.complexity[0] = if n { 1 } else { 0 };
            self.top[mbx].complexity[0] = if n { 1 } else { 0 };

            transform::iwht4x4(&mut block);

            for k in 0usize..16 {
                blocks[16 * k] = block[k];
            }

            plane = Plane::YCoeff1;
        }

        for y in 0usize..4 {
            let mut left = self.left.complexity[y + 1];
            for x in 0usize..4 {
                let i = x + y * 4;
                let block = &mut blocks[i * 16..][..16];
                let block: &mut [i32; 16] = block.try_into().unwrap();

                let complexity = self.top[mbx].complexity[x + 1] + left;
                let dcq = self.segment[sindex].ydc;
                let acq = self.segment[sindex].yac;

                let n = self.read_coefficients(block, p, plane, complexity as usize, dcq, acq)?;

                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    transform::idct4x4(block);
                }

                left = if n { 1 } else { 0 };
                self.top[mbx].complexity[x + 1] = if n { 1 } else { 0 };
            }

            self.left.complexity[y + 1] = left;
        }

        plane = Plane::Chroma;

        for &j in &[5usize, 7usize] {
            for y in 0usize..2 {
                let mut left = self.left.complexity[y + j];

                for x in 0usize..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let block = &mut blocks[i * 16..][..16];
                    let block: &mut [i32; 16] = block.try_into().unwrap();

                    let complexity = self.top[mbx].complexity[x + j] + left;
                    let dcq = self.segment[sindex].uvdc;
                    let acq = self.segment[sindex].uvac;

                    let n =
                        self.read_coefficients(block, p, plane, complexity as usize, dcq, acq)?;
                    if block[0] != 0 || n {
                        mb.non_zero_dct = true;
                        transform::idct4x4(block);
                    }

                    left = if n { 1 } else { 0 };
                    self.top[mbx].complexity[x + j] = if n { 1 } else { 0 };
                }

                self.left.complexity[y + j] = left;
            }
        }

        Ok(blocks)
    }

    /// Filters a row of macroblocks in the cache
    /// This operates on cache_y/u/v which have stride cache_y_stride/cache_uv_stride
    fn filter_row_in_cache(&mut self, mby: usize) {
        let mbwidth = self.mbwidth as usize;
        let cache_y_stride = self.cache_y_stride;
        let cache_uv_stride = self.cache_uv_stride;
        let extra_y_rows = self.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;

        for mbx in 0..mbwidth {
            let mb = self.macroblocks[mby * mbwidth + mbx];
            let (filter_level, interior_limit, hev_threshold) = self.calculate_filter_parameters(&mb);

            if filter_level == 0 {
                continue;
            }

            let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
            let sub_bedge_limit = (filter_level * 2) + interior_limit;
            let do_subblock_filtering =
                mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped && mb.non_zero_dct);

            // Filter across left of macroblock (horizontal filter on vertical edge)
            if mbx > 0 {
                if self.frame.filter_type {
                    // Simple filter
                    simple_filter_horizontal_16_rows(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        mbedge_limit,
                    );
                } else {
                    // Normal filter
                    for y in 0..16 {
                        let row = extra_y_rows + y;
                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.cache_y[row * cache_y_stride + mbx * 16 - 4..][..8],
                        );
                    }
                    for y in 0..8 {
                        let row = extra_uv_rows + y;
                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.cache_u[row * cache_uv_stride + mbx * 8 - 4..][..8],
                        );
                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.cache_v[row * cache_uv_stride + mbx * 8 - 4..][..8],
                        );
                    }
                }
            }

            // Filter across vertical subblocks
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for x in (4usize..16 - 1).step_by(4) {
                        simple_filter_horizontal_16_rows(
                            &mut self.cache_y[..],
                            extra_y_rows,
                            mbx * 16 + x,
                            cache_y_stride,
                            sub_bedge_limit,
                        );
                    }
                } else {
                    for x in (4usize..16 - 3).step_by(4) {
                        for y in 0..16 {
                            let row = extra_y_rows + y;
                            loop_filter::subblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.cache_y[row * cache_y_stride + mbx * 16 + x - 4..][..8],
                            );
                        }
                    }
                    for y in 0..8 {
                        let row = extra_uv_rows + y;
                        loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.cache_u[row * cache_uv_stride + mbx * 8 + 4 - 4..][..8],
                        );
                        loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.cache_v[row * cache_uv_stride + mbx * 8 + 4 - 4..][..8],
                        );
                    }
                }
            }

            // Filter across top of macroblock (vertical filter on horizontal edge)
            // For mby > 0, we filter between extra_rows area and current row
            if mby > 0 {
                // The edge is at row extra_y_rows (start of current MB row)
                // We need rows (extra_y_rows - 4) to (extra_y_rows + 4) approximately
                if self.frame.filter_type {
                    simple_filter_vertical_16_cols(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        mbedge_limit,
                    );
                } else {
                    // Use SIMD helper for luma
                    normal_filter_vertical_mb_16_cols(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        hev_threshold,
                        interior_limit,
                        mbedge_limit,
                    );
                    // Chroma
                    for x in 0..8 {
                        loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.cache_u[..],
                            extra_uv_rows * cache_uv_stride + mbx * 8 + x,
                            cache_uv_stride,
                        );
                        loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.cache_v[..],
                            extra_uv_rows * cache_uv_stride + mbx * 8 + x,
                            cache_uv_stride,
                        );
                    }
                }
            }

            // Filter across horizontal subblock edges
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for y in (4usize..16 - 1).step_by(4) {
                        simple_filter_vertical_16_cols(
                            &mut self.cache_y[..],
                            extra_y_rows + y,
                            mbx * 16,
                            cache_y_stride,
                            sub_bedge_limit,
                        );
                    }
                } else {
                    for y in (4usize..16 - 3).step_by(4) {
                        normal_filter_vertical_sub_16_cols(
                            &mut self.cache_y[..],
                            extra_y_rows + y,
                            mbx * 16,
                            cache_y_stride,
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                        );
                    }
                    for y in 0..1 {
                        for x in 0..8 {
                            loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.cache_u[..],
                                (extra_uv_rows + 4 + y * 4) * cache_uv_stride + mbx * 8 + x,
                                cache_uv_stride,
                            );
                            loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.cache_v[..],
                                (extra_uv_rows + 4 + y * 4) * cache_uv_stride + mbx * 8 + x,
                                cache_uv_stride,
                            );
                        }
                    }
                }
            }
        }
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
            // First row: output rows extra_y_rows to (extra_y_rows + 16 - extra_y_rows) = 16 - extra_y_rows rows
            // Starting from cache row extra_y_rows, output 16 - extra_y_rows rows to final row 0
            (extra_y_rows, 16 - extra_y_rows, 0usize)
        } else if is_last_row {
            // Last row: output extra area (extra_y_rows rows) + full current row (16 rows)
            // Starting from cache row 0, output extra_y_rows + 16 rows
            // Destination starts at mby*16 - extra_y_rows
            (0, extra_y_rows + 16, mby * 16 - extra_y_rows)
        } else {
            // Middle row: output extra area + (16 - extra_y_rows) rows of current
            // Starting from cache row 0, output extra_y_rows + 16 - extra_y_rows = 16 rows
            (0, 16, mby * 16 - extra_y_rows)
        };

        // Copy luma
        for y in 0..num_y_rows {
            let src_row = src_start_row + y;
            let dst_row = dst_start_y_row + y;
            let src_start = src_row * self.cache_y_stride;
            let dst_start = dst_row * luma_w;
            self.frame.ybuf[dst_start..dst_start + luma_w]
                .copy_from_slice(&self.cache_y[src_start..src_start + luma_w]);
        }

        // Same logic for chroma but with half the rows
        let (src_start_row_uv, num_uv_rows, dst_start_uv_row) = if is_first_row {
            (extra_uv_rows, 8 - extra_uv_rows, 0usize)
        } else if is_last_row {
            (0, extra_uv_rows + 8, mby * 8 - extra_uv_rows)
        } else {
            (0, 8, mby * 8 - extra_uv_rows)
        };

        // Copy chroma
        for y in 0..num_uv_rows {
            let src_row = src_start_row_uv + y;
            let dst_row = dst_start_uv_row + y;
            let src_start = src_row * self.cache_uv_stride;
            let dst_start = dst_row * chroma_w;
            self.frame.ubuf[dst_start..dst_start + chroma_w]
                .copy_from_slice(&self.cache_u[src_start..src_start + chroma_w]);
            self.frame.vbuf[dst_start..dst_start + chroma_w]
                .copy_from_slice(&self.cache_v[src_start..src_start + chroma_w]);
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
        self.cache_y.copy_within(src_start..src_start + copy_size, 0);

        // For chroma:
        let src_start_row_uv = 8; // = extra_uv_rows + 8 - extra_uv_rows = 8
        let src_start_uv = src_start_row_uv * self.cache_uv_stride;
        let copy_size_uv = extra_uv_rows * self.cache_uv_stride;
        self.cache_u.copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
        self.cache_v.copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
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
    pub fn decode_frame(r: R) -> Result<Frame, DecodingError> {
        let decoder = Self::new(r);
        decoder.decode_frame_()
    }

    fn decode_frame_(mut self) -> Result<Frame, DecodingError> {
        self.read_frame_header()?;

        for mby in 0..self.mbheight as usize {
            let p = mby % self.num_partitions as usize;
            self.left = PreviousMacroBlock::default();

            // Decode all macroblocks in this row (writes to cache)
            for mbx in 0..self.mbwidth as usize {
                let mut mb = self.read_macroblock_header(mbx)?;
                let blocks = if !mb.coeffs_skipped {
                    self.read_residual_data(&mut mb, mbx, p)?
                } else {
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        self.top[mbx].complexity[0] = 0;
                    }

                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        self.top[mbx].complexity[i] = 0;
                    }

                    [0i32; 384]
                };

                self.intra_predict_luma(mbx, mby, &mb, &blocks);
                self.intra_predict_chroma(mbx, mby, &mb, &blocks);

                self.macroblocks.push(mb);
            }

            // Row complete: filter in cache, output to final buffer, prepare for next row
            self.filter_row_in_cache(mby);
            self.output_row_from_cache(mby);
            self.rotate_extra_rows();

            self.left_border_y = vec![129u8; 1 + 16];
            self.left_border_u = vec![129u8; 1 + 8];
            self.left_border_v = vec![129u8; 1 + 8];
        }

        Ok(self.frame)
    }
}

// set border
fn set_chroma_border(
    left_border: &mut [u8],
    top_border: &mut [u8],
    chroma_block: &[u8],
    mbx: usize,
) {
    let stride = CHROMA_STRIDE;
    // top left is top right of previous chroma block
    left_border[0] = chroma_block[8];

    // left border
    for (i, left) in left_border[1..][..8].iter_mut().enumerate() {
        *left = chroma_block[(i + 1) * stride + 8];
    }

    for (top, &w) in top_border[mbx * 8..][..8]
        .iter_mut()
        .zip(&chroma_block[8 * stride + 1..][..8])
    {
        *top = w;
    }
}
