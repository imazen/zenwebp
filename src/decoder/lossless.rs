//! Decoding of lossless WebP images
//!
//! [Lossless spec](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::slice_reader::SliceReader;

use super::api::DecodeError;
use super::lossless_transform::{
    apply_color_indexing_transform, apply_color_transform, apply_predictor_transform,
    apply_subtract_green_transform,
};

use super::huffman::HuffmanTree;
use super::lossless_transform::TransformType;

const CODE_LENGTH_CODES: usize = 19;
const CODE_LENGTH_CODE_ORDER: [usize; CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];

const GREEN: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DIST: usize = 4;

const HUFFMAN_CODES_PER_META_CODE: usize = 5;

type HuffmanCodeGroup = [HuffmanTree; HUFFMAN_CODES_PER_META_CODE];

const ALPHABET_SIZE: [u16; HUFFMAN_CODES_PER_META_CODE] = [256 + 24, 256, 256, 256, 40];

#[inline]
pub(crate) fn subsample_size(size: u16, bits: u8) -> u16 {
    ((u32::from(size) + (1u32 << bits) - 1) >> bits)
        .try_into()
        .unwrap()
}

const NUM_TRANSFORM_TYPES: usize = 4;

//Decodes lossless WebP images
pub(crate) struct LosslessDecoder<'a> {
    bit_reader: BitReader<'a>,
    transforms: [Option<TransformType>; NUM_TRANSFORM_TYPES],
    transform_order: Vec<u8>,
    width: u16,
    height: u16,
    stop: Option<&'a dyn enough::Stop>,
}

impl<'a> LosslessDecoder<'a> {
    /// Create a new decoder
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            bit_reader: BitReader::new(SliceReader::new(data)),
            transforms: [None, None, None, None],
            transform_order: Vec::new(),
            width: 0,
            height: 0,
            stop: None,
        }
    }

    /// Set a cooperative cancellation token.
    pub(crate) fn set_stop(&mut self, stop: Option<&'a dyn enough::Stop>) {
        self.stop = stop;
    }

    /// Decodes a frame.
    ///
    /// In an alpha chunk the width and height are not included in the header, so they should be
    /// provided by setting the `implicit_dimensions` argument. Otherwise that argument should be
    /// `None` and the frame dimensions will be determined by reading the VP8L header.
    pub(crate) fn decode_frame(
        &mut self,
        width: u32,
        height: u32,
        implicit_dimensions: bool,
        buf: &mut [u8],
    ) -> Result<(), DecodeError> {
        if implicit_dimensions {
            self.width = width as u16;
            self.height = height as u16;
        } else {
            let signature = self.bit_reader.read_bits::<u8>(8)?;
            if signature != 0x2f {
                return Err(DecodeError::LosslessSignatureInvalid(signature));
            }

            self.width = self.bit_reader.read_bits::<u16>(14)? + 1;
            self.height = self.bit_reader.read_bits::<u16>(14)? + 1;
            if u32::from(self.width) != width || u32::from(self.height) != height {
                return Err(DecodeError::InconsistentImageSizes);
            }

            let _alpha_used = self.bit_reader.read_bits::<u8>(1)?;
            let version_num = self.bit_reader.read_bits::<u8>(3)?;
            if version_num != 0 {
                return Err(DecodeError::VersionNumberInvalid(version_num));
            }
        }

        let transformed_width = self.read_transforms()?;
        let transformed_size = usize::from(transformed_width) * usize::from(self.height) * 4;
        self.decode_image_stream(
            transformed_width,
            self.height,
            true,
            &mut buf[..transformed_size],
        )?;

        let mut image_size = transformed_size;
        let mut width = transformed_width;
        for &trans_index in self.transform_order.iter().rev() {
            let transform = self.transforms[usize::from(trans_index)].as_ref().unwrap();
            match transform {
                TransformType::PredictorTransform {
                    size_bits,
                    predictor_data,
                } => apply_predictor_transform(
                    &mut buf[..image_size],
                    width,
                    self.height,
                    *size_bits,
                    predictor_data,
                )?,
                TransformType::ColorTransform {
                    size_bits,
                    transform_data,
                } => {
                    apply_color_transform(
                        &mut buf[..image_size],
                        width,
                        *size_bits,
                        transform_data,
                    );
                }
                TransformType::SubtractGreen => {
                    apply_subtract_green_transform(&mut buf[..image_size]);
                }
                TransformType::ColorIndexingTransform {
                    table_size,
                    table_data,
                } => {
                    width = self.width;
                    image_size = usize::from(width) * usize::from(self.height) * 4;
                    apply_color_indexing_transform(
                        buf,
                        width,
                        self.height,
                        *table_size,
                        table_data,
                    );
                }
            }
        }

        Ok(())
    }

    /// Reads Image data from the bitstream
    ///
    /// Can be in any of the 5 roles described in the Specification. ARGB Image role has different
    /// behaviour to the other 4. xsize and ysize describe the size of the blocks where each block
    /// has its own entropy code
    fn decode_image_stream(
        &mut self,
        xsize: u16,
        ysize: u16,
        is_argb_img: bool,
        data: &mut [u8],
    ) -> Result<(), DecodeError> {
        let color_cache_bits = self.read_color_cache()?;
        let color_cache = color_cache_bits.map(ColorCache::new);

        let huffman_info = self.read_huffman_codes(is_argb_img, xsize, ysize, color_cache)?;
        self.decode_image_data(xsize, ysize, huffman_info, data)
    }

    /// Reads transforms and their data from the bitstream
    fn read_transforms(&mut self) -> Result<u16, DecodeError> {
        let mut xsize = self.width;

        while self.bit_reader.read_bits::<u8>(1)? == 1 {
            let transform_type_val = self.bit_reader.read_bits::<u8>(2)?;

            if self.transforms[usize::from(transform_type_val)].is_some() {
                //can only have one of each transform, error
                return Err(DecodeError::TransformError);
            }

            self.transform_order.push(transform_type_val);

            let transform_type = match transform_type_val {
                0 => {
                    //predictor

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let mut predictor_data =
                        vec![0; usize::from(block_xsize) * usize::from(block_ysize) * 4];
                    self.decode_image_stream(block_xsize, block_ysize, false, &mut predictor_data)?;

                    TransformType::PredictorTransform {
                        size_bits,
                        predictor_data,
                    }
                }
                1 => {
                    //color transform

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let mut transform_data =
                        vec![0; usize::from(block_xsize) * usize::from(block_ysize) * 4];
                    self.decode_image_stream(block_xsize, block_ysize, false, &mut transform_data)?;

                    TransformType::ColorTransform {
                        size_bits,
                        transform_data,
                    }
                }
                2 => {
                    //subtract green

                    TransformType::SubtractGreen
                }
                3 => {
                    let color_table_size = self.bit_reader.read_bits::<u16>(8)? + 1;

                    let mut color_map = vec![0; usize::from(color_table_size) * 4];
                    self.decode_image_stream(color_table_size, 1, false, &mut color_map)?;

                    let bits = if color_table_size <= 2 {
                        3
                    } else if color_table_size <= 4 {
                        2
                    } else if color_table_size <= 16 {
                        1
                    } else {
                        0
                    };
                    xsize = subsample_size(xsize, bits);

                    Self::adjust_color_map(&mut color_map);

                    TransformType::ColorIndexingTransform {
                        table_size: color_table_size,
                        table_data: color_map,
                    }
                }
                _ => unreachable!(),
            };

            self.transforms[usize::from(transform_type_val)] = Some(transform_type);
        }

        Ok(xsize)
    }

    /// Adjusts the color map since it's subtraction coded
    fn adjust_color_map(color_map: &mut [u8]) {
        for i in 4..color_map.len() {
            color_map[i] = color_map[i].wrapping_add(color_map[i - 4]);
        }
    }

    /// Reads huffman codes associated with an image
    #[inline(never)]
    fn read_huffman_codes(
        &mut self,
        read_meta: bool,
        xsize: u16,
        ysize: u16,
        color_cache: Option<ColorCache>,
    ) -> Result<HuffmanInfo, DecodeError> {
        let mut num_huff_groups = 1u32;

        let mut huffman_bits = 0;
        let mut huffman_xsize = 1;
        let mut huffman_ysize = 1;
        let mut entropy_image = Vec::new();

        if read_meta && self.bit_reader.read_bits::<u8>(1)? == 1 {
            //meta huffman codes
            huffman_bits = self.bit_reader.read_bits::<u8>(3)? + 2;
            huffman_xsize = subsample_size(xsize, huffman_bits);
            huffman_ysize = subsample_size(ysize, huffman_bits);

            let mut data = vec![0; usize::from(huffman_xsize) * usize::from(huffman_ysize) * 4];
            self.decode_image_stream(huffman_xsize, huffman_ysize, false, &mut data)?;

            entropy_image = data
                .chunks_exact(4)
                .map(|pixel| {
                    let meta_huff_code = (u16::from(pixel[0]) << 8) | u16::from(pixel[1]);
                    if u32::from(meta_huff_code) >= num_huff_groups {
                        num_huff_groups = u32::from(meta_huff_code) + 1;
                    }
                    meta_huff_code
                })
                .collect::<Vec<u16>>();
        }

        let mut hufftree_groups = Vec::new();

        for _i in 0..num_huff_groups {
            let mut group: HuffmanCodeGroup = Default::default();
            for j in 0..HUFFMAN_CODES_PER_META_CODE {
                let mut alphabet_size = ALPHABET_SIZE[j];
                if j == 0
                    && let Some(color_cache) = color_cache.as_ref()
                {
                    alphabet_size += 1 << color_cache.color_cache_bits;
                }

                let tree = self.read_huffman_code(alphabet_size)?;
                group[j] = tree;
            }
            hufftree_groups.push(group);
        }

        let huffman_mask = if huffman_bits == 0 {
            !0
        } else {
            (1 << huffman_bits) - 1
        };

        let mut info = HuffmanInfo {
            xsize: huffman_xsize,
            _ysize: huffman_ysize,
            color_cache,
            image: entropy_image,
            bits: huffman_bits,
            mask: huffman_mask,
            huffman_code_groups: hufftree_groups,
            group_meta: Vec::new(),
        };
        info.build_group_meta();

        Ok(info)
    }

    /// Decodes and returns a single huffman tree
    fn read_huffman_code(&mut self, alphabet_size: u16) -> Result<HuffmanTree, DecodeError> {
        let simple = self.bit_reader.read_bits::<u8>(1)? == 1;

        if simple {
            let num_symbols = self.bit_reader.read_bits::<u8>(1)? + 1;

            let is_first_8bits = self.bit_reader.read_bits::<u8>(1)?;
            let zero_symbol = self.bit_reader.read_bits::<u16>(1 + 7 * is_first_8bits)?;

            if zero_symbol >= alphabet_size {
                return Err(DecodeError::BitStreamError);
            }

            if num_symbols == 1 {
                Ok(HuffmanTree::build_single_node(zero_symbol))
            } else {
                let one_symbol = self.bit_reader.read_bits::<u16>(8)?;
                if one_symbol >= alphabet_size {
                    return Err(DecodeError::BitStreamError);
                }
                Ok(HuffmanTree::build_two_node(zero_symbol, one_symbol))
            }
        } else {
            let mut code_length_code_lengths = vec![0; CODE_LENGTH_CODES];

            let num_code_lengths = 4 + self.bit_reader.read_bits::<usize>(4)?;
            for i in 0..num_code_lengths {
                code_length_code_lengths[CODE_LENGTH_CODE_ORDER[i]] =
                    self.bit_reader.read_bits(3)?;
            }

            let new_code_lengths =
                self.read_huffman_code_lengths(code_length_code_lengths, alphabet_size)?;

            HuffmanTree::build_implicit(new_code_lengths)
        }
    }

    /// Reads huffman code lengths
    fn read_huffman_code_lengths(
        &mut self,
        code_length_code_lengths: Vec<u16>,
        num_symbols: u16,
    ) -> Result<Vec<u16>, DecodeError> {
        let table = HuffmanTree::build_implicit(code_length_code_lengths)?;

        let mut max_symbol = if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let length_nbits = 2 + 2 * self.bit_reader.read_bits::<u8>(3)?;
            let max_minus_two = self.bit_reader.read_bits::<u16>(length_nbits)?;
            if max_minus_two > num_symbols - 2 {
                return Err(DecodeError::BitStreamError);
            }
            2 + max_minus_two
        } else {
            num_symbols
        };

        let mut code_lengths = vec![0; usize::from(num_symbols)];
        let mut prev_code_len = 8; //default code length

        let mut symbol = 0;
        while symbol < num_symbols {
            if max_symbol == 0 {
                break;
            }
            max_symbol -= 1;

            self.bit_reader.fill()?;
            let code_len = table.read_symbol(&mut self.bit_reader)?;

            if code_len < 16 {
                code_lengths[usize::from(symbol)] = code_len;
                symbol += 1;
                if code_len != 0 {
                    prev_code_len = code_len;
                }
            } else {
                let use_prev = code_len == 16;
                let slot = code_len - 16;
                let extra_bits = match slot {
                    0 => 2,
                    1 => 3,
                    2 => 7,
                    _ => return Err(DecodeError::BitStreamError),
                };
                let repeat_offset = match slot {
                    0 | 1 => 3,
                    2 => 11,
                    _ => return Err(DecodeError::BitStreamError),
                };

                let mut repeat = self.bit_reader.read_bits::<u16>(extra_bits)? + repeat_offset;

                if symbol + repeat > num_symbols {
                    return Err(DecodeError::BitStreamError);
                }

                let length = if use_prev { prev_code_len } else { 0 };
                while repeat > 0 {
                    repeat -= 1;
                    code_lengths[usize::from(symbol)] = length;
                    symbol += 1;
                }
            }
        }

        Ok(code_lengths)
    }

    /// Decodes the image data using the huffman trees and either of the 3 methods of decoding.
    ///
    /// This is the hot loop. Optimizations matching libwebp's DecodeImageData:
    /// - `is_trivial_code`: all 5 trees single-symbol, no bits read
    /// - `use_packed_table`: single 6-bit lookup decodes entire ARGB pixel
    /// - `is_trivial_literal`: R/B/A are constant, only read green from bitstream
    /// - Incremental col/row tracking: avoid div/mod per block boundary
    fn decode_image_data(
        &mut self,
        width: u16,
        height: u16,
        mut huffman_info: HuffmanInfo,
        data: &mut [u8],
    ) -> Result<(), DecodeError> {
        let width_usize = usize::from(width);
        let num_values = width_usize * usize::from(height);
        let len_code_limit: u16 = 256 + 24;
        let color_cache_limit: u16 = len_code_limit
            + huffman_info
                .color_cache
                .as_ref()
                .map_or(0, |cc| 1u16 << cc.color_cache_bits);
        let mask = usize::from(huffman_info.mask);

        let mut col: usize = 0;
        let mut row: usize = 0;
        let mut index: usize = 0;

        let huff_index = huffman_info.get_huff_index(0, 0);
        let mut group_idx = huff_index;
        let mut tree = &huffman_info.huffman_code_groups[huff_index];
        let mut meta = &huffman_info.group_meta[huff_index];

        while index < num_values {
            // Update huffman group at tile boundaries (col & mask == 0)
            if (col & mask) == 0 {
                let new_idx = huffman_info.get_huff_index(col as u16, row as u16);
                if new_idx != group_idx {
                    group_idx = new_idx;
                    tree = &huffman_info.huffman_code_groups[group_idx];
                    meta = &huffman_info.group_meta[group_idx];
                }

                // Check for cooperative cancellation at block boundaries
                if let Some(stop) = self.stop {
                    stop.check()?;
                }
            }

            // Fast path 1: is_trivial_code - all trees are single-symbol, no bits read
            if meta.is_trivial_code {
                data[index * 4..][..4].copy_from_slice(&meta.trivial_pixel);
                if let Some(cc) = huffman_info.color_cache.as_mut() {
                    cc.insert(meta.trivial_pixel);
                }
                index += 1;
                col += 1;
                if col >= width_usize {
                    col = 0;
                    row += 1;
                }
                continue;
            }

            self.bit_reader.fill()?;

            // Fast path 2: use_packed_table - single 6-bit lookup decodes entire pixel
            let code;
            if meta.use_packed_table {
                let val =
                    (self.bit_reader.peek_full() as usize) & (PACKED_TABLE_SIZE - 1);
                let entry = meta.packed_table[val];
                if entry.bits < BITS_SPECIAL_MARKER {
                    // Literal pixel decoded in one lookup
                    self.bit_reader.consume(entry.bits)?;
                    data[index * 4..][..4].copy_from_slice(&entry.value);
                    if let Some(cc) = huffman_info.color_cache.as_mut() {
                        cc.insert(entry.value);
                    }
                    index += 1;
                    col += 1;
                    if col >= width_usize {
                        col = 0;
                        row += 1;
                    }
                    continue;
                }
                // Non-literal: extract green code, fall through to normal processing
                let actual_bits = entry.bits - BITS_SPECIAL_MARKER;
                self.bit_reader.consume(actual_bits)?;
                code = u16::from(entry.value[1]) | (u16::from(entry.value[2]) << 8);
            } else {
                code = tree[GREEN].read_symbol(&mut self.bit_reader)?;
            }

            if code < 256 {
                // Literal pixel
                let green = code as u8;
                if meta.is_trivial_literal {
                    // R, B, A are constant - only read green from bitstream
                    let pixel = [meta.literal_arb[0], green, meta.literal_arb[2], meta.literal_arb[3]];
                    data[index * 4..][..4].copy_from_slice(&pixel);
                    if let Some(cc) = huffman_info.color_cache.as_mut() {
                        cc.insert(pixel);
                    }
                } else {
                    let red = tree[RED].read_symbol(&mut self.bit_reader)? as u8;
                    if self.bit_reader.nbits < 15 {
                        self.bit_reader.fill()?;
                    }
                    let blue = tree[BLUE].read_symbol(&mut self.bit_reader)? as u8;
                    let alpha = tree[ALPHA].read_symbol(&mut self.bit_reader)? as u8;
                    data[index * 4] = red;
                    data[index * 4 + 1] = green;
                    data[index * 4 + 2] = blue;
                    data[index * 4 + 3] = alpha;
                    if let Some(cc) = huffman_info.color_cache.as_mut() {
                        cc.insert([red, green, blue, alpha]);
                    }
                }

                index += 1;
                col += 1;
                if col >= width_usize {
                    col = 0;
                    row += 1;
                }
            } else if code < len_code_limit {
                // Backward reference
                let length_symbol = code - 256;
                let length = Self::get_copy_distance(&mut self.bit_reader, length_symbol)?;

                let dist_symbol = tree[DIST].read_symbol(&mut self.bit_reader)?;
                let dist_code = Self::get_copy_distance(&mut self.bit_reader, dist_symbol)?;
                let dist = Self::plane_code_to_distance(width, dist_code);

                if index < dist || num_values - index < length {
                    return Err(DecodeError::BitStreamError);
                }

                // Copy block
                if dist == 1 {
                    let value: [u8; 4] = data[(index - dist) * 4..][..4].try_into().unwrap();
                    for i in 0..length {
                        data[index * 4 + i * 4..][..4].copy_from_slice(&value);
                    }
                } else {
                    if index + length + 3 <= num_values {
                        let start = (index - dist) * 4;
                        data.copy_within(start..start + 16, index * 4);

                        if length > 4 || dist < 4 {
                            for i in (0..length * 4).step_by((dist * 4).min(16)).skip(1) {
                                data.copy_within(start + i..start + i + 16, index * 4 + i);
                            }
                        }
                    } else {
                        for i in 0..length * 4 {
                            data[index * 4 + i] = data[index * 4 + i - dist * 4];
                        }
                    }

                    if let Some(cc) = huffman_info.color_cache.as_mut() {
                        for pixel in data[index * 4..][..length * 4].chunks_exact(4) {
                            cc.insert(pixel.try_into().unwrap());
                        }
                    }
                }

                index += length;
                col += length;
                while col >= width_usize {
                    col -= width_usize;
                    row += 1;
                }

                // Update huffman group if we crossed a tile boundary
                if (col & mask) != 0 {
                    let new_idx = huffman_info.get_huff_index(col as u16, row as u16);
                    if new_idx != group_idx {
                        group_idx = new_idx;
                        tree = &huffman_info.huffman_code_groups[group_idx];
                        meta = &huffman_info.group_meta[group_idx];
                    }
                }
            } else if code < color_cache_limit {
                // Color cache lookup
                let color_cache = huffman_info
                    .color_cache
                    .as_mut()
                    .ok_or(DecodeError::BitStreamError)?;
                let key = usize::from(code - len_code_limit);
                let cached_u32 = color_cache.lookup_u32(key);
                let pixel = u32_to_pixel(cached_u32);
                data[index * 4..][..4].copy_from_slice(&pixel);

                index += 1;
                col += 1;
                if col >= width_usize {
                    col = 0;
                    row += 1;
                }
            } else {
                return Err(DecodeError::BitStreamError);
            }
        }

        Ok(())
    }

    /// Reads color cache data from the bitstream
    fn read_color_cache(&mut self) -> Result<Option<u8>, DecodeError> {
        if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let code_bits = self.bit_reader.read_bits::<u8>(4)?;

            if !(1..=11).contains(&code_bits) {
                return Err(DecodeError::InvalidColorCacheBits(code_bits));
            }

            Ok(Some(code_bits))
        } else {
            Ok(None)
        }
    }

    /// Gets the copy distance from the prefix code and bitstream
    fn get_copy_distance(
        bit_reader: &mut BitReader<'_>,
        prefix_code: u16,
    ) -> Result<usize, DecodeError> {
        if prefix_code < 4 {
            return Ok(usize::from(prefix_code + 1));
        }
        let extra_bits: u8 = ((prefix_code - 2) >> 1).try_into().unwrap();
        let offset = (2 + (usize::from(prefix_code) & 1)) << extra_bits;

        let bits = bit_reader.peek(extra_bits) as usize;
        bit_reader.consume(extra_bits)?;

        Ok(offset + bits + 1)
    }

    /// Gets distance to pixel
    fn plane_code_to_distance(xsize: u16, plane_code: usize) -> usize {
        if plane_code > 120 {
            plane_code - 120
        } else {
            let (xoffset, yoffset) = DISTANCE_MAP[plane_code - 1];

            let dist = i32::from(xoffset) + i32::from(yoffset) * i32::from(xsize);
            if dist < 1 {
                return 1;
            }
            dist.try_into().unwrap()
        }
    }
}

/// Packed table size: 2^6 = 64 entries, matching libwebp's HUFFMAN_PACKED_BITS=6.
/// When all 4 channel codes (green+red+blue+alpha) fit in <= 6 total bits,
/// a single table lookup decodes the entire ARGB pixel.
const PACKED_BITS: u8 = 6;
const PACKED_TABLE_SIZE: usize = 1 << PACKED_BITS;
/// Marker in the `bits` field of a PackedEntry to signal that the green code
/// is a non-literal (length prefix or color cache code). The actual bits
/// consumed for the green code is `bits - BITS_SPECIAL_MARKER`.
const BITS_SPECIAL_MARKER: u8 = 64;

/// A single entry in the packed lookup table.
#[derive(Debug, Clone, Copy, Default)]
struct PackedEntry {
    /// Total bits consumed. If >= BITS_SPECIAL_MARKER, this is a non-literal
    /// green code and the actual bits consumed is `bits - BITS_SPECIAL_MARKER`.
    bits: u8,
    /// For literals: packed pixel as [R, G, B, A].
    /// For non-literals: green code stored in value[1] and value[0].
    value: [u8; 4],
}

/// Per-group metadata matching libwebp's HTreeGroup optimizations.
#[derive(Debug, Clone)]
struct GroupMeta {
    /// True if R, B, A Huffman trees each have a single symbol (bits==0).
    is_trivial_literal: bool,
    /// Pre-packed [R, 0, B, A] for trivial literals (green channel zeroed).
    literal_arb: [u8; 4],
    /// True if all 5 trees are single-symbol (no bits read at all).
    is_trivial_code: bool,
    /// If is_trivial_code, the complete pixel value.
    trivial_pixel: [u8; 4],
    /// True if a packed table is available for this group.
    use_packed_table: bool,
    /// Packed lookup table: 64 entries indexed by the next 6 bits.
    packed_table: [PackedEntry; PACKED_TABLE_SIZE],
}

#[derive(Debug, Clone)]
struct HuffmanInfo {
    xsize: u16,
    _ysize: u16,
    color_cache: Option<ColorCache>,
    image: Vec<u16>,
    bits: u8,
    mask: u16,
    huffman_code_groups: Vec<HuffmanCodeGroup>,
    /// Per-group metadata for fast-path optimizations.
    group_meta: Vec<GroupMeta>,
}

impl HuffmanInfo {
    fn get_huff_index(&self, x: u16, y: u16) -> usize {
        if self.bits == 0 {
            return 0;
        }
        let position =
            usize::from(y >> self.bits) * usize::from(self.xsize) + usize::from(x >> self.bits);
        usize::from(self.image[position])
    }

    /// Build per-group metadata after all Huffman trees have been read.
    fn build_group_meta(&mut self) {
        let num_groups = self.huffman_code_groups.len();
        self.group_meta = Vec::with_capacity(num_groups);

        for group in &self.huffman_code_groups {
            // Check is_trivial_literal: R, B, A trees each have single symbol
            let is_trivial_literal = group[RED].is_single_node()
                && group[BLUE].is_single_node()
                && group[ALPHA].is_single_node();

            let mut literal_arb = [0u8; 4];
            let mut is_trivial_code = false;
            let mut trivial_pixel = [0u8; 4];

            if is_trivial_literal {
                // Pre-pack the constant R, B, A values
                let (red, _) = group[RED].decode_primary(0);
                let (blue, _) = group[BLUE].decode_primary(0);
                let (alpha, _) = group[ALPHA].decode_primary(0);
                literal_arb = [red as u8, 0, blue as u8, alpha as u8];

                // Check if GREEN is also trivial
                if group[GREEN].is_single_node() {
                    let (green, _) = group[GREEN].decode_primary(0);
                    if green < 256 {
                        is_trivial_code = true;
                        trivial_pixel = [red as u8, green as u8, blue as u8, alpha as u8];
                    }
                }
            }

            // Compute max_bits across G+R+B+A (not DIST) for packed table decision
            let mut max_bits_total: u16 = 0;
            let mut can_pack = !is_trivial_code;
            for j in 0..4 {
                // GREEN=0, RED=1, BLUE=2, ALPHA=3
                let mb = group[j].max_code_bits();
                max_bits_total += u16::from(mb);
            }
            if max_bits_total >= u16::from(PACKED_BITS) {
                can_pack = false;
            }

            let mut packed_table = [PackedEntry::default(); PACKED_TABLE_SIZE];

            if can_pack {
                // Build packed table: for each possible 6-bit input, decode G+R+B+A
                for code in 0..PACKED_TABLE_SIZE {
                    let bits = code as u16;

                    // Decode green
                    let (green_sym, green_bits) = group[GREEN].decode_primary(bits);

                    if green_sym >= 256 {
                        // Non-literal green code (length prefix or color cache)
                        packed_table[code] = PackedEntry {
                            bits: green_bits + BITS_SPECIAL_MARKER,
                            value: [0, green_sym as u8, (green_sym >> 8) as u8, 0],
                        };
                    } else {
                        // Literal: decode R, B, A from remaining bits
                        let remaining = bits >> green_bits;
                        let (red_sym, red_bits) = group[RED].decode_primary(remaining);
                        let remaining2 = remaining >> red_bits;
                        let (blue_sym, blue_bits) = group[BLUE].decode_primary(remaining2);
                        let remaining3 = remaining2 >> blue_bits;
                        let (alpha_sym, alpha_bits) = group[ALPHA].decode_primary(remaining3);

                        let total_bits = green_bits + red_bits + blue_bits + alpha_bits;
                        packed_table[code] = PackedEntry {
                            bits: total_bits,
                            value: [red_sym as u8, green_sym as u8, blue_sym as u8, alpha_sym as u8],
                        };
                    }
                }
            }

            self.group_meta.push(GroupMeta {
                is_trivial_literal,
                literal_arb,
                is_trivial_code,
                trivial_pixel,
                use_packed_table: can_pack,
                packed_table,
            });
        }
    }
}

#[derive(Debug, Clone)]
struct ColorCache {
    color_cache_bits: u8,
    hash_shift: u8,
    color_cache: Vec<u32>,
}

impl ColorCache {
    fn new(bits: u8) -> Self {
        Self {
            color_cache_bits: bits,
            hash_shift: 32 - bits,
            color_cache: vec![0u32; 1 << bits],
        }
    }

    #[inline(always)]
    fn insert_u32(&mut self, argb: u32) {
        let index = (0x1e35a7bdu32.wrapping_mul(argb)) >> self.hash_shift;
        self.color_cache[index as usize] = argb;
    }

    #[inline(always)]
    fn insert(&mut self, color: [u8; 4]) {
        self.insert_u32(pixel_to_u32(color));
    }

    /// Insert a pixel read directly from a byte slice at the given byte offset.
    /// Avoids the overhead of try_into().unwrap() for [u8;4] conversion.
    #[inline(always)]
    fn insert_from_bytes(&mut self, data: &[u8], byte_offset: usize) {
        let r = data[byte_offset];
        let g = data[byte_offset + 1];
        let b = data[byte_offset + 2];
        let a = data[byte_offset + 3];
        let argb =
            (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
        self.insert_u32(argb);
    }

    #[inline(always)]
    fn lookup_u32(&self, index: usize) -> u32 {
        self.color_cache[index]
    }
}

/// Convert [R,G,B,A] to u32 in ARGB layout for hashing (matching libwebp's convention).
#[inline(always)]
fn pixel_to_u32(pixel: [u8; 4]) -> u32 {
    let [r, g, b, a] = pixel;
    (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
}

/// Convert u32 in ARGB layout back to [R,G,B,A].
#[inline(always)]
fn u32_to_pixel(v: u32) -> [u8; 4] {
    [
        (v >> 16) as u8,
        (v >> 8) as u8,
        v as u8,
        (v >> 24) as u8,
    ]
}

#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    reader: SliceReader<'a>,
    buffer: u64,
    nbits: u8,
}

impl<'a> BitReader<'a> {
    fn new(reader: SliceReader<'a>) -> Self {
        Self {
            reader,
            buffer: 0,
            nbits: 0,
        }
    }

    /// Fills the buffer with bits from the input stream.
    ///
    /// After this function, the internal buffer will contain 64-bits or have reached the end of
    /// the input stream.
    pub(crate) fn fill(&mut self) -> Result<(), DecodeError> {
        debug_assert!(self.nbits < 64);

        let mut buf = self.reader.fill_buf();
        if buf.len() >= 8 {
            let lookahead = u64::from_le_bytes(buf[..8].try_into().unwrap());
            self.reader.consume(usize::from((63 - self.nbits) / 8));
            self.buffer |= lookahead << self.nbits;
            self.nbits |= 56;
        } else {
            while !buf.is_empty() && self.nbits < 56 {
                self.buffer |= u64::from(buf[0]) << self.nbits;
                self.nbits += 8;
                self.reader.consume(1);
                buf = self.reader.fill_buf();
            }
        }

        Ok(())
    }

    /// Peeks at the next `num` bits in the buffer.
    pub(crate) const fn peek(&self, num: u8) -> u64 {
        self.buffer & ((1 << num) - 1)
    }

    /// Peeks at the full buffer.
    pub(crate) const fn peek_full(&self) -> u64 {
        self.buffer
    }

    /// Consumes `num` bits from the buffer returning an error if there are not enough bits.
    pub(crate) fn consume(&mut self, num: u8) -> Result<(), DecodeError> {
        if self.nbits < num {
            return Err(DecodeError::BitStreamError);
        }

        self.buffer >>= num;
        self.nbits -= num;
        Ok(())
    }

    /// Convenience function to read a number of bits and convert them to a type.
    pub(crate) fn read_bits<T: TryFrom<u32>>(&mut self, num: u8) -> Result<T, DecodeError> {
        debug_assert!(num as usize <= 8 * mem::size_of::<T>());
        debug_assert!(num <= 32);

        if self.nbits < num {
            self.fill()?;
        }
        let value = self.peek(num) as u32;
        self.consume(num)?;

        value.try_into().map_err(|_| {
            debug_assert!(false, "Value too large to fit in type");
            DecodeError::BitStreamError
        })
    }
}

#[cfg(test)]
mod test {

    use super::BitReader;
    use crate::slice_reader::SliceReader;

    #[test]
    fn bit_read_test() {
        //10011100 01000001 11100001
        let data = [0x9C, 0x41, 0xE1];
        let mut bit_reader = BitReader::new(SliceReader::new(&data));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 4); //100
        assert_eq!(bit_reader.read_bits::<u8>(2).unwrap(), 3); //11
        assert_eq!(bit_reader.read_bits::<u8>(6).unwrap(), 12); //001100
        assert_eq!(bit_reader.read_bits::<u16>(10).unwrap(), 40); //0000101000
        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 7); //111
    }

    #[test]
    fn bit_read_error_test() {
        //01101010
        let data = [0x6A];
        let mut bit_reader = BitReader::new(SliceReader::new(&data));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 2); //010
        assert_eq!(bit_reader.read_bits::<u8>(5).unwrap(), 13); //01101
        assert!(bit_reader.read_bits::<u8>(4).is_err()); //error
    }
}
