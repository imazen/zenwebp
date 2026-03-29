// VP8 boolean arithmetic encoder, matching libwebp's VP8BitWriter.
//
// Uses lookup-table normalization (kNorm/kNewRange) and run-length carry
// handling instead of per-bit loops and backwards carry propagation.
// This eliminates the while-loop in write_bool and the O(n) backwards walk
// in add_one_to_output, matching libwebp's VP8PutBit performance.

use alloc::vec;
use alloc::vec::Vec;

/// Normalization shift count: kNorm[range] = 8 - floor(log2(range+1))
/// For range in 0..127, gives the number of bits to shift out.
/// Matches libwebp's kNorm table in bit_writer_utils.c.
#[rustfmt::skip]
const K_NORM: [u8; 128] = [
     7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0,
];

/// New range after normalization: kNewRange[range] = ((range + 1) << kNorm[range]) - 1
/// Matches libwebp's kNewRange table in bit_writer_utils.c.
#[rustfmt::skip]
const K_NEW_RANGE: [u8; 128] = [
  127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
  127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
  247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
  183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
  243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
  151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
  181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
  211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
  241, 243, 245, 247, 249, 251, 253, 127,
];

pub(crate) struct ArithmeticEncoder {
    /// Output byte buffer.
    buf: Vec<u8>,
    /// Range-1 (libwebp convention: range is stored as range-1, always 0..254).
    range: i32,
    /// Accumulated value bits.
    value: i32,
    /// Number of outstanding 0xFF bytes waiting for carry resolution.
    run: i32,
    /// Number of pending bits (starts negative at -8, flush when > 0).
    nb_bits: i32,
}

impl Default for ArithmeticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithmeticEncoder {
    pub fn new() -> Self {
        Self {
            buf: vec![],
            range: 254, // 255 - 1
            value: 0,
            run: 0,
            nb_bits: -8,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            range: 254,
            value: 0,
            run: 0,
            nb_bits: -8,
        }
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.buf.clear();
        self.range = 254;
        self.value = 0;
        self.run = 0;
        self.nb_bits = -8;
    }

    /// Flush accumulated bits to the output buffer.
    /// Handles carry propagation via run-length encoding of 0xFF bytes.
    /// Matches libwebp's Flush() in bit_writer_utils.c.
    fn flush(&mut self) {
        let s = 8 + self.nb_bits;
        let bits = self.value >> s;
        self.value -= bits << s;
        self.nb_bits -= 8;
        if (bits & 0xff) != 0xff {
            if (bits & 0x100) != 0 {
                // Carry: increment the last written byte
                if let Some(last) = self.buf.last_mut() {
                    *last += 1;
                }
            }
            if self.run > 0 {
                // Resolve pending 0xFF bytes: they become 0x00 on carry, stay 0xFF otherwise
                let fill = if (bits & 0x100) != 0 { 0x00u8 } else { 0xFFu8 };
                for _ in 0..self.run {
                    self.buf.push(fill);
                }
                self.run = 0;
            }
            self.buf.push((bits & 0xff) as u8);
        } else {
            // Byte is 0xFF: defer writing, increment run counter for lazy carry resolution
            self.run += 1;
        }
    }

    pub(crate) fn write_flag(&mut self, flag_bool: bool) {
        self.write_bool(flag_bool, 128);
    }

    /// Encode a single boolean with the given probability (0-255).
    /// Uses lookup-table normalization matching libwebp's VP8PutBit.
    #[inline]
    pub(crate) fn write_bool(&mut self, bit: bool, probability: u8) {
        let split = (self.range * i32::from(probability)) >> 8;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            // Lookup-table normalization: one table access replaces a while-loop.
            // K_NORM gives the shift count, K_NEW_RANGE gives the final range.
            let shift = K_NORM[self.range as usize];
            self.range = K_NEW_RANGE[self.range as usize] as i32;
            self.value <<= shift;
            self.nb_bits += shift as i32;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
    }

    pub(crate) fn write_literal(&mut self, num_bits: u8, value: u8) {
        for bit in (0..num_bits).rev() {
            let bool_encode = (1 << bit) & value > 0;
            self.write_bool(bool_encode, 128);
        }
    }

    pub(crate) fn write_optional_signed_value(&mut self, num_bits: u8, value: Option<i8>) {
        self.write_flag(value.is_some());
        if let Some(value) = value {
            let abs_value = value.unsigned_abs();
            self.write_literal(num_bits, abs_value);
            // VP8 spec: sign bit 1 = negative, 0 = positive
            self.write_flag(value < 0);
        }
    }

    pub(crate) fn write_with_tree(&mut self, tree: &[i8], probabilities: &[u8], value: i8) {
        self.write_with_tree_start_index(tree, probabilities, value, 0);
    }

    // Could optimise to be faster by processing the trees as const, similar to the decoder
    // especially encoding of trees
    pub(crate) fn write_with_tree_start_index(
        &mut self,
        tree: &[i8],
        probabilities: &[u8],
        value: i8,
        start_index: usize,
    ) {
        assert_eq!(tree.len(), probabilities.len() * 2);
        // the values are encoded as negative or zero in the tree, positive values are indexes
        let mut current_index = tree.iter().position(|x| *x == -value).unwrap();

        // Use stack-allocated array instead of Vec - max tree depth is ~11
        let mut to_encode: [(bool, u8); 16] = [(false, 0); 16];
        let mut count = 0;

        loop {
            if current_index == start_index {
                // just write the 0 using the prob
                to_encode[count] = (false, probabilities[current_index / 2]);
                count += 1;
                break;
            }
            if current_index == start_index + 1 {
                to_encode[count] = (true, probabilities[current_index / 2]);
                count += 1;
                break;
            }

            // even => encode false
            let encode_val = if current_index % 2 == 0 {
                false
            } else {
                current_index -= 1;
                true
            };

            to_encode[count] = (encode_val, probabilities[current_index / 2]);
            count += 1;

            let previous_index = tree
                .iter()
                .position(|x| *x == (current_index as i8))
                .unwrap_or_else(|| {
                    panic!("Failed to encode {value} for tree {tree:?} and probs {probabilities:?}")
                });
            current_index = previous_index;
        }

        // write bools backwards
        for i in (0..count).rev() {
            let (encode_bool, prob) = to_encode[i];
            self.write_bool(encode_bool, prob);
        }
    }

    /// Flushes remaining bits and returns the output buffer.
    /// Matches libwebp's VP8BitWriterFinish.
    pub(crate) fn flush_and_get_buffer(mut self) -> Vec<u8> {
        // Flush remaining bits by writing enough zero-padding.
        // VP8BitWriterFinish calls VP8PutBits(bw, 0, 9 - bw->nb_bits)
        let pad_bits = 9 - self.nb_bits;
        for _ in 0..pad_bits {
            self.write_bool(false, 128);
        }
        self.nb_bits = 0;
        self.flush();
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use crate::common::types::*;
    use crate::decoder::arithmetic::ArithmeticDecoder;

    use super::*;

    fn convert_buffer_for_decoding(buffer: &[u8]) -> Vec<[u8; 4]> {
        let mut new_buf = vec![[0u8; 4]; buffer.len().div_ceil(4)];
        new_buf.as_mut_slice().as_flattened_mut()[..buffer.len()].copy_from_slice(buffer);
        new_buf
    }

    #[test]
    fn test_arithmetic_encoder_short() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_flag(false);
        encoder.write_bool(true, 10);
        encoder.write_bool(false, 250);
        encoder.write_literal(1, 1);
        encoder.write_literal(3, 5);
        encoder.write_literal(8, 64);
        encoder.write_literal(8, 185);
        let bytes = encoder.flush_and_get_buffer();
        // Meaningful content is the first 4 bytes; trailing padding may vary
        assert!(bytes.len() >= 4, "expected at least 4 bytes, got {}", bytes.len());
        assert_eq!(&[104, 101, 107, 128], &bytes[..4]);
    }

    #[test]
    fn test_arithmetic_encoder_hello() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_flag(false);
        encoder.write_bool(true, 10);
        encoder.write_bool(false, 250);
        encoder.write_literal(1, 1);
        encoder.write_literal(3, 5);
        encoder.write_literal(8, 64);
        encoder.write_literal(8, 185);
        encoder.write_literal(8, 31);
        encoder.write_literal(8, 134);
        encoder.write_optional_signed_value(2, None);
        encoder.write_optional_signed_value(2, Some(1));
        let data = b"hello";
        let bytes = encoder.flush_and_get_buffer();
        assert_eq!(data, &bytes[..data.len()]);
    }

    #[test]
    fn test_encoder_with_decoder() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_bool(true, 40);
        encoder.write_bool(true, 110);
        encoder.write_bool(false, 70);
        encoder.write_bool(false, 10);
        encoder.write_bool(true, 5);
        let write_buffer = encoder.flush_and_get_buffer();

        let decode_buffer = convert_buffer_for_decoding(&write_buffer);

        let mut decoder = ArithmeticDecoder::new();
        decoder.init(decode_buffer, write_buffer.len()).unwrap();

        let mut res = decoder.start_accumulated_result();
        assert!(decoder.read_bool(40).or_accumulate(&mut res));
        assert!(decoder.read_bool(110).or_accumulate(&mut res));
        assert!(!decoder.read_bool(70).or_accumulate(&mut res));
        assert!(!decoder.read_bool(10).or_accumulate(&mut res));
        assert!(decoder.read_bool(5).or_accumulate(&mut res));
        decoder.check(res, ()).unwrap();
    }

    #[test]
    fn test_encoder_tree() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_with_tree(&KEYFRAME_YMODE_TREE, &KEYFRAME_YMODE_PROBS, TM_PRED);
        let write_buffer = encoder.flush_and_get_buffer();
        // Meaningful content is the first 3 bytes; trailing padding may vary
        assert!(write_buffer.len() >= 3, "expected at least 3 bytes, got {}", write_buffer.len());
        assert_eq!(&[233, 64, 0], &write_buffer[..3]);
    }
}
