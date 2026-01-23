// VP8 Boolean Arithmetic Decoder - libwebp-rs style implementation
// Based on ~/work/webp-porting/libwebp-rs/src/utils/bit_reader.rs
//
// Key features:
// - Uses 56-bit buffer on 64-bit platforms (7 bytes at a time)
// - Stores range-1 internally (127-254 range)
// - Uses leading_zeros() for normalization (single CPU instruction)
// - VP8GetBitAlt algorithm

use crate::decoder::DecodingError;

/// BITS can be any multiple of 8 from 8 to 56 (inclusive).
#[cfg(target_pointer_width = "64")]
const BITS: i32 = 56;
#[cfg(not(target_pointer_width = "64"))]
const BITS: i32 = 24;

/// Number of bytes to read at once (BITS / 8)
#[cfg(target_pointer_width = "64")]
const BYTES_PER_LOAD: usize = 7;
#[cfg(not(target_pointer_width = "64"))]
const BYTES_PER_LOAD: usize = 3;

/// VP8 Boolean Arithmetic Decoder using libwebp's VP8GetBitAlt algorithm.
/// Borrows data slice directly - no pre-chunking needed.
pub struct VP8BitReader<'a> {
    /// Current accumulated value
    pub value: u64,
    /// Current range minus 1. In [127, 254] interval.
    pub range: u32,
    /// Number of valid bits left
    pub bits: i32,
    /// Remaining buffer to read from
    pub buf: &'a [u8],
    /// True if input is exhausted
    pub eof: bool,
}

#[allow(dead_code)]
impl<'a> VP8BitReader<'a> {
    /// Create a new bit reader from a byte slice
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut br = Self {
            value: 0,
            range: 255 - 1, // Store range-1
            bits: -8,       // to load the very first 8bits
            buf: data,
            eof: false,
        };
        br.load_new_bytes();
        br
    }

    #[cold]
    fn load_final_bytes(&mut self) {
        // Only read 8bits at a time
        if !self.buf.is_empty() {
            self.bits += 8;
            self.value = u64::from(self.buf[0]) | (self.value << 8);
            self.buf = &self.buf[1..];
        } else if !self.eof {
            self.value <<= 8;
            self.bits += 8;
            self.eof = true;
        } else {
            self.bits = 0; // To avoid undefined behaviour with shifts.
        }
    }

    #[inline(always)]
    pub fn load_new_bytes(&mut self) {
        // Read 'BITS' bits at a time if possible.
        if self.buf.len() >= BYTES_PER_LOAD {
            let bits: u64;

            #[cfg(target_pointer_width = "64")]
            {
                // We need to read 7 bytes (BITS=56).
                // Optimization: If we have at least 8 bytes, read u64 directly.
                if self.buf.len() >= 8 {
                    // Read 8 bytes as Big Endian to match the shift-accumulation logic
                    let in_bits_full =
                        u64::from_be_bytes(self.buf[..8].try_into().unwrap());
                    // We want the first 7 bytes - shift right by 8 to drop last byte
                    bits = in_bits_full >> 8;
                } else {
                    // Fallback for exactly 7 bytes available
                    let mut in_bits: u64 = 0;
                    for &byte in self.buf.iter().take(7) {
                        in_bits = (in_bits << 8) | u64::from(byte);
                    }
                    bits = in_bits;
                }
            }

            #[cfg(not(target_pointer_width = "64"))]
            {
                // BITS=24, 3 bytes.
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(self.buf[i]);
                }
                bits = in_bits;
            }

            self.value = bits | (self.value << BITS);
            self.bits += BITS;
            self.buf = &self.buf[BYTES_PER_LOAD..];
        } else {
            self.load_final_bytes();
        }
    }

    /// Read a bit with given probability (0-255).
    /// Uses VP8GetBitAlt algorithm from libwebp.
    #[inline(always)]
    pub fn get_bit(&mut self, prob: u8) -> i32 {
        let mut range = self.range;
        if self.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.bits;
        let split = (range.wrapping_mul(u32::from(prob))) >> 8;
        let value = (self.value >> pos) as u32;
        let bit = if value > split { 1 } else { 0 };

        if bit != 0 {
            range -= split;
            self.value = self.value.wrapping_sub((u64::from(split) + 1) << pos);
        } else {
            range = split + 1;
        }

        // Normalize using leading_zeros (compiles to single LZCNT instruction)
        let shift = 7 ^ (31 ^ range.leading_zeros() as i32);

        range <<= shift;
        self.bits -= shift;
        self.range = range.wrapping_sub(1);

        bit
    }

    /// Read a bit with probability 128 (50/50).
    #[inline(always)]
    pub fn get_flag(&mut self) -> bool {
        self.get_bit(128) != 0
    }

    /// VP8GetSigned - optimized sign bit reading
    #[inline(always)]
    pub fn get_signed(&mut self, v: i32) -> i32 {
        if self.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.bits;
        let split = self.range >> 1;
        let value = (self.value >> pos) as u32;
        // mask = -1 if value > split, else 0
        let mask = (split.wrapping_sub(value) as i32) >> 31;

        self.bits -= 1;
        self.range = self.range.wrapping_add(mask as u32);
        self.range |= 1;

        let term = (u64::from(split) + 1) & (mask as u32 as u64);
        self.value = self.value.wrapping_sub(term << pos);

        (v ^ mask) - mask
    }

    /// Read n bits as an unsigned value (MSB first)
    #[inline(always)]
    pub fn get_value(&mut self, n: u8) -> u32 {
        let mut v = 0u32;
        for i in (0..n).rev() {
            v |= (self.get_bit(0x80) as u32) << i;
        }
        v
    }

    /// Read n bits as a signed value
    #[inline]
    pub fn get_signed_value(&mut self, n: u8) -> i32 {
        let value = self.get_value(n) as i32;
        if self.get_bit(0x80) != 0 {
            -value
        } else {
            value
        }
    }

    /// Read optional signed value (flag + magnitude + sign)
    #[inline]
    pub fn get_optional_signed(&mut self, n: u8) -> i32 {
        if self.get_bit(0x80) == 0 {
            return 0;
        }
        let magnitude = self.get_value(n) as i32;
        if self.get_bit(0x80) != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Check if we've read past the end
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Check that reads were valid
    #[inline]
    #[allow(dead_code)]
    pub fn check(&self) -> Result<(), DecodingError> {
        if self.eof {
            Err(DecodingError::BitStreamError)
        } else {
            Ok(())
        }
    }

    /// Read from a probability tree starting at a specific node.
    /// The tree uses bit 0x80 to indicate leaf values.
    #[inline]
    #[allow(dead_code)]
    pub fn read_tree(&mut self, tree: &[crate::vp8::TreeNode], mut node: crate::vp8::TreeNode) -> i8 {
        loop {
            let prob = node.prob;
            let b = self.get_bit(prob) != 0;
            let i = if b { node.right } else { node.left };
            let Some(next_node) = tree.get(usize::from(i)) else {
                // Leaf node - extract value
                return (i & !0x80) as i8;
            };
            node = *next_node;
        }
    }
}

/// VP8 Boolean Arithmetic Decoder for header/mode parsing.
/// Owns its data buffer and uses VP8GetBitAlt algorithm.
/// This replaces ArithmeticDecoder with a faster implementation.
pub struct VP8HeaderBitReader {
    /// Owned data buffer
    data: Box<[u8]>,
    /// Current position in data
    pos: usize,
    /// Current accumulated value
    value: u64,
    /// Current range minus 1. In [127, 254] interval.
    range: u32,
    /// Number of valid bits left
    bits: i32,
    /// True if input is exhausted
    eof: bool,
}

impl VP8HeaderBitReader {
    pub fn new() -> Self {
        Self {
            data: Box::new([]),
            pos: 0,
            value: 0,
            range: 255 - 1,
            bits: -8,
            eof: false,
        }
    }

    /// Initialize the reader with data
    pub fn init(&mut self, data: Vec<u8>) -> Result<(), DecodingError> {
        if data.is_empty() {
            return Err(DecodingError::NotEnoughInitData);
        }
        self.data = data.into_boxed_slice();
        self.pos = 0;
        self.value = 0;
        self.range = 255 - 1;
        self.bits = -8;
        self.eof = false;
        self.load_new_bytes();
        Ok(())
    }

    #[cold]
    fn load_final_bytes(&mut self) {
        if self.pos < self.data.len() {
            self.bits += 8;
            self.value = u64::from(self.data[self.pos]) | (self.value << 8);
            self.pos += 1;
        } else if !self.eof {
            self.value <<= 8;
            self.bits += 8;
            self.eof = true;
        } else {
            self.bits = 0;
        }
    }

    #[inline(always)]
    fn load_new_bytes(&mut self) {
        let remaining = self.data.len() - self.pos;
        if remaining >= BYTES_PER_LOAD {
            let bits: u64;

            #[cfg(target_pointer_width = "64")]
            {
                if remaining >= 8 {
                    let in_bits_full =
                        u64::from_be_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
                    bits = in_bits_full >> 8;
                } else {
                    let mut in_bits: u64 = 0;
                    for &byte in self.data[self.pos..].iter().take(7) {
                        in_bits = (in_bits << 8) | u64::from(byte);
                    }
                    bits = in_bits;
                }
            }

            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(self.data[self.pos + i]);
                }
                bits = in_bits;
            }

            self.value = bits | (self.value << BITS);
            self.bits += BITS;
            self.pos += BYTES_PER_LOAD;
        } else {
            self.load_final_bytes();
        }
    }

    /// Read a bit with given probability (0-255).
    #[inline(always)]
    pub fn read_bool(&mut self, prob: u8) -> bool {
        let mut range = self.range;
        if self.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.bits;
        let split = (range.wrapping_mul(u32::from(prob))) >> 8;
        let value = (self.value >> pos) as u32;
        let bit = value > split;

        if bit {
            range -= split;
            self.value = self.value.wrapping_sub((u64::from(split) + 1) << pos);
        } else {
            range = split + 1;
        }

        let shift = 7 ^ (31 ^ range.leading_zeros() as i32);
        range <<= shift;
        self.bits -= shift;
        self.range = range.wrapping_sub(1);

        bit
    }

    /// Read a bit with probability 128 (50/50).
    #[inline(always)]
    pub fn read_flag(&mut self) -> bool {
        self.read_bool(128)
    }

    /// Read n bits as an unsigned value (MSB first)
    #[inline(always)]
    pub fn read_literal(&mut self, n: u8) -> u8 {
        let mut v = 0u8;
        for _ in 0..n {
            v = (v << 1) | (self.read_flag() as u8);
        }
        v
    }

    /// Read optional signed value (flag + magnitude + sign)
    #[inline]
    pub fn read_optional_signed_value(&mut self, n: u8) -> i32 {
        if !self.read_flag() {
            return 0;
        }
        let magnitude = self.read_literal(n) as i32;
        if self.read_flag() {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Read from a probability tree
    #[inline]
    pub fn read_with_tree<const N: usize>(&mut self, tree: &[crate::vp8::TreeNode; N]) -> i8 {
        let mut node = tree[0];
        loop {
            let prob = node.prob;
            let b = self.read_bool(prob);
            let i = if b { node.right } else { node.left };
            let Some(next_node) = tree.get(usize::from(i)) else {
                return crate::vp8::TreeNode::value_from_branch(i);
            };
            node = *next_node;
        }
    }

    /// Check if we've read past the end
    #[inline]
    #[allow(dead_code)]
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Check that reads were valid, returning an error if EOF was hit
    #[inline]
    pub fn check<T>(&self, value: T) -> Result<T, DecodingError> {
        if self.eof {
            Err(DecodingError::BitStreamError)
        } else {
            Ok(value)
        }
    }
}

impl Default for VP8HeaderBitReader {
    fn default() -> Self {
        Self::new()
    }
}

/// State for a single partition reader (can be saved/restored)
#[derive(Clone, Copy, Default)]
pub struct VP8BitReaderState {
    pub value: u64,
    pub range: u32,
    pub bits: i32,
    pub pos: usize, // Position in the buffer
    pub eof: bool,
}

/// Storage for multiple partitions with the new bit reader.
/// Owns the data and provides readers that borrow from it.
pub struct VP8Partitions {
    /// All partition data concatenated
    data: Box<[u8]>,
    /// (start offset, length) for each partition
    boundaries: [(usize, usize); 8],
    /// Current state for each partition
    states: [VP8BitReaderState; 8],
    /// Number of active partitions
    num_partitions: usize,
}

impl VP8Partitions {
    pub fn new() -> Self {
        Self {
            data: Box::new([]),
            boundaries: [(0, 0); 8],
            states: [VP8BitReaderState::default(); 8],
            num_partitions: 0,
        }
    }

    /// Initialize with partition data
    pub fn init(&mut self, data: Vec<u8>, boundaries: &[(usize, usize)]) {
        self.data = data.into_boxed_slice();
        self.num_partitions = boundaries.len().min(8);

        for (i, &(start, len)) in boundaries.iter().take(8).enumerate() {
            self.boundaries[i] = (start, len);
            // Initialize reader state for this partition
            let slice = &self.data[start..start + len];
            let reader = VP8BitReader::new(slice);
            self.states[i] = VP8BitReaderState {
                value: reader.value,
                range: reader.range,
                bits: reader.bits,
                pos: slice.len() - reader.buf.len(),
                eof: reader.eof,
            };
        }
    }

    /// Get a reader for partition p
    #[inline]
    pub fn reader(&mut self, p: usize) -> PartitionReader<'_> {
        let (start, len) = self.boundaries[p];
        let state = self.states[p];
        PartitionReader {
            data: &self.data[start..start + len],
            state,
            save_to: &mut self.states[p],
        }
    }

    /// Check if any partition hit EOF
    #[allow(dead_code)]
    pub fn check(&self) -> Result<(), DecodingError> {
        for i in 0..self.num_partitions {
            if self.states[i].eof {
                return Err(DecodingError::BitStreamError);
            }
        }
        Ok(())
    }
}

impl Default for VP8Partitions {
    fn default() -> Self {
        Self::new()
    }
}

/// A reader for a single partition that saves state on drop
pub struct PartitionReader<'a> {
    data: &'a [u8],
    state: VP8BitReaderState,
    save_to: &'a mut VP8BitReaderState,
}

impl<'a> PartitionReader<'a> {
    #[cold]
    fn load_final_bytes(&mut self) {
        if self.state.pos < self.data.len() {
            self.state.bits += 8;
            self.state.value = u64::from(self.data[self.state.pos]) | (self.state.value << 8);
            self.state.pos += 1;
        } else if !self.state.eof {
            self.state.value <<= 8;
            self.state.bits += 8;
            self.state.eof = true;
        } else {
            self.state.bits = 0;
        }
    }

    #[inline(always)]
    fn load_new_bytes(&mut self) {
        let remaining = self.data.len() - self.state.pos;
        if remaining >= BYTES_PER_LOAD {
            let bits: u64;

            #[cfg(target_pointer_width = "64")]
            {
                if remaining >= 8 {
                    let in_bits_full =
                        u64::from_be_bytes(self.data[self.state.pos..self.state.pos + 8].try_into().unwrap());
                    bits = in_bits_full >> 8;
                } else {
                    let mut in_bits: u64 = 0;
                    for &byte in self.data[self.state.pos..].iter().take(7) {
                        in_bits = (in_bits << 8) | u64::from(byte);
                    }
                    bits = in_bits;
                }
            }

            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(self.data[self.state.pos + i]);
                }
                bits = in_bits;
            }

            self.state.value = bits | (self.state.value << BITS);
            self.state.bits += BITS;
            self.state.pos += BYTES_PER_LOAD;
        } else {
            self.load_final_bytes();
        }
    }

    #[inline(always)]
    pub fn get_bit(&mut self, prob: u8) -> i32 {
        let mut range = self.state.range;
        if self.state.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.state.bits;
        let split = (range.wrapping_mul(u32::from(prob))) >> 8;
        let value = (self.state.value >> pos) as u32;
        let bit = if value > split { 1 } else { 0 };

        if bit != 0 {
            range -= split;
            self.state.value = self.state.value.wrapping_sub((u64::from(split) + 1) << pos);
        } else {
            range = split + 1;
        }

        let shift = 7 ^ (31 ^ range.leading_zeros() as i32);
        range <<= shift;
        self.state.bits -= shift;
        self.state.range = range.wrapping_sub(1);

        bit
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn get_signed(&mut self, v: i32) -> i32 {
        if self.state.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.state.bits;
        let split = self.state.range >> 1;
        let value = (self.state.value >> pos) as u32;
        let mask = (split.wrapping_sub(value) as i32) >> 31;

        self.state.bits -= 1;
        self.state.range = self.state.range.wrapping_add(mask as u32);
        self.state.range |= 1;

        let term = (u64::from(split) + 1) & (mask as u32 as u64);
        self.state.value = self.state.value.wrapping_sub(term << pos);

        (v ^ mask) - mask
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn read_tree(&mut self, tree: &[crate::vp8::TreeNode], mut node: crate::vp8::TreeNode) -> i8 {
        loop {
            let prob = node.prob;
            let b = self.get_bit(prob) != 0;
            let i = if b { node.right } else { node.left };
            let Some(next_node) = tree.get(usize::from(i)) else {
                return (i & !0x80) as i8;
            };
            node = *next_node;
        }
    }

    #[inline(always)]
    pub fn is_eof(&self) -> bool {
        self.state.eof
    }
}

impl Drop for PartitionReader<'_> {
    fn drop(&mut self) {
        *self.save_to = self.state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp8_arithmetic_decoder::ArithmeticDecoder;

    #[test]
    fn test_basic_reading() {
        let data = b"hello world and some more text";
        let mut br = VP8BitReader::new(data);

        // Read some bits
        for _ in 0..50 {
            let _ = br.get_flag();
        }
        assert!(!br.is_eof());
    }

    #[test]
    fn test_short_data() {
        let data = [0x55, 0xAA, 0x55];
        let mut br = VP8BitReader::new(&data);

        // Read until EOF
        for _ in 0..100 {
            let _ = br.get_flag();
        }
        assert!(br.is_eof());
    }

    #[test]
    fn test_get_value() {
        let data = b"test data for reading";
        let mut br = VP8BitReader::new(data);

        let v1 = br.get_value(4);
        let v2 = br.get_value(8);
        assert!(!br.is_eof());
        // Values depend on the algorithm, just verify no crash
        let _ = (v1, v2);
    }

    /// Compare VP8BitReader (libwebp-rs style) with ArithmeticDecoder
    /// They use different algorithms but should produce same results for valid data.
    #[test]
    fn test_compare_with_arithmetic_decoder() {
        // Use deterministic test data
        let data: Vec<u8> = (0..256).map(|i| (i * 17 + 31) as u8).collect();

        // Test with ArithmeticDecoder
        let mut old_decoder = ArithmeticDecoder::new();
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);
        old_decoder.init(chunks, data.len()).unwrap();

        // Test with VP8BitReader
        let mut new_reader = VP8BitReader::new(&data);

        // Compare flag reads (prob=128)
        let mut old_res = old_decoder.start_accumulated_result();
        let mut differences = 0;
        for i in 0..100 {
            let old_bit = old_decoder.read_flag().or_accumulate(&mut old_res);
            let new_bit = new_reader.get_flag();
            if old_bit != new_bit {
                differences += 1;
                if differences <= 5 {
                    eprintln!("Difference at flag {}: old={}, new={}", i, old_bit, new_bit);
                }
            }
        }

        eprintln!("Flag differences: {}/100", differences);
        assert_eq!(differences, 0, "Flag reads should match");
    }

    /// Test with various probabilities
    #[test]
    fn test_compare_various_probs() {
        let data: Vec<u8> = (0..512).map(|i| (i * 13 + 7) as u8).collect();

        let mut old_decoder = ArithmeticDecoder::new();
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);
        old_decoder.init(chunks, data.len()).unwrap();

        let mut new_reader = VP8BitReader::new(&data);

        let probs = [1, 10, 50, 100, 128, 150, 200, 240, 254];
        let mut old_res = old_decoder.start_accumulated_result();
        let mut total_diffs = 0;

        for &prob in &probs {
            let mut diffs = 0;
            for _ in 0..20 {
                let old_bit = old_decoder.read_bool(prob).or_accumulate(&mut old_res);
                let new_bit = new_reader.get_bit(prob) != 0;
                if old_bit != new_bit {
                    diffs += 1;
                }
            }
            eprintln!("Prob {}: {}/20 differences", prob, diffs);
            total_diffs += diffs;
        }

        assert_eq!(total_diffs, 0, "All probability reads should match");
    }
}

/// Micro-benchmark comparing ArithmeticDecoder vs VP8BitReader
#[cfg(all(test, feature = "_benchmarks"))]
mod benchmarks {
    use super::*;
    use crate::vp8_arithmetic_decoder::ArithmeticDecoder;
    use test::Bencher;

    fn make_test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i * 17 + 31) as u8).collect()
    }

    #[bench]
    fn bench_arithmetic_decoder_flags(b: &mut Bencher) {
        let data = make_test_data(4096);
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);

        b.iter(|| {
            let mut decoder = ArithmeticDecoder::new();
            decoder.init(chunks.clone(), data.len()).unwrap();
            let mut res = decoder.start_accumulated_result();
            let mut sum = 0u32;
            for _ in 0..1000 {
                sum += decoder.read_flag().or_accumulate(&mut res) as u32;
            }
            test::black_box(sum)
        });
    }

    #[bench]
    fn bench_vp8_bit_reader_flags(b: &mut Bencher) {
        let data = make_test_data(4096);

        b.iter(|| {
            let mut reader = VP8BitReader::new(&data);
            let mut sum = 0u32;
            for _ in 0..1000 {
                sum += reader.get_flag() as u32;
            }
            test::black_box(sum)
        });
    }

    #[bench]
    fn bench_arithmetic_decoder_mixed_probs(b: &mut Bencher) {
        let data = make_test_data(4096);
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);
        let probs: Vec<u8> = (0..100).map(|i| ((i * 7 + 50) % 255) as u8).collect();

        b.iter(|| {
            let mut decoder = ArithmeticDecoder::new();
            decoder.init(chunks.clone(), data.len()).unwrap();
            let mut res = decoder.start_accumulated_result();
            let mut sum = 0u32;
            for i in 0..1000 {
                sum += decoder.read_bool(probs[i % probs.len()]).or_accumulate(&mut res) as u32;
            }
            test::black_box(sum)
        });
    }

    #[bench]
    fn bench_vp8_bit_reader_mixed_probs(b: &mut Bencher) {
        let data = make_test_data(4096);
        let probs: Vec<u8> = (0..100).map(|i| ((i * 7 + 50) % 255) as u8).collect();

        b.iter(|| {
            let mut reader = VP8BitReader::new(&data);
            let mut sum = 0u32;
            for i in 0..1000 {
                sum += (reader.get_bit(probs[i % probs.len()]) != 0) as u32;
            }
            test::black_box(sum)
        });
    }
}
