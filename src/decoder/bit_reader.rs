// VP8 Boolean Arithmetic Decoder
//
// Two reader types:
// - VP8HeaderBitReader: Owns its data, used for header/mode parsing.
// - ActivePartitionReader: Borrows data + state, used for coefficient decoding.
//   Copies state fields into local storage for register-friendly access,
//   writes back on drop.
//
// Both use the VP8GetBitAlt algorithm:
// - 56-bit buffer on 64-bit platforms (7 bytes at a time)
// - Stores range-1 internally (127-254 range)
// - leading_zeros() for normalization (single CPU instruction)

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::api::DecodeError;
use super::internal_error::InternalDecodeError;

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

/// VP8 Boolean Arithmetic Decoder for header/mode parsing.
/// Owns its data buffer and uses VP8GetBitAlt algorithm.
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
    pub fn init(&mut self, data: Vec<u8>) -> Result<(), DecodeError> {
        if data.is_empty() {
            return Err(DecodeError::NotEnoughInitData);
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
    pub fn read_with_tree<const N: usize>(&mut self, tree: &[super::vp8::TreeNode; N]) -> i8 {
        let mut node = tree[0];
        loop {
            let prob = node.prob;
            let b = self.read_bool(prob);
            let i = if b { node.right } else { node.left };
            let Some(next_node) = tree.get(usize::from(i)) else {
                return super::vp8::TreeNode::value_from_branch(i);
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
    pub fn check<T>(&self, value: T) -> Result<T, InternalDecodeError> {
        if self.eof {
            Err(InternalDecodeError::BitStreamError)
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

/// State for a single partition reader (can be saved/restored).
#[derive(Clone, Copy, Default)]
pub struct VP8BitReaderState {
    pub value: u64,
    pub range: u32,
    pub bits: i32,
    pub pos: usize,
    pub eof: bool,
}

impl VP8BitReaderState {
    /// Initialize state from a data slice (reads the first batch of bytes).
    fn init_from_slice(data: &[u8]) -> Self {
        let mut state = Self {
            value: 0,
            range: 255 - 1,
            bits: -8,
            pos: 0,
            eof: false,
        };
        // Bootstrap: load initial bytes using the same logic as ActivePartitionReader.
        let tail = &data[state.pos..];
        if tail.len() >= 8 {
            #[cfg(target_pointer_width = "64")]
            {
                let chunk: [u8; 8] = tail[..8].try_into().unwrap();
                let bits = u64::from_be_bytes(chunk) >> 8;
                state.value = bits | (state.value << BITS);
                state.bits += BITS;
                state.pos += BYTES_PER_LOAD;
            }
            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(tail[i]);
                }
                state.value = in_bits | (state.value << BITS);
                state.bits += BITS;
                state.pos += BYTES_PER_LOAD;
            }
        } else if tail.len() >= BYTES_PER_LOAD {
            #[cfg(target_pointer_width = "64")]
            {
                let mut in_bits: u64 = 0;
                for &byte in tail.iter().take(7) {
                    in_bits = (in_bits << 8) | u64::from(byte);
                }
                state.value = in_bits | (state.value << BITS);
            }
            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(tail[i]);
                }
                state.value = in_bits | (state.value << BITS);
            }
            state.bits += BITS;
            state.pos += BYTES_PER_LOAD;
        } else if !data.is_empty() {
            // load_final_bytes path
            state.bits += 8;
            state.value = u64::from(data[0]) | (state.value << 8);
            state.pos = 1;
        } else {
            state.value <<= 8;
            state.bits += 8;
            state.eof = true;
        }
        state
    }
}

/// Storage for multiple partitions.
/// Owns the data and provides [`ActivePartitionReader`]s that borrow from it.
pub struct VP8Partitions {
    /// All partition data concatenated.
    pub(crate) data: Box<[u8]>,
    /// (start offset, length) for each partition.
    pub(crate) boundaries: [(usize, usize); 8],
    /// Current state for each partition.
    pub(crate) states: [VP8BitReaderState; 8],
    /// Number of active partitions.
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

    /// Initialize with partition data.
    pub fn init(&mut self, data: Vec<u8>, boundaries: &[(usize, usize)]) {
        self.data = data.into_boxed_slice();
        self.num_partitions = boundaries.len().min(8);

        for (i, &(start, len)) in boundaries.iter().take(8).enumerate() {
            self.boundaries[i] = (start, len);
            self.states[i] = VP8BitReaderState::init_from_slice(&self.data[start..start + len]);
        }
    }

    /// Get an active reader for partition p.
    /// Copies state fields into local storage for register-friendly access,
    /// writes back on drop.
    #[inline]
    #[allow(dead_code)]
    pub fn active_reader(&mut self, p: usize) -> ActivePartitionReader<'_> {
        let (start, len) = self.boundaries[p];
        ActivePartitionReader::new(&self.data[start..start + len], &mut self.states[p])
    }

    /// Check if any partition hit EOF.
    #[allow(dead_code)]
    pub fn check(&self) -> Result<(), DecodeError> {
        for i in 0..self.num_partitions {
            if self.states[i].eof {
                return Err(DecodeError::BitStreamError);
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

/// An active reader that borrows partition state directly.
/// Copies state fields into local storage for faster access (no pointer
/// indirection per `get_bit` call), and writes back on drop.
///
/// Uses sub-slice bounds check elimination in `load_new_bytes`.
pub struct ActivePartitionReader<'a> {
    data: &'a [u8],
    /// Local copies of hot state fields (written back on drop).
    value: u64,
    range: u32,
    bits: i32,
    pos: usize,
    eof: bool,
    /// Where to write back state on drop.
    save_to: &'a mut VP8BitReaderState,
}

impl<'a> ActivePartitionReader<'a> {
    /// Create a new active reader from explicit references.
    /// Copies state into local fields for register-friendly access.
    #[inline]
    pub fn new(data: &'a [u8], state: &'a mut VP8BitReaderState) -> Self {
        Self {
            data,
            value: state.value,
            range: state.range,
            bits: state.bits,
            pos: state.pos,
            eof: state.eof,
            save_to: state,
        }
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

    /// Load 7 bytes (56 bits) into the value buffer.
    ///
    /// Takes a sub-slice `tail = &data[pos..]` first (single bounds check on pos),
    /// then checks `tail.len() >= 8`. LLVM can prove the 8-byte read is in bounds.
    #[inline(always)]
    fn load_new_bytes(&mut self) {
        let tail = &self.data[self.pos..];
        if tail.len() >= 8 {
            #[cfg(target_pointer_width = "64")]
            {
                let chunk: [u8; 8] = tail[..8].try_into().unwrap();
                let bits = u64::from_be_bytes(chunk) >> 8;
                self.value = bits | (self.value << BITS);
                self.bits += BITS;
                self.pos += BYTES_PER_LOAD;
            }

            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(tail[i]);
                }
                self.value = in_bits | (self.value << BITS);
                self.bits += BITS;
                self.pos += BYTES_PER_LOAD;
            }
        } else if tail.len() >= BYTES_PER_LOAD {
            // Exactly 7 bytes remaining (rare, only happens once at partition end).
            #[cfg(target_pointer_width = "64")]
            {
                let mut in_bits: u64 = 0;
                for &byte in tail.iter().take(7) {
                    in_bits = (in_bits << 8) | u64::from(byte);
                }
                self.value = in_bits | (self.value << BITS);
            }

            #[cfg(not(target_pointer_width = "64"))]
            {
                let mut in_bits: u64 = 0;
                for i in 0..3 {
                    in_bits = (in_bits << 8) | u64::from(tail[i]);
                }
                self.value = in_bits | (self.value << BITS);
            }

            self.bits += BITS;
            self.pos += BYTES_PER_LOAD;
        } else {
            self.load_final_bytes();
        }
    }

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

        let shift = 7 ^ (31 ^ range.leading_zeros() as i32);
        range <<= shift;
        self.bits -= shift;
        self.range = range.wrapping_sub(1);

        bit
    }

    /// VP8GetSigned - optimized branchless sign bit reading with prob=128.
    /// Returns +v or -v based on the next bit.
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

    #[inline(always)]
    pub fn is_eof(&self) -> bool {
        self.eof
    }
}

impl Drop for ActivePartitionReader<'_> {
    #[inline]
    fn drop(&mut self) {
        self.save_to.value = self.value;
        self.save_to.range = self.range;
        self.save_to.bits = self.bits;
        self.save_to.pos = self.pos;
        self.save_to.eof = self.eof;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::arithmetic::ArithmeticDecoder;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Helper: create an ActivePartitionReader from a data slice for testing.
    /// Returns (state, data_box) so the reader can borrow them.
    fn make_test_reader(data: &[u8]) -> (VP8BitReaderState, Vec<u8>) {
        let state = VP8BitReaderState::init_from_slice(data);
        (state, data.to_vec())
    }

    #[test]
    fn test_basic_reading() {
        let data = b"hello world and some more text";
        let (mut state, owned) = make_test_reader(data);
        let mut reader = ActivePartitionReader::new(&owned, &mut state);

        // Read some bits
        for _ in 0..50 {
            let _ = reader.get_bit(128) != 0;
        }
        assert!(!reader.is_eof());
    }

    #[test]
    fn test_short_data() {
        let data = [0x55, 0xAA, 0x55];
        let (mut state, owned) = make_test_reader(&data);
        let mut reader = ActivePartitionReader::new(&owned, &mut state);

        // Read until EOF
        for _ in 0..100 {
            let _ = reader.get_bit(128) != 0;
        }
        assert!(reader.is_eof());
    }

    #[test]
    fn test_get_value() {
        let data = b"test data for reading";
        let (mut state, owned) = make_test_reader(data);
        let mut reader = ActivePartitionReader::new(&owned, &mut state);

        let mut v1 = 0u32;
        for i in (0..4).rev() {
            v1 |= (reader.get_bit(0x80) as u32) << i;
        }
        let mut v2 = 0u32;
        for i in (0..8).rev() {
            v2 |= (reader.get_bit(0x80) as u32) << i;
        }
        assert!(!reader.is_eof());
        let _ = (v1, v2);
    }

    /// Compare ActivePartitionReader with ArithmeticDecoder.
    /// They use different algorithms but should produce same results for valid data.
    #[test]
    fn test_compare_with_arithmetic_decoder() {
        let data: Vec<u8> = (0..256).map(|i| (i * 17 + 31) as u8).collect();

        let mut old_decoder = ArithmeticDecoder::new();
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);
        old_decoder.init(chunks, data.len()).unwrap();

        let (mut state, owned) = make_test_reader(&data);
        let mut reader = ActivePartitionReader::new(&owned, &mut state);

        let mut old_res = old_decoder.start_accumulated_result();
        let mut differences = 0;
        for _ in 0..100 {
            let old_bit = old_decoder.read_flag().or_accumulate(&mut old_res);
            let new_bit = reader.get_bit(128) != 0;
            if old_bit != new_bit {
                differences += 1;
            }
        }

        assert_eq!(
            differences, 0,
            "Flag reads should match (got {differences}/100 differences)"
        );
    }

    /// Test with various probabilities
    #[test]
    fn test_compare_various_probs() {
        let data: Vec<u8> = (0..512).map(|i| (i * 13 + 7) as u8).collect();

        let mut old_decoder = ArithmeticDecoder::new();
        let mut chunks = vec![[0u8; 4]; data.len().div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..data.len()].copy_from_slice(&data);
        old_decoder.init(chunks, data.len()).unwrap();

        let (mut state, owned) = make_test_reader(&data);
        let mut reader = ActivePartitionReader::new(&owned, &mut state);

        let probs = [1, 10, 50, 100, 128, 150, 200, 240, 254];
        let mut old_res = old_decoder.start_accumulated_result();
        let mut total_diffs = 0;

        for &prob in &probs {
            let mut diffs = 0;
            for _ in 0..20 {
                let old_bit = old_decoder.read_bool(prob).or_accumulate(&mut old_res);
                let new_bit = reader.get_bit(prob) != 0;
                if old_bit != new_bit {
                    diffs += 1;
                }
            }
            total_diffs += diffs;
        }

        assert_eq!(total_diffs, 0, "All probability reads should match");
    }
}

/// Micro-benchmark comparing ArithmeticDecoder vs ActivePartitionReader
#[cfg(all(test, feature = "_benchmarks"))]
mod benchmarks {
    use super::*;
    use crate::decoder::arithmetic::ArithmeticDecoder;
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
    fn bench_active_reader_flags(b: &mut Bencher) {
        let data = make_test_data(4096);

        b.iter(|| {
            let (mut state, owned) = (VP8BitReaderState::init_from_slice(&data), data.clone());
            let mut reader = ActivePartitionReader::new(&owned, &mut state);
            let mut sum = 0u32;
            for _ in 0..1000 {
                sum += (reader.get_bit(128) != 0) as u32;
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
                sum += decoder
                    .read_bool(probs[i % probs.len()])
                    .or_accumulate(&mut res) as u32;
            }
            test::black_box(sum)
        });
    }

    #[bench]
    fn bench_active_reader_mixed_probs(b: &mut Bencher) {
        let data = make_test_data(4096);
        let probs: Vec<u8> = (0..100).map(|i| ((i * 7 + 50) % 255) as u8).collect();

        b.iter(|| {
            let (mut state, owned) = (VP8BitReaderState::init_from_slice(&data), data.clone());
            let mut reader = ActivePartitionReader::new(&owned, &mut state);
            let mut sum = 0u32;
            for i in 0..1000 {
                sum += (reader.get_bit(probs[i % probs.len()]) != 0) as u32;
            }
            test::black_box(sum)
        });
    }
}
