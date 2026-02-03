//! Huffman tree construction and encoding for VP8L.
//!
//! Implements canonical Huffman codes as required by the VP8L bitstream.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use super::bitwriter::BitWriter;

/// Maximum Huffman code length (15 bits).
const MAX_CODE_LENGTH: u8 = 15;

/// Code length alphabet for encoding Huffman trees.
const CODE_LENGTH_CODES: usize = 19;

/// Order in which code length codes are written.
const CODE_LENGTH_CODE_ORDER: [usize; CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Huffman code (code word and length).
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanCode {
    pub code: u16,
    pub length: u8,
}

/// Build Huffman code lengths from symbol frequencies.
/// Uses package-merge algorithm for length-limited codes.
pub fn build_huffman_lengths(freq: &[u32], max_len: u8) -> Vec<u8> {
    let n = freq.len();
    let mut lengths = vec![0u8; n];

    // Count non-zero symbols
    let non_zero: Vec<(usize, u32)> = freq
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(i, &f)| (i, f))
        .collect();

    if non_zero.is_empty() {
        return lengths;
    }

    if non_zero.len() == 1 {
        // Single symbol gets length 1 for tree writing purposes.
        // When encoding image data, we'll check if tree is trivial
        // and skip writing bits.
        lengths[non_zero[0].0] = 1;
        return lengths;
    }

    if non_zero.len() == 2 {
        // Two symbols each get length 1
        lengths[non_zero[0].0] = 1;
        lengths[non_zero[1].0] = 1;
        return lengths;
    }

    // Build Huffman tree using heap
    #[derive(Eq, PartialEq)]
    struct Node {
        weight: u64,
        symbol: Option<usize>,
        children: Option<(Box<Node>, Box<Node>)>,
    }

    impl Ord for Node {
        fn cmp(&self, other: &Self) -> Ordering {
            // Min-heap: reverse comparison
            other.weight.cmp(&self.weight)
        }
    }

    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let _n_nonzero = non_zero.len();
    let mut heap = BinaryHeap::new();
    for (sym, freq) in non_zero {
        heap.push(Node {
            weight: freq as u64,
            symbol: Some(sym),
            children: None,
        });
    }

    // Build tree
    while heap.len() > 1 {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();
        heap.push(Node {
            weight: a.weight + b.weight,
            symbol: None,
            children: Some((Box::new(a), Box::new(b))),
        });
    }

    // Extract lengths from tree
    // Depth starts at 0 for the root. Leaves get depth = number of edges from root.
    // We DON'T clamp here - that will be done in ensure_valid_code_lengths so it can
    // properly fix the Kraft inequality.
    fn collect_lengths(node: &Node, depth: u8, lengths: &mut [u8], actual_max: &mut u8) {
        if let Some(sym) = node.symbol {
            // Leaf node: code length = depth (but at least 1 for valid codes)
            *actual_max = (*actual_max).max(depth);
            lengths[sym] = depth.max(1);
        } else if let Some((ref left, ref right)) = node.children {
            collect_lengths(left, depth + 1, lengths, actual_max);
            collect_lengths(right, depth + 1, lengths, actual_max);
        }
    }

    if let Some(root) = heap.pop() {
        let mut _max_depth = 0u8;
        collect_lengths(&root, 0, &mut lengths, &mut _max_depth);
    }

    // Limit code lengths and ensure valid Kraft sum
    ensure_valid_code_lengths(&mut lengths, max_len);

    lengths
}

/// Ensure code lengths form a valid Huffman tree with max length constraint.
/// Uses iterative adjustment to satisfy the Kraft inequality.
fn ensure_valid_code_lengths(lengths: &mut [u8], max_len: u8) {
    // Check if any length exceeds max_len
    let mut need_adjustment = false;
    for &len in lengths.iter() {
        if len > max_len {
            need_adjustment = true;
            break;
        }
    }

    if !need_adjustment {
        // No length limiting needed, tree should already be valid
        return;
    }

    // Cap lengths at max_len
    for len in lengths.iter_mut() {
        if *len > max_len {
            *len = max_len;
        }
    }

    let compute_kraft = |lengths: &[u8]| -> u64 {
        lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (max_len - l))
            .sum()
    };

    let target = 1u64 << max_len;
    let mut kraft = compute_kraft(lengths);

    // Adjustment loop
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10000;

    while kraft != target && iterations < MAX_ITERATIONS {
        iterations += 1;

        if kraft > target {
            // Too much codespace used: make some codes LONGER to free up space
            // A code at length L contributes 2^(max_len-L) to kraft.
            // Increasing to L+1 contributes 2^(max_len-L-1).
            // Net change: -2^(max_len-L) + 2^(max_len-L-1) = -2^(max_len-L-1)
            // Prioritize shortest codes (they free up the most space when lengthened)
            let mut best_idx = None;
            let mut best_len = max_len;
            for (i, &len) in lengths.iter().enumerate() {
                if len > 0 && len < max_len && len < best_len {
                    best_idx = Some(i);
                    best_len = len;
                }
            }

            if let Some(idx) = best_idx {
                let old_contrib = 1u64 << (max_len - best_len);
                let new_contrib = 1u64 << (max_len - best_len - 1);
                lengths[idx] += 1;
                kraft = kraft - old_contrib + new_contrib;
            } else {
                // All codes at max_len, can't fix
                break;
            }
        } else {
            // Not enough codespace used: make some codes SHORTER to use more space
            // A code at length L contributes 2^(max_len-L) to kraft.
            // Decreasing to L-1 contributes 2^(max_len-L+1).
            // Net change: -2^(max_len-L) + 2^(max_len-L+1) = +2^(max_len-L)
            // Prioritize longest codes (smallest impact per change)
            let mut best_idx = None;
            let mut best_len = 0u8;
            for (i, &len) in lengths.iter().enumerate() {
                if len > 1 && len > best_len {
                    best_idx = Some(i);
                    best_len = len;
                }
            }

            if let Some(idx) = best_idx {
                let old_contrib = 1u64 << (max_len - best_len);
                let new_contrib = 1u64 << (max_len - best_len + 1);
                lengths[idx] -= 1;
                kraft = kraft - old_contrib + new_contrib;
            } else {
                // All codes at length 1, can't fix
                break;
            }
        }
    }
}

/// Build canonical Huffman codes from code lengths.
/// Codes are bit-reversed for LSB-first VP8L bitstream (matching libwebp).
pub fn build_huffman_codes(lengths: &[u8]) -> Vec<HuffmanCode> {
    let n = lengths.len();
    let mut codes = vec![HuffmanCode::default(); n];

    if n == 0 {
        return codes;
    }

    // Count lengths
    let mut length_counts = [0u32; MAX_CODE_LENGTH as usize + 1];
    for &len in lengths {
        if len > 0 {
            length_counts[len as usize] += 1;
        }
    }

    // Compute starting codes for each length
    let mut next_code = [0u32; MAX_CODE_LENGTH as usize + 1];
    let mut code = 0u32;
    for bits in 1..=MAX_CODE_LENGTH as usize {
        code = (code + length_counts[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign reversed codes to symbols (matching libwebp's ConvertBitDepthsToSymbols)
    for (symbol, &len) in lengths.iter().enumerate() {
        if len > 0 {
            codes[symbol] = HuffmanCode {
                code: reverse_bits_16(len as u32, next_code[len as usize]) as u16,
                length: len,
            };
            next_code[len as usize] += 1;
        }
    }

    codes
}

/// Reverse bits for a code of given length (max 15 bits).
fn reverse_bits_16(num_bits: u32, bits: u32) -> u32 {
    const REVERSED_BITS: [u8; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];

    let mut retval = 0u32;
    let mut b = bits;
    let mut i = 0;
    while i < num_bits {
        i += 4;
        retval |= (REVERSED_BITS[(b & 0xf) as usize] as u32) << (16 - i);
        b >>= 4;
    }
    retval >> (16 - num_bits)
}

/// Write a Huffman tree to the bitstream.
pub fn write_huffman_tree(w: &mut BitWriter, lengths: &[u8]) {
    // Check for trivial cases
    let non_zero: Vec<(usize, u8)> = lengths
        .iter()
        .enumerate()
        .filter(|(_, &len)| len > 0)
        .map(|(i, &len)| (i, len))
        .collect();

    if non_zero.len() <= 2 && non_zero.iter().all(|(sym, _)| *sym < 256) {
        // Simple tree encoding
        w.write_bit(true); // simple tree flag

        if non_zero.len() == 1 || non_zero.is_empty() {
            // Single symbol (or empty - use symbol 0)
            w.write_bit(false); // 1 symbol (count - 1)
            let symbol = if non_zero.is_empty() {
                0
            } else {
                non_zero[0].0
            };
            // If symbol <= 1, use 1 bit; otherwise use 8 bits
            let is_8bit = symbol > 1;
            w.write_bit(is_8bit);
            if is_8bit {
                w.write_bits(symbol as u64, 8);
            } else {
                w.write_bits(symbol as u64, 1);
            }
        } else {
            // Two symbols
            w.write_bit(true); // 2 symbols (count - 1 = 1)
            let sym0 = non_zero[0].0;
            let sym1 = non_zero[1].0;
            // First symbol: if <= 1 use 1 bit, else 8 bits
            let first_is_8bit = sym0 > 1;
            w.write_bit(first_is_8bit);
            if first_is_8bit {
                w.write_bits(sym0 as u64, 8);
            } else {
                w.write_bits(sym0 as u64, 1);
            }
            // Second symbol always 8 bits
            w.write_bits(sym1 as u64, 8);
        }
    } else {
        // Normal tree encoding
        write_huffman_tree_complex(w, lengths);
    }
}

/// Token for RLE-encoded Huffman tree (matches libwebp's HuffmanTreeToken).
#[derive(Clone, Copy, Default)]
struct HuffmanTreeToken {
    code: u8, // 0-15 for code lengths, 16 = repeat, 17 = repeat 0s (3-10), 18 = repeat 0s (11-138)
    extra_bits: u8, // extra bits for codes 16, 17, 18
}

/// Write a complex Huffman tree using RLE encoding (matches libwebp exactly).
fn write_huffman_tree_complex(w: &mut BitWriter, lengths: &[u8]) {
    let n = lengths.len();

    w.write_bit(false); // not simple

    // Create RLE tokens from code lengths (matches VP8LCreateCompressedHuffmanTree)
    let mut tokens = Vec::with_capacity(n);
    let mut prev_value = 8u8; // Initial value for RLE (matches libwebp)
    let mut i = 0;

    while i < n {
        let value = lengths[i];
        // Count run of same value
        let mut k = i + 1;
        while k < n && lengths[k] == value {
            k += 1;
        }
        let runs = k - i;

        if value == 0 {
            // CodeRepeatedZeros
            code_repeated_zeros(runs, &mut tokens);
        } else {
            // CodeRepeatedValues
            code_repeated_values(runs, &mut tokens, value, prev_value);
            prev_value = value;
        }
        i += runs;
    }

    // Build histogram of tokens (0-18)
    let mut histogram = [0u32; CODE_LENGTH_CODES];
    for token in &tokens {
        histogram[token.code as usize] += 1;
    }

    // Build Huffman code for the code lengths (max depth 7)
    let mut code_length_bitdepth = [0u8; CODE_LENGTH_CODES];
    let mut code_length_codes = [0u16; CODE_LENGTH_CODES];
    build_code_length_huffman(
        &histogram,
        &mut code_length_bitdepth,
        &mut code_length_codes,
    );

    // Store the Huffman tree of Huffman tree (code length code lengths)
    // Throw away trailing zeros (matching StoreHuffmanTreeOfHuffmanTreeToBitMask)
    let mut codes_to_store = CODE_LENGTH_CODES;
    while codes_to_store > 4
        && code_length_bitdepth[CODE_LENGTH_CODE_ORDER[codes_to_store - 1]] == 0
    {
        codes_to_store -= 1;
    }

    w.write_bits((codes_to_store - 4) as u64, 4);
    for &order_idx in CODE_LENGTH_CODE_ORDER[..codes_to_store].iter() {
        w.write_bits(code_length_bitdepth[order_idx] as u64, 3);
    }

    // Clear huffman code if only one symbol (matching ClearHuffmanTreeIfOnlyOneSymbol)
    let symbol_count = code_length_bitdepth.iter().filter(|&&d| d != 0).count();
    if symbol_count <= 1 {
        code_length_bitdepth.fill(0);
        code_length_codes.fill(0);
    }

    // Calculate trimmed length and whether to write it
    // (matches StoreFullHuffmanCode logic)
    let num_tokens = tokens.len();
    let mut trimmed_length = num_tokens;
    let mut trailing_zero_bits = 0u32;

    // Count trailing zeros
    let mut idx = num_tokens;
    while idx > 0 {
        idx -= 1;
        let ix = tokens[idx].code as usize;
        if ix == 0 || ix == 17 || ix == 18 {
            trimmed_length -= 1;
            trailing_zero_bits += code_length_bitdepth[ix] as u32;
            if ix == 17 {
                trailing_zero_bits += 3;
            } else if ix == 18 {
                trailing_zero_bits += 7;
            }
        } else {
            break;
        }
    }

    let write_trimmed_length = trimmed_length > 1 && trailing_zero_bits > 12;
    let length = if write_trimmed_length {
        trimmed_length
    } else {
        num_tokens
    };

    w.write_bit(write_trimmed_length);
    if write_trimmed_length {
        if trimmed_length == 2 {
            w.write_bits(0, 3 + 2); // nbitpairs=1, trimmed_length=2
        } else {
            let nbits = 32 - ((trimmed_length - 2) as u32).leading_zeros();
            let nbitpairs = (nbits as usize).div_ceil(2);
            w.write_bits((nbitpairs - 1) as u64, 3);
            w.write_bits((trimmed_length - 2) as u64, (nbitpairs * 2) as u8);
        }
    }

    // Write the tokens (matching StoreHuffmanTreeToBitMask)
    for token in tokens[..length].iter() {
        let ix = token.code as usize;
        let extra_bits = token.extra_bits;

        w.write_bits(code_length_codes[ix] as u64, code_length_bitdepth[ix]);
        match ix {
            16 => w.write_bits(extra_bits as u64, 2),
            17 => w.write_bits(extra_bits as u64, 3),
            18 => w.write_bits(extra_bits as u64, 7),
            _ => {}
        }
    }
}

/// Encode repeated zero values using codes 0, 17, 18 (matches libwebp's CodeRepeatedZeros).
fn code_repeated_zeros(mut repetitions: usize, tokens: &mut Vec<HuffmanTreeToken>) {
    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: 0,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 11 {
            tokens.push(HuffmanTreeToken {
                code: 17,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else if repetitions < 139 {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: (repetitions - 11) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 18,
                extra_bits: 0x7f,
            }); // 138 repeated 0s
            repetitions -= 138;
        }
    }
}

/// Encode repeated non-zero values using codes 0-15 and 16 (matches libwebp's CodeRepeatedValues).
fn code_repeated_values(
    mut repetitions: usize,
    tokens: &mut Vec<HuffmanTreeToken>,
    value: u8,
    prev_value: u8,
) {
    if value != prev_value {
        tokens.push(HuffmanTreeToken {
            code: value,
            extra_bits: 0,
        });
        repetitions -= 1;
    }
    while repetitions >= 1 {
        if repetitions < 3 {
            for _ in 0..repetitions {
                tokens.push(HuffmanTreeToken {
                    code: value,
                    extra_bits: 0,
                });
            }
            break;
        } else if repetitions < 7 {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: (repetitions - 3) as u8,
            });
            break;
        } else {
            tokens.push(HuffmanTreeToken {
                code: 16,
                extra_bits: 3,
            }); // repeat 6 times
            repetitions -= 6;
        }
    }
}

/// Build Huffman code for code lengths (max depth 7).
fn build_code_length_huffman(
    histogram: &[u32; CODE_LENGTH_CODES],
    depths: &mut [u8; CODE_LENGTH_CODES],
    codes: &mut [u16; CODE_LENGTH_CODES],
) {
    // Build Huffman tree using existing function
    let freq_slice: Vec<u32> = histogram.to_vec();
    let computed_lengths = build_huffman_lengths(&freq_slice, 7);

    // Copy lengths
    depths[..CODE_LENGTH_CODES].copy_from_slice(&computed_lengths[..CODE_LENGTH_CODES]);

    // Generate reversed canonical codes (matching libwebp's ConvertBitDepthsToSymbols)
    let mut depth_count = [0u32; 16];
    for &d in depths.iter() {
        if d > 0 {
            depth_count[d as usize] += 1;
        }
    }

    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + depth_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for i in 0..CODE_LENGTH_CODES {
        let d = depths[i];
        if d > 0 {
            codes[i] = reverse_bits(d as u32, next_code[d as usize]) as u16;
            next_code[d as usize] += 1;
        }
    }
}

/// Reverse bits (matches libwebp's ReverseBits).
fn reverse_bits(num_bits: u32, bits: u32) -> u32 {
    const REVERSED_BITS: [u8; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];

    let mut retval = 0u32;
    let mut b = bits;
    let mut i = 0;
    while i < num_bits {
        i += 4;
        retval |= (REVERSED_BITS[(b & 0xf) as usize] as u32) << (16 - i);
        b >>= 4;
    }
    retval >> (16 - num_bits)
}

/// Write a single-entry Huffman tree (for trivial symbols).
pub fn write_single_entry_tree(w: &mut BitWriter, symbol: usize) {
    w.write_bit(true); // simple tree marker
    w.write_bit(false); // 1 symbol (count - 1)
                        // If symbol <= 1, use 1 bit; otherwise use 8 bits
    let is_8bit = symbol > 1;
    w.write_bit(is_8bit);
    if is_8bit {
        w.write_bits(symbol as u64, 8);
    } else {
        w.write_bits(symbol as u64, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_huffman_16_equal() {
        // 16 equally-likely symbols should each get length 4
        let mut freq = vec![0u32; 256];
        for f in freq[..16].iter_mut() {
            *f = 16;
        }
        let lengths = build_huffman_lengths(&freq, 15);

        println!("Lengths for 16 equal symbols:");
        for (i, &len) in lengths.iter().enumerate() {
            if len > 0 {
                println!("  symbol[{}] = {}", i, len);
            }
        }

        // Count how many at each length
        let mut length_counts = [0usize; 16];
        for &len in &lengths {
            if len > 0 {
                length_counts[len as usize] += 1;
            }
        }
        println!("Length histogram: {:?}", &length_counts[..8]);

        // Verify Kraft inequality
        let kraft: u32 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u32 << (15 - l))
            .sum();
        println!("Kraft sum (should be {}): {}", 1u32 << 15, kraft);

        // All 16 symbols should have length 4
        assert_eq!(length_counts[4], 16, "Expected 16 symbols with length 4");
    }

    #[test]
    fn test_build_huffman_single() {
        let freq = [0, 0, 100, 0, 0];
        let lengths = build_huffman_lengths(&freq, 15);
        assert_eq!(lengths[2], 1);
        assert_eq!(lengths.iter().filter(|&&l| l > 0).count(), 1);
    }

    #[test]
    fn test_build_huffman_two() {
        let freq = [100, 100, 0, 0, 0];
        let lengths = build_huffman_lengths(&freq, 15);
        assert_eq!(lengths[0], 1);
        assert_eq!(lengths[1], 1);
    }

    #[test]
    fn test_build_huffman_uniform() {
        let freq = [100, 100, 100, 100];
        let lengths = build_huffman_lengths(&freq, 15);
        // Uniform distribution should give length 2 for all, or close to it
        // The algorithm may produce slightly different results depending on tie-breaking
        for &l in &lengths {
            assert!((1..=3).contains(&l), "length {} out of expected range", l);
        }
        // Total should satisfy Kraft inequality
        let kraft: u32 = lengths.iter().map(|&l| 1u32 << (15 - l)).sum();
        assert!(kraft <= (1u32 << 15));
    }

    #[test]
    fn test_build_huffman_skewed() {
        let freq = [1, 10, 100, 1000];
        let lengths = build_huffman_lengths(&freq, 15);
        // Most frequent should have shortest code
        assert!(lengths[3] <= lengths[2]);
        assert!(lengths[2] <= lengths[1]);
        assert!(lengths[1] <= lengths[0]);
    }

    #[test]
    fn test_canonical_codes() {
        let lengths = [2u8, 2, 3, 3];
        let codes = build_huffman_codes(&lengths);

        println!("Generated codes:");
        for (i, code) in codes.iter().enumerate() {
            println!(
                "  symbol[{}]: code={:0width$b}, length={}",
                i,
                code.code,
                code.length,
                width = code.length as usize
            );
        }

        // Canonical codes (before bit reversal): 00, 01, 100, 101
        // After bit reversal for LSB-first VP8L:
        // 00 (2 bits) -> 00
        // 01 (2 bits) -> 10
        // 100 (3 bits) -> 001
        // 101 (3 bits) -> 101
        assert_eq!(codes[0].code, 0b00);
        assert_eq!(codes[1].code, 0b10);
        assert_eq!(codes[2].code, 0b001);
        assert_eq!(codes[3].code, 0b101);
    }
}
