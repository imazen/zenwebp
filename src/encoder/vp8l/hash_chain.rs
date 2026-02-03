//! Hash chain for LZ77 backward reference finding.
//!
//! Uses libwebp's hash function and chain structure for finding matches.
//! Includes left-extension optimization from libwebp's VP8LHashChainFill.

use alloc::vec;
use alloc::vec::Vec;

use super::types::{HASH_BITS, HASH_SIZE, MAX_LENGTH, MAX_LENGTH_BITS, WINDOW_SIZE};

/// Hash multipliers for two-pixel hashing (from libwebp).
const HASH_MULT_HI: u32 = 0xc6a4a793;
const HASH_MULT_LO: u32 = 0x5bd1e996;

/// Hash two adjacent pixels to get a bucket index.
#[inline]
fn hash_pix_pair(p0: u32, p1: u32) -> usize {
    let key = p1
        .wrapping_mul(HASH_MULT_HI)
        .wrapping_add(p0.wrapping_mul(HASH_MULT_LO));
    (key >> (32 - HASH_BITS)) as usize
}

/// Hash chain for efficient LZ77 match finding.
///
/// For each pixel position, stores the best (offset, length) match found.
/// offset = raw distance to matching position.
/// length = number of matching pixels.
#[derive(Debug)]
pub struct HashChain {
    /// For each pixel: (offset << MAX_LENGTH_BITS) | length
    /// offset = distance to best match, length = best match length
    offset_length: Vec<u32>,
}

impl HashChain {
    /// Build hash chain for the given ARGB pixels.
    ///
    /// Matches libwebp's VP8LHashChainFill including:
    /// - Two-pixel hash for normal positions
    /// - Color+run-length hash for constant-color runs
    /// - Heuristic row-above and previous-pixel initial guesses
    /// - Left-extension optimization (fills preceding positions without re-search)
    pub fn new(argb: &[u32], quality: u8, width: usize) -> Self {
        let size = argb.len();
        let mut offset_length = vec![0u32; size];

        if size <= 2 {
            return Self { offset_length };
        }

        let iter_max = get_max_iters(quality);
        let window_size = get_window_size(quality, width);

        // Temporarily use offset_length as chain storage (reinterpreted as i32).
        // This is safe because both are u32/i32 and same size.
        let mut chain: Vec<i32> = vec![-1; size];
        let mut hash_to_first: Vec<i32> = vec![-1; HASH_SIZE];

        // Fill chain linking pixels with same hash
        let mut argb_comp = argb[0] == argb[1];
        let mut pos = 0usize;
        while pos < size.saturating_sub(2) {
            let argb_comp_next = pos + 2 < size && argb[pos + 1] == argb[pos + 2];

            if argb_comp && argb_comp_next {
                // Consecutive identical pixels - use (color, run_length) hash
                let base_color = argb[pos];
                let mut len = 1usize;

                // Find run length
                while pos + len + 2 < size && argb[pos + len + 2] == base_color {
                    len += 1;
                }

                if len > MAX_LENGTH {
                    // Skip positions that will be covered by distance=1 matches
                    let skip = len - MAX_LENGTH;
                    for i in 0..skip {
                        chain[pos + i] = -1;
                    }
                    pos += skip;
                    len = MAX_LENGTH;
                }

                // Process remaining run positions
                while len > 0 {
                    let hash = hash_pix_pair(base_color, len as u32);
                    chain[pos] = hash_to_first[hash];
                    hash_to_first[hash] = pos as i32;
                    pos += 1;
                    len -= 1;
                }
                argb_comp = false;
            } else {
                // Normal case: hash adjacent pixels
                let hash = hash_pix_pair(argb[pos], argb[pos + 1]);
                chain[pos] = hash_to_first[hash];
                hash_to_first[hash] = pos as i32;
                pos += 1;
                argb_comp = argb_comp_next;
            }
        }

        // Handle penultimate pixel
        if size >= 2 {
            let p = size - 2;
            let hash = hash_pix_pair(argb[p], argb[p + 1]);
            chain[p] = hash_to_first[hash];
        }

        drop(hash_to_first);

        // Find best matches working backwards, with left-extension
        offset_length[0] = 0;
        offset_length[size - 1] = 0;

        let mut base_position = size - 2;
        while base_position > 0 {
            let max_len = max_find_copy_length(size - 1 - base_position);
            let argb_start = base_position;
            let mut best_length = 0usize;
            let mut best_distance = 0usize;
            let min_pos = base_position.saturating_sub(window_size);
            let length_max = max_len.min(256);

            // Heuristic: try row above as initial guess
            if base_position >= width {
                let curr_len = find_match_length(
                    argb,
                    base_position - width,
                    argb_start,
                    best_length,
                    max_len,
                );
                if curr_len > best_length {
                    best_length = curr_len;
                    best_distance = width;
                }
            }

            // Heuristic: try previous pixel
            if base_position >= 1 {
                let curr_len =
                    find_match_length(argb, base_position - 1, argb_start, best_length, max_len);
                if curr_len > best_length {
                    best_length = curr_len;
                    best_distance = 1;
                }
            }

            // Skip chain traversal if already maximal
            let mut chain_pos = if best_length == MAX_LENGTH {
                -1 // Skip chain
            } else {
                chain[base_position]
            };

            // Follow hash chain
            let mut iters = iter_max;
            let best_argb_init = if best_length < argb.len() - argb_start {
                argb[argb_start + best_length]
            } else {
                0
            };
            let mut best_argb = best_argb_init;

            while chain_pos >= min_pos as i32 && iters > 0 {
                iters -= 1;
                let p = chain_pos as usize;

                // Quick rejection: check if end matches
                if p + best_length < size {
                    if argb[p + best_length] != best_argb {
                        chain_pos = chain[p];
                        continue;
                    }
                } else {
                    chain_pos = chain[p];
                    continue;
                }

                let curr_len = vector_mismatch(argb, p, argb_start, max_len);
                if curr_len > best_length {
                    best_length = curr_len;
                    best_distance = base_position - p;
                    if argb_start + best_length < size {
                        best_argb = argb[argb_start + best_length];
                    }

                    if best_length >= length_max {
                        break;
                    }
                }

                chain_pos = chain[p];
            }

            // Left-extension optimization (from libwebp):
            // If the match extends to the left, fill in preceding positions
            // without re-searching the hash chain.
            let max_base_position = base_position;
            loop {
                debug_assert!(best_length <= MAX_LENGTH);
                debug_assert!(best_distance <= WINDOW_SIZE);
                offset_length[base_position] =
                    ((best_distance as u32) << MAX_LENGTH_BITS) | (best_length as u32);

                if base_position == 0 {
                    break;
                }
                base_position -= 1;

                // Stop if no match
                if best_distance == 0 {
                    break;
                }
                // Stop if we can't extend left
                if base_position < best_distance {
                    break;
                }
                if argb[base_position - best_distance] != argb[base_position] {
                    break;
                }
                // Stop if at max length with non-trivial distance, and there
                // could be a closer match
                if best_length == MAX_LENGTH
                    && best_distance != 1
                    && base_position + MAX_LENGTH < max_base_position
                {
                    break;
                }
                if best_length < MAX_LENGTH {
                    best_length += 1;
                }
            }
        }

        Self { offset_length }
    }

    /// Get the best match distance at a position.
    #[inline]
    pub fn offset(&self, pos: usize) -> usize {
        (self.offset_length[pos] >> MAX_LENGTH_BITS) as usize
    }

    /// Get the best match length at a position.
    #[inline]
    pub fn length(&self, pos: usize) -> usize {
        (self.offset_length[pos] & ((1 << MAX_LENGTH_BITS) - 1)) as usize
    }

    /// Get both offset and length at a position (avoids double lookup).
    #[inline]
    pub fn find_copy(&self, pos: usize) -> (usize, usize) {
        let val = self.offset_length[pos];
        let offset = (val >> MAX_LENGTH_BITS) as usize;
        let length = (val & ((1 << MAX_LENGTH_BITS) - 1)) as usize;
        (offset, length)
    }

    /// Number of positions in the chain.
    #[inline]
    pub fn size(&self) -> usize {
        self.offset_length.len()
    }

    /// Create an empty hash chain with zero-initialized entries.
    /// Used by LZ77 Box to build a custom chain.
    pub fn empty(size: usize) -> Self {
        Self {
            offset_length: vec![0u32; size],
        }
    }

    /// Set the offset/length pair at a position.
    #[inline]
    pub fn set(&mut self, pos: usize, offset: usize, length: usize) {
        self.offset_length[pos] = ((offset as u32) << MAX_LENGTH_BITS) | length as u32;
    }
}

/// Get max iterations for quality level.
#[inline]
fn get_max_iters(quality: u8) -> usize {
    8 + (quality as usize * quality as usize) / 128
}

/// Get window size for quality level.
#[inline]
fn get_window_size(quality: u8, width: usize) -> usize {
    let max = if quality > 75 {
        WINDOW_SIZE
    } else if quality > 50 {
        width << 8
    } else if quality > 25 {
        width << 6
    } else {
        width << 4
    };
    max.min(WINDOW_SIZE)
}

/// Limit copy length.
#[inline]
fn max_find_copy_length(len: usize) -> usize {
    len.min(MAX_LENGTH)
}

/// Find match length between two sequences starting at given offsets.
/// Returns 0 if the character at best_len doesn't match (quick rejection).
#[inline]
fn find_match_length(
    argb: &[u32],
    pos1: usize,
    pos2: usize,
    best_len: usize,
    max_len: usize,
) -> usize {
    // Quick check at best_len position
    let remaining1 = argb.len() - pos1;
    let remaining2 = argb.len() - pos2;
    if remaining1 <= best_len || remaining2 <= best_len {
        return 0;
    }
    if argb[pos1 + best_len] != argb[pos2 + best_len] {
        return 0;
    }
    vector_mismatch(argb, pos1, pos2, max_len)
}

/// Find first mismatch position between two subsequences.
#[inline]
fn vector_mismatch(argb: &[u32], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let remaining1 = argb.len() - pos1;
    let remaining2 = argb.len() - pos2;
    let len = remaining1.min(remaining2).min(max_len);
    for i in 0..len {
        if argb[pos1 + i] != argb[pos2 + i] {
            return i;
        }
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash_pix_pair(0xFF112233, 0xFF445566);
        let h2 = hash_pix_pair(0xFF112233, 0xFF445566);
        assert_eq!(h1, h2);
        assert!(h1 < HASH_SIZE);
    }

    #[test]
    fn test_hash_chain_simple() {
        // Simple test: repeated pixel should find match
        let pixels = vec![0xFF000000u32; 100];
        let chain = HashChain::new(&pixels, 75, 10);

        // Position 50 should find match with distance 1
        assert!(chain.length(50) > 0);
    }

    #[test]
    fn test_left_extension() {
        // Test that left-extension fills in consecutive positions
        let mut pixels = vec![0xFF112233u32; 50];
        pixels.extend_from_slice(&[0xFFAABBCCu32; 50]);
        pixels.extend_from_slice(&[0xFF112233u32; 50]);

        let chain = HashChain::new(&pixels, 75, 50);

        // Positions 100-149 should find matches to positions 0-49
        for i in 100..140 {
            let (offset, length) = chain.find_copy(i);
            assert!(length > 0, "Position {} should have a match", i);
            assert!(offset > 0, "Position {} should have non-zero offset", i);
        }
    }

    #[test]
    fn test_vector_mismatch() {
        let a = [1, 2, 3, 4, 5, 9, 7, 8];
        assert_eq!(vector_mismatch(&a, 0, 3, 5), 0); // 1 != 4
        let b = [1, 2, 3, 1, 2, 3, 7, 8];
        assert_eq!(vector_mismatch(&b, 0, 3, 5), 3); // matches 3 then differs
    }
}
