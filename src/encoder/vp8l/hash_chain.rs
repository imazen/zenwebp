//! Hash chain for LZ77 backward reference finding.
//!
//! Uses libwebp's hash function and chain structure for finding matches.

use alloc::vec;
use alloc::vec::Vec;

use super::types::{HASH_BITS, HASH_SIZE, MAX_LENGTH, MAX_LENGTH_BITS};

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
#[derive(Debug)]
pub struct HashChain {
    /// For each pixel: (offset << MAX_LENGTH_BITS) | length
    /// offset = distance to best match, length = best match length
    offset_length: Vec<u32>,
}

impl HashChain {
    /// Build hash chain for the given ARGB pixels.
    pub fn new(argb: &[u32], quality: u8, width: usize) -> Self {
        let size = argb.len();
        let mut offset_length = vec![0u32; size];

        if size <= 2 {
            return Self { offset_length };
        }

        let iter_max = get_max_iters(quality);
        let window_size = get_window_size(quality, width);

        // Build the chain: for each position, store the previous position with same hash
        let mut hash_to_first: Vec<i32> = vec![-1; HASH_SIZE];
        let mut chain: Vec<i32> = vec![-1; size];

        // Fill chain linking pixels with same hash
        let mut argb_comp = argb[0] == argb[1];
        for pos in 0..size.saturating_sub(2) {
            let argb_comp_next = argb.get(pos + 2).is_some_and(|&p| argb[pos + 1] == p);

            if argb_comp && argb_comp_next {
                // Consecutive identical pixels - use special hash including run length
                let base_color = argb[pos];
                let mut len = 1usize;

                // Find run length
                while pos + len + 2 < size && argb[pos + len + 2] == base_color {
                    len += 1;
                }

                // Skip long runs (they link to predecessor with distance=1)
                if len > MAX_LENGTH {
                    for i in 0..(len - MAX_LENGTH) {
                        chain[pos + i] = -1;
                    }
                    // Process remaining
                    let skip = len - MAX_LENGTH;
                    for i in 0..MAX_LENGTH.min(len) {
                        let p = pos + skip + i;
                        // Hash: (color, remaining_length)
                        let hash = hash_pix_pair(base_color, (MAX_LENGTH - i) as u32);
                        chain[p] = hash_to_first[hash];
                        hash_to_first[hash] = p as i32;
                    }
                    argb_comp = false;
                    continue;
                }

                // Process normal-length runs
                while len > 0 {
                    let p = pos + len - 1;
                    let hash = hash_pix_pair(base_color, len as u32);
                    chain[p] = hash_to_first[hash];
                    hash_to_first[hash] = p as i32;
                    len -= 1;
                }
                argb_comp = false;
            } else {
                // Normal case: hash adjacent pixels
                let hash = hash_pix_pair(argb[pos], argb[pos + 1]);
                chain[pos] = hash_to_first[hash];
                hash_to_first[hash] = pos as i32;
                argb_comp = argb_comp_next;
            }
        }

        // Handle penultimate pixel
        if size >= 2 {
            let pos = size - 2;
            let hash = hash_pix_pair(argb[pos], argb[pos + 1]);
            chain[pos] = hash_to_first[hash];
        }

        drop(hash_to_first);

        // Find best matches working backwards
        offset_length[size - 1] = 0;
        for base_pos in (1..size - 1).rev() {
            let max_len = max_find_copy_length(size - 1 - base_pos);
            let argb_start = &argb[base_pos..];
            let mut best_len = 0usize;
            let mut best_dist = 0usize;
            let min_pos = base_pos.saturating_sub(window_size);
            let length_max = max_len.min(256);
            let _max_base_pos = base_pos;

            // Heuristic: try row above
            if base_pos >= width {
                let curr_len =
                    find_match_length(&argb[base_pos - width..], argb_start, best_len, max_len);
                if curr_len > best_len {
                    best_len = curr_len;
                    best_dist = width;
                }
            }

            // Heuristic: try previous pixel
            if base_pos >= 1 {
                let curr_len =
                    find_match_length(&argb[base_pos - 1..], argb_start, best_len, max_len);
                if curr_len > best_len {
                    best_len = curr_len;
                    best_dist = 1;
                }
            }

            // Follow hash chain
            let mut pos = chain[base_pos];
            let mut iters = iter_max;
            let mut best_argb = argb_start.get(best_len).copied().unwrap_or(0);

            while pos >= min_pos as i32 && iters > 0 {
                iters -= 1;
                let p = pos as usize;

                // Quick rejection: check if end matches
                if argb.get(p + best_len).copied().unwrap_or(!best_argb) != best_argb {
                    pos = chain[p];
                    continue;
                }

                let curr_len = vector_mismatch(&argb[p..], argb_start, max_len);
                if curr_len > best_len {
                    best_len = curr_len;
                    best_dist = base_pos - p;
                    best_argb = argb_start.get(best_len).copied().unwrap_or(0);

                    if best_len >= length_max {
                        break;
                    }
                }

                pos = chain[p];
            }

            // Store best match
            debug_assert!(best_len <= MAX_LENGTH);
            offset_length[base_pos] = ((best_dist as u32) << MAX_LENGTH_BITS) | (best_len as u32);
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
        super::types::WINDOW_SIZE
    } else if quality > 50 {
        width << 8
    } else if quality > 25 {
        width << 6
    } else {
        width << 4
    };
    max.min(super::types::WINDOW_SIZE)
}

/// Limit copy length.
#[inline]
fn max_find_copy_length(len: usize) -> usize {
    len.min(MAX_LENGTH)
}

/// Find match length between two sequences.
#[inline]
fn find_match_length(array1: &[u32], array2: &[u32], best_len: usize, max_len: usize) -> usize {
    // Quick check at best_len position
    if array1.len() <= best_len || array2.len() <= best_len {
        return 0;
    }
    if array1[best_len] != array2[best_len] {
        return 0;
    }
    vector_mismatch(array1, array2, max_len)
}

/// Find first mismatch position (or max_len if all match).
#[inline]
fn vector_mismatch(a: &[u32], b: &[u32], max_len: usize) -> usize {
    let len = a.len().min(b.len()).min(max_len);
    for i in 0..len {
        if a[i] != b[i] {
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
    fn test_vector_mismatch() {
        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3, 9, 5];
        assert_eq!(vector_mismatch(&a, &b, 10), 3);

        let c = [1, 2, 3, 4, 5];
        let d = [1, 2, 3, 4, 5];
        assert_eq!(vector_mismatch(&c, &d, 10), 5);
    }
}
