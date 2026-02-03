//! Meta-Huffman encoding for spatially-varying codes.
//!
//! Allows different Huffman tables for different regions of the image.

use alloc::vec;
use alloc::vec::Vec;

use super::entropy::estimate_histogram_bits;
use super::histogram::Histogram;
use super::types::subsample_size;

/// Maximum number of histogram groups.
const MAX_HISTO_GROUPS: usize = 256;

/// Meta-Huffman configuration.
pub struct MetaHuffmanConfig {
    /// Block size bits (2-8).
    pub bits: u8,
    /// Number of histogram groups.
    pub num_groups: usize,
    /// Map from block index to histogram group.
    pub block_to_group: Vec<u16>,
    /// Histograms for each group.
    pub histograms: Vec<Histogram>,
}

impl MetaHuffmanConfig {
    /// Create a single-group configuration (no meta-Huffman).
    pub fn single(histogram: Histogram) -> Self {
        Self {
            bits: 0,
            num_groups: 1,
            block_to_group: Vec::new(),
            histograms: vec![histogram],
        }
    }
}

/// Build meta-Huffman configuration by clustering block histograms.
pub fn build_meta_huffman(
    pixels: &[u32],
    width: usize,
    height: usize,
    bits: u8,
    cache_bits: u8,
    quality: u8,
) -> MetaHuffmanConfig {
    let block_size = 1usize << bits;
    let blocks_x = subsample_size(width as u32, bits) as usize;
    let blocks_y = subsample_size(height as u32, bits) as usize;
    let num_blocks = blocks_x * blocks_y;

    // Build histogram for each block
    let mut block_histos: Vec<Histogram> = Vec::with_capacity(num_blocks);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut h = Histogram::new(cache_bits);

            let x_start = bx * block_size;
            let y_start = by * block_size;
            let x_end = (x_start + block_size).min(width);
            let y_end = (y_start + block_size).min(height);

            for y in y_start..y_end {
                for x in x_start..x_end {
                    let pixel = pixels[y * width + x];
                    h.add_literal(pixel);
                }
            }

            block_histos.push(h);
        }
    }

    // Determine target number of groups based on quality
    let max_groups = if quality > 90 {
        MAX_HISTO_GROUPS
    } else if quality > 75 {
        64
    } else if quality > 50 {
        16
    } else {
        4
    };

    // Cluster histograms using greedy merging
    let (histograms, block_to_group) = cluster_histograms(block_histos, max_groups, cache_bits);

    MetaHuffmanConfig {
        bits,
        num_groups: histograms.len(),
        block_to_group,
        histograms,
    }
}

/// Cluster histograms using greedy merging.
fn cluster_histograms(
    mut histos: Vec<Histogram>,
    max_groups: usize,
    cache_bits: u8,
) -> (Vec<Histogram>, Vec<u16>) {
    let n = histos.len();

    if n <= max_groups {
        // No clustering needed
        let mapping: Vec<u16> = (0..n as u16).collect();
        return (histos, mapping);
    }

    // Initialize mapping: each block maps to itself
    let mut mapping: Vec<usize> = (0..n).collect();

    // Compute initial costs
    let mut costs: Vec<u64> = histos.iter().map(estimate_histogram_bits).collect();

    // Greedy merge until we reach target number of groups
    let mut num_groups = n;
    while num_groups > max_groups {
        // Find best pair to merge
        let mut best_pair = (0, 1);
        let mut best_savings = i64::MIN;

        for i in 0..n {
            if mapping[i] != i {
                continue; // Skip merged clusters
            }

            for j in (i + 1)..n {
                if mapping[j] != j {
                    continue;
                }

                // Estimate cost of merging
                let combined_cost = estimate_combined_cost(&histos[i], &histos[j]);
                let savings = (costs[i] + costs[j]) as i64 - combined_cost as i64;

                if savings > best_savings {
                    best_savings = savings;
                    best_pair = (i, j);
                }
            }
        }

        // Merge best pair
        let (i, j) = best_pair;

        // Merge j into i
        // We need to clone j's histogram to avoid borrowing issues
        let j_histo = histos[j].clone();
        histos[i].add(&j_histo);
        costs[i] = estimate_histogram_bits(&histos[i]);

        // Update mapping: all blocks pointing to j now point to i
        for m in mapping.iter_mut() {
            if *m == j {
                *m = i;
            }
        }

        num_groups -= 1;
    }

    // Compact: renumber groups to be contiguous
    let mut group_remap: Vec<Option<u16>> = vec![None; n];
    let mut final_histos = Vec::new();

    for &group in &mapping {
        if group_remap[group].is_none() {
            group_remap[group] = Some(final_histos.len() as u16);
            final_histos.push(Histogram::new(cache_bits));
            final_histos.last_mut().unwrap().add(&histos[group]);
        }
    }

    // Build final mapping
    let final_mapping: Vec<u16> = mapping.iter().map(|&g| group_remap[g].unwrap()).collect();

    (final_histos, final_mapping)
}

/// Estimate cost of combining two histograms.
fn estimate_combined_cost(h1: &Histogram, h2: &Histogram) -> u64 {
    let mut combined = h1.clone();
    combined.add(h2);
    estimate_histogram_bits(&combined)
}

/// Encode the meta-Huffman image.
/// Returns pixels where green channel contains the group index.
pub fn encode_meta_huffman_image(config: &MetaHuffmanConfig, _blocks_x: usize) -> Vec<u32> {
    config
        .block_to_group
        .iter()
        .map(|&group| {
            // Pack group index into green (low byte) and red (high byte)
            let lo = group as u8;
            let hi = (group >> 8) as u8;
            super::types::make_argb(255, hi, lo, 0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::types::make_argb;
    use super::*;

    #[test]
    fn test_single_group() {
        let h = Histogram::new(0);
        let config = MetaHuffmanConfig::single(h);
        assert_eq!(config.num_groups, 1);
        assert!(config.block_to_group.is_empty());
    }

    #[test]
    fn test_cluster_small() {
        // Create 4 identical histograms
        let mut histos = Vec::new();
        for _ in 0..4 {
            let mut h = Histogram::new(0);
            h.add_literal(make_argb(255, 128, 128, 128));
            histos.push(h);
        }

        let (groups, mapping) = cluster_histograms(histos, 2, 0);

        assert!(groups.len() <= 2);
        assert_eq!(mapping.len(), 4);
        for &m in &mapping {
            assert!(m < groups.len() as u16);
        }
    }
}
