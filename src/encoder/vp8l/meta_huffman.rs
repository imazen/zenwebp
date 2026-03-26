//! Meta-Huffman encoding for spatially-varying codes.
//!
//! Builds per-tile histograms from backward references, clusters them
//! using entropy-based greedy merging (matching libwebp), and produces
//! a histogram index map for spatially-varying Huffman codes.

use alloc::vec;
use alloc::vec::Vec;

use super::entropy::{HistogramCosts, compute_histogram_cost, get_combined_histogram_cost};
use super::histogram::Histogram;
use super::types::{BackwardRefs, PixOrCopy, subsample_size};

/// Maximum Huffman group count for greedy combining.
const MAX_HISTO_GREEDY: usize = 100;

/// Number of entropy bins for initial binning (4×4×4 = 64 for 3D binning).
const NUM_PARTITIONS: usize = 4;
const BIN_SIZE: usize = NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS; // 64

/// Meta-Huffman configuration.
pub struct MetaHuffmanInfo {
    /// Histogram bits (block size = 2^bits pixels).
    pub histo_bits: u8,
    /// Number of final histogram groups.
    pub num_histograms: usize,
    /// Mapping from tile index to histogram group index.
    pub histogram_symbols: Vec<u16>,
    /// Final clustered histograms (one per group).
    pub histograms: Vec<Histogram>,
    /// Cached costs for each final histogram.
    pub costs: Vec<HistogramCosts>,
    /// Image width (for sub-image dimension computation).
    pub image_width: usize,
    /// Image height (for sub-image dimension computation).
    pub image_height: usize,
}

/// Build per-tile histograms from backward references.
///
/// Each tile covers a (2^histo_bits × 2^histo_bits) block of pixels.
/// Tokens are assigned to tiles based on the pixel position they encode.
/// Build per-tile histograms from backward references.
///
/// Each tile covers a (2^histo_bits × 2^histo_bits) block of pixels.
/// Tokens are assigned to tiles based on the pixel position they encode.
/// Assumes distances in refs are already plane codes (from apply_2d_locality).
fn build_tile_histograms(
    refs: &BackwardRefs,
    width: usize,
    _height: usize,
    histo_bits: u8,
    cache_bits: u8,
) -> Vec<Histogram> {
    let histo_xsize = subsample_size(width as u32, histo_bits) as usize;
    let histo_ysize = subsample_size(_height as u32, histo_bits) as usize;
    let num_tiles = histo_xsize * histo_ysize;

    let mut histos: Vec<Histogram> = (0..num_tiles).map(|_| Histogram::new(cache_bits)).collect();

    let mut x = 0usize;
    let mut y = 0usize;

    for token in refs.iter() {
        let tile_idx = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
        debug_assert!(tile_idx < num_tiles);

        match *token {
            PixOrCopy::Literal(argb) => {
                histos[tile_idx].add_literal(argb);
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
            }
            PixOrCopy::CacheIdx(idx) => {
                histos[tile_idx].add_cache_idx(idx);
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
            }
            PixOrCopy::Copy { len, dist } => {
                // Distances are already plane codes
                histos[tile_idx].add_copy(len, dist);

                for _ in 0..len {
                    x += 1;
                    if x >= width {
                        x = 0;
                        y += 1;
                    }
                }
            }
        }
    }

    histos
}

/// Get bin ID for entropy-based binning (matching libwebp's GetBinIdForEntropy).
fn get_bin_id_for_entropy(min: u64, max: u64, val: u64) -> usize {
    if min == max {
        return 0;
    }
    // Map val to [0, NUM_PARTITIONS-1] linearly
    let range = max - min;
    let offset = val.saturating_sub(min);
    let bin = ((NUM_PARTITIONS as u64 - 1) * offset) / range;
    bin.min(NUM_PARTITIONS as u64 - 1) as usize
}

/// Priority queue entry for histogram pair merging.
/// Matches libwebp's HistogramPair struct.
struct HistogramPair {
    idx1: usize,
    idx2: usize,
    cost_diff: i64,
    cost_combo: u64,
    per_type_costs: [u64; 5],
}

/// Min-priority-queue for histogram pairs, sorted by cost_diff (most negative first).
/// Uses a simple array with head tracking (matching libwebp's HistoQueue).
struct HistoQueue {
    queue: Vec<HistogramPair>,
    max_size: usize,
}

impl HistoQueue {
    fn new(max_size: usize) -> Self {
        Self {
            queue: Vec::with_capacity(max_size + 1),
            max_size,
        }
    }

    /// Push a pair if it improves on the threshold. Returns cost_diff if pushed, 0 otherwise.
    fn push(
        &mut self,
        histos: &[Histogram],
        costs: &[HistogramCosts],
        idx1: usize,
        idx2: usize,
        threshold: i64,
    ) -> i64 {
        if self.queue.len() >= self.max_size {
            return 0;
        }
        debug_assert!(threshold <= 0);

        let sum_cost = costs[idx1].total as i64 + costs[idx2].total as i64;
        let cost_threshold = sum_cost.saturating_add(threshold);
        if cost_threshold <= 0 {
            return 0;
        }

        if let Some(cost_combo) = get_combined_histogram_cost(
            &histos[idx1],
            &costs[idx1],
            &histos[idx2],
            &costs[idx2],
            cost_threshold as u64,
        ) {
            let cost_diff = cost_combo as i64 - sum_cost;
            let pair = HistogramPair {
                idx1,
                idx2,
                cost_diff,
                cost_combo,
                per_type_costs: [0; 5], // Not needed for stochastic
            };
            self.queue.push(pair);
            self.update_head(self.queue.len() - 1);
            return cost_diff;
        }
        0
    }

    /// Push a pair for greedy combining, storing per-type costs.
    fn push_greedy(
        &mut self,
        histos: &[Histogram],
        costs: &[HistogramCosts],
        idx1: usize,
        idx2: usize,
    ) {
        let sum_cost = costs[idx1].total as i64 + costs[idx2].total as i64;
        // Threshold = sum_cost (only accept if combined < sum)
        if sum_cost <= 0 {
            return;
        }

        if let Some(cost_combo) = get_combined_histogram_cost(
            &histos[idx1],
            &costs[idx1],
            &histos[idx2],
            &costs[idx2],
            sum_cost as u64,
        ) {
            let cost_diff = cost_combo as i64 - sum_cost;
            if cost_diff >= 0 {
                return;
            }
            let pair = HistogramPair {
                idx1,
                idx2,
                cost_diff,
                cost_combo,
                per_type_costs: [0; 5],
            };
            self.queue.push(pair);
            self.update_head(self.queue.len() - 1);
        }
    }

    /// Ensure queue[0] has the smallest cost_diff (most negative = best savings).
    fn update_head(&mut self, new_idx: usize) {
        if self.queue.is_empty() {
            return;
        }
        if new_idx < self.queue.len() && self.queue[new_idx].cost_diff < self.queue[0].cost_diff {
            self.queue.swap(0, new_idx);
        }
    }

    /// Remove a pair by index, replacing with last element.
    fn pop_at(&mut self, idx: usize) {
        let last = self.queue.len() - 1;
        if idx != last {
            self.queue.swap(idx, last);
        }
        self.queue.pop();
    }

    /// Replace bad_id with good_id in a pair's indices, keeping idx1 < idx2.
    fn fix_pair_idx(pair: &mut HistogramPair, bad_id: usize, good_id: usize) {
        if pair.idx1 == bad_id {
            pair.idx1 = good_id;
        }
        if pair.idx2 == bad_id {
            pair.idx2 = good_id;
        }
        if pair.idx1 > pair.idx2 {
            core::mem::swap(&mut pair.idx1, &mut pair.idx2);
        }
    }

    /// Re-evaluate pair cost. Returns false if pair should be removed.
    fn update_pair(
        histos: &[Histogram],
        costs: &[HistogramCosts],
        pair: &mut HistogramPair,
    ) -> bool {
        let sum_cost = costs[pair.idx1].total as i64 + costs[pair.idx2].total as i64;
        if sum_cost <= 0 {
            return false;
        }
        if let Some(cost_combo) = get_combined_histogram_cost(
            &histos[pair.idx1],
            &costs[pair.idx1],
            &histos[pair.idx2],
            &costs[pair.idx2],
            sum_cost as u64,
        ) {
            pair.cost_diff = cost_combo as i64 - sum_cost;
            pair.cost_combo = cost_combo;
            pair.cost_diff < 0
        } else {
            false
        }
    }
}

/// Cluster histograms using entropy binning + stochastic + greedy combining.
/// Matches libwebp's HistogramCombineEntropyBin + HistogramCombineStochastic
/// + HistogramCombineGreedy + HistogramRemap.
fn cluster_histograms(
    tile_histos: &[Histogram],
    quality: u8,
    cache_bits: u8,
) -> (Vec<Histogram>, Vec<u16>) {
    let n = tile_histos.len();

    if n <= 1 {
        if n == 1 {
            return (vec![tile_histos[0].clone()], vec![0]);
        }
        return (Vec::new(), Vec::new());
    }

    // Compute initial costs for all histograms
    let mut costs: Vec<HistogramCosts> = tile_histos.iter().map(compute_histogram_cost).collect();

    // Track which histograms are active and their mapping
    let mut active: Vec<bool> = vec![true; n];
    let mut mapping: Vec<usize> = (0..n).collect(); // tile -> cluster index
    let mut histos: Vec<Histogram> = tile_histos.to_vec();

    // Phase 1: Entropy binning - assign each histogram to an entropy bin
    // Find min/max costs for literal, red, blue types
    let mut lit_min = u64::MAX;
    let mut lit_max = 0u64;
    let mut red_min = u64::MAX;
    let mut red_max = 0u64;
    let mut blue_min = u64::MAX;
    let mut blue_max = 0u64;

    for (i, cost) in costs.iter().enumerate() {
        if !active[i] {
            continue;
        }
        lit_min = lit_min.min(cost.per_type[0]);
        lit_max = lit_max.max(cost.per_type[0]);
        red_min = red_min.min(cost.per_type[1]);
        red_max = red_max.max(cost.per_type[1]);
        blue_min = blue_min.min(cost.per_type[2]);
        blue_max = blue_max.max(cost.per_type[2]);
    }

    // Assign bin IDs (3D: literal × red × blue)
    let mut bin_ids: Vec<usize> = Vec::with_capacity(n);
    for cost in &costs {
        let lit_bin = get_bin_id_for_entropy(lit_min, lit_max, cost.per_type[0]);
        let red_bin = get_bin_id_for_entropy(red_min, red_max, cost.per_type[1]);
        let blue_bin = get_bin_id_for_entropy(blue_min, blue_max, cost.per_type[2]);
        bin_ids
            .push(lit_bin * NUM_PARTITIONS * NUM_PARTITIONS + red_bin * NUM_PARTITIONS + blue_bin);
    }

    // Phase 2: Merge within entropy bins (HistogramCombineEntropyBin)
    let num_active = active.iter().filter(|&&a| a).count();
    let num_bins = BIN_SIZE.min(num_active);
    let do_entropy_combine = num_active > num_bins * 2 && quality < 100;

    if do_entropy_combine {
        // Track first histogram per bin
        let mut bin_first: Vec<Option<usize>> = vec![None; BIN_SIZE];

        // Combine cost factor (matching libwebp's GetCombineCostFactor)
        let mut combine_cost_factor = 16i64;
        if quality < 90 {
            if num_active > 256 {
                combine_cost_factor /= 2;
            }
            if num_active > 512 {
                combine_cost_factor /= 2;
            }
            if num_active > 1024 {
                combine_cost_factor /= 2;
            }
            if quality <= 50 {
                combine_cost_factor /= 2;
            }
        }

        for i in 0..n {
            if !active[i] {
                continue;
            }

            let bin_id = bin_ids[i];
            if let Some(first) = bin_first[bin_id] {
                // Try to merge with first histogram in bin
                let bit_cost = costs[first].total;
                let threshold = bit_cost + costs[i].total;
                let cost_thresh_val = threshold.saturating_sub(
                    div_round_i64(bit_cost as i64 * combine_cost_factor, 100) as u64,
                );

                if get_combined_histogram_cost(
                    &histos[first],
                    &costs[first],
                    &histos[i],
                    &costs[i],
                    cost_thresh_val,
                )
                .is_some()
                {
                    // Merge i into first
                    let i_histo = histos[i].clone();
                    histos[first].add(&i_histo);
                    costs[first] = compute_histogram_cost(&histos[first]);
                    active[i] = false;

                    // Remap all tiles pointing to i → first
                    for m in mapping.iter_mut() {
                        if *m == i {
                            *m = first;
                        }
                    }
                }
            } else {
                bin_first[bin_id] = Some(i);
            }
        }
    }

    // Phase 2b: Stochastic combining (matching libwebp's HistogramCombineStochastic).
    // Uses priority queue of size 9 with Lehmer RNG, matching libwebp exactly.
    let target_size = {
        let q3 = (quality as u64) * (quality as u64) * (quality as u64);
        let t = 1 + div_round_u64(q3 * (MAX_HISTO_GREEDY as u64 - 1), 100u64 * 100 * 100);
        t.min(MAX_HISTO_GREEDY as u64) as usize
    };

    let num_active_pre_stochastic = active.iter().filter(|&&a| a).count();

    // do_greedy tracks whether we should run the greedy phase after stochastic
    let mut do_greedy = num_active_pre_stochastic <= target_size;

    if num_active_pre_stochastic > target_size {
        const HISTO_QUEUE_SIZE: usize = 9;
        let mut histo_queue = HistoQueue::new(HISTO_QUEUE_SIZE);

        // Build a compact index: position -> original index
        // This matches libwebp's approach where histograms are stored in a
        // compact array and HistogramSetRemoveHistogram swaps in the last element.
        let mut compact: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let mut compact_size = compact.len();

        let outer_iters = compact_size;
        let num_tries_no_success = outer_iters / 2;
        let mut tries_with_no_success = 0usize;
        let mut seed: u32 = 1;

        for _iter in 0..outer_iters {
            if compact_size < 2 || compact_size <= target_size {
                break;
            }
            tries_with_no_success += 1;
            if tries_with_no_success >= num_tries_no_success {
                break;
            }

            let best_cost = if histo_queue.queue.is_empty() {
                0i64
            } else {
                histo_queue.queue[0].cost_diff
            };
            let rand_range = (compact_size as u64) * (compact_size as u64 - 1);
            let num_tries = compact_size / 2;

            // Pick random samples (matching libwebp's inner loop)
            for _ in 0..num_tries {
                if compact_size < 2 {
                    break;
                }
                // Lehmer RNG
                seed = ((seed as u64 * 48271u64) % 2147483647u64) as u32;
                let tmp = (seed as u64) % rand_range;
                let ci1 = (tmp / (compact_size as u64 - 1)) as usize;
                let mut ci2 = (tmp % (compact_size as u64 - 1)) as usize;
                if ci2 >= ci1 {
                    ci2 += 1;
                }

                let idx1 = compact[ci1];
                let idx2 = compact[ci2];

                let curr_cost =
                    histo_queue.push(&histos, &costs, idx1, idx2, best_cost.min(0));
                if curr_cost < 0 {
                    // Break if queue reached full capacity
                    if histo_queue.queue.len() >= histo_queue.max_size {
                        break;
                    }
                }
            }

            if histo_queue.queue.is_empty() {
                continue;
            }

            // Get the best pair from the queue head
            let best_idx1 = histo_queue.queue[0].idx1;
            let merge_idx2 = histo_queue.queue[0].idx2;

            // Merge idx2 into idx1
            let j_histo = histos[merge_idx2].clone();
            histos[best_idx1].add(&j_histo);
            // Update cost from the queue's precomputed combined cost
            costs[best_idx1].total = histo_queue.queue[0].cost_combo;
            // Recompute per-type costs for the merged histogram
            costs[best_idx1] = compute_histogram_cost(&histos[best_idx1]);

            active[merge_idx2] = false;

            // Remap all tiles pointing to merge_idx2 → best_idx1
            for m in mapping.iter_mut() {
                if *m == merge_idx2 {
                    *m = best_idx1;
                }
            }

            // Remove merge_idx2 from compact array (swap with last, like libwebp)
            if let Some(pos) = compact.iter().position(|&x| x == merge_idx2) {
                compact.swap(pos, compact_size - 1);
                compact_size -= 1;
                compact.truncate(compact_size);
            }

            // Update queue: remove pairs involving the merged indices,
            // re-evaluate pairs that reference either idx
            let mut j = 0usize;
            while j < histo_queue.queue.len() {
                let p = &histo_queue.queue[j];
                let is_idx1_best = p.idx1 == best_idx1 || p.idx1 == merge_idx2;
                let is_idx2_best = p.idx2 == best_idx1 || p.idx2 == merge_idx2;

                if is_idx1_best && is_idx2_best {
                    // This is the pair we just merged (or a duplicate)
                    histo_queue.pop_at(j);
                    continue;
                }

                if is_idx1_best || is_idx2_best {
                    // Fix index references
                    HistoQueue::fix_pair_idx(
                        &mut histo_queue.queue[j],
                        merge_idx2,
                        best_idx1,
                    );
                    // Re-evaluate cost
                    if !HistoQueue::update_pair(&histos, &costs, &mut histo_queue.queue[j]) {
                        histo_queue.pop_at(j);
                        continue;
                    }
                }

                // Also fix if either index matches the "swapped-in last" index
                // (In libwebp, HistogramSetRemoveHistogram moves the last histo
                // into the removed slot. We don't compact our histos array, so we
                // only need to fix the merge_idx2 reference.)

                // Update head if this pair is now better
                histo_queue.update_head(j);
                j += 1;
            }

            tries_with_no_success = 0;
        }

        do_greedy = compact_size <= target_size;
    }

    // Phase 3: Greedy combining using priority queue (matching libwebp's
    // HistogramCombineGreedy). O(n^2) initial pair evaluation + O(n) per merge.
    // Only runs when stochastic brought count <= target_size (matching libwebp).
    let num_active = active.iter().filter(|&&a| a).count();

    if do_greedy && num_active > 1 {
        let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let active_n = active_indices.len();

        // Initialize priority queue with all pairs
        let mut histo_queue = HistoQueue::new(active_n * active_n);

        for ai in 0..active_n {
            for aj in (ai + 1)..active_n {
                histo_queue.push_greedy(
                    &histos,
                    &costs,
                    active_indices[ai],
                    active_indices[aj],
                );
            }
        }

        // Greedily merge the best pair until no beneficial pair remains
        while !histo_queue.queue.is_empty() {
            let best_idx1 = histo_queue.queue[0].idx1;
            let best_idx2 = histo_queue.queue[0].idx2;

            // Merge idx2 into idx1
            let j_histo = histos[best_idx2].clone();
            histos[best_idx1].add(&j_histo);
            costs[best_idx1] = compute_histogram_cost(&histos[best_idx1]);
            active[best_idx2] = false;

            for m in mapping.iter_mut() {
                if *m == best_idx2 {
                    *m = best_idx1;
                }
            }

            // Remove stale pairs and update pairs involving merged indices
            let mut j = 0usize;
            while j < histo_queue.queue.len() {
                let p = &histo_queue.queue[j];
                if p.idx1 == best_idx1 || p.idx2 == best_idx1
                    || p.idx1 == best_idx2 || p.idx2 == best_idx2
                {
                    histo_queue.pop_at(j);
                } else {
                    // No need to fix indices — we don't compact the array
                    histo_queue.update_head(j);
                    j += 1;
                }
            }

            // Add new pairs involving the merged histogram (best_idx1)
            for i in 0..n {
                if !active[i] || i == best_idx1 {
                    continue;
                }
                histo_queue.push_greedy(&histos, &costs, best_idx1, i);
            }
        }
    }

    // Phase 4: Remap - find best cluster for each original tile histogram
    // Matches libwebp's HistogramRemap with progressive threshold tightening.
    let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();

    if active_indices.len() > 1 {
        // Cache tile histogram costs to avoid recomputation
        let tile_costs: Vec<HistogramCosts> =
            tile_histos.iter().map(compute_histogram_cost).collect();

        for tile_idx in 0..n {
            let mut best_cluster = mapping[tile_idx];
            let mut best_bits = i64::MAX;

            for &cluster_idx in &active_indices {
                // Threshold: cluster.cost + best_bits (matching libwebp's
                // HistogramAddThresh which does SaturateAdd(a->bit_cost, &cost_threshold))
                let cost_threshold = if best_bits == i64::MAX {
                    u64::MAX
                } else {
                    let thresh = costs[cluster_idx].total as i64 + best_bits;
                    if thresh <= 0 {
                        continue;
                    }
                    thresh as u64
                };

                if let Some(combined_cost) = get_combined_histogram_cost(
                    &histos[cluster_idx],
                    &costs[cluster_idx],
                    &tile_histos[tile_idx],
                    &tile_costs[tile_idx],
                    cost_threshold,
                ) {
                    // Cost = C(cluster + tile) - C(cluster), matching HistogramAddThresh
                    let cost = combined_cost as i64 - costs[cluster_idx].total as i64;
                    if cost < best_bits {
                        best_bits = cost;
                        best_cluster = cluster_idx;
                    }
                }
            }

            mapping[tile_idx] = best_cluster;
        }
    } else if active_indices.len() == 1 {
        // Single cluster: all tiles map to it
        for m in mapping.iter_mut() {
            *m = active_indices[0];
        }
    }

    // Phase 5: Rebuild final histograms from remapped tiles
    let mut final_histos: Vec<Histogram> = active_indices
        .iter()
        .map(|_| Histogram::new(cache_bits))
        .collect();

    // Create index mapping from cluster_idx → final histogram index
    let mut cluster_to_final: Vec<Option<u16>> = vec![None; n];
    for (final_idx, &cluster_idx) in active_indices.iter().enumerate() {
        cluster_to_final[cluster_idx] = Some(final_idx as u16);
    }

    // Build final symbols and accumulate histograms
    let mut symbols: Vec<u16> = Vec::with_capacity(n);
    for tile_idx in 0..n {
        let cluster = mapping[tile_idx];
        let final_idx = cluster_to_final[cluster].unwrap_or(0);
        symbols.push(final_idx);
        final_histos[final_idx as usize].add(&tile_histos[tile_idx]);
    }

    (final_histos, symbols)
}

#[inline]
fn div_round_i64(num: i64, den: i64) -> i64 {
    if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    }
}

#[inline]
fn div_round_u64(num: u64, den: u64) -> u64 {
    (num + den / 2) / den
}

/// Build meta-Huffman info from backward references.
/// Returns info for spatially-varying Huffman codes.
pub fn build_meta_huffman(
    refs: &BackwardRefs,
    width: usize,
    height: usize,
    histo_bits: u8,
    cache_bits: u8,
    quality: u8,
) -> MetaHuffmanInfo {
    let histo_bits = histo_bits.clamp(2, 8);

    // Build per-tile histograms from backward reference tokens
    let tile_histos = build_tile_histograms(refs, width, height, histo_bits, cache_bits);

    // Cluster histograms
    let (histograms, symbols) = cluster_histograms(&tile_histos, quality, cache_bits);

    // Compute final costs
    let costs = histograms.iter().map(compute_histogram_cost).collect();

    MetaHuffmanInfo {
        histo_bits,
        num_histograms: histograms.len(),
        histogram_symbols: symbols,
        histograms,
        costs,
        image_width: width,
        image_height: height,
    }
}

/// Build a single-histogram MetaHuffmanInfo (no spatial variation).
/// Assumes distances in refs are already plane codes (from apply_2d_locality).
pub fn build_single_histogram(refs: &BackwardRefs, cache_bits: u8) -> MetaHuffmanInfo {
    let histogram = Histogram::from_refs(refs, cache_bits);
    let costs = vec![compute_histogram_cost(&histogram)];

    MetaHuffmanInfo {
        histo_bits: 0,
        num_histograms: 1,
        histogram_symbols: Vec::new(),
        histograms: vec![histogram],
        costs,
        image_width: 0,
        image_height: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::make_argb;
    use super::*;

    #[test]
    fn test_single_histogram() {
        let mut refs = BackwardRefs::new();
        refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        let info = build_single_histogram(&refs, 0);
        assert_eq!(info.num_histograms, 1);
        assert!(info.histogram_symbols.is_empty());
    }

    #[test]
    fn test_build_tile_histograms() {
        // 4x4 image with bits=2 → tile size 4x4, so 1 tile
        let mut refs = BackwardRefs::new();
        for _ in 0..16 {
            refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        }
        let histos = build_tile_histograms(&refs, 4, 4, 2, 0);
        assert_eq!(histos.len(), 1); // 4/4 = 1 tile for bits=2
    }

    #[test]
    fn test_cluster_identical() {
        // Two identical histograms should merge into one
        let mut h1 = Histogram::new(0);
        let mut h2 = Histogram::new(0);
        for _ in 0..100 {
            h1.add_literal(make_argb(255, 128, 64, 32));
            h2.add_literal(make_argb(255, 128, 64, 32));
        }
        let (groups, symbols) = cluster_histograms(&[h1, h2], 75, 0);
        assert_eq!(groups.len(), 1);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0], symbols[1]);
    }
}
