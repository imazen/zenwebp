//! Meta-Huffman encoding for spatially-varying codes.
//!
//! Builds per-tile histograms from backward references, clusters them
//! using entropy-based greedy merging (matching libwebp), and produces
//! a histogram index map for spatially-varying Huffman codes.

use alloc::vec;
use alloc::vec::Vec;

use super::entropy::{
    HistogramCosts, compute_histogram_cost, costs_from_merge, get_combined_histogram_cost,
    get_combined_histogram_cost_with_detail,
};
use super::histogram::Histogram;
use super::types::{BackwardRefs, PixOrCopy, subsample_size};

// === Instrumentation counters for tracing clustering behavior ===
#[cfg(feature = "std")]
mod cluster_trace {
    use core::sync::atomic::{AtomicU64, Ordering};

    static CLUSTER_CALLS: AtomicU64 = AtomicU64::new(0);
    static INITIAL_HISTOGRAMS: AtomicU64 = AtomicU64::new(0);
    static ENTROPY_BIN_MERGES: AtomicU64 = AtomicU64::new(0);
    static STOCHASTIC_OUTER_ITERS: AtomicU64 = AtomicU64::new(0);
    static STOCHASTIC_PAIR_EVALS: AtomicU64 = AtomicU64::new(0);
    static STOCHASTIC_MERGES: AtomicU64 = AtomicU64::new(0);
    static STOCHASTIC_QUEUE_UPDATES: AtomicU64 = AtomicU64::new(0);
    static GREEDY_INITIAL_PAIRS: AtomicU64 = AtomicU64::new(0);
    static GREEDY_MERGES: AtomicU64 = AtomicU64::new(0);
    static GREEDY_NEW_PAIRS: AtomicU64 = AtomicU64::new(0);
    static REMAP_EVALS: AtomicU64 = AtomicU64::new(0);
    static POST_ENTROPY_BIN_COUNT: AtomicU64 = AtomicU64::new(0);
    static POST_STOCHASTIC_COUNT: AtomicU64 = AtomicU64::new(0);
    static POST_GREEDY_COUNT: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn inc_cluster_calls() {
        CLUSTER_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn add_initial_histograms(n: u64) {
        INITIAL_HISTOGRAMS.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_entropy_bin_merges() {
        ENTROPY_BIN_MERGES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_stochastic_outer_iters() {
        STOCHASTIC_OUTER_ITERS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn add_stochastic_pair_evals(n: u64) {
        STOCHASTIC_PAIR_EVALS.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_stochastic_merges() {
        STOCHASTIC_MERGES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_stochastic_queue_updates() {
        STOCHASTIC_QUEUE_UPDATES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn add_greedy_initial_pairs(n: u64) {
        GREEDY_INITIAL_PAIRS.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_greedy_merges() {
        GREEDY_MERGES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn add_greedy_new_pairs(n: u64) {
        GREEDY_NEW_PAIRS.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn add_remap_evals(n: u64) {
        REMAP_EVALS.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn set_post_entropy_bin_count(n: u64) {
        POST_ENTROPY_BIN_COUNT.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn set_post_stochastic_count(n: u64) {
        POST_STOCHASTIC_COUNT.fetch_add(n, Ordering::Relaxed);
    }
    #[inline]
    pub fn set_post_greedy_count(n: u64) {
        POST_GREEDY_COUNT.fetch_add(n, Ordering::Relaxed);
    }

    pub fn print_and_reset() {
        eprintln!("=== Clustering Stats ===");
        eprintln!(
            "cluster_histograms calls: {}",
            CLUSTER_CALLS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "initial histograms (total): {}",
            INITIAL_HISTOGRAMS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "entropy bin merges: {}",
            ENTROPY_BIN_MERGES.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "post-entropy-bin active: {}",
            POST_ENTROPY_BIN_COUNT.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "stochastic outer iters: {}",
            STOCHASTIC_OUTER_ITERS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "stochastic pair evals (HistoQueuePush): {}",
            STOCHASTIC_PAIR_EVALS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "stochastic merges: {}",
            STOCHASTIC_MERGES.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "stochastic queue update re-evals: {}",
            STOCHASTIC_QUEUE_UPDATES.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "post-stochastic active: {}",
            POST_STOCHASTIC_COUNT.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "greedy initial pairs eval: {}",
            GREEDY_INITIAL_PAIRS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "greedy merges: {}",
            GREEDY_MERGES.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "greedy new pair evals: {}",
            GREEDY_NEW_PAIRS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "post-greedy active: {}",
            POST_GREEDY_COUNT.swap(0, Ordering::Relaxed)
        );
        eprintln!("remap evals: {}", REMAP_EVALS.swap(0, Ordering::Relaxed));
    }
}

#[cfg(not(feature = "std"))]
mod cluster_trace {
    #[inline]
    pub fn inc_cluster_calls() {}
    #[inline]
    pub fn add_initial_histograms(_n: u64) {}
    #[inline]
    pub fn inc_entropy_bin_merges() {}
    #[inline]
    pub fn inc_stochastic_outer_iters() {}
    #[inline]
    pub fn add_stochastic_pair_evals(_n: u64) {}
    #[inline]
    pub fn inc_stochastic_merges() {}
    #[inline]
    pub fn inc_stochastic_queue_updates() {}
    #[inline]
    pub fn add_greedy_initial_pairs(_n: u64) {}
    #[inline]
    pub fn inc_greedy_merges() {}
    #[inline]
    pub fn add_greedy_new_pairs(_n: u64) {}
    #[inline]
    pub fn add_remap_evals(_n: u64) {}
    #[inline]
    pub fn set_post_entropy_bin_count(_n: u64) {}
    #[inline]
    pub fn set_post_stochastic_count(_n: u64) {}
    #[inline]
    pub fn set_post_greedy_count(_n: u64) {}
    pub fn print_and_reset() {}
}

/// Print and reset clustering statistics.
pub fn print_clustering_stats() {
    cluster_trace::print_and_reset();
}

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

        if let Some((cost_combo, per_type)) = get_combined_histogram_cost_with_detail(
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
                per_type_costs: per_type,
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

        if let Some((cost_combo, per_type)) = get_combined_histogram_cost_with_detail(
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
                per_type_costs: per_type,
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
        if let Some((cost_combo, per_type)) = get_combined_histogram_cost_with_detail(
            &histos[pair.idx1],
            &costs[pair.idx1],
            &histos[pair.idx2],
            &costs[pair.idx2],
            sum_cost as u64,
        ) {
            pair.cost_diff = cost_combo as i64 - sum_cost;
            pair.cost_combo = cost_combo;
            pair.per_type_costs = per_type;
            pair.cost_diff < 0
        } else {
            false
        }
    }
}

/// Mutable state shared across clustering phases.
struct ClusterState {
    histos: Vec<Histogram>,
    costs: Vec<HistogramCosts>,
    active: Vec<bool>,
    mapping: Vec<usize>,
}

impl ClusterState {
    fn from_tiles(tile_histos: &[Histogram]) -> Self {
        let n = tile_histos.len();
        Self {
            histos: tile_histos.to_vec(),
            costs: tile_histos.iter().map(compute_histogram_cost).collect(),
            active: vec![true; n],
            mapping: (0..n).collect(),
        }
    }

    fn count_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    fn active_indices(&self) -> Vec<usize> {
        (0..self.active.len()).filter(|&i| self.active[i]).collect()
    }

    /// Merge histogram `src` into `dst` unconditionally, without evaluating or
    /// updating costs (libwebp's low-effort HistogramCombineEntropyBin path:
    /// plain HistogramAdd + remove). Costs of surviving histograms must be
    /// recomputed after the phase, before anything reads them.
    fn merge_pair_unconditional(&mut self, dst: usize, src: usize) {
        debug_assert!(dst != src);
        let src_histo = self.histos[src].clone();
        self.histos[dst].add(&src_histo);
        self.active[src] = false;
        for m in self.mapping.iter_mut() {
            if *m == src {
                *m = dst;
            }
        }
    }

    /// Merge histogram `src` into `dst`, using the precomputed combined cost.
    /// Marks `src` inactive and remaps any tile pointing to `src` → `dst`.
    /// This is the common merge primitive used by all three combining phases.
    fn merge_pair(&mut self, dst: usize, src: usize, combined_cost: u64, per_type: [u64; 5]) {
        debug_assert!(dst != src);
        let dst_costs = self.costs[dst].clone();
        let src_costs = self.costs[src].clone();
        let src_histo = self.histos[src].clone();
        self.histos[dst].add(&src_histo);
        self.costs[dst] = costs_from_merge(combined_cost, per_type, &dst_costs, &src_costs);
        self.active[src] = false;
        for m in self.mapping.iter_mut() {
            if *m == src {
                *m = dst;
            }
        }
    }
}

/// Combine cost factor (matching libwebp's GetCombineCostFactor).
fn combine_cost_factor(num_active: usize, quality: u8) -> i64 {
    let mut factor = 16i64;
    if quality < 90 {
        if num_active > 256 {
            factor /= 2;
        }
        if num_active > 512 {
            factor /= 2;
        }
        if num_active > 1024 {
            factor /= 2;
        }
        if quality <= 50 {
            factor /= 2;
        }
    }
    factor
}

/// Target cluster count after stochastic combining.
/// Matches libwebp's q^3 scaling: t = 1 + round(q^3 * (MAX-1) / 100^3).
fn stochastic_target_size(quality: u8) -> usize {
    let q3 = (quality as u64) * (quality as u64) * (quality as u64);
    let t = 1 + div_round_u64(q3 * (MAX_HISTO_GREEDY as u64 - 1), 100u64 * 100 * 100);
    t.min(MAX_HISTO_GREEDY as u64) as usize
}

/// Phase 1: Compute per-tile entropy bin IDs from literal/red/blue costs.
fn compute_entropy_bins(state: &ClusterState, low_effort: bool) -> Vec<usize> {
    let mut lit_min = u64::MAX;
    let mut lit_max = 0u64;
    let mut red_min = u64::MAX;
    let mut red_max = 0u64;
    let mut blue_min = u64::MAX;
    let mut blue_max = 0u64;

    for (i, cost) in state.costs.iter().enumerate() {
        if !state.active[i] {
            continue;
        }
        lit_min = lit_min.min(cost.per_type[0]);
        lit_max = lit_max.max(cost.per_type[0]);
        red_min = red_min.min(cost.per_type[1]);
        red_max = red_max.max(cost.per_type[1]);
        blue_min = blue_min.min(cost.per_type[2]);
        blue_max = blue_max.max(cost.per_type[2]);
    }

    let mut bin_ids: Vec<usize> = Vec::with_capacity(state.costs.len());
    for cost in &state.costs {
        let lit_bin = get_bin_id_for_entropy(lit_min, lit_max, cost.per_type[0]);
        if low_effort {
            // libwebp GetHistoBinIndex: literal-entropy bin only (4 bins).
            bin_ids.push(lit_bin);
            continue;
        }
        let red_bin = get_bin_id_for_entropy(red_min, red_max, cost.per_type[1]);
        let blue_bin = get_bin_id_for_entropy(blue_min, blue_max, cost.per_type[2]);
        bin_ids
            .push(lit_bin * NUM_PARTITIONS * NUM_PARTITIONS + red_bin * NUM_PARTITIONS + blue_bin);
    }
    bin_ids
}

/// Phase 2: Merge histograms within each entropy bin (matching libwebp's
/// HistogramCombineEntropyBin). Uses the incoming histogram's cost for the
/// threshold, not the accumulator's.
fn entropy_bin_combine_phase(
    state: &mut ClusterState,
    quality: u8,
    bin_ids: &[usize],
    low_effort: bool,
) {
    let n = state.histos.len();
    let factor = combine_cost_factor(state.count_active(), quality);
    let mut bin_first: Vec<Option<usize>> = vec![None; BIN_SIZE];

    for i in 0..n {
        if !state.active[i] {
            continue;
        }
        let bin_id = bin_ids[i];
        let Some(first) = bin_first[bin_id] else {
            bin_first[bin_id] = Some(i);
            continue;
        };

        if low_effort {
            // libwebp low-effort: merge same-bin histograms unconditionally,
            // no cost evaluation. Costs are recomputed after the loop.
            cluster_trace::inc_entropy_bin_merges();
            state.merge_pair_unconditional(first, i);
            continue;
        }

        // libwebp uses the incoming histogram's cost for the threshold
        // (not the accumulator's), matching HistogramCombineEntropyBin.
        let bit_cost_incoming = state.costs[i].total;
        let threshold = state.costs[first].total + bit_cost_incoming;
        let cost_thresh_val =
            threshold.saturating_sub(div_round_i64(bit_cost_incoming as i64 * factor, 100) as u64);

        if let Some((combined_cost, per_type)) = get_combined_histogram_cost_with_detail(
            &state.histos[first],
            &state.costs[first],
            &state.histos[i],
            &state.costs[i],
            cost_thresh_val,
        ) {
            cluster_trace::inc_entropy_bin_merges();
            state.merge_pair(first, i, combined_cost, per_type);
        }
    }

    if low_effort {
        // Unconditional merges above did not maintain costs; recompute for the
        // survivors before anything (remap) reads them. Matches libwebp's
        // "for low_effort case, update the final cost when everything is
        // merged" loop in HistogramCombineEntropyBin.
        for i in 0..n {
            if state.active[i] {
                state.costs[i] = compute_histogram_cost(&state.histos[i]);
            }
        }
    }
}

/// Phase 2b: Stochastic combining via size-9 priority queue with Lehmer RNG
/// (matching libwebp's HistogramCombineStochastic). Returns true if the greedy
/// phase should run (i.e., active count reached target_size).
fn stochastic_combine_phase(state: &mut ClusterState, target_size: usize) -> bool {
    const HISTO_QUEUE_SIZE: usize = 9;
    let mut histo_queue = HistoQueue::new(HISTO_QUEUE_SIZE);

    // Build a compact index: position -> original index. This matches libwebp's
    // approach where histograms are stored in a compact array and
    // HistogramSetRemoveHistogram swaps in the last element.
    let mut compact: Vec<usize> = state.active_indices();
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
        cluster_trace::inc_stochastic_outer_iters();

        let best_cost = if histo_queue.queue.is_empty() {
            0i64
        } else {
            histo_queue.queue[0].cost_diff
        };
        let rand_range = (compact_size as u64) * (compact_size as u64 - 1);
        let num_tries = compact_size / 2;

        // Pick random samples (matching libwebp's inner loop)
        let mut pair_evals_this_iter = 0u64;
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

            pair_evals_this_iter += 1;
            let curr_cost =
                histo_queue.push(&state.histos, &state.costs, idx1, idx2, best_cost.min(0));
            if curr_cost < 0 && histo_queue.queue.len() >= histo_queue.max_size {
                break;
            }
        }
        cluster_trace::add_stochastic_pair_evals(pair_evals_this_iter);

        if histo_queue.queue.is_empty() {
            continue;
        }

        // Get the best pair from the queue head and merge using its precomputed costs.
        let best_idx1 = histo_queue.queue[0].idx1;
        let merge_idx2 = histo_queue.queue[0].idx2;
        let combined_cost = histo_queue.queue[0].cost_combo;
        let per_type = histo_queue.queue[0].per_type_costs;
        cluster_trace::inc_stochastic_merges();
        state.merge_pair(best_idx1, merge_idx2, combined_cost, per_type);

        // Remove merge_idx2 from compact array (swap with last, like libwebp)
        if let Some(pos) = compact.iter().position(|&x| x == merge_idx2) {
            compact.swap(pos, compact_size - 1);
            compact_size -= 1;
            compact.truncate(compact_size);
        }

        // Update queue: remove pairs involving the merged indices,
        // re-evaluate pairs that reference either idx.
        update_queue_after_stochastic_merge(
            &mut histo_queue,
            &state.histos,
            &state.costs,
            best_idx1,
            merge_idx2,
        );

        tries_with_no_success = 0;
    }

    compact_size <= target_size
}

/// After a stochastic merge, prune/repair the priority queue: drop the pair
/// that was merged, fix dangling references to merge_idx2, and re-evaluate
/// pairs that now touch the accumulator.
fn update_queue_after_stochastic_merge(
    histo_queue: &mut HistoQueue,
    histos: &[Histogram],
    costs: &[HistogramCosts],
    best_idx1: usize,
    merge_idx2: usize,
) {
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
            HistoQueue::fix_pair_idx(&mut histo_queue.queue[j], merge_idx2, best_idx1);
            // Re-evaluate cost
            cluster_trace::inc_stochastic_queue_updates();
            if !HistoQueue::update_pair(histos, costs, &mut histo_queue.queue[j]) {
                histo_queue.pop_at(j);
                continue;
            }
        }

        // (In libwebp, HistogramSetRemoveHistogram moves the last histo
        // into the removed slot. We don't compact our histos array, so we
        // only need to fix the merge_idx2 reference above.)
        histo_queue.update_head(j);
        j += 1;
    }
}

/// Phase 3: Greedy combining via O(n^2)-initialized priority queue
/// (matching libwebp's HistogramCombineGreedy). Only runs when the active count
/// is small enough (typically after stochastic has reached target_size).
fn greedy_combine_phase(state: &mut ClusterState) {
    let n = state.histos.len();
    let active_indices = state.active_indices();
    let active_n = active_indices.len();

    let mut histo_queue = HistoQueue::new(active_n * active_n);
    let greedy_init_pairs = (active_n * (active_n - 1)) / 2;
    cluster_trace::add_greedy_initial_pairs(greedy_init_pairs as u64);

    for ai in 0..active_n {
        for aj in (ai + 1)..active_n {
            histo_queue.push_greedy(
                &state.histos,
                &state.costs,
                active_indices[ai],
                active_indices[aj],
            );
        }
    }

    // Greedily merge the best pair until no beneficial pair remains.
    while !histo_queue.queue.is_empty() {
        let best_idx1 = histo_queue.queue[0].idx1;
        let best_idx2 = histo_queue.queue[0].idx2;
        let combined_cost = histo_queue.queue[0].cost_combo;
        let per_type = histo_queue.queue[0].per_type_costs;
        cluster_trace::inc_greedy_merges();
        state.merge_pair(best_idx1, best_idx2, combined_cost, per_type);

        // Remove stale pairs involving merged indices (no index fixup needed
        // because we never compact our histos array).
        let mut j = 0usize;
        while j < histo_queue.queue.len() {
            let p = &histo_queue.queue[j];
            if p.idx1 == best_idx1
                || p.idx2 == best_idx1
                || p.idx1 == best_idx2
                || p.idx2 == best_idx2
            {
                histo_queue.pop_at(j);
            } else {
                histo_queue.update_head(j);
                j += 1;
            }
        }

        // Add new pairs involving the merged accumulator (best_idx1).
        let mut new_pairs = 0u64;
        for i in 0..n {
            if !state.active[i] || i == best_idx1 {
                continue;
            }
            new_pairs += 1;
            histo_queue.push_greedy(&state.histos, &state.costs, best_idx1, i);
        }
        cluster_trace::add_greedy_new_pairs(new_pairs);
    }
}

/// Phase 4: For each original tile, search active clusters for the one that
/// gives the smallest added cost (matching libwebp's HistogramRemap with
/// progressive threshold tightening).
fn remap_tiles_to_clusters(
    state: &mut ClusterState,
    tile_histos: &[Histogram],
    active_indices: &[usize],
) {
    let n = tile_histos.len();
    if active_indices.len() == 1 {
        for m in state.mapping.iter_mut() {
            *m = active_indices[0];
        }
        return;
    }

    // Cache tile histogram costs to avoid recomputation
    let tile_costs: Vec<HistogramCosts> = tile_histos.iter().map(compute_histogram_cost).collect();

    let remap_total = n as u64 * active_indices.len() as u64;
    cluster_trace::add_remap_evals(remap_total);

    for tile_idx in 0..n {
        let mut best_cluster = state.mapping[tile_idx];
        let mut best_bits = i64::MAX;

        for &cluster_idx in active_indices {
            // Threshold: cluster.cost + best_bits (matching libwebp's
            // HistogramAddThresh which does SaturateAdd(a->bit_cost, &cost_threshold))
            let cost_threshold = if best_bits == i64::MAX {
                u64::MAX
            } else {
                let thresh = state.costs[cluster_idx].total as i64 + best_bits;
                if thresh <= 0 {
                    continue;
                }
                thresh as u64
            };

            if let Some(combined_cost) = get_combined_histogram_cost(
                &state.histos[cluster_idx],
                &state.costs[cluster_idx],
                &tile_histos[tile_idx],
                &tile_costs[tile_idx],
                cost_threshold,
            ) {
                // Cost = C(cluster + tile) - C(cluster), matching HistogramAddThresh
                let cost = combined_cost as i64 - state.costs[cluster_idx].total as i64;
                if cost < best_bits {
                    best_bits = cost;
                    best_cluster = cluster_idx;
                }
            }
        }

        state.mapping[tile_idx] = best_cluster;
    }
}

/// Phase 5: Project active clusters into a dense [0, K) index space and rebuild
/// final histograms by summing the original tile histograms.
fn build_final_histograms(
    tile_histos: &[Histogram],
    mapping: &[usize],
    active_indices: &[usize],
    cache_bits: u8,
) -> (Vec<Histogram>, Vec<u16>) {
    let n = tile_histos.len();

    // Only clusters actually referenced by the post-remap mapping get a final
    // slot. The decoder derives the Huffman group count from the entropy
    // image's max symbol (max+1); an unreferenced trailing cluster would make
    // the encoder write more tree groups than the decoder reads, shifting the
    // rest of the bitstream. (Remap can strand a cluster with zero tiles —
    // easiest to hit with the low-effort unconditional bin merging, but
    // possible for any clustering outcome.)
    let mut referenced: Vec<bool> = vec![false; n];
    for tile_idx in 0..n {
        referenced[mapping[tile_idx]] = true;
    }

    // Index mapping from cluster_idx → dense final histogram index, in
    // active_indices order (stable when every active cluster is referenced).
    let mut cluster_to_final: Vec<Option<u16>> = vec![None; n];
    let mut num_final: u16 = 0;
    for &cluster_idx in active_indices.iter() {
        if referenced[cluster_idx] {
            cluster_to_final[cluster_idx] = Some(num_final);
            num_final += 1;
        }
    }

    let mut final_histos: Vec<Histogram> =
        (0..num_final).map(|_| Histogram::new(cache_bits)).collect();

    let mut symbols: Vec<u16> = Vec::with_capacity(n);
    for tile_idx in 0..n {
        let cluster = mapping[tile_idx];
        let final_idx = cluster_to_final[cluster].unwrap_or(0);
        symbols.push(final_idx);
        final_histos[final_idx as usize].add(&tile_histos[tile_idx]);
    }

    (final_histos, symbols)
}

/// Cluster histograms using entropy binning + stochastic + greedy combining.
/// Matches libwebp's HistogramCombineEntropyBin + HistogramCombineStochastic
/// + HistogramCombineGreedy + HistogramRemap.
fn cluster_histograms(
    tile_histos: &[Histogram],
    quality: u8,
    cache_bits: u8,
    low_effort: bool,
) -> (Vec<Histogram>, Vec<u16>) {
    let n = tile_histos.len();
    cluster_trace::inc_cluster_calls();
    cluster_trace::add_initial_histograms(n as u64);
    #[cfg(feature = "std")]
    if std::env::var("ZENWEBP_TRACE").is_ok() {
        let empty_count = tile_histos
            .iter()
            .filter(|h| {
                !h.literal.iter().any(|&c| c > 0)
                    && !h.red.iter().any(|&c| c > 0)
                    && !h.blue.iter().any(|&c| c > 0)
                    && !h.alpha.iter().any(|&c| c > 0)
                    && !h.distance.iter().any(|&c| c > 0)
            })
            .count();
        eprintln!(
            "[cluster_histograms] n={} quality={} cache_bits={} empty={}",
            n, quality, cache_bits, empty_count
        );
    }

    if n <= 1 {
        if n == 1 {
            return (vec![tile_histos[0].clone()], vec![0]);
        }
        return (Vec::new(), Vec::new());
    }

    let mut state = ClusterState::from_tiles(tile_histos);

    // Phase 1+2: Entropy binning + bin-internal merging.
    // Low effort (method 0) bins on literal entropy only (4 bins vs 64) and
    // merges same-bin histograms unconditionally, matching libwebp's
    // VP8LGetHistoImageSymbols with low_effort set.
    let bin_ids = compute_entropy_bins(&state, low_effort);
    let num_active = state.count_active();
    let num_bins = if low_effort { NUM_PARTITIONS } else { BIN_SIZE }.min(num_active);
    let entropy_combine = num_active > num_bins * 2 && quality < 100;
    if entropy_combine {
        entropy_bin_combine_phase(&mut state, quality, &bin_ids, low_effort);
    }

    // libwebp: "Don't combine the histograms using stochastic and greedy
    // heuristics for low-effort compression mode" — but if entropy combining
    // did not run (too few histograms), the normal phases still apply.
    if !low_effort || !entropy_combine {
        // Phase 2b: Stochastic combining
        let target_size = stochastic_target_size(quality);
        let num_active_pre_stochastic = state.count_active();
        cluster_trace::set_post_entropy_bin_count(num_active_pre_stochastic as u64);

        let do_greedy = if num_active_pre_stochastic > target_size {
            stochastic_combine_phase(&mut state, target_size)
        } else {
            true
        };

        // Phase 3: Greedy combining (only when stochastic reached target)
        cluster_trace::set_post_stochastic_count(state.count_active() as u64);
        if do_greedy && state.count_active() > 1 {
            greedy_combine_phase(&mut state);
        }
    } else {
        cluster_trace::set_post_entropy_bin_count(state.count_active() as u64);
        cluster_trace::set_post_stochastic_count(state.count_active() as u64);
    }

    // Phase 4: Remap tiles to best cluster
    let active_indices = state.active_indices();
    cluster_trace::set_post_greedy_count(active_indices.len() as u64);
    if !active_indices.is_empty() {
        remap_tiles_to_clusters(&mut state, tile_histos, &active_indices);
    }

    // Phase 5: Build final histogram outputs
    build_final_histograms(tile_histos, &state.mapping, &active_indices, cache_bits)
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
    low_effort: bool,
) -> MetaHuffmanInfo {
    let histo_bits = histo_bits.clamp(2, 8);
    #[cfg(feature = "std")]
    if std::env::var("ZENWEBP_TRACE").is_ok() {
        let xsize = super::types::subsample_size(width as u32, histo_bits);
        let ysize = super::types::subsample_size(height as u32, histo_bits);
        eprintln!(
            "[build_meta_huffman] {}x{} histo_bits={} tiles={}x{}={} cache_bits={} quality={}",
            width,
            height,
            histo_bits,
            xsize,
            ysize,
            xsize as usize * ysize as usize,
            cache_bits,
            quality
        );
    }

    // Build per-tile histograms from backward reference tokens
    let tile_histos = build_tile_histograms(refs, width, height, histo_bits, cache_bits);

    // Cluster histograms
    let (histograms, symbols) = cluster_histograms(&tile_histos, quality, cache_bits, low_effort);

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
        let (groups, symbols) = cluster_histograms(&[h1, h2], 75, 0, false);
        assert_eq!(groups.len(), 1);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0], symbols[1]);
    }
}
