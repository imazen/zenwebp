//! Entropy calculation for VP8L encoding.
//!
//! Matches libwebp's PopulationCost/BitsEntropyRefine approach
//! for accurate histogram cost estimation used in clustering.

#![allow(clippy::too_many_arguments)]

use super::histogram::Histogram;

// === Instrumentation counters for tracing call counts ===
#[cfg(feature = "std")]
mod trace {
    use core::sync::atomic::{AtomicU64, Ordering};

    static GET_COMBINED_HISTOGRAM_COST_CALLS: AtomicU64 = AtomicU64::new(0);
    static GET_COMBINED_COST_FOR_TYPE_CALLS: AtomicU64 = AtomicU64::new(0);
    static COMBINED_ENTROPY_CALLS: AtomicU64 = AtomicU64::new(0);
    static COMBINED_ENTROPY_ELEMENTS: AtomicU64 = AtomicU64::new(0);
    static POPULATION_COST_CALLS: AtomicU64 = AtomicU64::new(0);
    static COMPUTE_HISTOGRAM_COST_CALLS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn inc_get_combined_histogram_cost() {
        GET_COMBINED_HISTOGRAM_COST_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_get_combined_cost_for_type() {
        GET_COMBINED_COST_FOR_TYPE_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_combined_entropy(elements: u64) {
        COMBINED_ENTROPY_CALLS.fetch_add(1, Ordering::Relaxed);
        COMBINED_ENTROPY_ELEMENTS.fetch_add(elements, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_population_cost() {
        POPULATION_COST_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn inc_compute_histogram_cost() {
        COMPUTE_HISTOGRAM_COST_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn print_and_reset() {
        eprintln!("=== Entropy Call Stats ===");
        eprintln!(
            "get_combined_histogram_cost: {}",
            GET_COMBINED_HISTOGRAM_COST_CALLS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "get_combined_cost_for_type: {}",
            GET_COMBINED_COST_FOR_TYPE_CALLS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "combined_entropy (full calc): {}",
            COMBINED_ENTROPY_CALLS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "combined_entropy total elements: {}",
            COMBINED_ENTROPY_ELEMENTS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "population_cost: {}",
            POPULATION_COST_CALLS.swap(0, Ordering::Relaxed)
        );
        eprintln!(
            "compute_histogram_cost: {}",
            COMPUTE_HISTOGRAM_COST_CALLS.swap(0, Ordering::Relaxed)
        );
    }
}

#[cfg(not(feature = "std"))]
mod trace {
    #[inline]
    pub fn inc_get_combined_histogram_cost() {}
    #[inline]
    pub fn inc_get_combined_cost_for_type() {}
    #[inline]
    pub fn inc_combined_entropy(_elements: u64) {}
    #[inline]
    pub fn inc_population_cost() {}
    #[inline]
    pub fn inc_compute_histogram_cost() {}
    pub fn print_and_reset() {}
}

/// Fixed-point precision for entropy calculations (matching libwebp).
const LOG_2_PRECISION_BITS: u32 = 23;

/// Constants matching libwebp's lossless_common.h.
const LOG_2_RECIPROCAL_FIXED: u64 = 12102203;
const APPROX_LOG_WITH_CORRECTION_MAX: u32 = 65536;

/// Precomputed v * log2(v) * (1 << 23) for v in 0..256 (from libwebp kSLog2Table).
#[rustfmt::skip]
const K_SLOG2_TABLE: [u64; 256] = [
    0, 0, 16777216, 39886887, 67108864, 97388723, 130105423, 164848600,
    201326592, 239321324, 278663526, 319217973, 360874141, 403539997,
    447137711, 491600606, 536870912, 582898099, 629637592, 677049776,
    725099212, 773754010, 822985323, 872766924, 923074875, 973887230,
    1025183802, 1076945958, 1129156447, 1181799249, 1234859451, 1288323135,
    1342177280, 1396409681, 1451008871, 1505964059, 1561265072, 1616902301,
    1672866655, 1729149526, 1785742744, 1842638548, 1899829557, 1957308741,
    2015069397, 2073105127, 2131409817, 2189977618, 2248802933, 2307880396,
    2367204859, 2426771383, 2486575220, 2546611805, 2606876748, 2667365819,
    2728074942, 2789000187, 2850137762, 2911484006, 2973035382, 3034788471,
    3096739966, 3158886666, 3221225472, 3283753383, 3346467489, 3409364969,
    3472443085, 3535699182, 3599130679, 3662735070, 3726509920, 3790452862,
    3854561593, 3918833872, 3983267519, 4047860410, 4112610476, 4177515704,
    4242574127, 4307783833, 4373142952, 4438649662, 4504302186, 4570098787,
    4636037770, 4702117480, 4768336298, 4834692645, 4901184974, 4967811774,
    5034571569, 5101462912, 5168484389, 5235634615, 5302912235, 5370315922,
    5437844376, 5505496324, 5573270518, 5641165737, 5709180782, 5777314477,
    5845565671, 5913933235, 5982416059, 6051013057, 6119723161, 6188545324,
    6257478518, 6326521733, 6395673979, 6464934282, 6534301685, 6603775250,
    6673354052, 6743037185, 6812823756, 6882712890, 6952703725, 7022795412,
    7092987118, 7163278025, 7233667324, 7304154222, 7374737939, 7445417707,
    7516192768, 7587062379, 7658025806, 7729082328, 7800231234, 7871471825,
    7942803410, 8014225311, 8085736859, 8157337394, 8229026267, 8300802839,
    8372666477, 8444616560, 8516652476, 8588773618, 8660979393, 8733269211,
    8805642493, 8878098667, 8950637170, 9023257446, 9095958945, 9168741125,
    9241603454, 9314545403, 9387566451, 9460666086, 9533843800, 9607099093,
    9680431471, 9753840445, 9827325535, 9900886263, 9974522161, 10048232765,
    10122017615, 10195876260, 10269808253, 10343813150, 10417890516,
    10492039919, 10566260934, 10640553138, 10714916116, 10789349456,
    10863852751, 10938425600, 11013067604, 11087778372, 11162557513,
    11237404645, 11312319387, 11387301364, 11462350205, 11537465541,
    11612647010, 11687894253, 11763206912, 11838584638, 11914027082,
    11989533899, 12065104750, 12140739296, 12216437206, 12292198148,
    12368021795, 12443907826, 12519855920, 12595865759, 12671937032,
    12748069427, 12824262637, 12900516358, 12976830290, 13053204134,
    13129637595, 13206130381, 13282682202, 13359292772, 13435961806,
    13512689025, 13589474149, 13666316903, 13743217014, 13820174211,
    13897188225, 13974258793, 14051385649, 14128568535, 14205807192,
    14283101363, 14360450796, 14437855239, 14515314443, 14592828162,
    14670396151, 14748018167, 14825693972, 14903423326, 14981205995,
    15059041743, 15136930339, 15214871554, 15292865160, 15370910930,
    15449008641, 15527158071, 15605359001, 15683611210, 15761914485,
    15840268608, 15918673369, 15997128556, 16075633960, 16154189373,
    16232794589, 16311449405, 16390153617, 16468907026, 16547709431,
    16626560636, 16705460444, 16784408661, 16863405094, 16942449552,
    17021541845, 17100681785,
];

/// Precomputed log2(v) * (1 << 23) for v in 0..256 (from libwebp kLog2Table).
#[rustfmt::skip]
const K_LOG2_TABLE: [u32; 256] = [
    0, 0, 8388608, 13295629, 16777216, 19477745, 21684237, 23549800,
    25165824, 26591258, 27866353, 29019816, 30072845, 31041538, 31938408,
    32773374, 33554432, 34288123, 34979866, 35634199, 36254961, 36845429,
    37408424, 37946388, 38461453, 38955489, 39430146, 39886887, 40327016,
    40751698, 41161982, 41558811, 41943040, 42315445, 42676731, 43027545,
    43368474, 43700062, 44022807, 44337167, 44643569, 44942404, 45234037,
    45518808, 45797032, 46069003, 46334996, 46595268, 46850061, 47099600,
    47344097, 47583753, 47818754, 48049279, 48275495, 48497560, 48715624,
    48929828, 49140306, 49347187, 49550590, 49750631, 49947419, 50141058,
    50331648, 50519283, 50704053, 50886044, 51065339, 51242017, 51416153,
    51587818, 51757082, 51924012, 52088670, 52251118, 52411415, 52569616,
    52725775, 52879946, 53032177, 53182516, 53331012, 53477707, 53622645,
    53765868, 53907416, 54047327, 54185640, 54322389, 54457611, 54591338,
    54723604, 54854440, 54983876, 55111943, 55238669, 55364082, 55488208,
    55611074, 55732705, 55853126, 55972361, 56090432, 56207362, 56323174,
    56437887, 56551524, 56664103, 56775645, 56886168, 56995691, 57104232,
    57211808, 57318436, 57424133, 57528914, 57632796, 57735795, 57837923,
    57939198, 58039632, 58139239, 58238033, 58336027, 58433234, 58529666,
    58625336, 58720256, 58814437, 58907891, 59000628, 59092661, 59183999,
    59274652, 59364632, 59453947, 59542609, 59630625, 59718006, 59804761,
    59890898, 59976426, 60061354, 60145690, 60229443, 60312620, 60395229,
    60477278, 60558775, 60639726, 60720140, 60800023, 60879382, 60958224,
    61036555, 61114383, 61191714, 61268554, 61344908, 61420785, 61496188,
    61571124, 61645600, 61719620, 61793189, 61866315, 61939001, 62011253,
    62083076, 62154476, 62225457, 62296024, 62366182, 62435935, 62505289,
    62574248, 62642816, 62710997, 62778797, 62846219, 62913267, 62979946,
    63046260, 63112212, 63177807, 63243048, 63307939, 63372484, 63436687,
    63500551, 63564080, 63627277, 63690146, 63752690, 63814912, 63876816,
    63938405, 63999682, 64060650, 64121313, 64181673, 64241734, 64301498,
    64360969, 64420148, 64479040, 64537646, 64595970, 64654014, 64711782,
    64769274, 64826495, 64883447, 64940132, 64996553, 65052711, 65108611,
    65164253, 65219641, 65274776, 65329662, 65384299, 65438691, 65492840,
    65546747, 65600416, 65653847, 65707044, 65760008, 65812741, 65865245,
    65917522, 65969575, 66021404, 66073013, 66124403, 66175575, 66226531,
    66277275, 66327806, 66378127, 66428240, 66478146, 66527847, 66577345,
    66626641, 66675737, 66724635, 66773336, 66821842, 66870154, 66918274,
    66966204, 67013944, 67061497,
];

/// Compute v * log2(v) in fixed-point (matching libwebp's VP8LFastSLog2).
/// 256-entry LUT for v < 256, CLZ + kLog2Table for v >= 256. No log() calls.
#[inline]
fn fast_slog2(v: u32) -> u64 {
    if v < 256 {
        return K_SLOG2_TABLE[v as usize];
    }
    fast_slog2_slow(v)
}

/// Public wrapper for `fast_slog2` for use by other VP8L modules.
#[inline]
pub(super) fn fast_slog2_public(v: u32) -> u64 {
    fast_slog2(v)
}

/// Extended range SLog2 for v >= 256 (matches libwebp's FastSLog2Slow_C).
/// Uses bit-shifting + table lookup + linear correction — no log() calls
/// for v < 65536 (covers virtually all histogram entries).
#[inline]
fn fast_slog2_slow(v: u32) -> u64 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let orig_v = v as u64;
        // Find how many bits to shift down to get v into 0..255 range
        let log_cnt = (31 - v.leading_zeros()) as u64 - 7;
        let y = 1u64 << log_cnt;
        let shifted = (v >> log_cnt as u32) as usize; // now in 0..255
        // Linear correction: log2(1 + d) ≈ d / ln(2)
        let correction = LOG_2_RECIPROCAL_FIXED * (orig_v & (y - 1));
        orig_v * (K_LOG2_TABLE[shifted] as u64 + (log_cnt << LOG_2_PRECISION_BITS)) + correction
    } else {
        // Fallback for very large values (rare in practice)
        let vf = v as f64;
        (vf * libm::log2(vf) * (1u64 << LOG_2_PRECISION_BITS) as f64 + 0.5) as u64
    }
}

/// RLE streak statistics matching libwebp's VP8LStreaks.
#[derive(Debug, Default, Clone)]
struct Streaks {
    /// counts[0] = # of zero streaks > 3, counts[1] = # of nonzero streaks > 3
    counts: [u32; 2],
    /// streaks[is_nonzero][is_long]: total length of streaks
    /// [0][0] = zero streak length ≤3, [0][1] = zero streak length >3
    /// [1][0] = nonzero streak length ≤3, [1][1] = nonzero streak length >3
    streaks: [[u32; 2]; 2],
}

/// Bit entropy result matching libwebp's VP8LBitEntropy.
#[derive(Debug, Default, Clone)]
struct BitEntropy {
    entropy: u64,
    sum: u32,
    nonzeros: u32,
    max_val: u32,
    nonzero_code: u16,
}

/// Process a run-length streak (matches libwebp's GetEntropyUnrefinedHelper).
#[inline(always)]
fn entropy_unrefined_helper(
    val: u32,
    i: usize,
    val_prev: &mut u32,
    i_prev: &mut usize,
    bit_entropy: &mut BitEntropy,
    stats: &mut Streaks,
) {
    let streak = (i - *i_prev) as u32;

    if *val_prev != 0 {
        bit_entropy.sum += *val_prev * streak;
        bit_entropy.nonzeros += streak;
        bit_entropy.nonzero_code = *i_prev as u16;
        bit_entropy.entropy += fast_slog2(*val_prev) * streak as u64;
        if bit_entropy.max_val < *val_prev {
            bit_entropy.max_val = *val_prev;
        }
    }

    let is_nonzero = (*val_prev != 0) as usize;
    let is_long = (streak > 3) as usize;
    stats.counts[is_nonzero] += is_long as u32;
    stats.streaks[is_nonzero][is_long] += streak;

    *val_prev = val;
    *i_prev = i;
}

/// Calculate entropy + streak stats for a single distribution (matches VP8LGetEntropyUnrefined).
///
/// Simple loop matching libwebp's GetEntropyUnrefined_C exactly.
/// Uses iterator to eliminate bounds checks.
fn get_entropy_unrefined(x: &[u32]) -> (BitEntropy, Streaks) {
    let mut bit_entropy = BitEntropy::default();
    let mut stats = Streaks::default();

    if x.is_empty() {
        return (bit_entropy, stats);
    }

    let len = x.len();
    let mut i_prev = 0usize;
    let mut x_prev = x[0];

    for (i, &xv) in x.iter().enumerate().skip(1) {
        if xv != x_prev {
            entropy_unrefined_helper(
                xv,
                i,
                &mut x_prev,
                &mut i_prev,
                &mut bit_entropy,
                &mut stats,
            );
        }
    }

    entropy_unrefined_helper(
        0,
        len,
        &mut x_prev,
        &mut i_prev,
        &mut bit_entropy,
        &mut stats,
    );

    bit_entropy.entropy = fast_slog2(bit_entropy.sum).saturating_sub(bit_entropy.entropy);

    (bit_entropy, stats)
}

/// Calculate combined entropy for TWO distributions without merging them.
/// Matches libwebp's GetCombinedEntropyUnrefined_C.
///
/// Optimized with batch zero-skipping: for contiguous zero regions in both x
/// and y, skips chunks of 8 elements at a time using a single OR-reduction test.
/// Since histograms are typically 90%+ sparse, this avoids most per-element work.
fn get_combined_entropy_unrefined(x: &[u32], y: &[u32]) -> (BitEntropy, Streaks) {
    debug_assert_eq!(x.len(), y.len());
    let mut bit_entropy = BitEntropy::default();
    let mut stats = Streaks::default();

    if x.is_empty() {
        return (bit_entropy, stats);
    }

    let len = x.len().min(y.len());
    let mut i_prev = 0usize;
    let mut xy_prev = x[0] + y[0];

    // Process in chunks of 8 with batch zero-skipping.
    // When all 16 values (8 from x + 8 from y) OR to zero, the chunk is entirely
    // zero. If we're already in a zero streak, we can skip the entire chunk.
    let x = &x[..len];
    let y = &y[..len];
    let chunks_end = 1 + ((len.saturating_sub(2)) / 8) * 8;
    let mut i = 1usize;

    while i < chunks_end {
        let or_all = x[i]
            | x[i + 1]
            | x[i + 2]
            | x[i + 3]
            | x[i + 4]
            | x[i + 5]
            | x[i + 6]
            | x[i + 7]
            | y[i]
            | y[i + 1]
            | y[i + 2]
            | y[i + 3]
            | y[i + 4]
            | y[i + 5]
            | y[i + 6]
            | y[i + 7];

        if or_all == 0 {
            if xy_prev == 0 {
                // Zero streak continues - skip entire chunk
                i += 8;
                continue;
            }
            // Transition from nonzero to zero at first element
            entropy_unrefined_helper(
                0,
                i,
                &mut xy_prev,
                &mut i_prev,
                &mut bit_entropy,
                &mut stats,
            );
            i += 8;
            continue;
        }

        // Chunk has nonzero elements - process individually
        let end = i + 8;
        while i < end {
            let xy = x[i] + y[i];
            if xy != xy_prev {
                entropy_unrefined_helper(
                    xy,
                    i,
                    &mut xy_prev,
                    &mut i_prev,
                    &mut bit_entropy,
                    &mut stats,
                );
            }
            i += 1;
        }
    }

    // Tail: remaining elements
    while i < len {
        let xy = x[i] + y[i];
        if xy != xy_prev {
            entropy_unrefined_helper(
                xy,
                i,
                &mut xy_prev,
                &mut i_prev,
                &mut bit_entropy,
                &mut stats,
            );
        }
        i += 1;
    }

    // Final flush
    entropy_unrefined_helper(
        0,
        len,
        &mut xy_prev,
        &mut i_prev,
        &mut bit_entropy,
        &mut stats,
    );

    bit_entropy.entropy = fast_slog2(bit_entropy.sum).saturating_sub(bit_entropy.entropy);

    (bit_entropy, stats)
}

/// Fast combined Shannon entropy for histogram clustering comparison.
/// Matches libwebp's CombinedShannonEntropy_C — a tight loop that computes
/// sum(slog2(x) + slog2(x+y)) for nonzero entries, then subtracts from
/// slog2(sum_x) + slog2(sum_xy). Much faster than get_combined_entropy_unrefined
/// because it skips streak tracking and Huffman cost estimation.
///
/// Returns the combined entropy in fixed-point (scaled by 1 << LOG_2_PRECISION_BITS).
#[inline]
fn combined_shannon_entropy(x: &[u32], y: &[u32]) -> u64 {
    debug_assert_eq!(x.len(), y.len());
    let mut retval = 0u64;
    let mut sum_x = 0u32;
    let mut sum_xy = 0u32;

    for i in 0..x.len() {
        let xi = x[i];
        if xi != 0 {
            let xy = xi + y[i];
            sum_x += xi;
            retval += fast_slog2(xi);
            sum_xy += xy;
            retval += fast_slog2(xy);
        } else if y[i] != 0 {
            sum_xy += y[i];
            retval += fast_slog2(y[i]);
        }
    }

    fast_slog2(sum_x) + fast_slog2(sum_xy) - retval
}

/// Refine entropy using perceptual adjustments (matches BitsEntropyRefine).
#[inline]
fn bits_entropy_refine(entropy: &BitEntropy) -> u64 {
    if entropy.nonzeros < 5 {
        if entropy.nonzeros <= 1 {
            return 0; // Trivial: 0 or 1 symbol
        }
        if entropy.nonzeros == 2 {
            // Two symbols: mix 99% entropy + 1% bias
            return div_round(
                99 * ((entropy.sum as u64) << LOG_2_PRECISION_BITS) + entropy.entropy,
                100,
            );
        }
        let mix = if entropy.nonzeros == 3 {
            950u64
        } else {
            700u64
        };
        let min_limit = (2 * entropy.sum as u64 - entropy.max_val as u64) << LOG_2_PRECISION_BITS;
        let min_limit = div_round(mix * min_limit + (1000 - mix) * entropy.entropy, 1000);
        return if entropy.entropy < min_limit {
            min_limit
        } else {
            entropy.entropy
        };
    }

    // >= 5 symbols: use mix of 627/1000
    let mix = 627u64;
    let min_limit = (2 * entropy.sum as u64 - entropy.max_val as u64) << LOG_2_PRECISION_BITS;
    let min_limit = div_round(mix * min_limit + (1000 - mix) * entropy.entropy, 1000);
    if entropy.entropy < min_limit {
        min_limit
    } else {
        entropy.entropy
    }
}

/// Initial Huffman tree encoding cost (matches libwebp's InitialHuffmanCost).
fn initial_huffman_cost() -> u64 {
    const CODE_LENGTH_CODES: u64 = 19;
    let base = CODE_LENGTH_CODES * 3; // 57 bits for the code length codes
    (base << LOG_2_PRECISION_BITS) - div_round(91u64 << LOG_2_PRECISION_BITS, 10)
}

/// Huffman tree cost from RLE stats (matches libwebp's FinalHuffmanCost).
#[inline]
fn final_huffman_cost(stats: &Streaks) -> u64 {
    let mut retval = initial_huffman_cost();

    let retval_extra: u32 = stats.counts[0] * 1600
        + 240 * stats.streaks[0][1]
        + stats.counts[1] * 2640
        + 720 * stats.streaks[1][1]
        + 1840 * stats.streaks[0][0]
        + 3360 * stats.streaks[1][0];

    retval += (retval_extra as u64) << (LOG_2_PRECISION_BITS - 10);
    retval
}

/// Rounding division matching libwebp's DivRound.
#[inline]
fn div_round(num: u64, den: u64) -> u64 {
    (num + den / 2) / den
}

/// Bits entropy for a distribution (entropy only, no Huffman tree cost).
/// Matches libwebp's VP8LBitsEntropy.
pub fn vp8l_bits_entropy(population: &[u32]) -> u64 {
    let (bit_entropy, _stats) = get_entropy_unrefined(population);
    bits_entropy_refine(&bit_entropy)
}

/// Cost to encode a single population (entropy + Huffman tree cost).
/// Returns (cost, trivial_sym, is_used).
pub fn population_cost(population: &[u32]) -> (u64, Option<u16>, bool) {
    trace::inc_population_cost();
    let (bit_entropy, stats) = get_entropy_unrefined(population);

    let trivial_sym = if bit_entropy.nonzeros == 1 {
        Some(bit_entropy.nonzero_code)
    } else {
        None
    };

    let is_used = stats.streaks[1][0] != 0 || stats.streaks[1][1] != 0;
    let cost = bits_entropy_refine(&bit_entropy) + final_huffman_cost(&stats);

    (cost, trivial_sym, is_used)
}

/// Cost to encode two populations combined (without modifying them).
#[inline]
fn combined_entropy(x: &[u32], y: &[u32]) -> u64 {
    trace::inc_combined_entropy(x.len() as u64);
    let (bit_entropy, stats) = get_combined_entropy_unrefined(x, y);
    bits_entropy_refine(&bit_entropy) + final_huffman_cost(&stats)
}

/// Per-type cost info cached in histogram.
#[derive(Debug, Clone, Default)]
pub struct HistogramCosts {
    /// Total bit cost across all 5 types.
    pub total: u64,
    /// Per-type costs [literal, red, blue, alpha, distance].
    pub per_type: [u64; 5],
    /// Trivial symbol per type (Some(sym) if only one symbol).
    pub trivial_sym: [Option<u16>; 5],
    /// Whether each type has any nonzero entries.
    pub is_used: [bool; 5],
}

/// Extra bits cost for prefix-coded values (matching libwebp's VP8LExtraCost).
///
/// For prefix codes, codes 4..5 need 1 extra bit, 6..7 need 2, etc.
/// This returns the total extra bits weighted by population counts.
fn extra_cost(population: &[u32]) -> u64 {
    let length = population.len();
    if length < 6 {
        return 0;
    }
    // Codes 4,5 have 1 extra bit; 6,7 have 2; 8,9 have 3; etc.
    let mut cost = population[4] as u64 + population[5] as u64;
    let half_len = length / 2;
    for i in 2..half_len.saturating_sub(1) {
        cost += i as u64 * (population[2 * i + 2] as u64 + population[2 * i + 3] as u64);
    }
    cost
}

/// Compute full histogram cost (all 5 types) for clustering.
///
/// This is the per-type PopulationCost sum used by histogram clustering.
/// Does NOT include ExtraCost (that's only in estimate_histogram_bits).
pub fn compute_histogram_cost(h: &Histogram) -> HistogramCosts {
    trace::inc_compute_histogram_cost();
    let (lit_cost, lit_triv, lit_used) = population_cost(&h.literal);
    let (red_cost, red_triv, red_used) = population_cost(&h.red);
    let (blue_cost, blue_triv, blue_used) = population_cost(&h.blue);
    let (alpha_cost, alpha_triv, alpha_used) = population_cost(&h.alpha);
    let (dist_cost, dist_triv, dist_used) = population_cost(&h.distance);

    HistogramCosts {
        total: lit_cost + red_cost + blue_cost + alpha_cost + dist_cost,
        per_type: [lit_cost, red_cost, blue_cost, alpha_cost, dist_cost],
        trivial_sym: [lit_triv, red_triv, blue_triv, alpha_triv, dist_triv],
        is_used: [lit_used, red_used, blue_used, alpha_used, dist_used],
    }
}

/// Get combined entropy cost of two histograms for a single type index.
/// Fast path: handles trivial/unused cases without computation.
#[inline]
fn get_combined_cost_for_type(
    h1: &Histogram,
    h1_costs: &HistogramCosts,
    h2: &Histogram,
    h2_costs: &HistogramCosts,
    type_idx: usize,
) -> u64 {
    trace::inc_get_combined_cost_for_type();
    let h1_used = h1_costs.is_used[type_idx];
    let h2_used = h2_costs.is_used[type_idx];

    // Fast path: trivial symbol match
    let is_trivial = h1_costs.trivial_sym[type_idx].is_some()
        && h1_costs.trivial_sym[type_idx] == h2_costs.trivial_sym[type_idx];

    if is_trivial || !h1_used || !h2_used {
        return if h1_used {
            h1_costs.per_type[type_idx]
        } else {
            h2_costs.per_type[type_idx]
        };
    }

    // Full calculation: combine populations
    let (x, y) = get_populations_for_type(h1, h2, type_idx);
    combined_entropy(x, y)
}

/// Get population arrays for a given type index.
fn get_populations_for_type<'a>(
    h1: &'a Histogram,
    h2: &'a Histogram,
    type_idx: usize,
) -> (&'a [u32], &'a [u32]) {
    match type_idx {
        0 => (h1.literal.as_slice(), h2.literal.as_slice()),
        1 => (&h1.red, &h2.red),
        2 => (&h1.blue, &h2.blue),
        3 => (&h1.alpha, &h2.alpha),
        4 => (&h1.distance, &h2.distance),
        _ => unreachable!(),
    }
}

/// Evaluate merging two histograms with cost threshold (early bail-out).
/// Returns Some(combined_cost) if under threshold, None if exceeds.
pub fn get_combined_histogram_cost(
    h1: &Histogram,
    h1_costs: &HistogramCosts,
    h2: &Histogram,
    h2_costs: &HistogramCosts,
    cost_threshold: u64,
) -> Option<u64> {
    trace::inc_get_combined_histogram_cost();
    if cost_threshold == 0 {
        return None;
    }

    let mut cost = 0u64;
    for i in 0..5 {
        cost += get_combined_cost_for_type(h1, h1_costs, h2, h2_costs, i);
        if cost >= cost_threshold {
            return None; // Early bail-out
        }
    }

    Some(cost)
}

/// Estimate bit cost for a histogram (matching libwebp's VP8LHistogramEstimateBits).
///
/// Includes PopulationCost for all 5 types plus ExtraCost for prefix-coded
/// length and distance values. Used for strategy/cache comparison.
pub fn estimate_histogram_bits(h: &Histogram) -> u64 {
    use super::types::{NUM_DISTANCE_CODES, NUM_LENGTH_CODES, NUM_LITERAL_CODES};

    let costs = compute_histogram_cost(h);

    // Add extra bits for prefix-coded length and distance values
    let length_extra = extra_cost(&h.literal[NUM_LITERAL_CODES..][..NUM_LENGTH_CODES]);
    let distance_extra = extra_cost(&h.distance[..NUM_DISTANCE_CODES]);
    let extra_bits_cost = (length_extra + distance_extra) << LOG_2_PRECISION_BITS;

    (costs.total + extra_bits_cost) >> LOG_2_PRECISION_BITS
}

/// Estimate combined bit cost for merging two histograms.
pub fn estimate_combined_bits(h1: &Histogram, h2: &Histogram) -> u64 {
    let mut combined = h1.clone();
    combined.add(h2);
    estimate_histogram_bits(&combined)
}

/// Print and reset entropy call statistics.
pub fn print_entropy_stats() {
    trace::print_and_reset();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::vp8l::types::make_argb;

    #[test]
    fn test_fast_slog2() {
        // slog2(1) = 1 * log2(1) = 0
        assert_eq!(fast_slog2(0), 0);
        assert_eq!(fast_slog2(1), 0);
        // slog2(2) = 2 * log2(2) = 2 * 1 = 2 (scaled)
        let s2 = fast_slog2(2);
        let expected = 2.0 * (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((s2 as f64 - expected).abs() < expected * 0.01);
    }

    #[test]
    fn test_population_cost_trivial() {
        let mut counts = [0u32; 256];
        counts[42] = 100;
        let (cost, trivial, _) = population_cost(&counts);
        assert_eq!(trivial, Some(42));
        // Single symbol → entropy is 0, cost is only Huffman tree overhead
        assert!(cost > 0, "Tree overhead should be nonzero");
    }

    #[test]
    fn test_population_cost_two_symbols() {
        let mut counts = [0u32; 256];
        counts[0] = 50;
        counts[128] = 50;
        let (cost, trivial, is_used) = population_cost(&counts);
        assert!(trivial.is_none());
        assert!(is_used);
        assert!(cost > 0);
    }

    #[test]
    fn test_compute_histogram_cost() {
        let mut h = Histogram::new(0);
        h.add_literal(make_argb(255, 128, 64, 32));
        h.add_literal(make_argb(255, 128, 64, 32));
        let costs = compute_histogram_cost(&h);
        assert!(costs.total > 0);
        // All same pixel → each type should be trivial
        assert!(costs.trivial_sym[0].is_some()); // literal (green)
        assert!(costs.trivial_sym[1].is_some()); // red
        assert!(costs.trivial_sym[2].is_some()); // blue
        assert!(costs.trivial_sym[3].is_some()); // alpha
    }

    #[test]
    fn test_combined_cost_threshold() {
        let mut h1 = Histogram::new(0);
        let mut h2 = Histogram::new(0);
        for i in 0..100u32 {
            h1.add_literal(make_argb(255, (i % 16) as u8, (i % 8) as u8, (i % 4) as u8));
            h2.add_literal(make_argb(
                255,
                ((i + 50) % 16) as u8,
                ((i + 50) % 8) as u8,
                ((i + 50) % 4) as u8,
            ));
        }
        let c1 = compute_histogram_cost(&h1);
        let c2 = compute_histogram_cost(&h2);

        // With very high threshold, should succeed
        let result = get_combined_histogram_cost(&h1, &c1, &h2, &c2, u64::MAX);
        assert!(result.is_some());

        // With 0 threshold, should fail
        let result = get_combined_histogram_cost(&h1, &c1, &h2, &c2, 0);
        assert!(result.is_none());
    }
}
