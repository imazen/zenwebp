#!/usr/bin/env python3
"""Instrument libwebp's histogram_enc.c for the #71 clustering differential.

Applies string edits (asserting every anchor) to a COPY of the vendored
source so the trace prints, on stderr, one `LHIST` line per clustering
phase boundary and per HistoQueuePush evaluation. Pairs with zenwebp's
`HISTDBG=1` output (`ZHIST` lines, `--features mode_debug`).

usage: instrument.py <histogram_enc.c> (edited in place)
"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('#include "src/enc/histogram_enc.h"\n',
    '#include "src/enc/histogram_enc.h"\n#include <stdio.h>\n')

# Phase boundaries in VP8LGetHistoImageSymbols.
rep("""  HistogramBuild(xsize, histogram_bits, refs, orig_histo);
  HistogramCopyAndAnalyze(orig_histo, image_histo);
  entropy_combine =
      (image_histo->size > entropy_combine_num_bins * 2) && (quality < 100);
""", """  HistogramBuild(xsize, histogram_bits, refs, orig_histo);
  HistogramCopyAndAnalyze(orig_histo, image_histo);
  fprintf(stderr, "LHIST phase=copy raw=%d nonempty=%d cache_bits=%d histo_bits=%d quality=%d\\n",
          image_histo_raw_size, image_histo->size, cache_bits, histogram_bits, quality);
  entropy_combine =
      (image_histo->size > entropy_combine_num_bins * 2) && (quality < 100);
""")
rep("""    HistogramCombineEntropyBin(image_histo, tmp_histo, entropy_combine_num_bins,
                               combine_cost_factor, low_effort);
  }
""", """    HistogramCombineEntropyBin(image_histo, tmp_histo, entropy_combine_num_bins,
                               combine_cost_factor, low_effort);
    fprintf(stderr, "LHIST phase=bin size=%d factor=%d\\n", image_histo->size,
            (int)combine_cost_factor);
  }
""")
rep("""    if (!HistogramCombineStochastic(image_histo, threshold_size, &do_greedy)) {
      WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
      goto Error;
    }
    if (do_greedy) {
      if (!HistogramCombineGreedy(image_histo)) {
        WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
        goto Error;
      }
    }
""", """    if (!HistogramCombineStochastic(image_histo, threshold_size, &do_greedy)) {
      WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
      goto Error;
    }
    fprintf(stderr, "LHIST phase=stochastic size=%d target=%d do_greedy=%d\\n",
            image_histo->size, threshold_size, do_greedy);
    if (do_greedy) {
      if (!HistogramCombineGreedy(image_histo)) {
        WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
        goto Error;
      }
      fprintf(stderr, "LHIST phase=greedy size=%d\\n", image_histo->size);
    }
""")
rep("""  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(orig_histo, image_histo, histogram_symbols);
""", """  // Find the optimal map from original histograms to the final ones.
  HistogramRemap(orig_histo, image_histo, histogram_symbols);
  fprintf(stderr, "LHIST phase=remap size=%d\\n", image_histo->size);
""")

# Per-evaluation trace in HistoQueuePush (stochastic + greedy).
rep("""  // Do not even consider the pair if it does not improve the entropy.
  if (!HistoQueueUpdatePair(h1, h2, threshold, &pair)) return 0;

  histo_queue->queue[histo_queue->size++] = pair;
""", """  // Do not even consider the pair if it does not improve the entropy.
  if (!HistoQueueUpdatePair(h1, h2, threshold, &pair)) {
    fprintf(stderr, "LHIST push idx(%d,%d) cost1=%lld cost2=%lld rejected thresh=%lld\\n",
            idx1, idx2, (long long)h1->bit_cost, (long long)h2->bit_cost,
            (long long)threshold);
    return 0;
  }
  fprintf(stderr, "LHIST push idx(%d,%d) cost1=%lld cost2=%lld combo=%lld diff=%lld thresh=%lld\\n",
          idx1, idx2, (long long)h1->bit_cost, (long long)h2->bit_cost,
          (long long)pair.cost_combo, (long long)pair.cost_diff, (long long)threshold);

  histo_queue->queue[histo_queue->size++] = pair;
""")

# Entropy-bin decisions.
rep("""      if (HistogramAddEval(histograms[first], histograms[idx], cur_combo,
                           bit_cost_thresh)) {
""", """      {
        const int ok = HistogramAddEval(histograms[first], histograms[idx], cur_combo,
                                        bit_cost_thresh);
        fprintf(stderr, "LHIST bin idx=%d first=%d bin=%d cost=%lld thresh=%lld eval=%d\\n",
                idx, first, bin_id, (long long)bit_cost, (long long)bit_cost_thresh, ok);
        if (!ok) { ++idx; continue; }
      }
      {
""")
rep("""        if (try_combine ||
            bin_info[bin_id].num_combine_failures >= max_combine_failures) {
          // move the (better) merged histogram to its final slot
          HistogramSwap(&cur_combo, &histograms[first]);
          HistogramSetRemoveHistogram(image_histo, idx);
        } else {
          ++bin_info[bin_id].num_combine_failures;
          ++idx;
        }
      } else {
        ++idx;
      }
""", """        fprintf(stderr, "LHIST bin-merge idx=%d first=%d try_combine=%d failures=%d\\n",
                idx, first, try_combine, bin_info[bin_id].num_combine_failures);
        if (try_combine ||
            bin_info[bin_id].num_combine_failures >= max_combine_failures) {
          // move the (better) merged histogram to its final slot
          HistogramSwap(&cur_combo, &histograms[first]);
          HistogramSetRemoveHistogram(image_histo, idx);
        } else {
          ++bin_info[bin_id].num_combine_failures;
          ++idx;
        }
      }
""")
open(p, 'w').write(s)
print("instrumented", p)

# Optional second file: vp8l_enc.c — dump the final backward-reference stream
# of every image/sub-image (`LREF` lines, pairs with zenwebp's `REFDBG=1`
# `ZREF` lines) from StoreImageToBitMask.
if len(sys.argv) > 2:
    p2 = sys.argv[2]
    s = open(p2).read()
    old2 = """  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    if ((tile_x != (x & tile_mask)) || (tile_y != (y & tile_mask))) {"""
    new2 = """  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  fprintf(stderr, "LREF IMG w=%d\\n", width);
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    if (PixOrCopyIsLiteral(v)) {
      fprintf(stderr, "LREF L %08x\\n", v->argb_or_distance);
    } else if (PixOrCopyIsCacheIdx(v)) {
      fprintf(stderr, "LREF C %u\\n", v->argb_or_distance);
    } else {
      fprintf(stderr, "LREF M d=%u l=%u\\n", v->argb_or_distance, PixOrCopyLength(v));
    }
    if ((tile_x != (x & tile_mask)) || (tile_y != (y & tile_mask))) {"""
    assert s.count(old2) == 1, s.count(old2)
    s = s.replace(old2, new2)
    if '#include <stdio.h>' not in s:
        s = s.replace('#include <assert.h>\n', '#include <assert.h>\n#include <stdio.h>\n', 1)
    open(p2, 'w').write(s)
    print("instrumented", p2)
