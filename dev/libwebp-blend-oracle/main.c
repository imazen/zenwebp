// Pins zenwebp's animation alpha blend to libwebp's BlendPixelNonPremult.
// The blend is a static function, so include libwebp's own anim_decode.c
// instead of copying it. Prints an FNV-1a hash over every (src, dst) pair
// of red and alpha, sampled every STEP values per axis. G and B go through
// the same per-channel code as R.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/demux/anim_decode.c"

int main(int argc, char** argv) {
  const int step = argc > 1 ? atoi(argv[1]) : 1;
  uint64_t h = 1469598103934665603ULL;
  for (int r1 = 0; r1 < 256; r1 += step)
    for (int a1 = 0; a1 < 256; a1 += step)
      for (int r2 = 0; r2 < 256; r2 += step)
        for (int a2 = 0; a2 < 256; a2 += step) {
          const uint32_t src = (uint32_t)r1 | ((uint32_t)a1 << 24);
          const uint32_t dst = (uint32_t)r2 | ((uint32_t)a2 << 24);
          h = (h ^ BlendPixelNonPremult(src, dst)) * 1099511628211ULL;
        }
  printf("%016llx\n", (unsigned long long)h);
  return 0;
}
