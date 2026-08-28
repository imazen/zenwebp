// #71: encode a raw RGBA file losslessly with (instrumented) libwebp.
// usage: libwebp_trace <file.rgba> <width> <height> <method> [quality=75]
// Prints "size=<bytes>" on stdout; the LHIST trace goes to stderr.
#include <stdio.h>
#include <stdlib.h>
#include "src/webp/encode.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <file.rgba> <w> <h> <method> [quality]\n", argv[0]);
    return 2;
  }
  const int w = atoi(argv[2]), h = atoi(argv[3]), method = atoi(argv[4]);
  const float quality = argc > 5 ? (float)atof(argv[5]) : 75.0f;
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 1; }
  const size_t n = (size_t)w * h * 4;
  uint8_t* rgba = malloc(n);
  if (fread(rgba, 1, n, f) != n) { fprintf(stderr, "short read\n"); return 1; }
  fclose(f);

  WebPConfig config;
  WebPConfigInit(&config);
  config.lossless = 1;
  config.exact = 1;
  config.method = method;
  config.quality = quality;
  WebPPicture pic;
  WebPPictureInit(&pic);
  pic.use_argb = 1;
  pic.width = w;
  pic.height = h;
  WebPPictureImportRGBA(&pic, rgba, w * 4);
  WebPMemoryWriter wr;
  WebPMemoryWriterInit(&wr);
  pic.writer = WebPMemoryWrite;
  pic.custom_ptr = &wr;
  if (!WebPEncode(&config, &pic)) {
    fprintf(stderr, "encode failed: %d\n", pic.error_code);
    return 1;
  }
  printf("size=%zu\n", wr.size);
  WebPMemoryWriterClear(&wr);
  WebPPictureFree(&pic);
  free(rgba);
  return 0;
}
