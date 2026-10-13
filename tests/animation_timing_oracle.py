"""Independent libwebp decode through Pillow, invoked by just."""
import sys
from PIL import Image
with Image.open(sys.argv[1]) as image:
    assert image.n_frames == 3
    for index, duration in enumerate((33, 34, 33)):
        image.seek(index)
        image.load()
        assert image.info["duration"] == duration, image.info
        assert image.convert("RGB").getpixel((0, 0)) == (index * 40, 20, 60)
print("libwebp decoded all frames and confirmed 33/34/33 ms")
