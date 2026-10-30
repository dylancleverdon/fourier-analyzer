"""Draw the app icon (a sine over a spectrum) into an .iconset folder."""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def draw(size: int) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(s * 0.22)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=(20, 22, 30, 255))
    n = 14
    for i in range(n):
        h = (0.15 + 0.7 * abs(math.sin(i * 0.9 + 0.4)) * (1 - i / (n * 1.4))) * s * 0.55
        x0 = s * 0.14 + i * (s * 0.72 / n)
        d.rectangle([x0, s * 0.82 - h, x0 + s * 0.72 / n * 0.6, s * 0.82], fill=(176, 108, 255, 200))
    pts = [(s * 0.1 + t * s * 0.8, s * 0.45 - math.sin(t * 2 * math.pi * 1.5) * s * 0.18)
           for t in [k / 200 for k in range(201)]]
    d.line(pts, fill=(255, 138, 61, 255), width=max(2, s // 28))
    return img


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        draw(base).save(out / f"icon_{base}x{base}.png")
        draw(base * 2).save(out / f"icon_{base}x{base}@2x.png")


if __name__ == "__main__":
    if sys.argv[1].lower().endswith(".ico"):         # Windows icon
        draw(256).save(sys.argv[1], sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    else:
        main(sys.argv[1])
