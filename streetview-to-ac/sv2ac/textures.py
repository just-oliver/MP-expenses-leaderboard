"""Procedural base textures and Street View atlas packing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

RNG = np.random.default_rng(7)


def _noise(h, w, scales=((1, 1.0),), seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    for scale, amp in scales:
        small = rng.random((max(1, h // scale), max(1, w // scale)))
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += amp * (np.asarray(img, float) / 255 - 0.5)
    return out


def _to_img(rgb):
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def asphalt(width_m: float, tile_m: float, px_per_m: int = 64, centre_line: bool = True) -> Image.Image:
    """Road texture: u spans the full road width, v spans `tile_m` metres."""
    w = 1024
    h = int(min(4096, max(256, tile_m * px_per_m)))
    n = _noise(h, w, ((1, 40), (4, 25), (32, 18)), seed=1)
    base = 72 + n
    rgb = np.dstack([base, base, base * 1.02])
    img = _to_img(rgb)
    d = ImageDraw.Draw(img)
    mpp_x = width_m / w  # metres per pixel across
    line_w = max(2, int(0.12 / mpp_x))
    edge = int(0.3 / mpp_x)
    for x in (edge, w - edge - line_w):
        d.rectangle([x, 0, x + line_w, h], fill=(225, 225, 220))
    if centre_line:
        cx = w // 2 - line_w // 2
        dash = h // 2
        d.rectangle([cx, 0, cx + line_w, dash * 0.6], fill=(230, 230, 225))
        d.rectangle([cx, dash, cx + line_w, dash + dash * 0.6], fill=(230, 230, 225))
    return img.filter(ImageFilter.GaussianBlur(0.7))


def grass(size=512) -> Image.Image:
    n = _noise(size, size, ((1, 50), (8, 40), (64, 30)), seed=2)
    return _to_img(np.dstack([70 + n * 0.6, 115 + n, 55 + n * 0.5]))


def ground(size=512) -> Image.Image:
    n = _noise(size, size, ((2, 30), (16, 40), (128, 40)), seed=3)
    return _to_img(np.dstack([95 + n * 0.8, 110 + n, 70 + n * 0.6]))


def barrier(size=256) -> Image.Image:
    """Armco-style barrier: u along track, v up the wall."""
    img = Image.new("RGB", (size, size), (150, 150, 150))
    d = ImageDraw.Draw(img)
    for y0, y1 in ((0.25, 0.45), (0.5, 0.7)):
        d.rectangle([0, int(y0 * size), size, int(y1 * size)], fill=(200, 202, 205))
    d.rectangle([0, int(0.75 * size), size, size], fill=(110, 110, 110))
    for x in range(0, size, size // 2):
        d.rectangle([x, 0, x + 10, size], fill=(90, 90, 92))
    return img


def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


class Atlas:
    """Packs equally-sized tiles into one or more square atlas pages."""

    def __init__(self, prefix: str, tile_w: int, tile_h: int, page_size: int = 4096, alpha: bool = False):
        self.prefix = prefix
        self.mode = "RGBA" if alpha else "RGB"
        self.tw, self.th = tile_w, tile_h
        self.cols = page_size // tile_w
        self.rows = page_size // tile_h
        self.size = page_size
        self.pages: list[Image.Image] = []
        self.count = 0

    @property
    def per_page(self):
        return self.cols * self.rows

    def add(self, img: Image.Image):
        """Returns (page_index, (u0, v0, u1, v1)) in OBJ UV convention (v=0 at bottom)."""
        page, slot = divmod(self.count, self.per_page)
        if page == len(self.pages):
            self.pages.append(Image.new(self.mode, (self.size, self.size), (128, 128, 128, 255)[: len(self.mode)]))
        r, c = divmod(slot, self.cols)
        x, y = c * self.tw, r * self.th
        self.pages[page].paste(ImageOps.fit(img.convert(self.mode), (self.tw, self.th)), (x, y))
        self.count += 1
        # inset by half a pixel to stop neighbouring tiles bleeding in at mip levels
        u0, u1 = (x + 0.5) / self.size, (x + self.tw - 0.5) / self.size
        v_top, v_bot = 1 - (y + 0.5) / self.size, 1 - (y + self.th - 0.5) / self.size
        return page, (u0, v_bot, u1, v_top)

    def page_name(self, page):
        return f"{self.prefix}_{page:02d}"

    def save(self, tex_dir: Path):
        names = []
        for i, p in enumerate(self.pages):
            name = self.page_name(i)
            save(p, tex_dir / f"{name}.png")
            names.append(name)
        return names
