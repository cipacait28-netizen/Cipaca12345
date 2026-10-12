"""Kinetic lyric typography in the style of the reference: a small light line plus one big
condensed word, letters animated individually (scramble / pop / flicker / slide), placed in the
free space above/below the picture or *behind* the character so she occludes it.

Cues live in lyrics.json (see lyrics.example.json):
  {"t": 0.6, "end": 3.3, "small": "under the", "big": "STARS",
   "place": "top" | "bottom" | "behind", "at": [0.75, 0.25],   # "at" = panel fraction for behind
   "fx": "scramble" | "pop" | "flicker" | "slide", "color": "red" | "cream" | "brown"}
"""
import functools, json, math, os, random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_BIG = "fonts/Anton-Regular.ttf"
FONT_SMALL = "fonts/Montserrat.ttf"
COLORS = {"red": (232, 70, 40), "cream": (250, 238, 214), "brown": (60, 32, 24), "gold": (246, 190, 80)}
GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_cues():
    path = "lyrics.json" if os.path.exists("lyrics.json") else "lyrics.example.json"
    return json.load(open(path))


def c01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    x = c01(x)
    return x * x * (3 - 2 * x)


def back(x, s=2.2):
    x = c01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


@functools.lru_cache(maxsize=64)
def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    return f


@functools.lru_cache(maxsize=4096)
def glyph(ch, path, size, weight, rgb, shadow=True):
    """RGBA float sprite (premultiplied-ready) for one character, with a soft drop shadow."""
    f = font(path, size, weight)
    pad = max(6, size // 6)
    bb = f.getbbox(ch)
    w = max(1, f.getbbox(ch + "|")[2] - f.getbbox("|")[2] + 1, bb[2]) + 2 * pad
    asc, desc = f.getmetrics()
    h = asc + desc + 2 * pad
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    rgb_img = np.empty((h, w, 3), np.float32)
    rgb_img[:] = np.array(rgb[::-1], np.float32)
    if shadow:
        sh = cv2.GaussianBlur(np.roll(np.roll(a, size // 22 + 2, 0), size // 30 + 1, 1), (0, 0), max(1.5, size / 28)) * 0.6
        out_a = np.clip(a + sh * (1 - a), 0, 1)
        col = (rgb_img * a[..., None] + np.array([10, 8, 12], np.float32) * (sh * (1 - a))[..., None]) / np.maximum(out_a, 1e-4)[..., None]
        return col, out_a, pad
    return rgb_img, a, pad


def advance(ch, path, size, weight, spacing):
    return font(path, size, weight).getlength(ch) + spacing


def blit(img, col, a, x, y, alpha=1.0, scale=1.0, clip=None):
    if alpha <= 0.01 or scale <= 0.02:
        return
    if abs(scale - 1) > 0.01:
        col = cv2.resize(col, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = a.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(img.shape[0], y0 + h), min(img.shape[1], x0 + w)
    if clip:
        ys, ye = max(ys, clip[0]), min(ye, clip[1])
    if ye <= ys or xe <= xs:
        return
    aa = a[ys - y0:ye - y0, xs - x0:xe - x0, None] * alpha
    img[ys:ye, xs:xe] = img[ys:ye, xs:xe] * (1 - aa) + col[ys - y0:ye - y0, xs - x0:xe - x0] * aa


def layout(text, path, size, weight, spacing):
    adv = [advance(c, path, size, weight, spacing) for c in text]
    return adv, sum(adv) - spacing


def fit_size(text, path, base, maxw, weight=None, spacing=0):
    s = base
    while s > 20 and layout(text, path, s, weight, spacing)[1] > maxw:
        s -= 4
    return s


def draw_line(img, text, cx, cy, path, size, weight, rgb, u, end_u, fx, spacing, frame, align="center", left=None):
    adv, total = layout(text, path, size, weight, spacing)
    x = (left if align == "left" else cx - total / 2)
    n = max(1, len(text))
    rnd = random.Random(hash(text) & 0xFFFF)
    for i, ch in enumerate(text):
        w = adv[i]
        gx = x + (w - spacing) / 2
        x += w
        if ch == " ":
            continue
        out = c01((u - (end_u - 0.32) - 0.025 * i) / 0.2)       # staggered exit
        a_exit, dy_exit = 1 - smooth(out), -18 * smooth(out)
        if fx == "type":
            k = smooth((u - 0.03 * i) / 0.16)
            alpha, dy, sc = k, 12 * (1 - k), 1.0
            show = ch
        elif fx == "pop":
            k = (u - 0.10 - 0.05 * i) / 0.30
            alpha, dy, sc = smooth(k * 2), 0, back(k)
            show = ch
        elif fx == "flicker":
            lock = 0.12 + 0.5 * rnd.random()
            on = u > lock or (u > 0.05 and ((frame // 2) * 7 + i * 13) % 5 < 2)
            alpha, dy, sc, show = (1.0 if on else 0.0), 0, 1.0, ch
        elif fx == "slide":
            k = smooth((u - 0.08 - 0.045 * i) / 0.3)
            alpha, dy, sc, show = 1.0, size * 0.9 * (1 - k), 1.0, ch
        else:  # scramble: random letters on twos, each locking to its real glyph in turn
            lock = 0.10 + 0.055 * i
            if u < lock:
                show = ch if not ch.isalpha() else GLYPHS[((frame // 2) * 31 + i * 17) % 26]
            else:
                show = ch
            alpha, dy, sc = smooth((u - 0.02 * i) / 0.08), 0, 1.0
        col, a, pad = glyph(show, path, size, weight, rgb)
        clip = None
        if fx == "slide":
            asc, desc = font(path, size, weight).getmetrics()
            clip = (int(cy - size * 0.95), int(cy + size * 0.55))
        blit(img, col, a, gx, cy + dy + dy_exit, alpha * a_exit, sc, clip)


def draw_cue(img, cue, t, frame, zone, scale=1.0):
    """zone = (x0, y0, x1, y1) box to centre the cue in (image coords)."""
    u, end_u = t - cue["t"], cue["end"] - cue["t"]
    if u < 0 or u > end_u + 0.05:
        return
    x0, y0, x1, y1 = zone
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rgb_big = COLORS[cue.get("color", "red")]
    rgb_small = COLORS[cue.get("small_color", "cream" if cue.get("color", "red") != "cream" else "gold")]
    maxw = (x1 - x0) * 0.92
    big = cue.get("big", "")
    small = cue.get("small", "")
    bsize = fit_size(big, FONT_BIG, int(150 * scale), maxw) if big else 0
    ssize = fit_size(small, FONT_SMALL, int(36 * scale), maxw, "SemiBold", 3) if small else 0
    gap = 10 * scale
    total = (ssize + gap if small else 0) + bsize * 0.95
    top = cy - total / 2
    if small:
        draw_line(img, small, cx, top + ssize * 0.55, FONT_SMALL, ssize, "SemiBold", rgb_small, u, end_u, "type", 3, frame)
    if big:
        by = top + (ssize + gap if small else 0) + bsize * 0.42
        draw_line(img, big, cx, by, FONT_BIG, bsize, None, rgb_big, u - 0.12, end_u - 0.12, cue.get("fx", "scramble"), 2, frame)
