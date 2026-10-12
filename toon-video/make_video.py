"""Beat-synced 1930s-cartoon / halftone-dot-art photo video, drawn entirely in code.

    python analyze_audio.py            # once -> out/audio.json
    python make_video.py stills 3 10 40 90   # preview PNGs at those seconds -> out/still_*.png
    python make_video.py render        # full video -> out/video.mp4

Animation is "on twos" (12 drawings/second at 24 fps) like old hand-drawn cartoons.
Everything is derived from the photos in assets/photos and the beat grid of assets/song.mp3.
"""
import functools, glob, json, math, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 720, 1280, 24
OW, OH = 810, 1440            # working canvas (slightly larger than the frame for camera moves)
CELL = 9                      # halftone cell size on the canvas
GH, GW = OH // CELL, OW // CELL
LEVELS = 24
INK = (35, 25, 30)            # BGR
PAPER = (214, 232, 246)       # BGR cream

A = json.load(open("out/audio.json"))
DUR, PER = A["duration"], A["period"]
BEATS = np.array(A["beats"])
RMS = np.array(A["rms_half_s"])
NFRAMES = int(math.ceil(DUR * FPS))

# face centre as a fraction of the (cropped) photo, whether it is a white-background cut-out, zoom
PHOTOS = {
    1: dict(file="p1.jpg", face=(0.52, 0.46), cut=False, zoom=1.35),
    2: dict(file="p2.png", face=(0.50, 0.34), cut=False, zoom=1.1),
    3: dict(file="p3.png", face=(0.62, 0.31), cut=False, zoom=1.0),
    4: dict(file="p4.png", face=(0.53, 0.41), cut=False, zoom=1.0, crop_bottom=0.15),
    5: dict(file="p5.png", face=(0.55, 0.33), cut=True, zoom=1.0),
    6: dict(file="p6.png", face=(0.56, 0.33), cut=True, zoom=1.0),
    7: dict(file="p7.png", face=(0.50, 0.31), cut=True, zoom=1.0),
    8: dict(file="p8.png", face=(0.55, 0.31), cut=True, zoom=1.0),
    9: dict(file="p9.png", face=(0.51, 0.37), cut=True, zoom=1.0),
}
ORDER = [2, 5, 1, 3, 6, 4, 7, 8, 9]
# RGB pastel pairs for the sunburst backdrops
PALETTES = [((247, 200, 70), (240, 165, 40)), ((120, 200, 190), (85, 165, 160)), ((240, 130, 120), (222, 96, 92)),
            ((176, 154, 218), (140, 120, 195)), ((150, 205, 120), (112, 178, 96)), ((245, 225, 180), (232, 200, 146))]


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# ---------------------------------------------------------------- easing / beats
def clamp01(x):
    return min(1.0, max(0.0, x))


def ease_io(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    return 1 - (1 - clamp01(x)) ** 3


def ease_back(x):
    x = clamp01(x)
    c1 = 2.2
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def pulse(t):
    i = int(np.searchsorted(BEATS, t)) - 1
    if i < 0:
        return 0.0
    amp = 1.0 if i % 4 == 0 else 0.55
    return amp * math.exp(-(t - BEATS[i]) / 0.15)


# ---------------------------------------------------------------- schedule
def build_schedule():
    t0 = float(BEATS[np.searchsorted(BEATS, 6.0)])
    t_out = float(BEATS[np.searchsorted(BEATS, 177.0)])
    scenes, bi, k = [], int(np.searchsorted(BEATS, t0)), 0
    while True:
        t = float(BEATS[bi])
        loud = RMS[int(t * 2):int(t * 2) + 8].mean() if int(t * 2) < len(RMS) else 0
        nb = 6 if loud > 0.7 else 8
        end = float(BEATS[min(bi + nb, len(BEATS) - 1)])
        if end > t_out - 2 * PER:
            end = t_out
        if loud > 0.7:
            style = ["duotone", "halftone"][k % 2]
        else:
            style = ["halftone", "sketch"][k % 2]
        scenes.append(dict(s=t, e=end, photo=ORDER[k % len(ORDER)], style=style, pal=k % len(PALETTES), k=k))
        if end >= t_out:
            break
        bi += nb
        k += 1
    return t0, t_out, scenes


T_INTRO, T_OUT, SCENES = build_schedule()

# ---------------------------------------------------------------- shared arrays
YY, XX = np.mgrid[0:OH, 0:OW].astype(np.float32)
ANG = np.arctan2(YY - OH * 0.45, XX - OW * 0.5)
_rng = np.random.default_rng(7)
PAPER_C = (np.array(PAPER, np.float32)[None, None] + cv2.GaussianBlur(_rng.normal(0, 4, (OH, OW)).astype(np.float32), (0, 0), 1.2)[..., None])


def make_tiles():
    ss = 4
    t = np.zeros((LEVELS, CELL, CELL), np.float32)
    c = (CELL * ss - 1) / 2
    yy, xx = np.mgrid[0:CELL * ss, 0:CELL * ss]
    d = np.hypot(yy - c, xx - c)
    for k in range(LEVELS):
        r = k / (LEVELS - 1) * 0.74 * CELL * ss
        m = np.clip(r - d + 0.5, 0, 1) if k else np.zeros_like(d)
        t[k] = cv2.resize(m.astype(np.float32), (CELL, CELL), interpolation=cv2.INTER_AREA)
    return t


TILES = make_tiles()
VIG = None


def vignette():
    global VIG
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2))
        VIG = np.clip(1.08 - 0.32 * d ** 2, 0.55, 1.0)[..., None]
    return VIG


GRAIN = [cv2.GaussianBlur(np.random.default_rng(100 + i).normal(0, 9, (H, W)).astype(np.float32), (0, 0), 0.7)[..., None] for i in range(6)]


# ---------------------------------------------------------------- photo prep
def smooth_noise(seed, sigma):
    r = np.random.default_rng(seed).random((OH // 4, OW // 4)).astype(np.float32)
    r = cv2.GaussianBlur(r, (0, 0), sigma / 4)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)
    return cv2.resize(r, (OW, OH))


def bg_alpha(img):
    h, w = img.shape[:2]
    m = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if (img[sy, sx] > 235).all():
            cv2.floodFill(img.copy(), m, (sx, sy), 0, (18, 18, 18), (18, 18, 18),
                          4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
    a = 1.0 - (m[1:-1, 1:-1] > 0).astype(np.float32)
    a = cv2.erode(a, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(a, (0, 0), 1.2)


@functools.lru_cache(maxsize=4)
def prep(pid):
    cfg = PHOTOS[pid]
    img = cv2.imread(f"assets/photos/{cfg['file']}", cv2.IMREAD_COLOR)
    if max(img.shape[:2]) > 2200:
        sc = 2200 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    fx, fy = cfg["face"][0] * w, cfg["face"][1] * h
    if cfg.get("crop_bottom"):
        h = int(h * (1 - cfg["crop_bottom"]))
        img = img[:h]
    asp = OW / OH
    ch = h if w / h > asp else w / asp
    cw = ch * asp
    ch, cw = ch / cfg["zoom"], cw / cfg["zoom"]
    x0 = min(max(fx - cw / 2, 0), w - cw)
    y0 = min(max(fy - 0.38 * ch, 0), h - ch)
    crop = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    img = cv2.resize(crop, (OW, OH), interpolation=cv2.INTER_AREA if crop.shape[1] > OW else cv2.INTER_CUBIC)
    face = ((fx - x0) / cw * OW, (fy - y0) / ch * OH)

    alpha = bg_alpha(img) if cfg["cut"] else np.ones((OH, OW), np.float32)

    sm = cv2.bilateralFilter(img, 9, 40, 9)
    paint = cv2.stylization(sm, sigma_s=50, sigma_r=0.35)
    hsv = cv2.cvtColor(paint, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    v = hsv[..., 2]
    hsv[..., 2] = np.clip(0.5 * v + 0.5 * np.round(v / 34) * 34 + 6, 0, 255)
    paint = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
    lines = 255 - cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] > 30).astype(np.uint8) * 255
    lines = keep[lab]
    if cfg["cut"]:
        edge = cv2.dilate(alpha, np.ones((5, 5), np.uint8)) - cv2.erode(alpha, np.ones((5, 5), np.uint8))
        lines = np.maximum(lines, (np.clip(edge * 2, 0, 1) * 255).astype(np.uint8))
    lines = cv2.GaussianBlur(cv2.dilate(lines, np.ones((2, 2), np.uint8)).astype(np.float32) / 255, (0, 0), 0.8)
    lines *= alpha if cfg["cut"] else 1.0
    lines = np.clip(lines * 1.15, 0, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    glum = cv2.resize(gray, (GW, GH), interpolation=cv2.INTER_AREA)
    galpha = cv2.resize(alpha, (GW, GH), interpolation=cv2.INTER_AREA)
    gcol = cv2.resize(paint, (GW, GH), interpolation=cv2.INTER_AREA)
    ghsv = cv2.cvtColor(gcol, cv2.COLOR_BGR2HSV).astype(np.float32)
    ghsv[..., 1] = np.clip(ghsv[..., 1] * 1.2, 0, 255)
    gcol = cv2.cvtColor(ghsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    colorfull = np.repeat(np.repeat(gcol, CELL, 0), CELL, 1)

    dist = np.hypot(XX - face[0], YY - face[1])
    cdist = dist[CELL // 2::CELL, CELL // 2::CELL][:GH, :GW]
    maxd = float(dist.max())
    return dict(img=img, alpha=alpha, paint=paint, lines=lines, glum=glum, galpha=galpha, colorfull=colorfull,
                face=face, dist=dist, cdist=cdist, maxd=maxd,
                n1=smooth_noise(pid, 60), n2=smooth_noise(pid + 50, 45),
                ncell=cv2.resize(smooth_noise(pid + 99, 8), (GW, GH)))


# ---------------------------------------------------------------- warp maps ("boil")
def make_boil(seed, amp):
    r = np.random.default_rng(seed)
    fx = cv2.GaussianBlur(r.normal(0, 1, (OH // 4, OW // 4)).astype(np.float32), (0, 0), 6)
    fy = cv2.GaussianBlur(r.normal(0, 1, (OH // 4, OW // 4)).astype(np.float32), (0, 0), 6)
    fx = cv2.resize(fx / (fx.std() + 1e-6) * amp, (OW, OH))
    fy = cv2.resize(fy / (fy.std() + 1e-6) * amp, (OW, OH))
    return XX + fx, YY + fy


BOIL_LINE = [make_boil(s, 1.8) for s in (1, 2, 3)]
BOIL_COL = [make_boil(s, 2.6) for s in (11, 12, 13)]


def remap(a, maps):
    return cv2.remap(a, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- composing helpers
def mix(a, b, m):
    a = a.astype(np.float32)
    return a + (b.astype(np.float32) - a) * m[..., None]


def backdrop(pal, t):
    c1, c2 = bgr(PALETTES[pal][0]), bgr(PALETTES[pal][1])
    st = (((ANG + t * 0.35) * 14 / (2 * math.pi)).astype(np.int32) & 1)[..., None].astype(np.float32)
    return np.array(c1, np.float32) * (1 - st) + np.array(c2, np.float32) * st


def dots_mask(size_base, grow):
    idx = np.clip((size_base * grow * (LEVELS - 1) + 0.5).astype(np.int32), 0, LEVELS - 1)
    return TILES[idx].transpose(0, 2, 1, 3).reshape(GH * CELL, GW * CELL)


def painted(P, pal, t, ph, bidx):
    bd = backdrop(pal, t)
    person = remap(P["paint"], BOIL_COL[bidx])
    out = mix(bd, person, remap(P["alpha"], BOIL_COL[bidx]))
    ln = remap(P["lines"], BOIL_LINE[bidx])
    return out * (1 - ln[..., None] * 0.92) + np.array(INK, np.float32) * ln[..., None] * 0.92


def scene_canvas(sc, lt, D, ta, bidx):
    P = prep(sc["photo"])
    ph = lt / D
    fx, fy = P["face"]
    style = sc["style"]
    if style == "halftone":
        prog = ease_io(ph / 0.36)
        grow = np.clip((prog * 1.4 - (P["cdist"] / P["maxd"] + 0.10 * P["ncell"])) / 0.3, 0, 1)
        base = 0.34 + 0.66 * (1 - P["glum"])
        m = dots_mask(base, grow)
        dotscene = mix(PAPER_C, P["colorfull"], m)
        R = ease_io((ph - 0.34) / 0.2) * P["maxd"] * 1.02
        if R <= 1:
            return dotscene, P
        pa = painted(P, sc["pal"], ta, ph, bidx)
        inside = (P["dist"] < R).astype(np.float32)
        inside = cv2.GaussianBlur(inside, (0, 0), 1.0)
        out = mix(dotscene, pa, inside)
        ring = np.clip(1 - np.abs(P["dist"] - R) / 4.0, 0, 1)
        out = out * (1 - ring[..., None]) + np.array(INK, np.float32) * ring[..., None]
        return out, P
    if style == "sketch":
        yn = YY / OH
        p1 = ease_io(ph / 0.30)
        rv = np.clip((p1 * 1.7 - (0.55 * P["n1"] + 0.45 * yn)) * 7, 0, 1)
        ln = remap(P["lines"], BOIL_LINE[bidx]) * rv
        p2 = ease_io((ph - 0.26) / 0.34)
        cm = np.clip((p2 * 1.6 - (0.7 * P["n2"] + 0.3 * P["dist"] / P["maxd"])) * 6, 0, 1)
        person = remap(P["paint"], BOIL_COL[bidx])
        bd = backdrop(sc["pal"], ta)
        col = mix(bd, person, remap(P["alpha"], BOIL_COL[bidx]))
        out = mix(PAPER_C, col, cm)
        ink = np.array((60, 50, 58), np.float32) * (1 - cm[..., None]) + np.array(INK, np.float32) * cm[..., None]
        out = out * (1 - ln[..., None] * 0.9) + ink * ln[..., None] * 0.9
        return out, P
    # duotone pop-art poster: white halftone dots on red, black ink lines
    pl = pulse(ta)
    prog = ease_io(ph / 0.25)
    grow = np.clip((prog * 1.4 - (P["cdist"] / P["maxd"] + 0.08 * P["ncell"])) / 0.3, 0, 1) * (0.82 + 0.3 * pl)
    base = np.clip(P["glum"] ** 1.3, 0, 1) * P["galpha"]
    m = dots_mask(base, grow)
    red = np.array((30, 22, 205), np.float32)
    out = red + (np.array((240, 245, 250), np.float32) - red) * m[..., None]
    ln = np.clip(remap(P["lines"], BOIL_LINE[bidx]) * 1.25, 0, 1) * ease_out(ph / 0.3)
    out = out * (1 - ln[..., None]) + np.array((15, 10, 15), np.float32) * ln[..., None]
    return out, P


# ---------------------------------------------------------------- sprites / decor
def font(size, bold=True):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@functools.lru_cache(maxsize=None)
def text_sprite(text, size, fill, shadow=(200, 40, 40)):
    f = font(size)
    pad = 40
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    bb = tmp.textbbox((0, 0), text, font=f, stroke_width=10)
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = pad - bb[0], pad - bb[1]
    d.text((x + 9, y + 9), text, font=f, fill=shadow + (255,), stroke_width=10, stroke_fill=shadow + (255,))
    d.text((x, y), text, font=f, fill=fill + (255,), stroke_width=10, stroke_fill=(30, 22, 28, 255))
    a = np.array(im)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR), a[..., 3].astype(np.float32) / 255


def paste(frame, spr, cx, cy, scale=1.0, rot=0.0):
    img, a = spr
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    ci = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    ca = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
    frame[:] = frame * (1 - ca[..., None]) + ci * ca[..., None]


def draw_note(img, x, y, s, col):
    for thick, color in ((5, INK), (0, col)):
        t = thick
        cv2.ellipse(img, (x, y), (int(13 * s) + t, int(9 * s) + t), -20, 0, 360, color, -1, cv2.LINE_AA)
        cv2.line(img, (x + int(11 * s), y - int(3 * s)), (x + int(11 * s), y - int(44 * s)), color, int(4 * s) + t, cv2.LINE_AA)
        pts = np.array([[x + int(11 * s), y - int(44 * s)], [x + int(30 * s), y - int(36 * s)], [x + int(24 * s), y - int(22 * s)],
                        [x + int(11 * s), y - int(30 * s)]], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def draw_star(img, x, y, r, col, rot):
    pts = []
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.32
        pts.append([x + rr * math.cos(a), y + rr * math.sin(a)])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, INK, 4, cv2.LINE_AA)
    cv2.fillPoly(img, [pts], col, cv2.LINE_AA)


def decor(frame, ta):
    gold, cream = (60, 190, 245), (225, 240, 250)
    p = pulse(ta)
    for i in range(7):
        ph = (i * 0.37 + ta * 0.045 * (1 + i % 3 * 0.3)) % 1.0
        x = int(W * (0.08 + 0.84 * ((i * 0.61) % 1.0)) + 22 * math.sin(ta * 1.6 + i * 2))
        y = int(H * (1.05 - ph * 1.15))
        if i % 2 == 0:
            draw_note(frame, x, y, 0.8 + 0.25 * p, gold if i % 4 == 0 else cream)
        else:
            draw_star(frame, x, y, int(18 + 8 * p), cream, ta * 1.5 + i)


# ---------------------------------------------------------------- intro / outro cards
def card(ta, t0, title_lines, sub):
    bd = backdrop(0, ta - t0)
    ang = ANG
    out = cv2.resize(bd, (W, H)).astype(np.float32)
    # double border like an old title card
    o8 = np.clip(out, 0, 255).astype(np.uint8)
    cv2.rectangle(o8, (28, 28), (W - 28, H - 28), INK, 8, cv2.LINE_AA)
    cv2.rectangle(o8, (52, 52), (W - 52, H - 52), INK, 3, cv2.LINE_AA)
    out = o8.astype(np.float32)
    lt = ta - t0
    cy = H * 0.42
    for i, (txt, size) in enumerate(title_lines):
        s = ease_back((lt - 0.25 * i - 0.2) / 0.55)
        if s > 0.01:
            paste(out, text_sprite(txt, size, (225, 240, 250)), W / 2, cy + i * 160, s * (1 + 0.06 * pulse(ta)), 3 * math.sin(ta * 2 + i))
    s = ease_back((lt - 1.2) / 0.5)
    if s > 0.01:
        paste(out, text_sprite(sub, 64, (60, 190, 245), (90, 40, 40)), W / 2, H * 0.74, s)
    decor(out, ta)
    return out


# ---------------------------------------------------------------- frame assembly
SECTIONS = [dict(kind="intro", s=0.0, e=T_INTRO)] + [dict(sc, kind="scene") for sc in SCENES] + [dict(kind="outro", s=T_OUT, e=DUR + 1)]


def section_at(t):
    for sec in SECTIONS:
        if sec["s"] <= t < sec["e"]:
            return sec
    return SECTIONS[-1]


def camera(P, ph, ta):
    p = pulse(ta)
    z = (W / OW) * (1.06 + 0.06 * ph) * (1 + 0.012 * p)
    fx, fy = P["face"]
    cx = OW / 2 + (fx - OW / 2) * 0.12
    cy = OH / 2 + (fy - OH * 0.4) * 0.08
    rot = 1.1 * math.sin(ta * math.pi / PER / 2)
    sy, sx = 1 + 0.035 * p, 1 - 0.02 * p
    M0 = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy], [0, 0, 1]], np.float64)
    R = np.vstack([cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0), [0, 0, 1]])
    S = np.array([[sx, 0, W / 2 * (1 - sx)], [0, sy, H * (1 - sy)], [0, 0, 1]], np.float64)
    return S @ R @ M0


def iris(frame, ta, s, e, center):
    r = min(ease_io((ta - s) / 0.45), ease_io((e - ta) / 0.45))
    if r >= 0.999:
        return frame
    maxr = math.hypot(W, H) * 0.62
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(r * maxr), 255, -1, cv2.LINE_AA)
    return (frame.astype(np.float32) * (mask[..., None] / 255.0)).astype(np.uint8)


_last = {}


def base_frame(f):
    """Pre-post-processing frame; computed on even frames only (animation on twos)."""
    fe = f - (f % 2)
    if _last.get("f") == fe:
        return _last["img"]
    ta = fe / FPS
    sec = section_at(ta)
    bidx = (fe // 2) % 3
    if sec["kind"] == "intro":
        out = card(ta, 0.0, [("PIYA GHAR", 98), ("AAYA", 150)], "~ Homecoming ~")
        img = np.clip(out, 0, 255).astype(np.uint8)
        center = (W / 2, H / 2)
    elif sec["kind"] == "outro":
        out = card(ta, T_OUT, [("THE", 150), ("END", 190)], "~ thank you ~")
        img = np.clip(out, 0, 255).astype(np.uint8)
        center = (W / 2, H / 2)
    else:
        D = sec["e"] - sec["s"]
        lt = ta - sec["s"]
        canvas, P = scene_canvas(sec, lt, D, ta, bidx)
        M = camera(P, lt / D, ta)
        img = cv2.warpAffine(np.clip(canvas, 0, 255).astype(np.uint8), M[:2], (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
        fc = M @ np.array([P["face"][0], P["face"][1], 1.0])
        center = (fc[0], fc[1])
        decor(img, ta)
    img = iris(img, ta, sec["s"], min(sec["e"], DUR), center)
    _last.update(f=fe, img=img)
    return img


def post(img, f):
    r = np.random.default_rng(f * 7919 + 13)
    out = img.astype(np.float32)
    out *= vignette()
    out *= np.array([1.02, 1.05, 1.10], np.float32)
    out += GRAIN[f % 6] * 0.9
    out *= 1 + 0.02 * math.sin(f * 1.7) + r.uniform(-0.012, 0.012)
    if r.random() < 0.35:
        x = int(r.integers(0, W))
        cv2.line(out, (x, 0), (x + int(r.integers(-3, 3)), H), (235, 235, 235), 1, cv2.LINE_AA)
    for _ in range(int(r.integers(0, 5))):
        cv2.circle(out, (int(r.integers(0, W)), int(r.integers(0, H))), int(r.integers(1, 3)), (25, 25, 25), -1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame(f):
    return post(base_frame(f), f)


# ---------------------------------------------------------------- drivers
def render_chunk(args):
    i, a, b = args
    path = f"out/chunk_{i:02d}.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for f in range(a, b):
        p.stdin.write(frame(f).tobytes())
        if (f - a) % 480 == 0:
            print(f"chunk {i}: {f - a}/{b - a}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def main():
    os.makedirs("out", exist_ok=True)
    if sys.argv[1] == "stills":
        for s in sys.argv[2:]:
            f = int(float(s) * FPS)
            cv2.imwrite(f"out/still_{float(s):06.2f}.png", frame(f))
        return
    n = 4
    per = (NFRAMES // n + 1) // 2 * 2
    jobs = [(i, i * per, min(NFRAMES, (i + 1) * per)) for i in range(n) if i * per < NFRAMES]
    with ProcessPoolExecutor(n) as ex:
        chunks = list(ex.map(render_chunk, jobs))
    with open("out/list.txt", "w") as fh:
        fh.writelines(f"file '{os.path.basename(c)}'\n" for c in chunks)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "out/list.txt", "-i", "assets/song.mp3",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "out/video.mp4"], check=True)
    print("done -> out/video.mp4")


if __name__ == "__main__":
    print(f"{len(SCENES)} scenes, intro until {T_INTRO:.2f}s, outro from {T_OUT:.2f}s, {NFRAMES} frames", flush=True) if sys.argv[1:] else None
    main()
