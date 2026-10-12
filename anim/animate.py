"""2D character animation of the collage panels -> output/humnava_toon.mp4

    python crop_panels.py && python analyze_panels.py   # assets
    python animate.py stills 1.0 5.0 ...                 # preview frames -> output/frame_*.png
    python animate.py render                             # full MP4 with the song audio

Each shot separates the character (matte) from an inpainted background plate, then drives the
character with a displacement-field rig built from the face landmarks:
  head tilt / nod around a neck pivot, hair sway, breathing, squash-and-stretch hops with
  anticipation + overshoot, and painted-in eyelid blinks. Character drawings update on twos
  (12 fps) like hand-drawn animation; camera, parallax and particles run on ones.
"""
import json, math, os, subprocess, sys

import cv2
import numpy as np

import lyrics_fx

CUES = lyrics_fx.load_cues()
STICKER = True   # white paper-cut-out outline around the character

W, H, FPS = 720, 1280, 24
AUDIO = "assets/original/song.wav"
DUR = 22.45
NF = int(DUR * FPS)
WARM = np.array([0.93, 1.0, 1.08], np.float32)   # BGR grade


# ------------------------------------------------------------------ easing
def c01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    x = c01(x)
    return x * x * (3 - 2 * x)


def back(x, s=1.9):
    x = c01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def keys(t, ks, ease=smooth):
    """Piecewise interpolation over [(time, value), ...]."""
    if t <= ks[0][0]:
        return ks[0][1]
    for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
        if t < t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
    return ks[-1][1]


def smoothstep_arr(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ acting primitives
def hop(t, t0, amp=1.0):
    """Squash-and-stretch hop: anticipation squash, stretch + lift, overshoot settle.
    Returns (sy, sx, lift_px)."""
    u = t - t0
    if u < -0.18 or u > 0.75:
        return 1.0, 1.0, 0.0
    if u < 0:                       # anticipation: squash down
        k = smooth((u + 0.18) / 0.18)
        return 1 - 0.05 * amp * k, 1 + 0.03 * amp * k, -3 * amp * k
    if u < 0.14:                    # release: stretch up
        k = smooth(u / 0.14)
        return 1 - 0.05 * amp + 0.10 * amp * k, 1 + 0.03 * amp - 0.05 * amp * k, -3 * amp + 15 * amp * k
    k = (u - 0.14) / 0.61           # fall back with a damped overshoot
    osc = math.exp(-5 * k) * math.cos(9 * k)
    return 1 + 0.05 * amp * osc, 1 - 0.02 * amp * osc, 12 * amp * osc


def blink_amount(t, times, hold=0.0):
    """Eyelid closure 0..1 for blinks starting at `times` (timed in drawings on twos)."""
    for b in times:
        u = t - b
        seq = [0.45, 1.0] + [1.0] * int(hold * 12) + [0.55, 0.15]
        i = int(math.floor(u * 12))
        if 0 <= i < len(seq):
            return seq[i]
    return 0.0


# ------------------------------------------------------------------ shots
# Panel choices follow the music: night opening -> café daydream -> longing in bed -> happy
# memory -> windy sunset -> wistful train ride -> playful wink to close.
SHOTS = [
    dict(panel="panel_01", s=0.0, e=3.6, tilt=[(0, 0), (1.1, -2.6), (2.6, -1.4), (3.6, -2.2)], blinks=[1.55],
         hair=4.5, breath=1.6, hops=[], particles="bokeh", zoom=(1.0, 1.07), drift=(-10, 0), glow=None),
    dict(panel="panel_03", s=3.6, e=6.34, sway=(2.2, 0.38), blinks=[0.9, 2.2], hair=3.0, breath=1.4, hops=[],
         particles="dust", zoom=(1.04, 1.0), drift=(8, -4), glow=((0.5, 0.06), 140, (90, 190, 255), "flicker"), hair_ymax=0.7),
    dict(panel="panel_05", s=6.34, e=8.75, tilt=[(0, 0), (0.7, 1.8), (1.6, 0.6), (2.4, 1.5)], blinks=[0.85], blink_hold=0.35,
         hair=3.0, breath=2.6, hops=[], particles="hearts", zoom=(1.0, 1.05), drift=(-6, 0), glow=None, hair_ymax=0.5),
    dict(panel="panel_09", s=8.75, e=11.73, sway=(1.6, 0.6), blinks=[1.45], hair=3.0, breath=1.2, hops=[0.42, 1.95],
         particles="bokeh", zoom=(1.06, 1.0), drift=(0, 6), glow=None, hair_ymax=0.35),
    dict(panel="panel_08", s=11.73, e=14.74, tilt=[(0, 0), (1.4, -1.5), (3.0, -0.6)], blinks=[1.75], hair=10.0, wind=1.0,
         breath=1.4, hops=[], particles="petals", zoom=(1.0, 1.06), drift=(-12, 0), glow=((0.71, 0.58), 230, (80, 170, 255), "pulse")),
    dict(panel="panel_11", s=14.74, e=18.34, tilt=[(0, 0), (1.2, 1.6), (2.4, 0.9), (3.6, 1.8)], blinks=[0.95, 2.75], hair=3.5,
         breath=1.3, hops=[], rumble=1.3, particles="streaks", zoom=(1.03, 1.0), drift=(10, 0), glow=None),
    dict(panel="panel_10", s=18.34, e=DUR, sway=(1.4, 0.7), blinks=[], hair=4.0, breath=1.2, hops=[0.55, 1.75],
         flash=0.86, particles="sparkles", zoom=(1.0, 1.08), drift=(0, -6), glow=None, iris_out=(21.35, 22.3)),
]
XFADE = 0.36


# ------------------------------------------------------------------ shot assets
class Shot:
    def __init__(self, cfg):
        self.cfg = cfg
        d = f"assets/animated/{cfg['panel']}"
        self.art = cv2.imread(f"{d}/art.png")
        self.bg = cv2.imread(f"{d}/bg.png")
        m = cv2.imread(f"{d}/matte.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        self.rig = json.load(open(f"{d}/rig.json"))["face"]
        h, w = self.art.shape[:2]
        self.h, self.w = h, w
        self.Y, self.X = np.mgrid[0:h, 0:w].astype(np.float32)
        self.matte = m
        mb = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 6)
        self.mb = mb
        f = self.rig
        oval = np.array(f["oval"], np.float32)
        self.top_y, self.chin = f["top"][1], np.array(f["chin"], np.float32)
        self.fh = self.chin[1] - self.top_y
        self.fw = oval[:, 0].max() - oval[:, 0].min()
        self.fc = oval.mean(0)
        self.pivot = np.array([self.chin[0], self.chin[1] + 0.35 * self.fh], np.float32)
        X, Y = self.X, self.Y
        # head weight: whole head above the neck, fading out across the neck/shoulders
        vert = 1 - smoothstep_arr(self.chin[1] - 0.05 * self.fh, self.chin[1] + 0.55 * self.fh, Y)
        horz = np.exp(-((X - self.fc[0]) / (1.9 * self.fw)) ** 2)
        self.w_head = vert * horz * mb
        # hair: dark pixels of the character outside the (dilated) face
        face = np.zeros((h, w), np.uint8)
        cv2.fillPoly(face, [oval.astype(np.int32)], 1)
        face = cv2.dilate(face, np.ones((int(self.fw * 0.12) | 1,) * 2, np.uint8)).astype(np.float32)
        face = cv2.GaussianBlur(face, (0, 0), 5)
        v = cv2.cvtColor(self.art, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
        dark = cv2.GaussianBlur((v < 0.33).astype(np.float32), (0, 0), 4)
        length = np.clip((Y - self.top_y + 0.2 * self.fh) / (1.6 * self.fh), 0, 1)
        ymax = self.chin[1] + cfg.get("hair_ymax", 5.0) * self.fh
        self.w_hair = dark * m * (1 - face) * length * (1 - smoothstep_arr(ymax - 0.2 * self.fh, ymax + 0.2 * self.fh, Y))
        self.w_hair = cv2.GaussianBlur(self.w_hair, (0, 0), 3)
        # body: everything above the bottom edge rides the breath / hops, fading to 0 at the cut edge
        self.hf = np.clip((h - Y) / (h - self.top_y + 1e-3), 0, 1) ** 0.9 * mb
        self.w_breath = np.clip((h - Y) / max(h - self.chin[1], 1), 0, 1) * mb
        # eyes for blinking
        self.eyes = []
        if cfg["panel"] != "panel_10":       # panel 10 already winks; its open eye is behind the camera hand
            for k in ("l_eye", "r_eye"):
                pts = np.array(f[k], np.float32)
                up, lo = pts[0:9], np.vstack([pts[8:], pts[:1]])[::-1]
                self.eyes.append((up, lo))
        self.skin = self._skin()
        # layout: scale so the panel fills the width and most of the height
        self.sc0 = max(W * 1.02 / w, 860 / h)
        self.dw, self.dh = W, int(min(h * self.sc0, 960))
        cover = max(W / w, H / h) * 1.15
        bb = cv2.resize(self.bg, None, fx=cover, fy=cover, interpolation=cv2.INTER_LINEAR)
        bb = cv2.GaussianBlur(bb, (0, 0), 22).astype(np.float32) * 0.5
        self.backdrop = bb
        self.rng = np.random.default_rng(abs(hash(cfg["panel"])) % 2 ** 32)
        self.parts = self.rng.random((40, 5)).astype(np.float32)

    def _skin(self):
        cols = []
        for up, lo in self.eyes:
            x0, x1 = int(lo[:, 0].min()), int(lo[:, 0].max())
            y0 = int(lo[:, 1].max() + 4)
            patch = self.art[y0:y0 + 10, x0:x1].reshape(-1, 3)
            if len(patch):
                cols.append(np.median(patch, 0))
        return np.mean(cols, 0) if cols else np.array([120, 160, 220], np.float32)

    # -- eyelids painted over the artwork
    def blink(self, img, amt):
        if amt <= 0.02 or not self.eyes:
            return img
        img = img.copy()
        for up, lo in self.eyes:
            xs = np.linspace(up[:, 0].min(), up[:, 0].max(), 24)
            ou = np.argsort(up[:, 0]); ol = np.argsort(lo[:, 0])
            yu = np.interp(xs, up[ou, 0], up[ou, 1]) - 2
            yl = np.interp(xs, lo[ol, 0], lo[ol, 1])
            ylid = yu + (yl - yu + 1) * amt
            poly = np.vstack([np.c_[xs, yu - 2], np.c_[xs, ylid][::-1]]).astype(np.int32)
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32)[..., None] / 255
            lid = np.empty_like(img, np.float32)
            lid[:] = self.skin * 0.93
            img = (img * (1 - mask) + lid * mask).astype(np.uint8)
            ew = xs[-1] - xs[0]
            curve = np.c_[xs, ylid + (1.5 if amt > 0.9 else 0)].astype(np.int32)
            cv2.polylines(img, [curve], False, (28, 22, 30), max(2, int(ew * 0.07)), cv2.LINE_AA)
            if amt > 0.9:   # little lashes at the outer corner
                outer = curve[-1] if xs[-1] > self.fc[0] else curve[0]
                dx = 1 if xs[-1] > self.fc[0] else -1
                cv2.line(img, tuple(outer), (int(outer[0] + dx * ew * 0.12), int(outer[1] - ew * 0.04)), (28, 22, 30), 2, cv2.LINE_AA)
        return img

    # -- one drawing of the character (on twos)
    def character(self, t):
        c = self.cfg
        if "tilt" in c:
            ang = keys(t, c["tilt"], back)
        else:
            a, f = c["sway"]
            ang = a * math.sin(2 * math.pi * f * t) * smooth(t / 0.6)
        sy = sx = 1.0
        lift = 0.0
        for h0 in c["hops"]:
            a, b, l = hop(t, h0)
            sy, sx, lift = sy * a, sx * b, lift + l
        if c.get("flash") is not None:
            # surprise take before the flash: quick anticipation lean + pop
            a, b, l = hop(t, c["flash"] - 0.05, 0.6)
            sy, sx, lift = sy * a, sx * b, lift + l
        breath = c["breath"] * math.sin(2 * math.pi * 0.32 * t)
        nod = 2.5 * math.exp(-((t * 1.0) % (60 / 99.4) / 0.12)) if c.get("nod", True) else 0.0
        X, Y = self.X, self.Y
        # head rotation about the neck pivot (inverse map: sample source at R(-a)(p - P) + P)
        th = math.radians(ang) * self.w_head
        px, py = self.pivot
        cx, cy = X - px, Y - py
        cs, sn = np.cos(-th), np.sin(-th)
        dx = (cx * cs - cy * sn + px) - X
        dy = (cx * sn + cy * cs + py) - Y
        dy += nod * 0.6 * self.w_head
        # hair sway with travelling waves + wind lean
        A = c["hair"]
        wave = np.sin(2 * math.pi * 0.42 * t + Y * 0.011 + X * 0.003) + 0.45 * np.sin(2 * math.pi * 0.95 * t + Y * 0.027)
        lean = c.get("wind", 0.0) * (0.6 + 0.4 * math.sin(2 * math.pi * 0.23 * t))
        dx += A * self.w_hair * (wave + lean)
        dy += 0.3 * A * self.w_hair * np.cos(2 * math.pi * 0.42 * t + Y * 0.011)
        # breathing + hops: stretch about the bottom edge
        dy += breath * self.w_breath
        dy += ((sy - 1) * (self.h - Y) + lift) * self.hf
        dx += -(X - self.fc[0]) * (sx - 1) * self.hf
        art = self.blink(self.art, blink_amount(t, c["blinks"], c.get("blink_hold", 0.0)))
        mx, my = X + dx, Y + dy
        ch = cv2.remap(art, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        al = cv2.remap(self.matte, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return ch, al

    # -- particles drawn in panel space (behind or in front of the character)
    def particles(self, img, t, front):
        kind = self.cfg["particles"]
        h, w = img.shape[:2]
        over = np.zeros_like(img, np.float32)
        for i, row in enumerate(self.parts):
            a, b, c, d, e = map(float, row)
            if (i % 3 == 0) != front:
                continue
            if kind == "bokeh":
                x = (a * w + 18 * math.sin(t * 0.7 + i)) % w
                y = (b * h - t * (12 + 20 * c)) % h
                r = int(6 + 22 * c)
                col = (60 + 60 * d, 170 + 50 * d, 255)
                cv2.circle(over, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
            elif kind == "dust":
                x = (a * w + 10 * math.sin(t * 0.9 + i * 2)) % w
                y = (b * h - t * (6 + 10 * c)) % h
                cv2.circle(over, (int(x), int(y)), int(1 + 3 * c), (150, 210, 255), -1, cv2.LINE_AA)
            elif kind == "hearts":
                if i % 4:
                    continue
                x = a * w + 25 * math.sin(t * 1.3 + i)
                y = h * (1.05 - ((b + t * (0.10 + 0.08 * c)) % 1.1))
                s = 10 + 12 * c
                heart(over, x, y, s, (120, 120, 245))
            elif kind == "petals":
                x = (a * w + t * (140 + 90 * c)) % (w + 60) - 30
                y = (b * h + t * (40 + 30 * d) + 20 * math.sin(t * 2 + i)) % h
                ang = t * (90 + 120 * e) + i * 40
                cv2.ellipse(over, (int(x), int(y)), (int(7 + 6 * c), int(3 + 3 * c)), ang, 0, 360, (150, 140, 250), -1, cv2.LINE_AA)
            elif kind == "streaks":
                if front:
                    continue
                y = int(b * h * 0.8)
                x = (a * w * 3 - t * (900 + 600 * c)) % (w * 3) - w
                cv2.line(over, (int(x), y), (int(x + 80 + 120 * d), y), (110, 200, 255), int(2 + 3 * c), cv2.LINE_AA)
            elif kind == "sparkles":
                u = (t * (0.5 + 0.4 * c) + a) % 1.0
                x, y = a * w, b * h
                r = 14 * math.sin(math.pi * u)
                if r > 2:
                    star(over, x, y, r, (200, 245, 255))
        k = 0.55 if front else 0.75
        over = cv2.GaussianBlur(over, (0, 0), 1.2 if front or kind != "bokeh" else 3.5)
        return np.clip(img.astype(np.float32) + over * k, 0, 255)

    def glow(self, img, t):
        g = self.cfg.get("glow")
        if not g:
            return img
        (fx, fy), r, col, mode = g
        amp = 0.55 + (0.2 * math.sin(t * 2.2) if mode == "pulse" else 0.12 * math.sin(t * 23) * math.sin(t * 7.1) + 0.1)
        y, x = self.Y, self.X
        d2 = ((x - fx * self.w) ** 2 + (y - fy * self.h) ** 2) / (r * r)
        return np.clip(img + (np.exp(-d2) * amp)[..., None] * np.array(col, np.float32) * 0.5, 0, 255)

    def frame(self, t, tc, f=0):
        """t: shot-local time (ones) ; tc: character time (twos)."""
        c = self.cfg
        rum = c.get("rumble", 0.0)
        jy = rum * math.sin(tc * 2 * math.pi * 6.0) * (0.6 + 0.4 * math.sin(tc * 3))
        ox = c["drift"][0] * (t / (c["e"] - c["s"]) - 0.5) * 0.5   # extra background parallax
        bg = cv2.warpAffine(self.bg, np.float32([[1, 0, -ox], [0, 1, -jy * 0.5]]), (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        bg = self.glow(bg.astype(np.float32), t)
        bg = self.particles(bg, t, front=False)
        tg = t + c["s"]
        for cue in CUES:   # lyric words placed *behind* the character
            if cue.get("place") == "behind" and cue["t"] - 0.1 <= tg <= cue["end"] + 0.1:
                fx_, fy_ = cue.get("at", [0.5, 0.2])
                hw, hh = cue.get("w", 0.6) / 2 * self.w, 0.16 * self.h
                cx_, cy_ = fx_ * self.w, fy_ * self.h
                lyrics_fx.draw_cue(bg, cue, tg, f, (cx_ - hw, cy_ - hh, cx_ + hw, cy_ + hh), scale=1.25 / self.sc0)
        ch, al = self.character(tc)
        if STICKER:
            k = max(5, int(self.w * 0.009)) | 1
            ring = cv2.GaussianBlur(cv2.dilate(al, np.ones((k, k), np.uint8)), (0, 0), 1.0)
            shadow = cv2.GaussianBlur(np.roll(np.roll(ring, 6, 0), 4, 1), (0, 0), 6)
            bg = bg * (1 - 0.45 * shadow[..., None])
            bg = bg * (1 - ring[..., None]) + np.array([238, 246, 252], np.float32) * ring[..., None]
        panel = bg * (1 - al[..., None]) + ch.astype(np.float32) * al[..., None]
        panel = self.particles(panel, t, front=True)
        if c.get("flash") is not None:
            u = t - c["flash"]
            if 0 <= u < 0.6:
                lx, ly = 0.33 * self.w, 0.36 * self.h
                k = math.exp(-u / 0.12)
                d2 = ((self.X - lx) ** 2 + (self.Y - ly) ** 2) / (160 ** 2)
                panel = np.clip(panel + (np.exp(-d2) * 255 * k)[..., None] + 90 * k, 0, 255)
                star(panel, lx, ly, 60 + 120 * u, (235, 250, 255))
        # camera: push in / out with drift, centered between frame and face
        p = t / (c["e"] - c["s"])
        z = self.sc0 * (c["zoom"][0] + (c["zoom"][1] - c["zoom"][0]) * smooth(p))
        cx = self.w / 2 + (self.fc[0] - self.w / 2) * 0.35 + c["drift"][0] * (p - 0.5)
        cy = self.h / 2 + (self.fc[1] - self.h / 2) * 0.25 + c["drift"][1] * (p - 0.5) + jy
        M = np.float32([[z, 0, self.dw / 2 - z * cx], [0, z, self.dh / 2 - z * cy]])
        disp = cv2.warpAffine(np.clip(panel, 0, 255).astype(np.uint8), M, (self.dw, self.dh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
        face = M @ np.array([self.fc[0], self.fc[1], 1.0])
        # scene extension above/below the panel: the panel's own colours, stretched and blurred
        y0 = (H - self.dh) // 2
        self.y0 = y0
        small = cv2.resize(disp, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        out = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) * 0.62
        fe = 110
        a = np.ones((self.dh, 1), np.float32)
        ramp = np.linspace(0, 1, fe, dtype=np.float32)[:, None]
        a[:fe] = smoothstep_arr(0, 1, ramp)
        a[-fe:] = smoothstep_arr(0, 1, ramp[::-1])
        out[y0:y0 + self.dh] = out[y0:y0 + self.dh] * (1 - a[..., None]) + disp.astype(np.float32) * a[..., None]
        return out, (face[0], face[1] + y0)


def heart(img, x, y, s, col):
    t = np.linspace(0, 2 * np.pi, 40)
    hx = 16 * np.sin(t) ** 3
    hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.c_[x + hx * s / 16, y + hy * s / 16].astype(np.int32)
    cv2.fillPoly(img, [pts], col, cv2.LINE_AA)


def star(img, x, y, r, col):
    pts = []
    for i in range(8):
        a = i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.18
        pts.append([x + rr * math.cos(a), y + rr * math.sin(a)])
    cv2.fillPoly(img, [np.array(pts, np.int32)], col, cv2.LINE_AA)


# ------------------------------------------------------------------ film look
_y, _x = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip(1.12 - 0.38 * (((_x - W / 2) / (W / 2)) ** 2 + ((_y - H / 2) / (H / 2)) ** 2), 0.5, 1)[..., None]
GRAIN = [cv2.GaussianBlur(np.random.default_rng(i).normal(0, 7, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None] for i in range(6)]


def film(img, f):
    r = np.random.default_rng(f)
    out = img * VIG * WARM
    out = out + GRAIN[f % 6]
    out *= 1 + 0.015 * math.sin(f * 1.3) + r.uniform(-0.01, 0.01)
    if (f // 2) % 7 == 0:   # gate weave on twos
        out = np.roll(out, int(r.integers(-1, 2)), axis=0)
    for _ in range(int(r.integers(0, 3))):
        cv2.circle(out, (int(r.integers(0, W)), int(r.integers(0, H))), 1, (20, 20, 20), -1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_leak(img, k, t):
    if k <= 0:
        return img
    gx = (_x / W + 0.3 * math.sin(t * 3)) % 1.0
    leak = np.exp(-((gx - 0.5) / 0.22) ** 2)[..., None] * np.array([60, 140, 255], np.float32)
    return np.clip(img + leak * k * 0.8, 0, 255)


def iris(img, t, win, center):
    if not win or t < win[0]:
        return img
    r = (1 - smooth((t - win[0]) / (win[1] - win[0]))) * math.hypot(W, H) * 0.65
    m = np.zeros((H, W), np.float32)
    if r > 1:
        cv2.circle(m, (int(center[0]), int(center[1])), int(r), 1.0, -1, cv2.LINE_AA)
    return img * m[..., None]


# ------------------------------------------------------------------ driver
_shots = {}


def shot(i):
    if i not in _shots:
        _shots[i] = Shot(SHOTS[i])
    return _shots[i]


def render(f):
    t = f / FPS
    tc_glob = (f // 2) * 2 / FPS
    i = max(k for k, s in enumerate(SHOTS) if s["s"] <= t)
    sc = SHOTS[i]
    img, face = shot(i).frame(t - sc["s"], tc_glob - sc["s"], f)
    u = t - sc["s"]
    if i > 0 and u < XFADE:   # dissolve from the previous shot, with a warm light leak
        prev = SHOTS[i - 1]
        pimg, _ = shot(i - 1).frame(t - prev["s"], tc_glob - prev["s"], f)
        k = smooth(u / XFADE)
        img = pimg * (1 - k) + img * k
        img = light_leak(img, math.sin(math.pi * k), t)
    if i == 0 and u < 0.7:
        img = img * smooth(u / 0.7)
    S = shot(i)
    zones = {"top": (24, 24, W - 24, S.y0 + 80), "bottom": (24, S.y0 + S.dh - 80, W - 24, H - 34)}
    img = np.ascontiguousarray(img, np.float32)
    for cue in CUES:
        if cue.get("place", "top") in zones:
            lyrics_fx.draw_cue(img, cue, t, f, zones[cue.get("place", "top")])
    img = iris(img, t, sc.get("iris_out"), face)
    return film(img, f)


def main():
    os.makedirs("output", exist_ok=True)
    if sys.argv[1] == "stills":
        for s in sys.argv[2:]:
            cv2.imwrite(f"output/frame_{float(s):05.2f}.png", render(int(float(s) * FPS)))
        return
    tmp = "output/_video.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for f in range(NF):
        p.stdin.write(render(f).tobytes())
        if f % 96 == 0:
            print(f"{f}/{NF}", flush=True)
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", "output/humnava_toon.mp4"], check=True)
    os.remove(tmp)
    print("done -> output/humnava_toon.mp4")


if __name__ == "__main__":
    main()
