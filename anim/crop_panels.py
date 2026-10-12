"""Split the collage into individual panels.

    python crop_panels.py            # auto-detect -> assets/cropped/panels.json + PNGs + contact sheet
    python crop_panels.py --manual   # re-crop from the (hand-edited) assets/cropped/panels.json

Each panel is stored as 4 corner points (clockwise from top-left) in collage pixels, so tilted
panels are de-skewed with a perspective warp. The original collage is never modified.
"""
import json, sys
import cv2
import numpy as np

SRC = "assets/original/collage.png"
OUT = "assets/cropped"
GUTTER_RGB = (250, 215, 160)   # cream border colour between panels
INSET = 7                      # px trimmed inside each detected edge to drop leftover border
# Rough panel corners (TL, TR, BR, BL) read off the collage; snap_edges() refines each edge onto the
# cream gutter line. Edit these (or assets/cropped/panels.json with --manual) to adjust a cut.
SEEDS = [
    [(0, 0), (431, 0), (431, 388), (0, 388)],
    [(434, 0), (834, 0), (834, 391), (434, 391)],
    [(838, 0), (1196, 0), (1196, 393), (838, 393)],
    [(1201, 0), (1536, 0), (1536, 440), (1201, 438)],
    [(0, 392), (553, 392), (553, 692), (0, 690)],
    [(558, 392), (1003, 405), (1003, 708), (558, 702)],
    [(1008, 445), (1536, 447), (1536, 705), (1008, 710)],
    [(0, 697), (440, 705), (440, 1024), (0, 1024)],
    [(444, 708), (797, 712), (797, 1024), (431, 1024)],
    [(800, 712), (1128, 716), (1112, 1024), (782, 1024)],
    [(1132, 716), (1536, 708), (1536, 1024), (1118, 1024)],
]


def order(pts):
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def gutter_score(im):
    g = np.array(GUTTER_RGB[::-1], np.float32)
    d = np.linalg.norm(im.astype(np.float32) - g, axis=2)
    return np.clip(1 - d / 60, 0, 1)


def snap_edges(im, seed, R=14):
    """Move each non-border edge (both endpoints independently) onto the best gutter line."""
    h, w = im.shape[:2]
    sc = gutter_score(im)
    q = np.array(seed, np.float32)
    lines = []
    for i in range(4):
        a, b = q[i].copy(), q[(i + 1) % 4].copy()
        horiz = abs(b[0] - a[0]) > abs(b[1] - a[1])
        k = 1 if horiz else 0
        on_border = all(p[k] <= 1 or p[k] >= (h if horiz else w) - 1 for p in (a, b))
        if not on_border:
            t = np.linspace(0.04, 0.96, 160)[:, None]
            best = (-1, 0, 0)
            for da in range(-R, R + 1):
                for db in range(-R, R + 1):
                    aa, bb = a.copy(), b.copy()
                    aa[k] += da; bb[k] += db
                    p = aa + (bb - aa) * t
                    xs = np.clip(p[:, 0].round().astype(int), 0, w - 1)
                    ys = np.clip(p[:, 1].round().astype(int), 0, h - 1)
                    v = sc[ys, xs].mean()
                    if v > best[0]:
                        best = (v, da, db)
            a[k] += best[1]; b[k] += best[2]
        lines.append((a, b))
    # corners = intersections of consecutive edges
    def inter(l1, l2):
        (x1, y1), (x2, y2) = l1; (x3, y3), (x4, y4) = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return (px, py)
    out = [inter(lines[3], lines[0]), inter(lines[0], lines[1]), inter(lines[1], lines[2]), inter(lines[2], lines[3])]
    return np.clip(np.array(out, np.float32), 0, [w - 1, h - 1])


def detect(im):
    return [snap_edges(im, s) for s in SEEDS]


def detect_auto(im):
    h, w = im.shape[:2]
    g = np.array(GUTTER_RGB[::-1], np.float32)
    dist = np.linalg.norm(im.astype(np.float32) - g, axis=2)
    gutter = (dist < 45).astype(np.uint8)
    # keep only long thin line-like gutter runs (not cream lamps inside pictures)
    hor = cv2.morphologyEx(gutter, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    ver = cv2.morphologyEx(gutter, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    lines = cv2.dilate(np.maximum(hor, ver), np.ones((5, 5), np.uint8))
    lines[:2], lines[-2:], lines[:, :2], lines[:, -2:] = 1, 1, 1, 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - lines, connectivity=4)
    panels = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.02 * h * w:
            continue
        m = (lab == i).astype(np.uint8)
        cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(quad) != 4:
            quad = cv2.boxPoints(cv2.minAreaRect(cnt))
        panels.append(order(quad))
    # reading order: rows by centre y, then x
    panels.sort(key=lambda q: (round(q[:, 1].mean() / 300), q[:, 0].mean()))
    return panels


def warp(im, q):
    q = order(q)
    c = q.mean(0)
    q = q + (c - q) / np.linalg.norm(c - q, axis=1, keepdims=True) * INSET * 1.41
    wd = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    ht = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [wd - 1, 0], [wd - 1, ht - 1], [0, ht - 1]]))
    return cv2.warpPerspective(im, M, (wd, ht), flags=cv2.INTER_LANCZOS4)


def contact_sheet(crops, path):
    th = 300
    tiles = []
    for i, c in enumerate(crops, 1):
        t = cv2.resize(c, (int(c.shape[1] * th / c.shape[0]), th), interpolation=cv2.INTER_AREA)
        t = cv2.copyMakeBorder(t, 34, 8, 8, 8, cv2.BORDER_CONSTANT, value=(40, 30, 30))
        cv2.putText(t, f"panel_{i:02d}  {c.shape[1]}x{c.shape[0]}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 240, 250), 1, cv2.LINE_AA)
        tiles.append(t)
    rows, row, wsum = [], [], 0
    for t in tiles:
        if wsum + t.shape[1] > 1700 and row:
            rows.append(row); row, wsum = [], 0
        row.append(t); wsum += t.shape[1]
    rows.append(row)
    W = max(sum(t.shape[1] for t in r) for r in rows)
    out = [cv2.copyMakeBorder(np.hstack(r), 0, 0, 0, W - sum(t.shape[1] for t in r), cv2.BORDER_CONSTANT, value=(40, 30, 30)) for r in rows]
    cv2.imwrite(path, np.vstack(out))


def overlay(im, quads, path):
    o = im.copy()
    for i, q in enumerate(quads, 1):
        cv2.polylines(o, [q.astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
        c = q.mean(0).astype(int)
        cv2.putText(o, str(i), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.imwrite(path, o)


def main():
    im = cv2.imread(SRC)
    cfg = f"{OUT}/panels.json"
    if "--manual" in sys.argv:
        quads = [np.array(p["corners"], np.float32) for p in json.load(open(cfg))]
    else:
        quads = detect(im)
        json.dump([{"name": f"panel_{i:02d}", "corners": q.round(1).tolist()} for i, q in enumerate(quads, 1)], open(cfg, "w"), indent=1)
    crops = [warp(im, q) for q in quads]
    for i, c in enumerate(crops, 1):
        cv2.imwrite(f"{OUT}/panel_{i:02d}.png", c)
    contact_sheet(crops, "output/contact_sheet.png")
    overlay(im, quads, "output/crop_overlay.png")
    print(f"{len(crops)} panels:", [c.shape[1::-1] for c in crops])


if __name__ == "__main__":
    main()
