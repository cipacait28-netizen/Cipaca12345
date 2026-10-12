"""Per-panel layer separation + face rig.

For every assets/cropped/panel_XX.png this writes to assets/animated/panel_XX/:
  art.png     2x upscaled artwork
  matte.png   character matte (isnet-anime segmentation)
  bg.png      background with the character painted out (for parallax / deformation margins)
  rig.json    face landmarks used by the animator (eyes, brows, mouth, chin, face oval)
and a debug sheet output/rig_check.png.

Models (downloaded once into models/):
  isnet-anime.onnx     https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx
  face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import glob, json, os

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

UP = 2
L_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
R_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
        136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

seg = ort.InferenceSession("models/isnet-anime.onnx", providers=["CPUExecutionProvider"])
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"), num_faces=1,
    min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))


def matte(img):
    h, w = img.shape[:2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - 0.5).transpose(2, 0, 1)[None]
    p = seg.run(None, {"img": x})[0][0, 0]
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)
    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip((p - 0.15) / 0.7, 0, 1)


def landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for scale in (1.0, 1.5, 0.7):
        im = cv2.resize(rgb, None, fx=scale, fy=scale) if scale != 1 else rgb
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(im)))
        if r.face_landmarks:
            h, w = img.shape[:2]
            return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32)
    return None


def main():
    os.makedirs("output", exist_ok=True)
    tiles = []
    for path in sorted(glob.glob("assets/cropped/panel_*.png")):
        name = os.path.basename(path)[:-4]
        out = f"assets/animated/{name}"
        os.makedirs(out, exist_ok=True)
        src = cv2.imread(path)
        art = cv2.resize(src, None, fx=UP, fy=UP, interpolation=cv2.INTER_LANCZOS4)
        art = cv2.addWeighted(art, 1.35, cv2.GaussianBlur(art, (0, 0), 1.6), -0.35, 0)
        m = matte(art)
        pts = landmarks(art)
        # background plate: inpaint under a dilated matte at half res, then soften
        hole = (cv2.dilate((m > 0.2).astype(np.uint8), np.ones((25, 25), np.uint8)) * 255)
        small = cv2.resize(art, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        hs = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        bg = cv2.inpaint(small, hs, 9, cv2.INPAINT_TELEA)
        bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 3), (art.shape[1], art.shape[0]), interpolation=cv2.INTER_CUBIC)
        bg = np.where(hole[..., None] > 0, bg, art)
        cv2.imwrite(f"{out}/art.png", art)
        cv2.imwrite(f"{out}/matte.png", (m * 255).astype(np.uint8))
        cv2.imwrite(f"{out}/bg.png", bg)
        rig = {"size": [art.shape[1], art.shape[0]], "face": None}
        dbg = art.copy()
        dbg[m < 0.5] = (dbg[m < 0.5] * 0.35).astype(np.uint8)
        if pts is not None:
            rig["face"] = {k: pts[idx].round(1).tolist() for k, idx in
                           (("l_eye", L_EYE), ("r_eye", R_EYE), ("oval", OVAL), ("mouth", MOUTH))}
            rig["face"]["chin"] = pts[152].round(1).tolist()
            rig["face"]["top"] = pts[10].round(1).tolist()
            for k in ("l_eye", "r_eye", "mouth", "oval"):
                cv2.polylines(dbg, [np.array(rig["face"][k], np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
        json.dump(rig, open(f"{out}/rig.json", "w"))
        t = cv2.resize(dbg, (int(dbg.shape[1] * 300 / dbg.shape[0]), 300))
        cv2.putText(t, name + ("" if pts is not None else "  NO FACE"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
        print(name, art.shape[1::-1], "face" if pts is not None else "NO FACE", f"matte {m.mean():.2f}", flush=True)
    rows = [tiles[i:i + 4] for i in range(0, len(tiles), 4)]
    wmax = max(sum(t.shape[1] for t in r) for r in rows)
    cv2.imwrite("output/rig_check.png", np.vstack([cv2.copyMakeBorder(np.hstack(r), 0, 0, 0, wmax - sum(t.shape[1] for t in r),
                                                                         cv2.BORDER_CONSTANT) for r in rows]))


if __name__ == "__main__":
    main()
