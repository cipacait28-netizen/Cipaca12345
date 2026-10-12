"""Beat / energy analysis of assets/song.mp3 -> out/audio.json (numpy only)."""
import json, subprocess, numpy as np

SR = 22050
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "assets/song.mp3", "-ac", "1", "-ar", str(SR),
                      "-f", "f32le", "-"], capture_output=True).stdout
y = np.frombuffer(raw, np.float32)
dur = len(y) / SR
N, H = 1024, 256
win = np.hanning(N)
nfr = 1 + (len(y) - N) // H
idx = np.arange(N)[None, :] + H * np.arange(nfr)[:, None]
S = np.abs(np.fft.rfft(y[idx] * win, axis=1))
S = np.log1p(10 * S)
flux = np.maximum(0, np.diff(S, axis=0)).sum(1)
flux = np.concatenate([[0], flux])
flux = (flux - flux.mean()) / (flux.std() + 1e-9)
fps = SR / H

# tempo by autocorrelation of the onset envelope
ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
lo, hi = int(fps * 60 / 150), int(fps * 60 / 60)
lag = lo + np.argmax(ac[lo:hi])
# prefer the double-time candidate if it is nearly as strong (avoid half-tempo picks)
if lag * 2 < hi and ac[lag * 2] > 0.8 * ac[lag]:
    pass
bpm = 60 * fps / lag
period = lag / fps

# beat phase: pick offset that maximises onset strength on the grid
best, off = -1e9, 0
for o in np.arange(0, period, 1 / fps):
    ts = np.arange(o, dur, period)
    v = flux[np.minimum((ts * fps).astype(int), len(flux) - 1)].sum()
    if v > best:
        best, off = v, o
beats = np.arange(off, dur, period)

# loudness per 0.5 s for section intensity
rms = np.sqrt((y[: len(y) // (SR // 2) * (SR // 2)].reshape(-1, SR // 2) ** 2).mean(1))
rms = rms / rms.max()
json.dump({"duration": dur, "bpm": float(bpm), "period": float(period), "beats": beats.round(4).tolist(),
           "rms_half_s": rms.round(3).tolist()}, open("out/audio.json", "w"))
print(f"duration {dur:.1f}s  bpm {bpm:.1f}  beats {len(beats)}  first beat {off:.2f}s")
print("loudness per 10s:", [round(float(rms[i:i+20].mean()), 2) for i in range(0, len(rms), 20)])
