"""Pixel-diff matrix for the batch-size paradox test.

Usage:
    python diff_check.py A.png B.png C.png D.png

    A = extension OFF, batch size 1      (single image)
    B = extension ON,  batch size 1      (single image)
    C = extension OFF, batch size 4      (grid png; top-left quadrant auto-cropped)
    D = extension ON,  batch size 4      (grid png; top-left quadrant auto-cropped)

All four runs: same seed, same prompt/CFG/sampler/resolution/steps.
Prints whole-image and center-face MAE (0-255) for every pair, plus a
verdict: D-B's face MAE should sit at the A-C (same face, different
batch numerics) level, NOT at the A-B (cond noise ON vs OFF) level.
"""
import itertools
import sys

import numpy as np
from PIL import Image

NAMES = ["A", "B", "C", "D"]
# face box: center x, center y, width frac, height frac
FACE_CX, FACE_CY, FACE_WF, FACE_HF = 0.50, 0.40, 0.40, 0.40
try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


def load_all(paths):
    raw = [Image.open(p).convert("RGB") for p in paths]
    target_w = min(im.size[0] for im in raw)
    imgs = []
    for name, im in zip(NAMES, raw):
        w, h = im.size
        if w >= 2 * target_w - 8:
            im = im.crop((0, 0, w // 2, h // 2))
            print(f"{name}: grid {w}x{h} -> cropped top-left quadrant ({im.size[0]}x{im.size[1]})")
        imgs.append(im)
    return imgs


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1
    imgs = load_all(sys.argv[1:5])
    w, h = min(im.size for im in imgs)
    arrs = {}
    for name, im in zip(NAMES, imgs):
        a = np.asarray(im, dtype=np.float32)
        if a.shape[1:] != (h, w):
            a = np.asarray(im.resize((w, h), _RESAMPLE), dtype=np.float32)
        arrs[name] = a

    cx, cy = int(w * FACE_CX), int(h * FACE_CY)
    fw, fh = int(w * FACE_WF), int(h * FACE_HF)
    fx0, fy0 = cx - fw // 2, cy - fh // 2
    faces = {n: a[fy0:fy0 + fh, fx0:fx0 + fw] for n, a in arrs.items()}

    print(f"common size {w}x{h}, face box {fw}x{fh} centered at ({cx},{cy})")
    print(f"{'pair':<6s} {'whole MAE':>10s} {'face MAE':>10s} {'face px>32':>12s}")
    results = {}
    for x, y in itertools.combinations(NAMES, 2):
        d = np.abs(arrs[x] - arrs[y])
        df = np.abs(faces[x] - faces[y])
        whole = float(d.mean())
        face = float(df.mean())
        frac = float((df.max(axis=2) > 32).mean())
        results[(x, y)] = (whole, face, frac)
        print(f"{x}-{y:<4s} {whole:10.2f} {face:10.2f} {frac:11.1%}")

    def face_mae(x, y):
        return results.get((x, y), results.get((y, x)))[1]

    a_c = face_mae("A", "C")
    a_b = face_mae("A", "B")
    d_b = face_mae("D", "B")
    d_c = face_mae("D", "C")
    print()
    print("face-MAE readout:")
    print(f"  A-C = {a_c:6.2f}   <- same-face baseline (OFF size1 vs OFF size4 image1)")
    print(f"  A-B = {a_b:6.2f}   <- cond-noise baseline (OFF size1 vs ON size1)")
    print(f"  D-B = {d_b:6.2f}   <- EXPECTED ~ A-C if row-0 inputs are identical across sizes")
    print(f"  D-C = {d_c:6.2f}   <- EXPECTED ~ A-B")
    span = a_b - a_c
    if span < 1.0:
        print("  verdict: baselines too close to separate - does B look different from A at all?")
        return 0
    pos = (d_b - a_c) / span
    if pos < 0.34:
        verdict = "D's face matches B -> batch-size parity holds, the 'D~C' report was an artifact"
    elif pos > 0.66:
        verdict = "D's face matches C -> real bug in the batched path, start hunting"
    else:
        verdict = "D-B sits in between -> inspect the four images; also check run settings drift"
    print(f"  verdict: {verdict}")
    print(f"  (D-B position on the A-C..A-B scale: {pos:.2f}, 0 = like A-C, 1 = like A-B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
