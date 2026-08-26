"""
Preprocessing technique: Fourier Domain Adaptation (FDA).
Replace the low-frequency amplitude spectrum of each DeepWeeds (target) image with that of a
random GBIF (source) image, keeping the target's phase. Aligns global colour/illumination
style toward the source domain. Label-free, no training. (Yang & Soatto, CVPR 2020.)
"""
import os, json
import numpy as np
from bench_common import deepweeds_arrays, gbif_arrays

NAME = "Fourier Domain Adaptation (beta=0.01)"
BETA = 0.01

def fda_pair(src, ref, beta=BETA):
    src = src.astype(np.float32); ref = ref.astype(np.float32)
    out = np.empty_like(src)
    for c in range(3):
        fs = np.fft.fft2(src[:,:,c]); fr = np.fft.fft2(ref[:,:,c])
        amp_s, pha_s = np.abs(fs), np.angle(fs)
        amp_s = np.fft.fftshift(amp_s); amp_r = np.fft.fftshift(np.abs(fr))
        h, w = amp_s.shape; b = max(1, int(min(h,w)*beta)); cy, cx = h//2, w//2
        amp_s[cy-b:cy+b+1, cx-b:cx+b+1] = amp_r[cy-b:cy+b+1, cx-b:cx+b+1]
        amp_s = np.fft.ifftshift(amp_s)
        out[:,:,c] = np.real(np.fft.ifft2(amp_s * np.exp(1j*pha_s)))
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    from prep_common import score_deepweeds
    imgs, labels = deepweeds_arrays()
    gbif,_ = gbif_arrays("global_train", size=224)
    rng = np.random.default_rng(42)
    proc = np.stack([fda_pair(im, gbif[rng.integers(len(gbif))]) for im in imgs])
    res = score_deepweeds(proc, labels)
    os.makedirs("results_prep", exist_ok=True)
    json.dump({"technique": NAME, "deepweeds": res}, open("results_prep/fda.json","w"), indent=2)
    print(f"[{NAME}]")
    for k,v in res.items(): print(f"  {k:20s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
