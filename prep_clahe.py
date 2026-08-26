"""
Preprocessing technique: CLAHE (Contrast-Limited Adaptive Histogram Equalisation).
Normalise local contrast/illumination on each DeepWeeds image (applied to the L channel in
LAB). Does not align to the source domain; it removes lighting/contrast variation that differs
between curated GBIF shots and Australian field imagery. Label-free.
"""
import os, json
import numpy as np
import cv2
from bench_common import deepweeds_arrays

NAME = "CLAHE contrast normalisation"

def clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = cl.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def main():
    from prep_common import score_deepweeds
    imgs, labels = deepweeds_arrays()
    proc = np.stack([clahe(im) for im in imgs])
    res = score_deepweeds(proc, labels)
    os.makedirs("results_prep", exist_ok=True)
    json.dump({"technique": NAME, "deepweeds": res}, open("results_prep/clahe.json","w"), indent=2)
    print(f"[{NAME}]")
    for k,v in res.items(): print(f"  {k:20s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
