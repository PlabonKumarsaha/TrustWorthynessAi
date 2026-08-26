"""
Preprocessing technique: Reinhard colour transfer.
Recolour each DeepWeeds (target) image so its LAB colour mean/std match the GBIF
(source) domain the models were trained on. Label-free; uses source colour statistics only.
"""
import os, json
import numpy as np
import cv2
from bench_common import deepweeds_arrays, gbif_arrays
from prep_common import score_deepweeds

NAME = "Reinhard colour transfer"

def lab(im):  return cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
def unlab(l): return cv2.cvtColor(np.clip(l,0,255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def gbif_lab_stats():
    X,_ = gbif_arrays("global_train", size=224)
    labs = np.stack([lab(im) for im in X]).reshape(-1,3)
    return labs.mean(0), labs.std(0)

def transfer(im, ref_mean, ref_std):
    l = lab(im); m, s = l.reshape(-1,3).mean(0), l.reshape(-1,3).std(0)
    out = (l - m) / (s + 1e-6) * ref_std + ref_mean
    return unlab(out)

def main():
    imgs, labels = deepweeds_arrays()
    ref_mean, ref_std = gbif_lab_stats()
    proc = np.stack([transfer(im, ref_mean, ref_std) for im in imgs])
    res = score_deepweeds(proc, labels)
    os.makedirs("results_prep", exist_ok=True)
    json.dump({"technique": NAME, "deepweeds": res}, open("results_prep/reinhard.json","w"), indent=2)
    print(f"[{NAME}]")
    for k,v in res.items(): print(f"  {k:20s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
