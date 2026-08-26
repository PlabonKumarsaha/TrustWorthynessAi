"""
Preprocessing technique: histogram matching.
Remap each DeepWeeds (target) image's per-channel intensity histogram onto a random GBIF
(source) reference image, so the target's tonal/colour distribution matches the source. Label-free.
"""
import os, json
import numpy as np
from skimage.exposure import match_histograms
from bench_common import deepweeds_arrays, gbif_arrays

NAME = "Histogram matching (to GBIF)"

def main():
    from prep_common import score_deepweeds
    imgs, labels = deepweeds_arrays()
    gbif,_ = gbif_arrays("global_train", size=224)
    rng = np.random.default_rng(42)
    proc = np.stack([match_histograms(im, gbif[rng.integers(len(gbif))], channel_axis=-1).astype(np.uint8)
                     for im in imgs])
    res = score_deepweeds(proc, labels)
    os.makedirs("results_prep", exist_ok=True)
    json.dump({"technique": NAME, "deepweeds": res}, open("results_prep/histmatch.json","w"), indent=2)
    print(f"[{NAME}]")
    for k,v in res.items(): print(f"  {k:20s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
