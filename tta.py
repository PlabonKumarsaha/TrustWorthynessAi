"""
Step 3 — multi-view (test-time-augmentation-style) feature averaging for the top fusion.

Each image is embedded under four views — original, horizontal flip, 90% centre crop, and
90% centre crop + flip — and the embeddings are averaged. Applied to every GBIF image (train and
test alike) so the classifier sees features from the same distribution it is evaluated on.
Backbones: DINOv2-448 and CLIP ViT-L/14 (the best fusion pair). GBIF only: this is Phase 1
(in-domain accuracy), so DeepWeeds is not extracted for these variants.

Writes results_tta/ (features cached for fusion_cv.py).
"""
import os, json, time, warnings
import numpy as np
import torch.nn.functional as F
from PIL import ImageOps

from fusion import build_features, dedup_info, run_model, cached, DEVICE
from clip336 import evaluate, report
from hd_common import list_dir, GBIF_DIR

warnings.filterwarnings("ignore")
OUT = "results_tta"
VIEWS = ["orig", "hflip", "zoom", "zoom_hflip"]
# each TTA variant sits next to its single-view counterpart for a direct comparison
CONFIGS = [(("dinov2_448",), "SVM"), (("dinov2_448_tta",), "LR"), (("dinov2_448_tta",), "SVM"),
           (("clip_l14",), "SVM"), (("clip_l14_tta",), "LR"), (("clip_l14_tta",), "SVM"),
           (("dinov2_448", "clip_l14"), "SVM"), (("dinov2_448", "clip_l14"), "Stack"),
           (("dinov2_448_tta", "clip_l14_tta"), "LR"), (("dinov2_448_tta", "clip_l14_tta"), "SVM"),
           (("dinov2_448_tta", "clip_l14_tta"), "Stack")]

class View:
    """Picklable PIL view followed by the backbone's own preprocessing."""
    def __init__(self, name, tf): self.name, self.tf = name, tf
    def __call__(self, img):
        if self.name.startswith("zoom"):
            w, h = img.size; dw, dh = int(w * 0.05), int(h * 0.05)
            img = img.crop((dw, dh, w - dw, h - dh))
        if self.name.endswith("hflip"):
            img = ImageOps.mirror(img)
        return self.tf(img)

def dinov2_448():
    import timm
    from timm.data import resolve_data_config, create_transform
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=448).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3, 448, 448)
    return m, create_transform(**cfg, is_training=False), 32

def clip_l14():
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    m = m.to(DEVICE).eval()
    return (lambda x: F.normalize(m.encode_image(x), dim=-1)), pre, 64

def multiview(build, fs):
    fn, tf, bs = build()
    return np.mean([run_model(fn, View(v, tf), fs, bs) for v in VIEWS], axis=0)

if __name__ == "__main__":
    os.makedirs(f"{OUT}/feats", exist_ok=True); t0 = time.time()
    S, gys, ytr, yte, ydw, itr, ite = build_features()
    gfs, _ = list_dir(GBIF_DIR)
    for name, build in [("dinov2_448_tta", dinov2_448), ("clip_l14_tta", clip_l14)]:
        X = cached(f"{OUT}/feats/{name}_gbif.npz", lambda: multiview(build, gfs))
        S[name] = dict(all=X, tr=X[itr], te=X[ite], dw=None)
    _, clean = dedup_info(S["dinov2_224"]["all"], itr, ite)
    R = evaluate(S, CONFIGS, ytr, yte, ydw, clean)
    json.dump(dict(results=R, views=VIEWS, seconds=round(time.time() - t0, 1)), open(f"{OUT}/results.json", "w"), indent=2)
    report(R, "Step 3 — multi-view feature averaging (fixed GBIF-400 split)", f"{OUT}/report.md")
