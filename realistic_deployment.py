"""
Annotation-free deployment evaluation on the FULL, REALISTIC DeepWeeds target.

Two fixes over all earlier experiments:
  1. The 'negative' class is no longer dropped -- all 9,106 non-target images are included.
     The model is trained on GBIF's 8 weed species only and must REJECT anything else
     (open-set), which is the real field condition.
  2. No artificial class balancing -- the natural DeepWeeds distribution is used, and the
     class prior is ESTIMATED rather than assumed uniform.

Stage A streams TFDS -> DINOv2 features (never holds the image set in RAM).
Stage B evaluates closed-set accuracy, open-set rejection, and whether the earlier
distribution-alignment gain survives a realistic prior.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = "results_novel/realistic"
MODEL = "vit_base_patch14_dinov2.lvd142m"
CLASSES = ["chinee_apple","lantana","parkinsonia","parthenium",
           "prickly_acacia","rubber_vine","siam_weed","snake_weed"]   # label 8 = negative

def build(size):
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3,size,size)
    return m, create_transform(**cfg, is_training=False)

@torch.no_grad()
def stream_deepweeds(size, bs=16):
    """Stream the FULL DeepWeeds split through DINOv2, keeping only features."""
    import tensorflow_datasets as tfds
    m, tf = build(size)
    ds = tfds.load("deep_weeds", split="train", as_supervised=True)
    feats, labels, buf, n = [], [], [], 0
    for img, lbl in tfds.as_numpy(ds):
        buf.append(tf(Image.fromarray(img).convert("RGB"))); labels.append(int(lbl)); n += 1
        if len(buf) == bs:
            feats.append(m(torch.stack(buf).to(DEVICE)).float().cpu()); buf = []
            if n % 1600 == 0: print(f"   ...{n} images", flush=True)
    if buf: feats.append(m(torch.stack(buf).to(DEVICE)).float().cpu())
    F = torch.cat(feats).numpy(); y = np.array(labels)
    print(f"  full DeepWeeds: {F.shape}, class counts={np.bincount(y)}", flush=True)
    return F, y

@torch.no_grad()
def gbif_feats(size, bs=16):
    import glob
    m, tf = build(size)
    xs, ys = [], []
    for ci,c in enumerate(CLASSES):
        for fp in sorted(glob.glob(f"weeds/global_train/{c}/*")):
            xs.append(fp); ys.append(ci)
    out=[]
    for i in range(0,len(xs),bs):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in xs[i:i+bs]]).to(DEVICE)
        out.append(m(batch).float().cpu())
    return torch.cat(out).numpy(), np.array(ys)

def main():
    size = int(sys.argv[1]) if len(sys.argv)>1 else 448
    os.makedirs(OUT, exist_ok=True)
    fdw, fdwl = f"{OUT}/dw_full_{size}.npy", f"{OUT}/dw_full_labels.npy"
    fsrc, fsrcl = f"{OUT}/gbif_{size}.npy", f"{OUT}/gbif_labels.npy"

    if not os.path.exists(fsrc):
        print(f"[build] GBIF source features @{size}", flush=True)
        F, y = gbif_feats(size); np.save(fsrc, F); np.save(fsrcl, y)
        print(f"  GBIF: {F.shape}", flush=True)
    if not os.path.exists(fdw):
        print(f"[build] FULL DeepWeeds features @{size} (17,509 imgs, streaming)", flush=True)
        F, y = stream_deepweeds(size); np.save(fdw, F); np.save(fdwl, y)
    print("features ready ->", OUT, flush=True)

if __name__ == "__main__":
    main()
