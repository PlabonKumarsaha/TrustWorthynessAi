"""
Stage 1 of the factorial study: extract frozen features for every (backbone, resolution) cell.

Using a frozen-feature + linear-probe protocol for ALL backbones controls for the training
recipe, so any difference in out-of-domain accuracy is attributable to the representation
itself -- which is exactly the claim under test.

Backbones : ResNet-50 (supervised CNN) | CLIP ViT-B/32 (VLM) | DINOv2 ViT-B/14 (self-supervised)
Splits    : GBIF train (source), GBIF test (in-domain), DeepWeeds test (out-of-domain), 8 species
"""
import os, sys, glob
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = "results_novel/factorial_cache"
CLASSES = ["chinee_apple","lantana","parkinsonia","parthenium",
           "prickly_acacia","rubber_vine","siam_weed","snake_weed"]
IMNET = ([0.485,0.456,0.406],[0.229,0.224,0.225])

def gbif8(split):
    xs, ys = [], []
    for ci,c in enumerate(CLASSES):
        for fp in sorted(glob.glob(os.path.join(f"weeds/{split}", c, "*"))):
            xs.append(np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8)); ys.append(ci)
    return np.stack(xs), np.array(ys)

def splits():
    Xtr,ytr = gbif8("global_train"); Xin,yin = gbif8("global_test")
    Xte = np.load("weeds/_arrays/au_imgs.npy"); yte = np.load("weeds/_arrays/au_labels.npy")
    return {"gbif_train":(Xtr,ytr), "gbif_test":(Xin,yin), "dw_test":(Xte,yte)}

# ---------------- backbone feature extractors ----------------
def make_resnet50(size):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    fe = nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
    tf = transforms.Compose([transforms.Resize(int(size*1.14)), transforms.CenterCrop(size),
                             transforms.ToTensor(), transforms.Normalize(*IMNET)])
    def go(x): return fe(x).flatten(1)
    return go, tf

def make_clip(size):
    import open_clip
    m,_,pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    m = m.to(DEVICE).eval()
    def go(x): return m.encode_image(x)
    return go, pre                      # CLIP ViT-B/32 is fixed at its native 224

def make_dinov2(size):
    import timm
    from timm.data import resolve_data_config, create_transform
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                          num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"]=(3,size,size)
    tf = create_transform(**cfg, is_training=False)
    def go(x): return m(x)
    return go, tf

BUILDERS = {"resnet50": make_resnet50, "clip_vitb32": make_clip, "dinov2_vitb14": make_dinov2}
GRID = {"resnet50":[224,336,448], "clip_vitb32":[224], "dinov2_vitb14":[224,336,448]}

@torch.no_grad()
def extract(go, tf, imgs, bs=16):
    out=[]
    for i in range(0,len(imgs),bs):
        xb = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i+bs]]).to(DEVICE)
        out.append(go(xb).float().cpu())
    return torch.cat(out).numpy()

def main():
    os.makedirs(CACHE, exist_ok=True)
    data = splits()
    only = sys.argv[1:] or list(BUILDERS)
    for bb in only:
        for size in GRID[bb]:
            need = {s: f"{CACHE}/{bb}_{size}_{s}.npy" for s in data}
            if all(os.path.exists(p) for p in need.values()):
                print(f"[skip] {bb}@{size} cached", flush=True); continue
            print(f"[build] {bb}@{size}", flush=True)
            go, tf = BUILDERS[bb](size)
            for s,(X,_) in data.items():
                if os.path.exists(need[s]): continue
                F = extract(go, tf, X); np.save(need[s], F)
                print(f"   {s}: {F.shape}", flush=True)
            del go, tf
    # labels once
    for s,(_,y) in data.items(): np.save(f"{CACHE}/labels_{s}.npy", y)
    print("done", flush=True)

if __name__ == "__main__":
    main()
