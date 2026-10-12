"""
CLIP as a backbone for GBIF -> DeepWeeds weed classification.

Frozen CLIP image encoders (no fine-tuning), evaluated two ways:
  - zero-shot: class-name prompts, no training at all;
  - frozen features + the 12 classical heads from the HD(LM)^2D study (hd_heads.HEADS).
Same GBIF-400 split (train 1,920 / test 640) and DeepWeeds test (8,403) as the main study, so
results line up with results_hd/ and with blip_backbone.py (identical output schema).
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hd_common import gbif_splits, deepweeds_all, metrics, CLASSES
from hd_heads import HEADS

OUT = "results_clip_backbone"
VARIANTS = {"clip_vitb32_openai": ("ViT-B-32", "openai"),
            "clip_vitl14_laion2b": ("ViT-L-14", "laion2b_s32b_b82k")}
NAMES = ["chinee apple", "lantana", "parkinsonia", "parthenium weed",
         "prickly acacia", "rubber vine", "siam weed", "snake weed"]
TEMPLATES = ["a photo of {}.", "a photo of {}, a type of weed.", "a close-up photo of a {} plant.",
             "a field photograph of {}.", "an image of the invasive weed {}."]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class Files(Dataset):
    def __init__(self, fs, tf): self.fs, self.tf = fs, tf
    def __len__(self): return len(self.fs)
    def __getitem__(self, i): return self.tf(Image.open(self.fs[i]).convert("RGB"))

@torch.no_grad()
def embed(model, pre, fs):
    out = []
    for xb in DataLoader(Files(fs, pre), batch_size=64, num_workers=4):
        out.append(F.normalize(model.encode_image(xb.to(DEVICE)).float(), dim=-1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def text_classifier(model, tok):
    w = []
    for n in NAMES:
        t = F.normalize(model.encode_text(tok([s.format(n) for s in TEMPLATES]).to(DEVICE)).float(), dim=-1)
        w.append(F.normalize(t.mean(0), dim=-1))
    return torch.stack(w).cpu().numpy()

def run(key, data, smoke=False):
    path = f"{OUT}/{key}.json"
    if os.path.exists(path) and not smoke: print(f"[{key}] done, skipping"); return
    arch, pretrained = VARIANTS[key]; t0 = time.time()
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tok = open_clip.get_tokenizer(arch); model = model.to(DEVICE).eval()
    (ftr, ytr), (fte, yte), (fdw, ydw) = data
    Xtr, Xte, Xdw = embed(model, pre, ftr), embed(model, pre, fte), embed(model, pre, fdw)
    W = text_classifier(model, tok)
    res = dict(family="CLIP", backbone=key, feat_dim=int(Xtr.shape[1]), classes=CLASSES,
               zero_shot=dict(in_domain=metrics(yte, (Xte @ W.T).argmax(1)),
                              shift=metrics(ydw, (Xdw @ W.T).argmax(1))), heads={})
    for hn, mk in HEADS.items():
        clf = make_pipeline(StandardScaler(), mk()).fit(Xtr, ytr)
        res["heads"][hn] = dict(in_domain=metrics(yte, clf.predict(Xte)), shift=metrics(ydw, clf.predict(Xdw)))
    res["seconds"] = round(time.time() - t0, 1)
    if not smoke:
        os.makedirs(f"{OUT}/feats", exist_ok=True)
        np.savez(f"{OUT}/feats/{key}.npz", Xtr=Xtr, Xte=Xte, Xdw=Xdw, ytr=ytr, yte=yte, ydw=ydw)
        json.dump(res, open(path, "w"), indent=2)
    best = max(res["heads"], key=lambda h: res["heads"][h]["shift"]["acc"])
    print(f"[{key}] dim={res['feat_dim']} zero-shot GBIF={res['zero_shot']['in_domain']['acc']:.4f} "
          f"DW={res['zero_shot']['shift']['acc']:.4f} | best head {best}: "
          f"GBIF={res['heads'][best]['in_domain']['acc']:.4f} DW={res['heads'][best]['shift']['acc']:.4f} "
          f"({res['seconds']:.0f}s)", flush=True)

def report():
    L = ["# CLIP as backbone (frozen), GBIF-400 → DeepWeeds\n",
         "| Backbone | Dim | Zero-shot GBIF | Zero-shot DeepWeeds | Best head | Head GBIF | **Head DeepWeeds** | Mean of 12 heads (DW) |",
         "|---|---|---|---|---|---|---|---|"]
    for k in VARIANTS:
        if not os.path.exists(f"{OUT}/{k}.json"): continue
        r = json.load(open(f"{OUT}/{k}.json")); h = r["heads"]
        b = max(h, key=lambda x: h[x]["shift"]["acc"]); m = np.mean([h[x]["shift"]["acc"] for x in h])
        L.append(f"| {k} | {r['feat_dim']} | {r['zero_shot']['in_domain']['acc']*100:.1f}% | "
                 f"{r['zero_shot']['shift']['acc']*100:.1f}% | {b} | {h[b]['in_domain']['acc']*100:.1f}% | "
                 f"**{h[b]['shift']['acc']*100:.1f}%** | {m*100:.1f}% |")
    L.append("\nReference (fine-tuned CNN, same split): RegNetY-32GF + SVM = 88.1% GBIF / 45.7% DeepWeeds.")
    open(f"{OUT}/report.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    (ftr, ytr), _, (fte, yte) = gbif_splits(); fdw, ydw = deepweeds_all()
    if a.smoke:
        s = lambda f, y, n: (f[::len(f)//n], y[::len(f)//n])
        data = (s(ftr, ytr, 64), s(fte, yte, 16), s(fdw, ydw, 16)); run("clip_vitb32_openai", data, smoke=True)
    else:
        for k in VARIANTS: run(k, ((ftr, ytr), (fte, yte), (fdw, ydw)))
        report()
