"""
BLIP as a backbone for GBIF -> DeepWeeds weed classification.

Frozen BLIP image-text retrieval models (no fine-tuning), evaluated two ways:
  - zero-shot: image-text contrastive similarity with class-name prompts;
  - frozen features + the 12 classical heads from the HD(LM)^2D study (hd_heads.HEADS).
Heads use the vision encoder's [CLS] token before projection (768-d base, 1024-d large);
zero-shot uses BLIP's projected, normalised image and text embeddings.
Same GBIF-400 split and DeepWeeds test as the main study; output schema matches clip_backbone.py.
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForImageTextRetrieval, BlipProcessor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hd_common import gbif_splits, deepweeds_all, metrics, CLASSES
from hd_heads import HEADS

OUT = "results_blip_backbone"
VARIANTS = {"blip_base_coco": "Salesforce/blip-itm-base-coco",
            "blip_large_coco": "Salesforce/blip-itm-large-coco"}
NAMES = ["chinee apple", "lantana", "parkinsonia", "parthenium weed",
         "prickly acacia", "rubber vine", "siam weed", "snake weed"]
TEMPLATES = ["a photo of {}.", "a photo of {}, a type of weed.", "a close-up photo of a {} plant.",
             "a field photograph of {}.", "an image of the invasive weed {}."]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class Files(Dataset):
    def __init__(self, fs, proc): self.fs, self.proc = fs, proc
    def __len__(self): return len(self.fs)
    def __getitem__(self, i):
        return self.proc(images=Image.open(self.fs[i]).convert("RGB"), return_tensors="pt")["pixel_values"][0]

@torch.no_grad()
def embed(model, proc, fs):
    """Returns (pre-projection CLS features for heads, projected normalised features for zero-shot)."""
    cls, proj = [], []
    for xb in DataLoader(Files(fs, proc.image_processor), batch_size=32, num_workers=4):
        h = model.vision_model(pixel_values=xb.to(DEVICE)).last_hidden_state[:, 0, :]
        cls.append(h.float().cpu()); proj.append(F.normalize(model.vision_proj(h), dim=-1).float().cpu())
    return torch.cat(cls).numpy(), torch.cat(proj).numpy()

@torch.no_grad()
def text_classifier(model, proc):
    w = []
    for n in NAMES:
        t = proc.tokenizer([s.format(n) for s in TEMPLATES], padding=True, return_tensors="pt").to(DEVICE)
        h = model.text_encoder(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        e = F.normalize(model.text_proj(h), dim=-1).mean(0)
        w.append(F.normalize(e, dim=-1).float().cpu())
    return torch.stack(w).numpy()

def run(key, data, smoke=False):
    path = f"{OUT}/{key}.json"
    if os.path.exists(path) and not smoke: print(f"[{key}] done, skipping"); return
    t0 = time.time()
    proc = BlipProcessor.from_pretrained(VARIANTS[key])
    model = BlipForImageTextRetrieval.from_pretrained(VARIANTS[key]).to(DEVICE).eval()
    (ftr, ytr), (fte, yte), (fdw, ydw) = data
    (Ctr, _), (Cte, Pte), (Cdw, Pdw) = embed(model, proc, ftr), embed(model, proc, fte), embed(model, proc, fdw)
    W = text_classifier(model, proc)
    res = dict(family="BLIP", backbone=key, feat_dim=int(Ctr.shape[1]), classes=CLASSES,
               zero_shot=dict(in_domain=metrics(yte, (Pte @ W.T).argmax(1)),
                              shift=metrics(ydw, (Pdw @ W.T).argmax(1))), heads={})
    for hn, mk in HEADS.items():
        clf = make_pipeline(StandardScaler(), mk()).fit(Ctr, ytr)
        res["heads"][hn] = dict(in_domain=metrics(yte, clf.predict(Cte)), shift=metrics(ydw, clf.predict(Cdw)))
    res["seconds"] = round(time.time() - t0, 1)
    if not smoke:
        os.makedirs(f"{OUT}/feats", exist_ok=True)
        np.savez(f"{OUT}/feats/{key}.npz", Xtr=Ctr, Xte=Cte, Xdw=Cdw, ytr=ytr, yte=yte, ydw=ydw)
        json.dump(res, open(path, "w"), indent=2)
    best = max(res["heads"], key=lambda h: res["heads"][h]["shift"]["acc"])
    print(f"[{key}] dim={res['feat_dim']} zero-shot GBIF={res['zero_shot']['in_domain']['acc']:.4f} "
          f"DW={res['zero_shot']['shift']['acc']:.4f} | best head {best}: "
          f"GBIF={res['heads'][best]['in_domain']['acc']:.4f} DW={res['heads'][best]['shift']['acc']:.4f} "
          f"({res['seconds']:.0f}s)", flush=True)

def report():
    L = ["# BLIP as backbone (frozen), GBIF-400 → DeepWeeds\n",
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
        data = (s(ftr, ytr, 64), s(fte, yte, 16), s(fdw, ydw, 16)); run("blip_base_coco", data, smoke=True)
    else:
        for k in VARIANTS: run(k, ((ftr, ytr), (fte, yte), (fdw, ydw)))
        report()
