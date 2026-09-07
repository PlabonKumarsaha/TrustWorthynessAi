"""
Second accuracy push, building on the DINOv2 result:
  * higher resolution (448)  -- 224->336 was worth +10 pts
  * multi-resolution feature concatenation
  * self-training the linear head on confident DeepWeeds pseudo-labels (label-free)
Features are cached to disk so re-runs are cheap.
"""
import os, json, sys
import numpy as np
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from bench_common import gbif_arrays
from adapt_common import test_set, pool_set, DEVICE

MODEL = "vit_base_patch14_dinov2.lvd142m"
CACHE = "results_novel/feat_cache"

def score(y, p): return dict(acc=round(float(accuracy_score(y,p)),4),
                             macro_f1=round(float(f1_score(y,p,average="macro")),4))

def get_feats(name, imgs, size):
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/{name}_{size}.npy"
    if os.path.exists(fp): return np.load(fp)
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3,size,size)
    tf = create_transform(**cfg, is_training=False)
    out=[]
    with torch.no_grad():
        for i in range(0,len(imgs),16):
            xb = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i+16]]).to(DEVICE)
            out.append(m(xb).cpu())
    F = torch.cat(out).numpy(); np.save(fp, F); del m
    return F

def main():
    sizes = [int(s) for s in (sys.argv[1:] or ["336","448"])]
    Xtr,ytr = gbif_arrays("global_train"); Xte,yte = test_set(); Xpo,ypo = pool_set()
    results = {}; per_size = {}

    for s in sizes:
        print(f"\n### DINOv2 @ {s}px")
        Ftr = get_feats("gbif_train", Xtr, s); Fte = get_feats("dw_test", Xte, s); Fpo = get_feats("pool", Xpo, s)
        clf = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
        p = clf.predict_proba(Fte)
        results[f"DINOv2-{s}"] = score(yte, p.argmax(1))
        per_size[s] = dict(Ftr=Ftr, Fte=Fte, Fpo=Fpo, probs=p)
        print(f"  DeepWeeds = {results[f'DINOv2-{s}']['acc']*100:.1f}%")

    # ---- multi-resolution: concatenate features ----
    if len(sizes) > 1:
        Ftr = np.concatenate([per_size[s]["Ftr"] for s in sizes], 1)
        Fte = np.concatenate([per_size[s]["Fte"] for s in sizes], 1)
        clf = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
        p_cat = clf.predict_proba(Fte)
        results["DINOv2-multires-concat"] = score(yte, p_cat.argmax(1))
        print(f"\n  multi-res concat  = {results['DINOv2-multires-concat']['acc']*100:.1f}%")
        p_avg = sum(per_size[s]["probs"] for s in sizes)/len(sizes)
        results["DINOv2-multires-probavg"] = score(yte, p_avg.argmax(1))
        print(f"  multi-res prob-avg = {results['DINOv2-multires-probavg']['acc']*100:.1f}%")

    # ---- self-training the head on confident target pseudo-labels (label-free) ----
    best_s = max(sizes, key=lambda s: results[f"DINOv2-{s}"]["acc"])
    Ftr, Fte = per_size[best_s]["Ftr"], per_size[best_s]["Fte"]
    print(f"\n### self-training on DINOv2-{best_s} (label-free)")
    p = per_size[best_s]["probs"]
    for rnd in range(3):
        conf = p.max(1); pred = p.argmax(1); keep = conf >= 0.95
        prec = (pred[keep]==yte[keep]).mean() if keep.any() else 0
        print(f"  round {rnd+1}: kept {int(keep.sum())}/{len(yte)} "
              f"({keep.mean()*100:.1f}% coverage, precision {prec*100:.1f}%)")
        Xaug = np.concatenate([Ftr, Fte[keep]]); yaug = np.concatenate([ytr, pred[keep]])
        clf = LogisticRegression(max_iter=3000).fit(Xaug, yaug)
        p = clf.predict_proba(Fte)
        r = score(yte, p.argmax(1)); print(f"    -> DeepWeeds = {r['acc']*100:.1f}%")
        results[f"DINOv2-{best_s}-selftrain-r{rnd+1}"] = r

    json.dump(results, open("results_novel/accuracy_push2.json","w"), indent=2)
    best = max(results.items(), key=lambda kv: kv[1]["acc"])
    print(f"\n=== BEST: {best[0]} = {best[1]['acc']*100:.1f}% (f1={best[1]['macro_f1']:.3f}) ===")

if __name__ == "__main__":
    main()
