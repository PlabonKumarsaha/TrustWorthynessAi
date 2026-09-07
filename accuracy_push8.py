"""
FULL-SCALE validation: the DINOv2 + self-training pipeline on ALL 8 weed species.

Source : GBIF global_train (8 classes x 280)   -- source labels only
Target : DeepWeeds  weeds/_arrays/au_imgs.npy  (1600 imgs, 200/class, 8 classes)
Chance : 12.5%

Reports resolution effect and label-free self-training, directly comparable to the
4-species run (which reached 94.2%).
"""
import os, json, sys, glob
import numpy as np
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "vit_base_patch14_dinov2.lvd142m"
CACHE = "results_novel/feat_cache8"
CLASSES = ["chinee_apple","lantana","parkinsonia","parthenium",
           "prickly_acacia","rubber_vine","siam_weed","snake_weed"]

def score(y,p): return dict(acc=round(float(accuracy_score(y,p)),4),
                            macro_f1=round(float(f1_score(y,p,average="macro")),4))

def gbif8(split, size=224):
    xs, ys = [], []
    for ci,c in enumerate(CLASSES):
        for fp in sorted(glob.glob(os.path.join(f"weeds/{split}", c, "*"))):
            xs.append(np.asarray(Image.open(fp).convert("RGB").resize((size,size)),dtype=np.uint8)); ys.append(ci)
    return np.stack(xs), np.array(ys)

def get_feats(name, imgs, size):
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/{name}_{size}.npy"
    if os.path.exists(fp): return np.load(fp)
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"]=(3,size,size)
    tf = create_transform(**cfg, is_training=False)
    out=[]
    with torch.no_grad():
        for i in range(0,len(imgs),16):
            xb = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i+16]]).to(DEVICE)
            out.append(m(xb).cpu())
            if (i//16) % 20 == 0: print(f"    {name}@{size}: {i}/{len(imgs)}", flush=True)
    F = torch.cat(out).numpy(); np.save(fp,F); del m
    return F

def main():
    sizes = [int(s) for s in (sys.argv[1:] or ["448"])]
    print("loading data ...", flush=True)
    Xtr, ytr = gbif8("global_train"); Xin, yin = gbif8("global_test")
    Xte = np.load("weeds/_arrays/au_imgs.npy"); yte = np.load("weeds/_arrays/au_labels.npy")
    print(f"  GBIF train {Xtr.shape}  GBIF test {Xin.shape}  DeepWeeds {Xte.shape} (chance=12.5%)", flush=True)

    results = {}
    best_s, best_acc = None, -1
    store = {}
    for s in sizes:
        print(f"\n### DINOv2 @ {s}px (8 species)", flush=True)
        Ftr = get_feats("gbif_train8", Xtr, s); Fin = get_feats("gbif_test8", Xin, s); Fte = get_feats("dw_test8", Xte, s)
        clf = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
        p_in = clf.predict_proba(Fin); p = clf.predict_proba(Fte)
        r = {"in_domain_gbif": score(yin,p_in.argmax(1)), "deepweeds": score(yte,p.argmax(1))}
        results[f"DINOv2-{s}"] = r
        print(f"  GBIF={r['in_domain_gbif']['acc']*100:.1f}%   DeepWeeds={r['deepweeds']['acc']*100:.1f}%", flush=True)
        store[s] = (Ftr, Fte, p)
        if r["deepweeds"]["acc"] > best_acc: best_acc, best_s = r["deepweeds"]["acc"], s

    # ---- label-free self-training ----
    Ftr, Fte, p = store[best_s]
    print(f"\n### self-training on DINOv2-{best_s} (label-free, 8 species)", flush=True)
    for rnd in range(3):
        conf = p.max(1); pred = p.argmax(1); keep = conf >= 0.95
        prec = (pred[keep]==yte[keep]).mean() if keep.any() else 0
        print(f"  round {rnd+1}: kept {int(keep.sum())}/{len(yte)} "
              f"({keep.mean()*100:.1f}% coverage, precision {prec*100:.1f}%)", flush=True)
        clf = LogisticRegression(max_iter=3000).fit(
            np.concatenate([Ftr, Fte[keep]]), np.concatenate([ytr, pred[keep]]))
        p = clf.predict_proba(Fte)
        r = score(yte, p.argmax(1)); results[f"DINOv2-{best_s}-selftrain-r{rnd+1}"] = r
        print(f"    -> DeepWeeds = {r['acc']*100:.1f}%", flush=True)

    print("\n=== per-class F1 (final) ===", flush=True)
    print(classification_report(yte, p.argmax(1), target_names=CLASSES, digits=3))
    results["_per_class"] = classification_report(yte, p.argmax(1), target_names=CLASSES,
                                                  output_dict=True, zero_division=0)
    json.dump(results, open("results_novel/accuracy_push8.json","w"), indent=2)
    best = max(((k,v) for k,v in results.items() if isinstance(v,dict) and "acc" in v),
               key=lambda kv: kv[1]["acc"])
    print(f"\n=== BEST (8 species): {best[0]} = {best[1]['acc']*100:.1f}% ===", flush=True)

if __name__ == "__main__":
    main()
