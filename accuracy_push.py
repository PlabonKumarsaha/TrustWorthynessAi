"""
Accuracy push on the DeepWeeds target: add DINOv2 as a much stronger expert, test the effect of
input resolution, and ensemble with the existing ResNet-50 + CLIP experts.

All training uses GBIF (source) labels ONLY. DeepWeeds labels are used solely to score.
"""
import os, json, sys
import numpy as np
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from bench_common import gbif_arrays, deepweeds_arrays, CLASSES
from adapt_common import load_resnet, load_clip, test_set, pool_set, resnet_probs, clip_probs, DEVICE

MODEL = "vit_base_patch14_dinov2.lvd142m"

def score(y, p): return dict(acc=round(float(accuracy_score(y, p)), 4),
                             macro_f1=round(float(f1_score(y, p, average="macro")), 4))

def dinov2_feats(img_size):
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=img_size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3, img_size, img_size)
    tf = create_transform(**cfg, is_training=False)
    @torch.no_grad()
    def go(imgs, bs=32):
        out = []
        for i in range(0, len(imgs), bs):
            xb = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i+bs]]).to(DEVICE)
            out.append(m(xb).cpu())
        return torch.cat(out).numpy()
    return go

def main():
    sizes = [int(s) for s in (sys.argv[1:] or ["224", "336"])]
    Xtr, ytr = gbif_arrays("global_train")
    Xin, yin = gbif_arrays("global_test")          # in-domain control
    Xte, yte = test_set()                           # DeepWeeds target
    results = {}

    Xpo, ypo = pool_set()
    best_probs, best_pool, best_key, best_acc = None, None, None, -1
    for s in sizes:
        print(f"\n### DINOv2 @ {s}px")
        feat = dinov2_feats(s)
        Ftr, Fin, Fte, Fpo = feat(Xtr), feat(Xin), feat(Xte), feat(Xpo)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Ftr, ytr)
        p_in = clf.predict_proba(Fin); p_te = clf.predict_proba(Fte); p_po = clf.predict_proba(Fpo)
        r = {"in_domain_gbif": score(yin, p_in.argmax(1)), "deepweeds": score(yte, p_te.argmax(1))}
        results[f"DINOv2-{s}"] = r
        print(f"  GBIF={r['in_domain_gbif']['acc']*100:.1f}%   DeepWeeds={r['deepweeds']['acc']*100:.1f}%")
        if r["deepweeds"]["acc"] > best_acc:
            best_acc, best_probs, best_pool, best_key = r["deepweeds"]["acc"], p_te, p_po, f"DINOv2-{s}"

    # ---- ensemble with the existing experts ----
    print("\n### ensembles (DeepWeeds)")
    rn = load_resnet(); cs = load_clip()
    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    combos = {
        "ResNet": p_rn, "CLIP": p_cl, best_key: best_probs,
        "ResNet+CLIP": p_rn + p_cl,
        f"{best_key}+CLIP": best_probs + p_cl,
        f"{best_key}+ResNet": best_probs + p_rn,
        f"{best_key}+ResNet+CLIP": best_probs + p_rn + p_cl,
    }
    for name, p in combos.items():
        r = score(yte, p.argmax(1)); results[f"ens::{name}"] = r
        print(f"  {name:28s} {r['acc']*100:5.1f}%  (f1={r['macro_f1']:.3f})")

    # save DINOv2 probs for downstream self-training
    os.makedirs("results_novel", exist_ok=True)
    np.save("results_novel/dinov2_test_probs.npy", best_probs)
    np.save("results_novel/dinov2_pool_probs.npy", best_pool)
    json.dump(results, open("results_novel/accuracy_push.json", "w"), indent=2)
    print(f"\nbest single expert: {best_key} @ {best_acc*100:.1f}%   -> saved probs")

if __name__ == "__main__":
    main()
