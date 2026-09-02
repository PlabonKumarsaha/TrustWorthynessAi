"""
Few-shot adaptation (the realistic ceiling). Uses k labelled DeepWeeds images per class from a
HELD-OUT pool (disjoint from the 1200-image test set), so there is no test leakage.
Methods: CLIP Tip-Adapter (training-free), linear probes on CLIP & ResNet features, and an
ensemble. Swept over k = 5, 10, 20.  (Zhang et al., Tip-Adapter, ECCV 2022)
"""
import os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from adapt_common import (load_resnet, load_clip, test_set, few_shot, eval_tf,
                          score, clip_feats, DEVICE)
from bench_common import NUM

@torch.no_grad()
def resnet_feats(model, imgs, bs=64):
    fe = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
    xb = torch.stack([eval_tf(Image.fromarray(im)) for im in imgs]); out=[]
    for i in range(0,len(xb),bs): out.append(fe(xb[i:i+bs].to(DEVICE)).flatten(1).cpu())
    return torch.cat(out).numpy()

def tip_adapter(cs, Xk, yk, Xte, alpha=1.0, beta=5.0):
    keys = clip_feats(cs, Xk); vals = F.one_hot(torch.tensor(yk), NUM).float()
    ft = clip_feats(cs, Xte)
    zs = (100.0 * ft @ cs["text_w"].cpu().T).softmax(1)
    A = torch.exp(-beta*(1 - ft @ keys.T))
    cache = (A @ vals); cache = cache / cache.sum(1, keepdim=True)
    return (zs + alpha*cache).numpy()

def main():
    Xte, yte = test_set()
    cs = load_clip(); rn = load_resnet()
    Fte_rn = resnet_feats(rn, Xte); Fte_cl = clip_feats(cs, Xte).numpy()

    out = {}
    for k in [5, 10, 20]:
        Xk, yk = few_shot(k)
        Fk_rn = resnet_feats(rn, Xk); Fk_cl = clip_feats(cs, Xk).numpy()
        lp_rn = LogisticRegression(max_iter=2000, C=1.0).fit(Fk_rn, yk)
        lp_cl = LogisticRegression(max_iter=2000, C=1.0).fit(Fk_cl, yk)
        p_rn = lp_rn.predict_proba(Fte_rn)
        p_cl = lp_cl.predict_proba(Fte_cl)
        p_tip = tip_adapter(cs, Xk, yk, Xte)
        ens = p_rn + p_cl + p_tip
        res = {
            f"ResNet linear-probe":  score(yte, p_rn.argmax(1)),
            f"CLIP linear-probe":    score(yte, p_cl.argmax(1)),
            f"CLIP Tip-Adapter":     score(yte, p_tip.argmax(1)),
            f"Ensemble (all 3)":     score(yte, ens.argmax(1)),
        }
        out[f"k={k}"] = res
        print(f"--- {k} shots/class ---")
        for n,v in res.items(): print(f"  {n:22s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

    os.makedirs("results_adapt", exist_ok=True)
    json.dump({"technique":"Few-shot (k labelled target/class)","by_k":out},
              open("results_adapt/fewshot.json","w"), indent=2)

if __name__ == "__main__":
    main()
