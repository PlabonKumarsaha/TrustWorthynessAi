"""
Proposed technique vs baseline, same self-training pipeline (isolates the selection rule).

Baseline rule : keep pseudo-labels where ensemble confidence >= 0.9   (what we did before -> 84.1%)
Proposed rule : keep where the experts AGREE and pass a CONFORMAL threshold calibrated on the
                disjoint few-shot pool (cross-architecture conformal agreement).

Both rules feed identical ResNet self-training; we report final DeepWeeds accuracy.
"""
import os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from adapt_common import (load_resnet, load_clip, test_set, pool_set, eval_tf, score,
                          resnet_probs, clip_probs, DEVICE)

ALPHA = 0.3          # conformal risk level for the proposed rule
ROUNDS, EPOCHS, LR = 2, 3, 1e-4

def experts_probs(rn, cs, imgs):
    return {"ResNet": resnet_probs(rn, imgs), "CLIP": clip_probs(cs, imgs)}

def ensemble(P): return sum(P.values())/len(P)

def conformal_agreement_mask(P, ens, Ppool, ens_pool, ypool, alpha=ALPHA):
    """Keep test items where all experts agree AND conformal nonconformity <= pool-calibrated q."""
    experts = list(P.keys())
    preds = {e: P[e].argmax(1) for e in experts}
    agree = np.all([preds[e]==preds[experts[0]] for e in experts], axis=0)
    preds_pool = {e: Ppool[e].argmax(1) for e in experts}
    agree_pool = np.all([preds_pool[e]==preds_pool[experts[0]] for e in experts], axis=0)
    s_cal = 1 - ens_pool[agree_pool][np.arange(agree_pool.sum()), ypool[agree_pool]]
    q = np.quantile(s_cal, 1-alpha, method="higher")
    pred = ens.argmax(1); s = 1 - ens[np.arange(len(ens)), pred]
    keep = agree & (s <= q)
    return keep, pred

def self_train(rn, cs, Xte, yte, select_fn, tag):
    Xt = torch.stack([eval_tf(Image.fromarray(im)) for im in Xte])
    for rnd in range(ROUNDS):
        P = experts_probs(rn, cs, Xte); ens = ensemble(P)
        keep, pred = select_fn(P, ens)
        prec = (pred[keep]==yte[keep]).mean() if keep.any() else 0
        print(f"  [{tag}] round {rnd+1}: kept {int(keep.sum())}/{len(yte)} "
              f"(coverage={keep.mean()*100:.1f}%, pseudo-label precision={prec*100:.1f}%)")
        idx = np.where(keep)[0]
        rn.train()
        for m in rn.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        opt = torch.optim.AdamW([p for p in rn.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
        for ep in range(EPOCHS):
            perm = np.random.permutation(len(idx))
            for i in range(0, len(idx), 32):
                bi = idx[perm[i:i+32]]
                xb = Xt[bi].to(DEVICE); yb = torch.tensor(pred[bi]).to(DEVICE)
                loss = F.cross_entropy(rn(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        rn.eval()
    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    return {"ResNet": score(yte, p_rn.argmax(1)),
            "ResNet+CLIP": score(yte, (p_rn+p_cl).argmax(1))}

def main():
    Xte, yte = test_set(); Xpo, ypo = pool_set()
    cs = load_clip()
    Ppool = experts_probs(load_resnet(), cs, Xpo); ens_pool = ensemble(Ppool)

    results = {}
    # ---- baseline: confidence >= 0.9 ----
    def sel_conf(P, ens):
        pred = ens.argmax(1); return (ens.max(1) >= 0.9), pred
    results["baseline_confidence>=0.9"] = self_train(load_resnet(), cs, Xte, yte, sel_conf, "conf")

    # ---- proposed: conformal + cross-architecture agreement ----
    def sel_prop(P, ens):
        return conformal_agreement_mask(P, ens, Ppool, ens_pool, ypo, ALPHA)
    results["proposed_conformal+agreement"] = self_train(load_resnet(), cs, Xte, yte, sel_prop, "prop")

    os.makedirs("results_novel", exist_ok=True)
    json.dump(results, open("results_novel/method.json","w"), indent=2)
    print("\n=== FINAL DeepWeeds accuracy ===")
    for rule, r in results.items():
        print(f"  {rule:32s} ResNet={r['ResNet']['acc']*100:.1f}%  ResNet+CLIP={r['ResNet+CLIP']['acc']*100:.1f}%")

if __name__ == "__main__":
    main()
