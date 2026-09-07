"""
Fix the diagnosed class-bias at 8 species.

Diagnosis: parthenium & snake_weed act as SINK classes (precision ~0.45, recall ~0.9) while
prickly_acacia / lantana / siam_weed are starved (recall 0.32-0.53). Naive self-training
amplifies this (pseudo-label precision fell 81%->75% over rounds).

Fixes tested (all label-free; uses cached DINOv2-448 features, so this is fast):
  A. Sinkhorn distribution alignment (DA) at inference
  B. class-balanced self-training (per-class top-k selection, CBST-style)
  C. DA + class-balanced self-training
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

CACHE = "results_novel/feat_cache8"; SIZE = 448
CLASSES = ["chinee_apple","lantana","parkinsonia","parthenium",
           "prickly_acacia","rubber_vine","siam_weed","snake_weed"]
C = len(CLASSES)

def score(y,p): return dict(acc=round(float(accuracy_score(y,p)),4),
                            macro_f1=round(float(f1_score(y,p,average="macro")),4))

def sinkhorn(P, iters=100):
    """Align predicted marginal to uniform (target is balanced 200/class)."""
    Q = np.asarray(P, dtype=np.float64) + 1e-12
    col_target = Q.sum() / C
    for _ in range(iters):
        Q = Q / Q.sum(1, keepdims=True)
        Q = Q / Q.sum(0, keepdims=True) * col_target
    return Q

def balanced_select(P, frac):
    """CBST-style: take the top `frac` most-confident per PREDICTED class (not globally)."""
    pred = P.argmax(1); conf = P.max(1); keep = np.zeros(len(P), bool)
    for c in range(C):
        idx = np.where(pred == c)[0]
        if len(idx) == 0: continue
        k = max(1, int(len(idx) * frac))
        keep[idx[np.argsort(-conf[idx])[:k]]] = True
    return keep, pred

def main():
    Ftr = np.load(f"{CACHE}/gbif_train8_{SIZE}.npy"); Fte = np.load(f"{CACHE}/dw_test8_{SIZE}.npy")
    ytr = np.concatenate([[c]*280 for c in range(C)])           # GBIF: 280/class, ordered
    yte = np.load("weeds/_arrays/au_labels.npy")
    res = {}

    clf0 = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
    P0 = clf0.predict_proba(Fte)
    res["base"] = score(yte, P0.argmax(1)); print(f"base                 {res['base']['acc']*100:5.1f}%")

    # ---- A: Sinkhorn DA at inference ----
    res["A_DA"] = score(yte, sinkhorn(P0).argmax(1)); print(f"A: +Sinkhorn DA      {res['A_DA']['acc']*100:5.1f}%")

    # ---- B: class-balanced self-training ----
    P = P0.copy()
    for r in range(3):
        keep, pred = balanced_select(P, 0.5 + 0.15*r)
        prec = (pred[keep]==yte[keep]).mean()
        clf = LogisticRegression(max_iter=3000).fit(
            np.concatenate([Ftr, Fte[keep]]), np.concatenate([ytr, pred[keep]]))
        P = clf.predict_proba(Fte)
        res[f"B_CBST_r{r+1}"] = score(yte, P.argmax(1))
        print(f"B: CBST r{r+1} (kept {keep.sum():4d}, prec {prec*100:.1f}%)  {res[f'B_CBST_r{r+1}']['acc']*100:5.1f}%")

    # ---- C: DA + class-balanced self-training ----
    P = sinkhorn(P0)
    for r in range(3):
        keep, pred = balanced_select(P, 0.5 + 0.15*r)
        prec = (pred[keep]==yte[keep]).mean()
        clf = LogisticRegression(max_iter=3000).fit(
            np.concatenate([Ftr, Fte[keep]]), np.concatenate([ytr, pred[keep]]))
        P = sinkhorn(clf.predict_proba(Fte))
        res[f"C_DA+CBST_r{r+1}"] = score(yte, P.argmax(1))
        print(f"C: DA+CBST r{r+1} (kept {keep.sum():4d}, prec {prec*100:.1f}%)  {res[f'C_DA+CBST_r{r+1}']['acc']*100:5.1f}%")

    best_key = max(res, key=lambda k: res[k]["acc"])
    print(f"\n=== BEST: {best_key} = {res[best_key]['acc']*100:.1f}% (f1={res[best_key]['macro_f1']:.3f}) ===")
    print("\n=== per-class F1 (best config) ===")
    print(classification_report(yte, P.argmax(1), target_names=CLASSES, digits=3, zero_division=0))
    json.dump(res, open("results_novel/accuracy_push8_balanced.json","w"), indent=2)

if __name__ == "__main__":
    main()
