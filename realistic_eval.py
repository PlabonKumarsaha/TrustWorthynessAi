"""
Stage B: does the 80.6% survive REALISTIC deployment conditions?

Three things change versus every earlier experiment:
  1. NEGATIVES INCLUDED  - all 9,106 non-target images. The model saw only 8 GBIF weed
                           species in training, so rejecting them is genuine open-set.
  2. NATURAL IMBALANCE   - no 200/class subsampling.
  3. ESTIMATED PRIOR     - the earlier Sinkhorn DA assumed a uniform target prior (true only
                           because I had constructed it). Here the prior is estimated from
                           unlabelled data by EM (Saerens-Latinne-Decaestecker), so the
                           assumption is removed.

Reports closed-set accuracy, open-set rejection quality (AUROC), and whether the DA gain holds.
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

OUT = "results_novel/realistic"; SIZE = 448
CLASSES = ["chinee_apple","lantana","parkinsonia","parthenium",
           "prickly_acacia","rubber_vine","siam_weed","snake_weed"]
C = 8

def em_prior(P, src_prior, iters=200, tol=1e-7):
    """Label-free target class-prior estimation (Saerens et al. 2002)."""
    pi = np.full(C, 1.0/C)
    for _ in range(iters):
        w = (pi/src_prior)[None, :] * P
        w = w / w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new-pi).max() < tol: break
        pi = new
    return pi

def align(P, target_prior, iters=100):
    """Sinkhorn to an ARBITRARY target marginal (not just uniform)."""
    Q = np.asarray(P, np.float64) + 1e-12
    col_target = target_prior * Q.sum()
    for _ in range(iters):
        Q = Q / Q.sum(1, keepdims=True)
        Q = Q / Q.sum(0, keepdims=True) * col_target
    return Q

def balanced_select(P, frac, prior):
    """Class-balanced selection, quota proportional to the (estimated) prior."""
    pred = P.argmax(1); conf = P.max(1); keep = np.zeros(len(P), bool)
    total = int(len(P)*frac)
    for c in range(C):
        idx = np.where(pred==c)[0]
        if len(idx)==0: continue
        k = min(len(idx), max(1, int(total*prior[c])))
        keep[idx[np.argsort(-conf[idx])[:k]]] = True
    return keep, pred

def mahalanobis_scores(Ftr, ytr, Fte):
    """Distance to nearest class centroid (shared covariance) - open-set score."""
    mus = np.stack([Ftr[ytr==c].mean(0) for c in range(C)])
    Xc = np.concatenate([Ftr[ytr==c]-mus[c] for c in range(C)])
    cov = np.cov(Xc.T) + np.eye(Ftr.shape[1])*1e-3
    P = np.linalg.pinv(cov)
    d = np.stack([np.einsum('ij,jk,ik->i', Fte-mus[c], P, Fte-mus[c]) for c in range(C)], 1)
    return -d.min(1)          # higher = more in-distribution

def main():
    Ftr = np.load(f"{OUT}/gbif_{SIZE}.npy"); ytr = np.load(f"{OUT}/gbif_labels.npy")
    F   = np.load(f"{OUT}/dw_full_{SIZE}.npy"); y = np.load(f"{OUT}/dw_full_labels.npy")
    is_weed = y < C
    print(f"DeepWeeds full: {F.shape}, weeds={is_weed.sum()}, negatives={(~is_weed).sum()}")
    print(f"natural weed distribution: {np.bincount(y[is_weed], minlength=C)}")

    clf = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
    P_all = clf.predict_proba(F)
    Pw, yw = P_all[is_weed], y[is_weed]
    src_prior = np.bincount(ytr, minlength=C)/len(ytr)
    res = {}

    # ---------- 1. CLOSED-SET on the natural, imbalanced weed subset ----------
    res["closed_none"] = dict(acc=round(float(accuracy_score(yw, Pw.argmax(1))),4),
                              macro_f1=round(float(f1_score(yw, Pw.argmax(1), average="macro")),4))
    true_prior = np.bincount(yw, minlength=C)/len(yw)
    est = em_prior(Pw, src_prior)
    res["prior_estimation"] = dict(true=[round(float(v),4) for v in true_prior],
                                   estimated=[round(float(v),4) for v in est],
                                   L1_error=round(float(np.abs(est-true_prior).sum()),4))
    for tag, pri in [("uniform", np.full(C,1.0/C)), ("estimated", est), ("oracle", true_prior)]:
        Pa = align(Pw, pri)
        res[f"closed_DA_{tag}"] = dict(acc=round(float(accuracy_score(yw, Pa.argmax(1))),4),
                                       macro_f1=round(float(f1_score(yw, Pa.argmax(1), average="macro")),4))
    # DA + class-balanced self-training with the ESTIMATED prior
    P = align(Pw, est)
    for r in range(3):
        keep, pred = balanced_select(P, 0.5+0.15*r, est)
        if keep.sum() < C: break
        c2 = LogisticRegression(max_iter=3000).fit(
            np.concatenate([Ftr, F[is_weed][keep]]), np.concatenate([ytr, pred[keep]]))
        P = align(c2.predict_proba(F[is_weed]), est)
    res["closed_DA+CBST_estimated"] = dict(acc=round(float(accuracy_score(yw, P.argmax(1))),4),
                                           macro_f1=round(float(f1_score(yw, P.argmax(1), average="macro")),4))

    # ---------- 2. OPEN-SET: reject the 9,106 unseen negatives ----------
    msp = P_all.max(1)
    energy = np.log(np.exp(clf.decision_function(F)).sum(1))
    maha = mahalanobis_scores(Ftr, ytr, F)
    for tag, s in [("MSP", msp), ("energy", energy), ("mahalanobis", maha)]:
        res[f"openset_auroc_{tag}"] = round(float(roc_auc_score(is_weed.astype(int), s)), 4)

    print("\n=== CLOSED-SET (natural imbalance, weeds only) ===")
    for k in ["closed_none","closed_DA_uniform","closed_DA_estimated","closed_DA_oracle","closed_DA+CBST_estimated"]:
        print(f"  {k:28s} acc={res[k]['acc']*100:5.1f}%  macroF1={res[k]['macro_f1']:.3f}")
    print(f"\n  prior L1 error (EM vs true) = {res['prior_estimation']['L1_error']:.3f}")
    print("\n=== OPEN-SET rejection of 9,106 unseen negatives (AUROC) ===")
    for t in ["MSP","energy","mahalanobis"]:
        print(f"  {t:12s} {res[f'openset_auroc_{t}']:.3f}")
    print("\n=== per-class (closed-set, final) ===")
    print(classification_report(yw, P.argmax(1), target_names=CLASSES, digits=3, zero_division=0))

    json.dump(res, open(f"{OUT}/realistic_results.json","w"), indent=2)
    print(f"saved -> {OUT}/realistic_results.json")

if __name__ == "__main__":
    main()
