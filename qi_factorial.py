"""
Stage 2: the factorial study.

For every (backbone, resolution) cell we fit an identical linear probe on GBIF (source labels
only) and then apply each label-free adaptation method, scoring on DeepWeeds.

The claim under test:
  (a) backbone + resolution explain far more out-of-domain accuracy than the adaptation method;
  (b) the RANK ORDER of adaptation methods is not stable across backbones -- i.e. conclusions
      drawn on ResNet-50@224 do not transfer to a modern representation.
"""
import os, json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr

CACHE = "results_novel/factorial_cache"
C = 8
GRID = [("resnet50",224),("resnet50",336),("resnet50",448),
        ("clip_vitb32",224),
        ("dinov2_vitb14",224),("dinov2_vitb14",336),("dinov2_vitb14",448)]
SEEDS = [0,1,2]

def load(bb,size):
    f = lambda s: np.load(f"{CACHE}/{bb}_{size}_{s}.npy")
    y = lambda s: np.load(f"{CACHE}/labels_{s}.npy")
    return f("gbif_train"), y("gbif_train"), f("gbif_test"), y("gbif_test"), f("dw_test"), y("dw_test")

def sinkhorn(P, iters=100):
    Q = np.asarray(P, np.float64) + 1e-12; tgt = Q.sum()/C
    for _ in range(iters):
        Q = Q/Q.sum(1, keepdims=True); Q = Q/Q.sum(0, keepdims=True)*tgt
    return Q

def balanced_select(P, frac):
    pred = P.argmax(1); conf = P.max(1); keep = np.zeros(len(P), bool)
    for c in range(C):
        idx = np.where(pred==c)[0]
        if len(idx)==0: continue
        k = max(1,int(len(idx)*frac))
        keep[idx[np.argsort(-conf[idx])[:k]]] = True
    return keep, pred

def selftrain(Ftr,ytr,Fte,P,rounds=3,balanced=False,align=False,seed=0):
    for r in range(rounds):
        if balanced: keep,pred = balanced_select(P, 0.5+0.15*r)
        else:
            pred = P.argmax(1); keep = P.max(1) >= 0.95
        if keep.sum() < C: break
        clf = LogisticRegression(max_iter=3000, random_state=seed).fit(
            np.concatenate([Ftr,Fte[keep]]), np.concatenate([ytr,pred[keep]]))
        P = clf.predict_proba(Fte)
        if align: P = sinkhorn(P)
    return P

METHODS = ["none","DA","ST","CBST","DA+CBST"]

def run_cell(bb,size,seed):
    Ftr,ytr,Fin,yin,Fte,yte = load(bb,size)
    clf = LogisticRegression(max_iter=3000, random_state=seed).fit(Ftr,ytr)
    P0 = clf.predict_proba(Fte)
    indom = accuracy_score(yin, clf.predict(Fin))
    out = {}
    out["none"]    = accuracy_score(yte, P0.argmax(1))
    out["DA"]      = accuracy_score(yte, sinkhorn(P0).argmax(1))
    out["ST"]      = accuracy_score(yte, selftrain(Ftr,ytr,Fte,P0.copy(),seed=seed).argmax(1))
    out["CBST"]    = accuracy_score(yte, selftrain(Ftr,ytr,Fte,P0.copy(),balanced=True,seed=seed).argmax(1))
    out["DA+CBST"] = accuracy_score(yte, selftrain(Ftr,ytr,Fte,sinkhorn(P0),balanced=True,align=True,seed=seed).argmax(1))
    return indom, out

def main():
    cells, raw = {}, []
    for bb,size in GRID:
        if not os.path.exists(f"{CACHE}/{bb}_{size}_dw_test.npy"):
            print(f"[missing] {bb}@{size} - skipping"); continue
        accs = {m: [] for m in METHODS}; indoms = []
        for seed in SEEDS:
            indom, out = run_cell(bb,size,seed)
            indoms.append(indom)
            for m in METHODS: accs[m].append(out[m])
            raw.append(dict(backbone=bb,res=size,seed=seed,in_domain=indom,**out))
        cells[f"{bb}@{size}"] = dict(
            in_domain=round(float(np.mean(indoms)),4),
            **{m: dict(mean=round(float(np.mean(accs[m])),4), std=round(float(np.std(accs[m])),4))
               for m in METHODS})
        row = cells[f"{bb}@{size}"]
        print(f"{bb}@{size:<4} in-dom={row['in_domain']*100:5.1f} | " +
              "  ".join(f"{m}={row[m]['mean']*100:5.1f}" for m in METHODS), flush=True)

    # ---- (a) variance decomposition: what explains OOD accuracy? ----
    import pandas as pd
    df = pd.DataFrame(raw)
    grand = df["none"].mean()
    contrib = {
        "backbone": float(df.groupby("backbone")[METHODS].mean().mean(1).max() -
                          df.groupby("backbone")[METHODS].mean().mean(1).min()),
        "resolution": float(df[df.backbone!="clip_vitb32"].groupby("res")[METHODS].mean().mean(1).max() -
                            df[df.backbone!="clip_vitb32"].groupby("res")[METHODS].mean().mean(1).min()),
        "adaptation_method": float(df[METHODS].mean().max() - df[METHODS].mean().min()),
    }
    print("\n=== spread in OOD accuracy attributable to each factor (pp) ===")
    for k,v in contrib.items(): print(f"  {k:20s} {v*100:5.1f}")

    # ---- (b) does the method ranking survive a backbone change? ----
    print("\n=== method ranking per backbone (best->worst) ===")
    rankings = {}
    for bb in df.backbone.unique():
        sub = df[df.backbone==bb][METHODS].mean()
        rankings[bb] = list(sub.sort_values(ascending=False).index)
        print(f"  {bb:16s} {' > '.join(rankings[bb])}")
    pairs = {}
    for a,b in itertools.combinations(rankings,2):
        ra = [rankings[a].index(m) for m in METHODS]; rb = [rankings[b].index(m) for m in METHODS]
        rho = spearmanr(ra,rb).statistic
        pairs[f"{a} vs {b}"] = round(float(rho),3)
        print(f"  rank correlation {a} vs {b}: rho={rho:.3f}")

    json.dump(dict(cells=cells, factor_spread=contrib, rankings=rankings,
                   rank_correlation=pairs, raw=raw),
              open("results_novel/factorial.json","w"), indent=2, default=float)
    print("\nsaved -> results_novel/factorial.json")

if __name__ == "__main__":
    main()
