"""
HD(LM)^2D plus a label-free adaptation stage.

Keeps the paper's pipeline unchanged — fine-tuned CNN backbone, penultimate features, classical
head with the same default settings as hd_heads.py — and adds three techniques between the
features and the classifier, using only *unlabelled* DeepWeeds features:
  restd   re-standardise target features with DeepWeeds' own mean/variance
  align   Sinkhorn prior alignment of the head's outputs (assumes ~uniform class prior)
  st      class-balanced self-training of the head (20/40/60% most confident per class, 3 rounds)
  all     all three together
Run on the cached forward-direction features (GBIF-400 -> DeepWeeds) for all six backbones and
the paper's five consistent heads (SVM, LR, HGB, kNN, RF).
"""
import os, json, time, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

from hd_common import BACKBONES, metrics, NUM
from hd_heads import HEADS

warnings.filterwarnings("ignore")
OUT, SRC = "results_hd_adapt", "results_hd"
USE = ["SVM", "LR", "HGB", "kNN", "RF"]
VARIANTS = ["base", "restd", "align", "st", "all"]

def probs(clf, X):
    if hasattr(clf, "predict_proba"): return clf.predict_proba(X)
    s = clf.decision_function(X); e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def sinkhorn(P, iters=100):
    Q = P.astype(np.float64) + 1e-9; col = Q.sum() / NUM
    for _ in range(iters):
        Q /= Q.sum(1, keepdims=True); Q = Q / Q.sum(0, keepdims=True) * col
    return Q

def fit_predict(head, Xs, ys, Xt, restd):
    """Paper pipeline: StandardScaler fit on source -> head. With restd, target is standardised
    with its own statistics instead of the source's."""
    ss = StandardScaler().fit(Xs); clf = HEADS[head]().fit(ss.transform(Xs), ys)
    Zt = StandardScaler().fit_transform(Xt) if restd else ss.transform(Xt)
    return probs(clf, Zt), clf, ss

def adapt(head, Xtr, ytr, Xt, restd=False, align=False, st=False):
    P, _, _ = fit_predict(head, Xtr, ytr, Xt, restd)
    if st:
        for frac in (0.2, 0.4, 0.6):
            Q = sinkhorn(P) if align else P
            pred, conf = Q.argmax(1), Q.max(1); keep = []
            for c in range(NUM):
                ix = np.where(pred == c)[0]
                if len(ix): keep += list(ix[np.argsort(-conf[ix])][:max(1, int(frac * len(ix)))])
            keep = np.array(keep)
            P, _, _ = fit_predict(head, np.vstack([Xtr, Xt[keep]]), np.r_[ytr, pred[keep]], Xt, restd)
    return sinkhorn(P) if align else P

def run(b):
    path = f"{OUT}/{b}.json"
    if os.path.exists(path): print(f"[{b}] done, skipping"); return json.load(open(path))
    d = np.load(f"{SRC}/feats/{b}.npz"); sk = "Xsh" if "Xsh" in d else "Xdw"
    Xtr, Xte, Xt = (d[k].astype(np.float32) for k in ("Xtr", "Xte", sk))
    ytr, yte, yt = d["ytr"], d["yte"], d["ysh" if "ysh" in d else "ydw"]
    R, t0 = {}, time.time()
    for h in USE:
        P_te, _, _ = fit_predict(h, Xtr, ytr, Xte, restd=False)
        R[h] = {"in_domain": metrics(yte, P_te.argmax(1))}
        cfg = {"base": {}, "restd": dict(restd=True), "align": dict(align=True), "st": dict(st=True),
               "all": dict(restd=True, align=True, st=True)}
        for v in VARIANTS:
            R[h][v] = metrics(yt, adapt(h, Xtr, ytr, Xt, **cfg[v]).argmax(1))
        print(f"  [{b}] {h:4s} " + "  ".join(f"{v}={R[h][v]['acc']*100:5.1f}" for v in VARIANTS), flush=True)
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(path, "w"), indent=2); return R

def report(all_R):
    L = ["# HD(LM)²D + label-free adaptation stage (GBIF-400 → DeepWeeds)\n",
         "Same fine-tuned backbones and classical heads as the paper replication; adaptation uses "
         "unlabelled DeepWeeds features only.\n",
         "## DeepWeeds accuracy: paper pipeline vs + all three techniques\n",
         "| Backbone | " + " | ".join(f"{h} base → all" for h in USE) + " |", "|---|" + "---|" * len(USE)]
    for b, R in all_R.items():
        L.append(f"| {b} | " + " | ".join(f"{R[h]['base']['acc']*100:.1f} → **{R[h]['all']['acc']*100:.1f}**"
                                          for h in USE) + " |")
    L += ["\n## Mean gain over the paper pipeline, across 6 backbones (DeepWeeds, points)\n",
          "| Head | + restd | + align | + st | **+ all** | all improves |", "|---|---|---|---|---|---|"]
    for h in USE:
        g = {v: [all_R[b][h][v]["acc"] - all_R[b][h]["base"]["acc"] for b in all_R] for v in VARIANTS[1:]}
        L.append(f"| {h} | " + " | ".join(f"{np.mean(g[v])*100:+.1f}" for v in VARIANTS[1:-1])
                 + f" | **{np.mean(g['all'])*100:+.1f}** | {sum(x > 0 for x in g['all'])}/{len(all_R)} |")
    best = max(((b, h) for b in all_R for h in USE), key=lambda k: all_R[k[0]][k[1]]["all"]["acc"])
    L.append(f"\nBest: **{best[0]} + {best[1]} + all three = {all_R[best[0]][best[1]]['all']['acc']*100:.1f}%** "
             f"on DeepWeeds (paper pipeline best: RegNetY-32GF + SVM = 45.7%). Prior alignment assumes "
             f"a roughly uniform DeepWeeds class prior.")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    report({b: run(b) for b in BACKBONES})
