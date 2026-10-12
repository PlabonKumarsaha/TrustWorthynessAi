"""
Classifier search on fixed, cached features: which classifier gives the best GBIF -> DeepWeeds
accuracy when the backbone is held constant?

Feature sets (all cached, GBIF-400 split, DeepWeeds 8,403 as the shift test):
  clip_vitl14  frozen CLIP ViT-L/14 (results_clip_backbone)
  regnet_y     RegNetY-32GF fine-tuned end to end on GBIF (results_hd)
  dinov2       frozen DINOv2 ViT-B/14 at 224 px (results_shift)

A. Source-only classifiers, each tuned by 5-fold CV on the GBIF training split. DeepWeeds labels
   are used only for final scoring.
B. Label-free shift-aware variants built on the tuned logistic regression, using *unlabelled*
   DeepWeeds features (transductive, as in SHOT):
     target re-standardisation, Sinkhorn prior alignment (assumes ~uniform class prior),
     class-balanced self-training, and all three combined.
"""
import os, json, time, warnings
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from hd_common import gbif_splits, deepweeds_all, list_dir, GBIF_DIR, metrics, NUM

warnings.filterwarnings("ignore")
OUT = "results_clfsearch"
CV = StratifiedKFold(5, shuffle=True, random_state=0)

# ---------------- feature sets ----------------
def load_sets():
    S = {}
    d = np.load("results_clip_backbone/feats/clip_vitl14_laion2b.npz")
    S["clip_vitl14"] = (d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["Xdw"], d["ydw"])
    d = np.load("results_hd/feats/regnet_y_32gf.npz")
    S["regnet_y"] = tuple(d[k].astype(np.float32) for k in ("Xtr", "ytr", "Xte", "yte", "Xdw", "ydw"))
    X = np.load("results_shift/feats_dinov2_GBIF-400.npz")["X"]; Xd = np.load("results_shift/feats_dinov2_DeepWeeds.npz")["X"]
    fs, _ = list_dir(GBIF_DIR); idx = {f: i for i, f in enumerate(fs)}
    (ftr, ytr), _, (fte, yte) = gbif_splits(); _, ydw = deepweeds_all()
    S["dinov2"] = (X[[idx[f] for f in ftr]], ytr, X[[idx[f] for f in fte]], yte, Xd, ydw)
    return S

# ---------------- A: tuned source-only classifiers ----------------
def lin(clf): return make_pipeline(Normalizer(), StandardScaler(), clf)

SEARCH = {
    "LR":          (lin(LogisticRegression(max_iter=3000)), {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "SVM-linear":  (lin(SVC(kernel="linear")),               {"svc__C": [0.001, 0.01, 0.1, 1]}),
    "SVM-RBF":     (lin(SVC(kernel="rbf")),                  {"svc__C": [1, 10, 100]}),
    "kNN-cosine":  (make_pipeline(Normalizer(), KNeighborsClassifier(metric="cosine", weights="distance")),
                    {"kneighborsclassifier__n_neighbors": [5, 10, 20, 40]}),
    "MLP":         (lin(MLPClassifier((512,), max_iter=400, early_stopping=True, random_state=0)),
                    {"mlpclassifier__alpha": [1e-4, 1e-2, 1.0]}),
}

def prototypes(Xtr, ytr):
    Z = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    P = np.stack([Z[ytr == c].mean(0) for c in range(NUM)])
    return P / np.linalg.norm(P, axis=1, keepdims=True)

def proto_proba(P, X, T=20.0):
    s = (X / np.linalg.norm(X, axis=1, keepdims=True)) @ P.T * T
    e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def proba(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)
    s = model.decision_function(X); e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

# ---------------- B: label-free shift-aware variants ----------------
def sinkhorn(P, iters=100):
    Q = P.astype(np.float64) + 1e-9; col = Q.sum() / NUM
    for _ in range(iters):
        Q /= Q.sum(1, keepdims=True); Q = Q / Q.sum(0, keepdims=True) * col
    return Q

def lr_fit(X, y, C):
    return make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y)

def lr_restd(Xtr, ytr, Xt, C):
    """Fit on source with source statistics; predict target standardised with *target* statistics."""
    n = Normalizer(); m = LogisticRegression(C=C, max_iter=3000)
    m.fit(StandardScaler().fit_transform(n.transform(Xtr)), ytr)
    return m.predict_proba(StandardScaler().fit_transform(n.transform(Xt)))

def self_train(Xtr, ytr, Xt, C, restd=False, align=False, rounds=3):
    """Class-balanced self-training: each round adds the most confident target samples per
    predicted class (20/40/60%) as pseudo-labels and refits on source + pseudo-labelled target."""
    P = lr_restd(Xtr, ytr, Xt, C) if restd else lr_fit(Xtr, ytr, C).predict_proba(Xt)
    for r, frac in enumerate([0.2, 0.4, 0.6][:rounds]):
        Q = sinkhorn(P) if align else P
        pred, conf = Q.argmax(1), Q.max(1); keep = []
        for c in range(NUM):
            ix = np.where(pred == c)[0]
            if len(ix): keep += list(ix[np.argsort(-conf[ix])][:max(1, int(frac * len(ix)))])
        keep = np.array(keep)
        X2, y2 = np.vstack([Xtr, Xt[keep]]), np.r_[ytr, pred[keep]]
        P = lr_restd(X2, y2, Xt, C) if restd else lr_fit(X2, y2, C).predict_proba(Xt)
    return sinkhorn(P) if align else P

def run(name, data):
    Xtr, ytr, Xte, yte, Xdw, ydw = data; t0 = time.time(); R = {"A_source_only": {}, "B_shift_aware": {}}
    fitted = {}
    for k, (est, grid) in SEARCH.items():
        gs = GridSearchCV(est, grid, cv=CV, n_jobs=-1).fit(Xtr, ytr); fitted[k] = gs.best_estimator_
        R["A_source_only"][k] = dict(best_params=gs.best_params_, cv_acc=round(float(gs.best_score_), 4),
                                     in_domain=metrics(yte, gs.predict(Xte)), shift=metrics(ydw, gs.predict(Xdw)))
        print(f"  [{name}] {k:12s} cv={gs.best_score_:.4f} GBIF={R['A_source_only'][k]['in_domain']['acc']:.4f} "
              f"DW={R['A_source_only'][k]['shift']['acc']:.4f}  {gs.best_params_}", flush=True)
    P = prototypes(Xtr, ytr)
    R["A_source_only"]["Prototype"] = dict(best_params={}, cv_acc=None,
                                           in_domain=metrics(yte, proto_proba(P, Xte).argmax(1)),
                                           shift=metrics(ydw, proto_proba(P, Xdw).argmax(1)))
    # soft-vote ensemble of the probabilistic classifiers
    def ens(X): return sum(proba(fitted[k], X) for k in ("LR", "MLP", "kNN-cosine")) + proto_proba(P, X)
    R["A_source_only"]["Ensemble"] = dict(best_params={"members": ["LR", "MLP", "kNN-cosine", "Prototype"]},
                                          cv_acc=None, in_domain=metrics(yte, ens(Xte).argmax(1)),
                                          shift=metrics(ydw, ens(Xdw).argmax(1)))
    C = R["A_source_only"]["LR"]["best_params"]["logisticregression__C"]
    base = lr_fit(Xtr, ytr, C).predict_proba(Xdw)
    R["B_shift_aware"] = {
        "LR (tuned, source-only)":        metrics(ydw, base.argmax(1)),
        "+ target re-standardisation":     metrics(ydw, lr_restd(Xtr, ytr, Xdw, C).argmax(1)),
        "+ prior alignment (Sinkhorn)":    metrics(ydw, sinkhorn(base).argmax(1)),
        "+ class-balanced self-training":  metrics(ydw, self_train(Xtr, ytr, Xdw, C).argmax(1)),
        "+ all three":                     metrics(ydw, self_train(Xtr, ytr, Xdw, C, restd=True, align=True).argmax(1)),
    }
    for k, v in R["B_shift_aware"].items(): print(f"  [{name}] B {k:32s} DW={v['acc']:.4f}", flush=True)
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{OUT}/{name}.json", "w"), indent=2, default=str)
    return R

def report(all_R):
    L = ["# Classifier search on fixed features (GBIF-400 → DeepWeeds)\n",
         "## A. Source-only classifiers (tuned by 5-fold CV on GBIF train)\n",
         "| Classifier | " + " | ".join(f"{n} GBIF | {n} **DW**" for n in all_R) + " |",
         "|---|" + "---|---|" * len(all_R)]
    for k in list(next(iter(all_R.values()))["A_source_only"]):
        L.append(f"| {k} | " + " | ".join(
            f"{R['A_source_only'][k]['in_domain']['acc']*100:.1f} | **{R['A_source_only'][k]['shift']['acc']*100:.1f}**"
            for R in all_R.values()) + " |")
    L += ["\n## B. Label-free shift-aware variants (DeepWeeds accuracy)\n",
          "| Variant | " + " | ".join(all_R) + " |", "|---|" + "---|" * len(all_R)]
    for k in next(iter(all_R.values()))["B_shift_aware"]:
        L.append(f"| {k} | " + " | ".join(f"{R['B_shift_aware'][k]['acc']*100:.1f}" for R in all_R.values()) + " |")
    L.append("\nReference, untuned defaults: CLIP L/14 + SVM 43.6%; RegNetY + SVM 45.7%; "
             "Sinkhorn alignment assumes a roughly uniform DeepWeeds class prior.")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    all_R = {}
    for name, data in load_sets().items():
        print(f"=== {name} (dim {data[0].shape[1]}) ===", flush=True)
        all_R[name] = run(name, data)
    report(all_R)
