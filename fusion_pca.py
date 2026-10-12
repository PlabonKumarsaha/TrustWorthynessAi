"""
Step 4 — dimension-balanced fusion: reduce every backbone to the same size before fusing.

In fusion.py, adding RegNetY-32GF lowered accuracy (96.6 -> 89.1): its 3,712 dims were 71% of
the fused vector, so it dominated the classifier. Here each backbone block is L2-normalised,
standardised and reduced to 256 dims by PCA (fitted on the training fold only, inside the
pipeline), so every backbone contributes equally. Run with and without RegNetY, and without PCA
for reference. Fixed GBIF-400 split only, because RegNetY's features were fine-tuned on the
training split. Writes results_fusion_pca/.
"""
import os, json, time, warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import StackingClassifier

from fusion import (build_features, dedup_info, model, concat, BlockNorm, Cols,
                    STACK_LR_C, STACK_SVM_C)
from hd_common import metrics

warnings.filterwarnings("ignore")
OUT, K = "results_fusion_pca", 256
COMBOS = [("dinov2_448", "clip_l14"), ("dinov2_448", "clip_l14", "regnet_y"), ("dinov2_448", "regnet_y")]

def block_pca(a, b):
    return make_pipeline(BlockNorm([(0, b - a)]), StandardScaler(), PCA(min(K, b - a), random_state=0))

def pca_model(kind, blocks):
    ct = ColumnTransformer([(f"b{i}", block_pca(a, b), slice(a, b)) for i, (a, b) in enumerate(blocks)])
    if kind == "LR":
        return GridSearchCV(make_pipeline(ct, StandardScaler(), LogisticRegression(max_iter=3000)),
                            {"logisticregression__C": [0.01, 0.1, 1]}, cv=3, n_jobs=-1)
    if kind == "SVM":
        return GridSearchCV(make_pipeline(ct, StandardScaler(), SVC()), {"svc__C": [1, 10, 100]}, cv=3, n_jobs=-1)
    est = []
    for i, (a, b) in enumerate(blocks):
        est += [(f"lr{i}", make_pipeline(Cols(a, b), block_pca(a, b), LogisticRegression(C=STACK_LR_C, max_iter=3000))),
                (f"svm{i}", make_pipeline(Cols(a, b), block_pca(a, b), SVC(C=STACK_SVM_C)))]
    return StackingClassifier(est, final_estimator=LogisticRegression(max_iter=3000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    S, gys, ytr, yte, ydw, itr, ite = build_features()
    _, clean = dedup_info(S["dinov2_224"]["all"], itr, ite)
    R = {}
    for names in COMBOS:
        Xtr, blocks = concat(S, names, "tr"); Xte, _ = concat(S, names, "te"); Xdw, _ = concat(S, names, "dw")
        for pca in (False, True):
            for kind in ("LR", "SVM", "Stack"):
                m = (pca_model if pca else model)(kind, blocks).fit(Xtr, ytr); p = m.predict(Xte)
                key = f"{'+'.join(names)} | {'PCA-256 ' if pca else ''}{kind}"
                R[key] = dict(test=metrics(yte, p), test_dedup=metrics(yte[clean], p[clean]),
                              deepweeds=metrics(ydw, m.predict(Xdw)))
                print(f"  {key:46s} test={R[key]['test']['acc']*100:5.1f}  de-dup={R[key]['test_dedup']['acc']*100:5.1f}"
                      f"  DW={R[key]['deepweeds']['acc']*100:5.1f}", flush=True)
    json.dump(dict(results=R, pca_dims=K, seconds=round(time.time() - t0, 1)), open(f"{OUT}/results.json", "w"), indent=2)
    from clip336 import report
    report(R, f"Step 4 — PCA-{K} dimension-balanced fusion (fixed GBIF-400 split)", f"{OUT}/report.md")
