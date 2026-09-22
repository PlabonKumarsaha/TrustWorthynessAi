"""
Stage 2 of the HD(LM)^2D replication: refit the paper's 12 classical heads on the cached
penultimate features of each fine-tuned backbone, and score each on GBIF test (in-domain)
and DeepWeeds (shift). Heads are fit on the GBIF training split only.

All heads use scikit-learn/XGBoost defaults (the paper reports no tuning) behind a
StandardScaler. Resumable per backbone.
"""
import os, json, time, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from hd_common import BACKBONES, OUT, metrics

warnings.filterwarnings("ignore")
SEED = 42
# order follows the paper's Table 14 ranking
HEADS = {
    "SVM":       lambda: SVC(),
    "LR":        lambda: LogisticRegression(max_iter=3000),
    "HGB":       lambda: HistGradientBoostingClassifier(random_state=SEED),
    "kNN":       lambda: KNeighborsClassifier(),
    "RF":        lambda: RandomForestClassifier(n_jobs=-1, random_state=SEED),
    "XGBoost":   lambda: XGBClassifier(n_jobs=-1, random_state=SEED, tree_method="hist"),
    "LDA":       lambda: LinearDiscriminantAnalysis(),
    "Bagging":   lambda: BaggingClassifier(n_jobs=-1, random_state=SEED),
    "GradBoost": lambda: GradientBoostingClassifier(random_state=SEED),
    "AdaBoost":  lambda: AdaBoostClassifier(random_state=SEED),
    "DT":        lambda: DecisionTreeClassifier(random_state=SEED),
    "NB":        lambda: GaussianNB(),
}

def run(name):
    feat_path, out_path = f"{OUT}/feats/{name}.npz", f"{OUT}/heads/{name}.json"
    if not os.path.exists(feat_path): print(f"[{name}] no features yet, skipping"); return
    if os.path.exists(out_path): print(f"[{name}] heads already done, skipping"); return
    d = np.load(feat_path)
    sh_key = "Xsh" if "Xsh" in d else "Xdw"        # older forward-direction caches used Xdw
    Xtr, Xte, Xsh = (d[k].astype(np.float32) for k in ("Xtr", "Xte", sh_key))
    ytr, yte, ysh = d["ytr"], d["yte"], d["ysh" if "ysh" in d else "ydw"]
    res = {}
    for hn, mk in HEADS.items():
        t0 = time.time()
        clf = make_pipeline(StandardScaler(), mk()).fit(Xtr, ytr)
        res[hn] = dict(in_domain=metrics(yte, clf.predict(Xte)),
                       shift=metrics(ysh, clf.predict(Xsh)),
                       seconds=round(time.time() - t0, 1))
        print(f"  [{name}] {hn:9s} in-domain={res[hn]['in_domain']['acc']:.4f}  "
              f"shift={res[hn]['shift']['acc']:.4f}  ({res[hn]['seconds']}s)", flush=True)
    os.makedirs(f"{OUT}/heads", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)

if __name__ == "__main__":
    for nm in BACKBONES:
        run(nm)
