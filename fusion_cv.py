"""
Step 1 — repeated grouped cross-validation with paired significance tests.

Settles which in-domain differences are real. Every frozen candidate (single backbones, fusions,
CLIP-336, multi-view TTA and PCA-balanced variants, whichever caches exist) is scored on the SAME
25 folds: 5 repeats x 5-fold StratifiedGroupKFold over all 3,200 GBIF images, with near-duplicate
groups kept inside one fold. Each candidate is compared with the reference, DINOv2-448 + SVM
(the best single model), fold by fold:
  - corrected resampled t-test (Nadeau & Bengio, 2003), which accounts for the overlap between
    training sets across folds that makes a naive paired t-test over-confident;
  - Wilcoxon signed-rank test as a non-parametric check;
  - Holm correction across all comparisons.
RegNetY is excluded: its features were fine-tuned on the fixed training split.
Writes results_fusion_cv/.
"""
import os, json, time, warnings
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

from fusion import build_features, dedup_info, model, concat
from fusion_pca import pca_model

warnings.filterwarnings("ignore")
OUT, REPEATS, FOLDS = "results_fusion_cv", 5, 5
REFERENCE = "dinov2_448 | SVM"
EXTRA = {"clip336": "results_clip336/feats/clip336_gbif.npz",
         "dinov2_448_tta": "results_tta/feats/dinov2_448_tta_gbif.npz",
         "clip_l14_tta": "results_tta/feats/clip_l14_tta_gbif.npz"}
CANDIDATES = [
    (("dinov2_448",), "SVM", False), (("dinov2_224",), "SVM", False), (("dinov2_448",), "LR", False),
    (("clip_l14",), "SVM", False), (("clip336",), "SVM", False),
    (("dinov2_448_tta",), "SVM", False), (("clip_l14_tta",), "SVM", False),
    (("dinov2_448", "clip_l14"), "SVM", False), (("dinov2_448", "clip_l14"), "LR", False),
    (("dinov2_448", "clip_l14"), "Stack", False),
    (("dinov2_448", "clip336"), "SVM", False), (("dinov2_448", "clip336"), "Stack", False),
    (("dinov2_448_tta", "clip_l14_tta"), "SVM", False), (("dinov2_448_tta", "clip_l14_tta"), "Stack", False),
    (("dinov2_448", "clip_l14"), "SVM", True), (("dinov2_448", "clip_l14"), "Stack", True),
]

def corrected_t(d, test_train_ratio):
    k = len(d); v = np.var(d, ddof=1)
    if v == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * v)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))

def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j])); adj[j] = run
    return adj

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    S, gys, *_ , itr, ite = build_features()
    for n, p in EXTRA.items():
        if os.path.exists(p): S[n] = dict(all=np.load(p)["X"])
    groups, _ = dedup_info(S["dinov2_224"]["all"], itr, ite)
    folds = [f for r in range(REPEATS)
             for f in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=r).split(np.zeros(len(gys)), gys, groups)]
    ratio = np.mean([len(te) / len(tr) for tr, te in folds])
    acc = {}
    for names, kind, pca in CANDIDATES:
        if any(n not in S or S[n].get("all") is None for n in names): print(f"  skip {names}: no cache"); continue
        X, blocks = concat(S, names, "all"); key = f"{'+'.join(names)} | {'PCA-256 ' if pca else ''}{kind}"
        acc[key] = [float(((pca_model if pca else model)(kind, blocks).fit(X[tr], gys[tr]).predict(X[te]) == gys[te]).mean())
                    for tr, te in folds]
        print(f"  {key:46s} {np.mean(acc[key])*100:5.2f} ± {np.std(acc[key])*100:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    ref = np.array(acc[REFERENCE]); comps = [k for k in acc if k != REFERENCE]
    rows = []
    for k in comps:
        d = np.array(acc[k]) - ref; t, p = corrected_t(d, ratio)
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append(dict(key=k, mean=float(np.mean(acc[k])), std=float(np.std(acc[k])), delta=float(d.mean()),
                         t=t, p_corrected=p, p_wilcoxon=float(w)))
    for r, a in zip(rows, holm(np.array([r["p_corrected"] for r in rows]))): r["p_holm"] = float(a)
    json.dump(dict(folds=len(folds), repeats=REPEATS, reference=REFERENCE, accuracies=acc, comparisons=rows,
                   seconds=round(time.time() - t0, 1)), open(f"{OUT}/results.json", "w"), indent=2)

    L = ["# Repeated grouped CV with paired tests (in-domain GBIF accuracy)\n",
         f"{REPEATS}×{FOLDS} = {len(folds)} folds over all 3,200 GBIF images, near-duplicate groups kept within one "
         f"fold; every candidate on identical folds. Reference: **{REFERENCE}** = "
         f"{np.mean(ref)*100:.2f} ± {np.std(ref)*100:.2f}%. p-values: corrected resampled t-test "
         "(Nadeau & Bengio 2003), Holm-adjusted; Wilcoxon signed-rank as a check.\n",
         "| Model | Classifier | Accuracy (mean ± std) | Δ vs reference | p (corrected t) | p (Holm) | p (Wilcoxon) | Significant at 0.05 (Holm)? |",
         "|---|---|---|---|---|---|---|---|",
         f"| {REFERENCE.split(' | ')[0]} | {REFERENCE.split(' | ')[1]} | **{np.mean(ref)*100:.2f} ± {np.std(ref)*100:.2f}** | reference | — | — | — | — |"]
    for r in sorted(rows, key=lambda r: -r["mean"]):
        n, c = r["key"].split(" | ")
        sig = ("yes, better" if r["delta"] > 0 else "yes, worse") if r["p_holm"] < 0.05 else "no"
        L.append(f"| {n} | {c} | {r['mean']*100:.2f} ± {r['std']*100:.2f} | {r['delta']*100:+.2f} | "
                 f"{r['p_corrected']:.3f} | {r['p_holm']:.3f} | {r['p_wilcoxon']:.3f} | {sig} |")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)
