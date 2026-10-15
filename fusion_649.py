"""
Multi-backbone feature fusion on GBIF-649 (in-domain accuracy), with DeepWeeds as the shift test.

Same models as fusion.py (per-block L2 norm -> concat -> StandardScaler -> LR / RBF-SVM with C
tuned by inner CV, or stacking), moved from GBIF-400 to the larger GBIF-649 set so every result in
the final report uses one GBIF dataset.

Split: the GBIF-649 60/20/20 split used by the HD(LM)^2D GBIF-649 run (hd_common._stratified,
seed 42): train 3,112 / val 1,032 / test 1,048. Models are fit on train only, as the 12 classical
heads were. The fine-tuned RegNetY-32GF features come from that run (results_hd_900).

Reported: test accuracy, de-duplicated test accuracy (test images with a DINOv2 cosine > 0.95
near-copy in train removed), 5-fold grouped CV over all 5,192 images (frozen backbones only),
and DeepWeeds accuracy (8,403 images, no adaptation). Writes results_fusion_649/.
"""
import os, json, time, argparse, warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from fusion import (extract_dinov2, extract_clip_l14, cached, dedup_info, concat, model)
from hd_common import list_dir, _stratified, GBIF_LARGE_DIR, DW_DIR, metrics

warnings.filterwarnings("ignore")
OUT = "results_fusion_649"
SINGLES = ["dinov2_224", "dinov2_448", "clip_l14", "regnet_y"]
FUSIONS = [("dinov2_448", "clip_l14"), ("dinov2_224", "clip_l14"), ("dinov2_448", "clip_l14", "regnet_y")]

def configs():
    for n in SINGLES:
        for k in ("LR", "SVM"): yield (n,), k
    for c in FUSIONS:
        for k in ("LR", "SVM", "Stack"): yield c, k

def build_features():
    gfs, gys = list_dir(GBIF_LARGE_DIR); _, dys = list_dir(DW_DIR)
    idx = {f: i for i, f in enumerate(gfs)}
    (ftr, ytr), _, (fte, yte) = _stratified(GBIF_LARGE_DIR)
    itr, ite = np.array([idx[f] for f in ftr]), np.array([idx[f] for f in fte])
    os.makedirs(f"{OUT}/feats", exist_ok=True)
    G = {
        "dinov2_224": (np.load("results_shift/feats_dinov2_GBIF-649.npz")["X"],
                       np.load("results_shift/feats_dinov2_DeepWeeds.npz")["X"]),
        "dinov2_448": (np.load("results_shot_full/feats/dinov2_448_gbif649.npz")["X"],
                       np.load("results_fusion/feats/dinov2_448_dw.npz")["X"]),
        "clip_l14": (cached(f"{OUT}/feats/clip_l14_gbif649.npz", lambda: extract_clip_l14(gfs)),
                     np.load("results_clip_backbone/feats/clip_vitl14_laion2b.npz")["Xdw"]),
    }
    for k, (a, _) in G.items(): assert len(a) == len(gfs), (k, a.shape)
    S = {k: dict(all=a, tr=a[itr], te=a[ite], dw=d) for k, (a, d) in G.items()}
    r = np.load("results_hd_900/feats/regnet_y_32gf.npz")
    assert (r["ytr"] == ytr).all() and (r["yte"] == yte).all()
    S["regnet_y"] = dict(all=None, tr=r["Xtr"].astype(np.float32), te=r["Xte"].astype(np.float32),
                         dw=r["Xsh"].astype(np.float32))
    return S, gys, ytr, yte, dys, itr, ite

def fixed_split(S, ytr, yte, ydw, clean, cfgs=None):
    R = {}
    for names, kind in (cfgs or configs()):
        Xtr, blocks = concat(S, names, "tr"); Xte, _ = concat(S, names, "te"); Xdw, _ = concat(S, names, "dw")
        t0 = time.time(); m = model(kind, blocks).fit(Xtr, ytr); pte = m.predict(Xte)
        key = f"{'+'.join(names)} | {kind}"
        R[key] = dict(test=metrics(yte, pte), test_dedup=metrics(yte[clean], pte[clean]),
                      deepweeds=metrics(ydw, m.predict(Xdw)), seconds=round(time.time() - t0, 1))
        print(f"  [fixed] {key:42s} test={R[key]['test']['acc']*100:5.1f}  dedup={R[key]['test_dedup']['acc']*100:5.1f}"
              f"  DW={R[key]['deepweeds']['acc']*100:5.1f}  ({R[key]['seconds']:.0f}s)", flush=True)
    return R

def cross_val(S, gys, groups, cfgs=None):
    R = {}; cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for names, kind in (cfgs or configs()):
        if any(S[n]["all"] is None for n in names): continue
        X, blocks = concat(S, names, "all"); accs = []
        for tr, te in cv.split(X, gys, groups):
            accs.append(float((model(kind, blocks).fit(X[tr], gys[tr]).predict(X[te]) == gys[te]).mean()))
        key = f"{'+'.join(names)} | {kind}"
        R[key] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), folds=accs)
        print(f"  [cv]    {key:42s} {np.mean(accs)*100:5.1f} ± {np.std(accs)*100:.1f}", flush=True)
    return R

def report(F, C, n_clean, n_test):
    L = ["# Multi-backbone feature fusion — trained on GBIF-649, tested on GBIF-649 test and DeepWeeds\n",
         f"Train 3,112 → test {n_test:,}; de-duplicated test = {n_clean:,} test images with no near-copy in training. "
         "5-fold CV: all 5,192 GBIF-649 images, near-duplicate groups kept within one fold, frozen backbones only. "
         "DeepWeeds: all 8,403 weed images, no adaptation.\n",
         "| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | 5-fold CV | DeepWeeds |",
         "|---|---|---|---|---|---|---|"]
    for key, r in sorted(F.items(), key=lambda kv: -kv[1]["test"]["acc"]):
        names, kind = key.split(" | ")
        cv = f"{C[key]['mean']*100:.1f} ± {C[key]['std']*100:.1f}" if key in C else "—"
        L.append(f"| {names} | {kind} | **{r['test']['acc']*100:.1f}** | **{r['test_dedup']['acc']*100:.1f}** | "
                 f"{r['test']['macro_f1']:.3f} | {cv} | {r['deepweeds']['acc']*100:.1f} |")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-cv", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    S, gys, ytr, yte, ydw, itr, ite = build_features()
    groups, clean = dedup_info(S["dinov2_224"]["all"], itr, ite)
    print(f"de-dup: {clean.sum()}/{len(clean)} test images kept; {len(set(groups))} near-duplicate groups "
          f"over {len(groups)} GBIF-649 images", flush=True)
    F = fixed_split(S, ytr, yte, ydw, clean)
    C = {} if a.no_cv else cross_val(S, gys, groups)
    json.dump(dict(fixed_split=F, cv=C, n_test=int(len(yte)), n_test_dedup=int(clean.sum()),
                   seconds=round(time.time() - t0, 1)), open(f"{OUT}/results.json", "w"), indent=2)
    report(F, C, int(clean.sum()), int(len(yte)))
