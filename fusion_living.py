"""
Multi-backbone feature fusion on GBIF-Living (living-plant photos only), DeepWeeds as shift test.

Same models and evaluation as fusion_649.py, on the GBIF-Living 60/20/20 split (seed 42) used by
the HD(LM)^2D living2dw run. Backbones: frozen DINOv2 ViT-B/14 @448 and CLIP ViT-L/14 (LAION-2B),
plus the fine-tuned RegNetY-32GF features from results_hd_living once that run exists.
Near-duplicate groups use DINOv2-448 features (cosine > 0.95). Writes results_fusion_living/.
"""
import os, json, time, argparse, warnings
import numpy as np

from fusion import extract_dinov2, extract_clip_l14, cached, dedup_info
from fusion_649 import fixed_split, cross_val
from hd_common import list_dir, _stratified, GBIF_LIVING_DIR, DW_DIR

warnings.filterwarnings("ignore")
OUT = "results_fusion_living"
REGNET = "results_hd_living/feats/regnet_y_32gf.npz"

def build_features():
    gfs, gys = list_dir(GBIF_LIVING_DIR); _, dys = list_dir(DW_DIR)
    idx = {f: i for i, f in enumerate(gfs)}
    (ftr, ytr), _, (fte, yte) = _stratified(GBIF_LIVING_DIR)
    itr, ite = np.array([idx[f] for f in ftr]), np.array([idx[f] for f in fte])
    os.makedirs(f"{OUT}/feats", exist_ok=True)
    G = {"dinov2_448": (cached(f"{OUT}/feats/dinov2_448_living.npz", lambda: extract_dinov2(448, gfs)),
                        np.load("results_fusion/feats/dinov2_448_dw.npz")["X"]),
         "clip_l14": (cached(f"{OUT}/feats/clip_l14_living.npz", lambda: extract_clip_l14(gfs)),
                      np.load("results_clip_backbone/feats/clip_vitl14_laion2b.npz")["Xdw"])}
    S = {k: dict(all=a, tr=a[itr], te=a[ite], dw=d) for k, (a, d) in G.items()}
    if os.path.exists(REGNET):
        r = np.load(REGNET); assert (r["ytr"] == ytr).all() and (r["yte"] == yte).all()
        S["regnet_y"] = dict(all=None, tr=r["Xtr"].astype(np.float32), te=r["Xte"].astype(np.float32),
                             dw=r["Xsh"].astype(np.float32))
    return S, gys, ytr, yte, dys, itr, ite

def configs(S):
    singles = [n for n in ("dinov2_448", "clip_l14", "regnet_y") if n in S]
    fusions = [("dinov2_448", "clip_l14")] + ([("dinov2_448", "clip_l14", "regnet_y")] if "regnet_y" in S else [])
    return [((n,), k) for n in singles for k in ("LR", "SVM")] + [(c, k) for c in fusions for k in ("LR", "SVM", "Stack")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-cv", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    S, gys, ytr, yte, ydw, itr, ite = build_features(); cfgs = configs(S)
    groups, clean = dedup_info(S["dinov2_448"]["all"], itr, ite)
    print(f"GBIF-Living {len(gys)} images; de-dup: {clean.sum()}/{len(clean)} test images kept", flush=True)
    F = fixed_split(S, ytr, yte, ydw, clean, cfgs)
    C = {} if a.no_cv else cross_val(S, gys, groups, cfgs)
    json.dump(dict(fixed_split=F, cv=C, n_images=int(len(gys)), n_test=int(len(yte)), n_test_dedup=int(clean.sum()),
                   with_regnet="regnet_y" in S, seconds=round(time.time() - t0, 1)),
              open(f"{OUT}/results.json", "w"), indent=2)
    L = [f"# Feature fusion — trained on GBIF-Living ({len(ytr):,} train), tested on GBIF-Living test and DeepWeeds\n",
         "| Model | Classifier | Test | Test (de-dup) | 5-fold CV | DeepWeeds |", "|---|---|---|---|---|---|"]
    for key, r in sorted(F.items(), key=lambda kv: -kv[1]["test"]["acc"]):
        n, k = key.split(" | "); cv = f"{C[key]['mean']*100:.1f} ± {C[key]['std']*100:.1f}" if key in C else "—"
        L.append(f"| {n} | {k} | {r['test']['acc']*100:.1f} | {r['test_dedup']['acc']*100:.1f} | {cv} | {r['deepweeds']['acc']*100:.1f} |")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)
