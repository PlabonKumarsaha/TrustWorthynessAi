"""
Phase 1 — multi-backbone deep-feature fusion for in-domain (GBIF) classification accuracy.

Extends the base paper's "deep features + classical classifier" from one backbone to several
complementary ones:
  dinov2_224 / dinov2_448   self-supervised ViT-B/14 (frozen; 448 = finer 14-px patch grid on
                            upsampled 224-px GBIF images)
  clip_l14                  language-supervised ViT-L/14, LAION-2B (frozen)
  regnet_y                  supervised RegNetY-32GF fine-tuned on GBIF train (results_hd)

Fusion:
  A. early fusion  — L2-normalise each backbone block, concatenate, standardise, LR or RBF-SVM
                     (C tuned by CV on the training data only)
  B. stacking      — per-backbone LR + RBF-SVM, meta-LR on their out-of-fold outputs

Evaluation:
  fixed split      train 1,920 -> test 640, plus a de-duplicated test subset (test images with a
                   near-copy in training, DINOv2 cosine > 0.95, removed); DeepWeeds as secondary
  5-fold CV        over all 3,200 GBIF images for frozen backbones only, near-duplicate groups kept
                   inside one fold (StratifiedGroupKFold). regnet_y is excluded from CV because its
                   features were fine-tuned on the training split.
"""
import os, json, time, argparse, warnings
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import StackingClassifier

from hd_common import list_dir, gbif_splits, GBIF_DIR, DW_DIR, metrics

warnings.filterwarnings("ignore")
OUT = "results_fusion"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SINGLES = ["dinov2_224", "dinov2_448", "clip_l14", "regnet_y"]
FUSIONS = [("dinov2_448", "clip_l14"), ("dinov2_224", "clip_l14"), ("dinov2_448", "clip_l14", "regnet_y")]
# fixed base-learner settings for stacking: best C found by CV on GBIF train in classifier_search.py
STACK_LR_C, STACK_SVM_C = 0.01, 10

# ---------------- feature extraction (cached) ----------------
class Files(Dataset):
    def __init__(self, fs, tf): self.fs, self.tf = fs, tf
    def __len__(self): return len(self.fs)
    def __getitem__(self, i): return self.tf(Image.open(self.fs[i]).convert("RGB"))

@torch.no_grad()
def run_model(fn, tf, fs, bs):
    return torch.cat([fn(x.to(DEVICE)).float().cpu() for x in DataLoader(Files(fs, tf), batch_size=bs, num_workers=4)]).numpy()

def extract_dinov2(size, fs):
    import timm
    from timm.data import resolve_data_config, create_transform
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3, size, size)
    return run_model(m, create_transform(**cfg, is_training=False), fs, 32 if size > 224 else 64)

def extract_clip_l14(fs):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    m = m.to(DEVICE).eval()
    return run_model(lambda x: F.normalize(m.encode_image(x), dim=-1), pre, fs, 64)

def cached(path, make):
    if os.path.exists(path): return np.load(path)["X"]
    X = make(); np.savez(path, X=X); print(f"  cached {path} {X.shape}", flush=True); return X

def build_features():
    """Return {name: dict(all=GBIF 3,200 in list_dir order or None, tr, te, dw)} plus split indices."""
    gfs, gys = list_dir(GBIF_DIR); dfs, dys = list_dir(DW_DIR)
    idx = {f: i for i, f in enumerate(gfs)}
    (ftr, ytr), _, (fte, yte) = gbif_splits()
    itr, ite = np.array([idx[f] for f in ftr]), np.array([idx[f] for f in fte])
    os.makedirs(f"{OUT}/feats", exist_ok=True)
    G = {
        "dinov2_224": (np.load("results_shift/feats_dinov2_GBIF-400.npz")["X"],
                       np.load("results_shift/feats_dinov2_DeepWeeds.npz")["X"]),
        "dinov2_448": (cached(f"{OUT}/feats/dinov2_448_gbif.npz", lambda: extract_dinov2(448, gfs)),
                       cached(f"{OUT}/feats/dinov2_448_dw.npz", lambda: extract_dinov2(448, dfs))),
        "clip_l14": (cached(f"{OUT}/feats/clip_l14_gbif.npz", lambda: extract_clip_l14(gfs)),
                     np.load("results_clip_backbone/feats/clip_vitl14_laion2b.npz")["Xdw"]),
    }
    S = {k: dict(all=a, tr=a[itr], te=a[ite], dw=d) for k, (a, d) in G.items()}
    r = np.load("results_hd/feats/regnet_y_32gf.npz")
    S["regnet_y"] = dict(all=None, tr=r["Xtr"].astype(np.float32), te=r["Xte"].astype(np.float32),
                         dw=r["Xdw"].astype(np.float32))
    return S, gys, ytr, yte, dys, itr, ite

def dedup_info(Z, itr, ite, thr=0.95):
    """Near-duplicate groups over all GBIF images, and a mask of test images with no near-copy in train."""
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Z @ Z.T; np.fill_diagonal(S, 0)
    _, groups = connected_components(csr_matrix(S > thr), directed=False)
    clean = (S[np.ix_(ite, itr)] <= thr).all(1)
    return groups, clean

# ---------------- models ----------------
class BlockNorm(BaseEstimator, TransformerMixin):
    """L2-normalise each backbone's feature block separately so no backbone dominates by scale."""
    def __init__(self, blocks=None): self.blocks = blocks
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for a, b in self.blocks: X[:, a:b] /= np.linalg.norm(X[:, a:b], axis=1, keepdims=True) + 1e-9
        return X

class Cols(BaseEstimator, TransformerMixin):
    def __init__(self, a=0, b=None): self.a, self.b = a, b
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.a:self.b]

def concat(S, names, part):
    Xs = [S[n][part] for n in names]; ends = np.cumsum([x.shape[1] for x in Xs])
    return np.hstack(Xs), [(int(e - x.shape[1]), int(e)) for x, e in zip(Xs, ends)]

def model(kind, blocks, inner_cv=3):
    if kind == "LR":
        return GridSearchCV(make_pipeline(BlockNorm(blocks), StandardScaler(), LogisticRegression(max_iter=3000)),
                            {"logisticregression__C": [0.01, 0.1, 1]}, cv=inner_cv, n_jobs=-1)
    if kind == "SVM":
        return GridSearchCV(make_pipeline(BlockNorm(blocks), StandardScaler(), SVC()),
                            {"svc__C": [1, 10, 100]}, cv=inner_cv, n_jobs=-1)
    est = []
    for i, (a, b) in enumerate(blocks):
        pre = lambda: [Cols(a, b), BlockNorm([(0, b - a)]), StandardScaler()]
        est += [(f"lr{i}", make_pipeline(*pre(), LogisticRegression(C=STACK_LR_C, max_iter=3000))),
                (f"svm{i}", make_pipeline(*pre(), SVC(C=STACK_SVM_C)))]
    return StackingClassifier(est, final_estimator=LogisticRegression(max_iter=3000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)

def configs():
    for n in SINGLES:
        for k in ("LR", "SVM"): yield (n,), k
    for c in FUSIONS:
        for k in ("LR", "SVM", "Stack"): yield c, k

# ---------------- evaluation ----------------
def fixed_split(S, ytr, yte, ydw, clean):
    R = {}
    for names, kind in configs():
        Xtr, blocks = concat(S, names, "tr"); Xte, _ = concat(S, names, "te"); Xdw, _ = concat(S, names, "dw")
        t0 = time.time(); m = model(kind, blocks).fit(Xtr, ytr); pte = m.predict(Xte)
        key = f"{'+'.join(names)} | {kind}"
        R[key] = dict(test=metrics(yte, pte), test_dedup=metrics(yte[clean], pte[clean]),
                      deepweeds=metrics(ydw, m.predict(Xdw)), seconds=round(time.time() - t0, 1))
        print(f"  [fixed] {key:42s} test={R[key]['test']['acc']*100:5.1f}  dedup={R[key]['test_dedup']['acc']*100:5.1f}"
              f"  DW={R[key]['deepweeds']['acc']*100:5.1f}", flush=True)
    return R

def cross_val(S, gys, groups):
    R = {}; cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for names, kind in configs():
        if any(S[n]["all"] is None for n in names): continue
        X, blocks = concat(S, names, "all"); accs = []
        for tr, te in cv.split(X, gys, groups):
            accs.append(float((model(kind, blocks).fit(X[tr], gys[tr]).predict(X[te]) == gys[te]).mean()))
        key = f"{'+'.join(names)} | {kind}"
        R[key] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), folds=accs)
        print(f"  [cv]    {key:42s} {np.mean(accs)*100:5.1f} ± {np.std(accs)*100:.1f}", flush=True)
    return R

def report(F, C, n_clean):
    L = ["# Multi-backbone feature fusion — in-domain GBIF accuracy (Phase 1)\n",
         f"Fixed split: train 1,920 → test 640; de-duplicated test = {n_clean} test images with no near-copy "
         "in training. 5-fold CV: all 3,200 GBIF images, near-duplicate groups kept within one fold, "
         "frozen backbones only. DeepWeeds is a secondary column (no adaptation).\n",
         "| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | 5-fold CV | DeepWeeds |",
         "|---|---|---|---|---|---|---|"]
    for key, r in sorted(F.items(), key=lambda kv: -kv[1]["test_dedup"]["acc"]):
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
          f"over {len(groups)} GBIF images", flush=True)
    F = fixed_split(S, ytr, yte, ydw, clean)
    C = {} if a.no_cv else cross_val(S, gys, groups)
    json.dump(dict(fixed_split=F, cv=C, n_test_dedup=int(clean.sum()), seconds=round(time.time() - t0, 1)),
              open(f"{OUT}/results.json", "w"), indent=2)
    report(F, C, int(clean.sum()))
