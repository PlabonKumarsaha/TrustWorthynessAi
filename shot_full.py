"""
DINOv2-448 + SHOT with more source data: how far does the GBIF -> DeepWeeds accuracy go?

Reproduces the SHOT setup that gave 85.7% (baselines.py: 768->256 bottleneck + BatchNorm + ReLU,
linear classifier; source training 60 epochs; SHOT 40 epochs with the classifier frozen, loss =
entropy - diversity + 0.3 x centroid pseudo-label CE) and varies only the source data:

  original   the 2,240-image GBIF training split behind the 85.7% result (control)
  gbif400    all 3,200 GBIF-400 images
  combined   GBIF-400 + GBIF-649 (5,192), GBIF-649 images with a near-copy in GBIF-400 removed

All sources use the same frozen DINOv2 ViT-B/14 @448 features (fusion.py extractor), so only the
data changes. Target: the 8,403 DeepWeeds weed images (negatives excluded), unlabelled during
adaptation. Three seeds per source. Writes results_shot_full/.
"""
import os, json, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from fusion import extract_dinov2, cached
from hd_common import list_dir, GBIF_DIR, DW_DIR, metrics, NUM

warnings.filterwarnings("ignore")
OUT, DEV, SEEDS = "results_shot_full", "cpu", [0, 1, 2]
GBIF649_DIR, TRAIN_SPLIT_DIR = "weeds/global_900", "weeds/global_train"

class Net(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU())
        self.c = nn.Linear(h, NUM)
    def forward(self, x): return self.c(self.g(x))

def train_source(Ts, Ys, epochs=60):
    m = Net(Ts.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Ts))
        for i in range(0, len(Ts), 128):
            b = perm[i:i + 128]
            if len(b) < 2: continue
            loss = F.cross_entropy(m(Ts[b]), Ys[b]); opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def predict(m, X): m.eval(); return m(X).argmax(1).numpy()

def shot(src_model, Tt, epochs=40, beta=0.3):
    m = Net(Tt.shape[1]).to(DEV); m.load_state_dict(src_model.state_dict())
    for p in m.c.parameters(): p.requires_grad_(False)                 # freeze source hypothesis
    opt = torch.optim.AdamW(m.g.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.eval()
        with torch.no_grad():                                           # centroid pseudo-labels
            f = m.g(Tt); p = m.c(f).softmax(1)
            cent = (p.T @ f) / (p.sum(0)[:, None] + 1e-8)
            pl = torch.cdist(F.normalize(f, dim=1), F.normalize(cent, dim=1)).argmin(1)
            for _ in range(2):
                oh = F.one_hot(pl, NUM).float(); cent = (oh.T @ f) / (oh.sum(0)[:, None] + 1e-8)
                pl = torch.cdist(F.normalize(f, dim=1), F.normalize(cent, dim=1)).argmin(1)
        m.train(); perm = torch.randperm(len(Tt))
        for i in range(0, len(Tt), 128):
            b = perm[i:i + 128]
            if len(b) < 2: continue
            out = m(Tt[b]); pr = out.softmax(1)
            ent = -(pr * torch.log(pr + 1e-8)).sum(1).mean()
            mp = pr.mean(0); div = -(mp * torch.log(mp + 1e-8)).sum()
            loss = ent - div + beta * F.cross_entropy(out, pl[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def sources():
    gfs, gys = list_dir(GBIF_DIR)
    G = np.load("results_fusion/feats/dinov2_448_gbif.npz")["X"]
    base = {os.path.basename(f): i for i, f in enumerate(gfs)}
    tr_fs, _ = list_dir(TRAIN_SPLIT_DIR)
    itr = np.array([base[os.path.basename(f)] for f in tr_fs])
    nfs, nys = list_dir(GBIF649_DIR)
    N = cached(f"{OUT}/feats/dinov2_448_gbif649.npz", lambda: extract_dinov2(448, nfs))
    a = G / np.linalg.norm(G, axis=1, keepdims=True); b = N / np.linalg.norm(N, axis=1, keepdims=True)
    keep = np.concatenate([(b[i:i + 512] @ a.T).max(1) for i in range(0, len(b), 512)]) <= 0.95
    print(f"GBIF-649: dropped {int((~keep).sum())} near-copies of GBIF-400 images", flush=True)
    return {"original": (G[itr], gys[itr]),
            "gbif400": (G, gys),
            "combined": (np.vstack([G, N[keep]]), np.r_[gys, nys[keep]])}

if __name__ == "__main__":
    os.makedirs(f"{OUT}/feats", exist_ok=True); t0 = time.time()
    _, yt = list_dir(DW_DIR); Xt = np.load("results_fusion/feats/dinov2_448_dw.npz")["X"]
    R = {}
    for name, (Xs, ys) in sources().items():
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
        Ts, Tt = torch.tensor((Xs - mu) / sd).float(), torch.tensor((Xt - mu) / sd).float()
        Ys = torch.tensor(ys).long()
        lp = metrics(yt, LogisticRegression(max_iter=3000).fit(Xs, ys).predict(Xt))
        runs = []
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            src = train_source(Ts, Ys)
            runs.append(dict(seed=s, source_only=metrics(yt, predict(src, Tt)), shot=metrics(yt, predict(shot(src, Tt), Tt))))
        acc = lambda k: [r[k]["acc"] for r in runs]; f1 = [r["shot"]["macro_f1"] for r in runs]
        R[name] = dict(n_source=int(len(ys)), per_class=np.bincount(ys).tolist(), linear_probe=lp, runs=runs,
                       source_only_mean=float(np.mean(acc("source_only"))), source_only_std=float(np.std(acc("source_only"))),
                       shot_mean=float(np.mean(acc("shot"))), shot_std=float(np.std(acc("shot"))),
                       shot_f1_mean=float(np.mean(f1)))
        print(f"[{name:8s}] n={len(ys):5d}  linear={lp['acc']*100:5.1f}  source-only={R[name]['source_only_mean']*100:5.1f}"
              f"  SHOT={R[name]['shot_mean']*100:5.2f} ± {R[name]['shot_std']*100:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(results=R, seeds=SEEDS, n_target=int(len(yt)), seconds=round(time.time() - t0, 1)),
              open(f"{OUT}/results.json", "w"), indent=2)
    L = ["# DINOv2-448 + SHOT with more GBIF source data → DeepWeeds (8,403 weed images)\n",
         "Same frozen DINOv2-448 features, SHOT settings and target for every row; only the source data changes. "
         f"{len(SEEDS)} seeds (mean ± std).\n",
         "| Source | Images | Linear probe | Source-only MLP | **SHOT** | SHOT macro-F1 |", "|---|---|---|---|---|---|"]
    for n, r in R.items():
        L.append(f"| {n} | {r['n_source']:,} | {r['linear_probe']['acc']*100:.1f}% | "
                 f"{r['source_only_mean']*100:.1f} ± {r['source_only_std']*100:.1f}% | "
                 f"**{r['shot_mean']*100:.1f} ± {r['shot_std']*100:.1f}%** | {r['shot_f1_mean']:.3f} |")
    L.append("\nReference: the earlier run of this setup (original split, single seed) gave 85.7%.")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)
