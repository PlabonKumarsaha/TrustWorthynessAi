"""
Neighbourhood-based, label-free adaptation on cached features (GBIF-400 -> DeepWeeds).

Motivation: in hd_adapt/vlm_adapt the kNN head gained far more than any other (+12 points on
average), i.e. DeepWeeds images cluster well in feature space even where GBIF-trained decision
boundaries do not fit them. Graph label propagation (Zhou et al., 2004) exploits exactly that:
predictions are smoothed over a kNN graph of the *unlabelled* target features.

Seed predictions: LR (hd_heads default) with target re-standardisation + Sinkhorn alignment —
the same techniques as hd_adapt. Hyperparameters fixed in advance (k=10, alpha=0.9, 30
iterations; self-training 20/40/60%), never tuned on DeepWeeds.

Variants: seed | + label propagation | + label propagation alternated with class-balanced
self-training. Runs on the 6 fine-tuned CNNs and the 4 frozen CLIP/BLIP backbones.
"""
import os, json, time, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from hd_common import BACKBONES, metrics, NUM
from hd_adapt import fit_predict, sinkhorn

warnings.filterwarnings("ignore")
OUT = "results_neighbour"
K, ALPHA, ITERS = 10, 0.9, 30
VLM = {"clip_vitb32_openai": "results_clip_backbone/feats/clip_vitb32_openai.npz",
       "clip_vitl14_laion2b": "results_clip_backbone/feats/clip_vitl14_laion2b.npz",
       "blip_base_coco": "results_blip_backbone/feats/blip_base_coco.npz",
       "blip_large_coco": "results_blip_backbone/feats/blip_large_coco.npz"}

def feature_sets():
    S = {b: (f"results_hd/feats/{b}.npz", "fine-tuned CNN") for b in BACKBONES}
    S.update({k: (v, "frozen VLM") for k, v in VLM.items()})
    return S

def load(path):
    d = np.load(path); return [d[k].astype(np.float32) if k.startswith("X") else d[k]
                               for k in ("Xtr", "ytr", "Xte", "yte", "Xdw", "ydw")]

def graph(Xt):
    """Symmetric, normalised kNN affinity on L2-normalised, target-standardised features."""
    Z = StandardScaler().fit_transform(Xt); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(Z)
    dist, idx = nn.kneighbors(Z)
    n = len(Z); W = np.zeros((n, n), np.float32)
    rows = np.repeat(np.arange(n), K); cols = idx[:, 1:].ravel(); w = np.clip(1 - dist[:, 1:].ravel(), 0, None)
    W[rows, cols] = w; W = np.maximum(W, W.T)
    d = W.sum(1); d[d == 0] = 1; Dm = 1 / np.sqrt(d)
    return W * Dm[:, None] * Dm[None, :]

def propagate(S, Y0):
    F = Y0.copy()
    for _ in range(ITERS): F = ALPHA * (S @ F) + (1 - ALPHA) * Y0
    F = np.clip(F, 1e-9, None); return F / F.sum(1, keepdims=True)

def seed_probs(Xtr, ytr, Xt):
    P, _, _ = fit_predict("LR", Xtr, ytr, Xt, restd=True); return sinkhorn(P)

def lp_self_train(Xtr, ytr, Xt, S):
    P = propagate(S, seed_probs(Xtr, ytr, Xt))
    for frac in (0.2, 0.4, 0.6):
        Q = sinkhorn(P); pred, conf = Q.argmax(1), Q.max(1); keep = []
        for c in range(NUM):
            ix = np.where(pred == c)[0]
            if len(ix): keep += list(ix[np.argsort(-conf[ix])][:max(1, int(frac * len(ix)))])
        keep = np.array(keep)
        P0, _, _ = fit_predict("LR", np.vstack([Xtr, Xt[keep]]), np.r_[ytr, pred[keep]], Xt, restd=True)
        P = propagate(S, sinkhorn(P0))
    return sinkhorn(P)

def run(name, path, kind):
    out = f"{OUT}/{name}.json"
    if os.path.exists(out): print(f"[{name}] done, skipping"); return json.load(open(out))
    Xtr, ytr, Xte, yte, Xt, yt = load(path); t0 = time.time()
    S = graph(Xt); seed = seed_probs(Xtr, ytr, Xt)
    R = dict(kind=kind, in_domain=metrics(yte, fit_predict("LR", Xtr, ytr, Xte, restd=False)[0].argmax(1)),
             seed=metrics(yt, seed.argmax(1)),
             lp=metrics(yt, sinkhorn(propagate(S, seed)).argmax(1)),
             lp_st=metrics(yt, lp_self_train(Xtr, ytr, Xt, S).argmax(1)),
             seconds=round(time.time() - t0, 1))
    json.dump(R, open(out, "w"), indent=2)
    print(f"[{name}] GBIF={R['in_domain']['acc']*100:.1f} | DW seed={R['seed']['acc']*100:.1f} "
          f"LP={R['lp']['acc']*100:.1f} LP+ST={R['lp_st']['acc']*100:.1f}  ({R['seconds']:.0f}s)", flush=True)
    return R

def report(all_R):
    L = ["# Neighbourhood adaptation: graph label propagation (GBIF-400 → DeepWeeds)\n",
         f"k={K}, alpha={ALPHA}, {ITERS} iterations; seed = LR + target re-standardisation + Sinkhorn. "
         "No DeepWeeds labels used; transductive.\n",
         "| Backbone | Type | GBIF test | DW: seed (LR+restd+align) | DW: + label propagation | **DW: + LP + self-training** | macro-F1 |",
         "|---|---|---|---|---|---|---|"]
    for n, R in sorted(all_R.items(), key=lambda kv: -kv[1]["lp_st"]["acc"]):
        L.append(f"| {n} | {R['kind']} | {R['in_domain']['acc']*100:.1f} | {R['seed']['acc']*100:.1f} | "
                 f"{R['lp']['acc']*100:.1f} | **{R['lp_st']['acc']*100:.1f}** | {R['lp_st']['macro_f1']:.3f} |")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    report({n: run(n, p, k) for n, (p, k) in feature_sets().items()})
