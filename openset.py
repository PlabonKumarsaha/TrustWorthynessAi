"""
Open-set rejection for annotation-free deployment.

Task: the model is trained on GBIF's 8 weed species only. At test time it must reject the
9,106 DeepWeeds images that are NOT target weeds. No negative is ever seen in training and no
DeepWeeds label is used. Baseline from the previous run: AUROC 0.718 (MSP) / 0.772 (energy).

Scores compared (all label-free):
  classical : MSP, MaxLogit, Energy, Entropy, Mahalanobis, Relative-Mahalanobis
  strong    : kNN distance in feature space (Sun et al. 2022)
  VLM       : CLIP text-defined negatives -- describe non-target vegetation in language
  ours      : cross-architecture disagreement (DINOv2 vs CLIP) as an OOD signal
  fusion    : z-scored combination of the best complementary scores
"""
import os, json
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import softmax, logsumexp

OUT = "results_novel/realistic"; FC = "results_novel/factorial_cache"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
C = 8
READABLE = ["chinee apple","lantana","parkinsonia","parthenium weed",
            "prickly acacia","rubber vine","siam weed","snake weed"]
NEGATIVE_PROMPTS = [
    "a photo of grass", "a photo of bare soil", "a photo of dry leaves and mulch",
    "a photo of a tree canopy", "a photo of native australian bushland",
    "a photo of rocks and gravel", "a photo of a dirt track", "a photo of scrub vegetation",
]

# ---------------- CLIP features over the full DeepWeeds set ----------------
@torch.no_grad()
def clip_full():
    fp = f"{OUT}/dw_full_clip.npy"
    if os.path.exists(fp): return np.load(fp)
    import open_clip, tensorflow_datasets as tfds
    m,_,pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    m = m.to(DEVICE).eval()
    ds = tfds.load("deep_weeds", split="train", as_supervised=True)
    out, buf, n = [], [], 0
    for img,_ in tfds.as_numpy(ds):
        buf.append(pre(Image.fromarray(img).convert("RGB"))); n += 1
        if len(buf)==64:
            out.append(m.encode_image(torch.stack(buf).to(DEVICE)).float().cpu()); buf=[]
            if n % 3200 == 0: print(f"   clip ...{n}", flush=True)
    if buf: out.append(m.encode_image(torch.stack(buf).to(DEVICE)).float().cpu())
    F = torch.cat(out).numpy(); np.save(fp, F); return F

@torch.no_grad()
def clip_text_scores(Fclip):
    """P(image is one of the 8 weeds) vs text-defined non-target vegetation."""
    import open_clip
    m,_,_ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); m = m.to(DEVICE).eval()
    prompts = [f"a photo of {n}, a type of weed." for n in READABLE] + NEGATIVE_PROMPTS
    T = m.encode_text(tok(prompts).to(DEVICE))
    T = (T/T.norm(dim=-1, keepdim=True)).cpu().numpy()
    X = Fclip/np.linalg.norm(Fclip, axis=1, keepdims=True)
    P = softmax(100.0 * X @ T.T, axis=1)
    return P[:, :C].sum(1)          # mass on weed prompts; low => reject

# ---------------- OOD scores ----------------
def knn_score(Ftr, Fte, k=10):
    A = Ftr/np.linalg.norm(Ftr,axis=1,keepdims=True)
    B = Fte/np.linalg.norm(Fte,axis=1,keepdims=True)
    out = np.empty(len(B))
    for i in range(0, len(B), 2048):
        sim = B[i:i+2048] @ A.T
        out[i:i+2048] = np.partition(-sim, k, axis=1)[:, k] * -1   # k-th largest similarity
    return out

def maha(Ftr, ytr, Fte, relative=False):
    mus = np.stack([Ftr[ytr==c].mean(0) for c in range(C)])
    Xc = np.concatenate([Ftr[ytr==c]-mus[c] for c in range(C)])
    P = np.linalg.pinv(np.cov(Xc.T) + np.eye(Ftr.shape[1])*1e-3)
    d = np.stack([np.einsum('ij,jk,ik->i', Fte-mus[c], P, Fte-mus[c]) for c in range(C)],1).min(1)
    if not relative: return -d
    mu0 = Ftr.mean(0); P0 = np.linalg.pinv(np.cov((Ftr-mu0).T)+np.eye(Ftr.shape[1])*1e-3)
    d0 = np.einsum('ij,jk,ik->i', Fte-mu0, P0, Fte-mu0)
    return -(d - d0)

def js_div(P, Q):
    M = 0.5*(P+Q); eps=1e-12
    kl = lambda A,B: (A*np.log((A+eps)/(B+eps))).sum(1)
    return 0.5*kl(P,M) + 0.5*kl(Q,M)

def z(x): return (x-x.mean())/(x.std()+1e-9)

def main():
    Fd  = np.load(f"{OUT}/dw_full_448.npy")                 # DINOv2 @448, full DeepWeeds
    y   = np.load(f"{OUT}/dw_full_labels.npy")
    Ftr = np.load(f"{OUT}/gbif_448.npy");  ytr = np.load(f"{OUT}/gbif_labels.npy")
    Fc_tr = np.load(f"{FC}/clip_vitb32_224_gbif_train.npy")
    is_weed = (y < C).astype(int)
    print(f"full set {Fd.shape}, weeds={is_weed.sum()}, negatives={(1-is_weed).sum()}", flush=True)

    print("[clip] features over full DeepWeeds ...", flush=True)
    Fc = clip_full()

    clf_d = LogisticRegression(max_iter=3000).fit(Ftr, ytr)
    logits = clf_d.decision_function(Fd); Pd = softmax(logits, axis=1)
    clf_c = LogisticRegression(max_iter=3000).fit(Fc_tr, ytr)
    Pc = clf_c.predict_proba(Fc)

    S = {}
    S["MSP"]            = Pd.max(1)
    S["MaxLogit"]       = logits.max(1)
    S["Energy"]         = logsumexp(logits, axis=1)
    S["Entropy"]        = (Pd*np.log(Pd+1e-12)).sum(1)          # negative entropy
    S["Mahalanobis"]    = maha(Ftr, ytr, Fd)
    S["RelMahalanobis"] = maha(Ftr, ytr, Fd, relative=True)
    S["kNN"]            = knn_score(Ftr, Fd, k=10)
    S["CLIP-text-neg"]  = clip_text_scores(Fc)
    S["XArch-agree"]    = -js_div(Pd, Pc)                        # low divergence => in-dist
    S["fuse kNN+CLIPtext"]        = z(S["kNN"]) + z(S["CLIP-text-neg"])
    S["fuse kNN+CLIPtext+XArch"]  = z(S["kNN"]) + z(S["CLIP-text-neg"]) + z(S["XArch-agree"])

    res = {}
    print(f"\n{'score':28s}{'AUROC':>8s}{'AUPR':>8s}{'FPR@95TPR':>11s}")
    print("-"*55)
    for k, s in sorted(S.items(), key=lambda kv: -roc_auc_score(is_weed, kv[1])):
        auroc = roc_auc_score(is_weed, s); aupr = average_precision_score(is_weed, s)
        thr = np.percentile(s[is_weed==1], 5)                    # 95% of weeds retained
        fpr = float((s[is_weed==0] >= thr).mean())               # negatives wrongly accepted
        res[k] = dict(auroc=round(float(auroc),4), aupr=round(float(aupr),4), fpr95=round(fpr,4))
        print(f"{k:28s}{auroc:8.3f}{aupr:8.3f}{fpr*100:10.1f}%")

    json.dump(res, open(f"{OUT}/openset_results.json","w"), indent=2)
    print(f"\nsaved -> {OUT}/openset_results.json")

if __name__ == "__main__":
    main()
