"""
Does domain adaptation DEGRADE open-set rejection?

Hypothesis: SHOT minimises prediction entropy on every target image. When the unlabelled
target stream contains non-target vegetation (9,106 negatives in DeepWeeds), adaptation is
explicitly training the model to be CONFIDENTLY WRONG about them - pulling their features into
the eight source class clusters. Closed-set accuracy should rise while the ability to reject
should fall.

Three conditions:
  A. source-only                  - no adaptation (rejection baseline)
  B. SHOT adapted on weeds only   - what we ran in baselines.py
  C. SHOT adapted on FULL stream  - realistic deployment: negatives are in the unlabelled data

For each we report closed-set accuracy on the 8,403 weeds AND rejection quality against the
9,106 negatives, so the tension (if any) is visible in one table.
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import logsumexp

OUT = "results_novel/realistic"; C = 8; DEV = "cpu"
torch.manual_seed(0); np.random.seed(0)

Xs = np.load(f"{OUT}/gbif_448.npy"); ys = np.load(f"{OUT}/gbif_labels.npy")
Fa = np.load(f"{OUT}/dw_full_448.npy"); ya = np.load(f"{OUT}/dw_full_labels.npy")
mu, sd = Xs.mean(0), Xs.std(0)+1e-6
Xs_n = (Xs-mu)/sd; Fa_n = (Fa-mu)/sd
is_weed = (ya < C)
Ts = torch.tensor(Xs_n).float(); Ys = torch.tensor(ys).long()
Tall = torch.tensor(Fa_n).float(); Tw = torch.tensor(Fa_n[is_weed]).float()
yw = ya[is_weed]
print(f"source {Xs.shape} | target full {Fa.shape}: {is_weed.sum()} weeds, {(~is_weed).sum()} negatives")

class Net(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU())
        self.c = nn.Linear(h, C)
    def forward(self,x): return self.c(self.g(x))

def train_source(epochs=60):
    m = Net().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Ts))
        for i in range(0,len(Ts),128):
            b = perm[i:i+128]
            loss = F.cross_entropy(m(Ts[b]), Ys[b]); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

def shot(m, Tadapt, epochs=40, beta=0.3):
    """SHOT adaptation on whatever unlabelled set Tadapt is."""
    for p in m.c.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(m.g.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        m.eval()
        with torch.no_grad():
            f = m.g(Tadapt); p = m.c(f).softmax(1)
            cent = (p.T @ f)/(p.sum(0)[:,None]+1e-8)
            pl = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1)).argmin(1)
            for _ in range(2):
                oh = F.one_hot(pl, C).float()
                cent = (oh.T @ f)/(oh.sum(0)[:,None]+1e-8)
                pl = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1)).argmin(1)
        m.train(); perm = torch.randperm(len(Tadapt))
        for i in range(0,len(Tadapt),128):
            b = perm[i:i+128]
            if len(b) < 2: continue
            out = m(Tadapt[b]); pr = out.softmax(1)
            ent = -(pr*torch.log(pr+1e-8)).sum(1).mean()
            mp = pr.mean(0); div = -(mp*torch.log(mp+1e-8)).sum()
            loss = ent - div + beta*F.cross_entropy(out, pl[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

@torch.no_grad()
def analyse(m, tag):
    m.eval()
    fs = m.g(Ts).numpy()                       # source features in bottleneck space
    fa = m.g(Tall).numpy()                     # all target features
    logits = m.c(torch.tensor(fa).float()).numpy()
    P = torch.tensor(logits).softmax(1).numpy()
    acc = accuracy_score(yw, P[is_weed].argmax(1))

    scores = {}
    scores["MSP"] = P.max(1)
    scores["energy"] = logsumexp(logits, axis=1)
    scores["neg-entropy"] = (P*np.log(P+1e-12)).sum(1)
    A = fs/np.linalg.norm(fs,axis=1,keepdims=True); B = fa/np.linalg.norm(fa,axis=1,keepdims=True)
    knn = np.empty(len(B))
    for i in range(0,len(B),4096):
        sim = B[i:i+4096] @ A.T
        knn[i:i+4096] = np.partition(-sim,10,axis=1)[:,10]*-1
    scores["kNN"] = knn
    mus = np.stack([fs[ys==c].mean(0) for c in range(C)])
    Xc = np.concatenate([fs[ys==c]-mus[c] for c in range(C)])
    Pm = np.linalg.pinv(np.cov(Xc.T)+np.eye(fs.shape[1])*1e-3)
    scores["Mahalanobis"] = -np.stack([np.einsum('ij,jk,ik->i', fa-mus[c], Pm, fa-mus[c])
                                       for c in range(C)],1).min(1)

    out = {"closed_set_acc": round(float(acc),4)}
    for k,s in scores.items():
        au = roc_auc_score(is_weed.astype(int), s)
        thr = np.percentile(s[is_weed], 5)
        out[k] = dict(auroc=round(float(au),4), fpr95=round(float((s[~is_weed]>=thr).mean()),4))
    # how confident is the model on things it has never seen?
    out["mean_conf_on_negatives"] = round(float(P[~is_weed].max(1).mean()),4)
    out["mean_conf_on_weeds"]     = round(float(P[is_weed].max(1).mean()),4)
    print(f"\n=== {tag} ===")
    print(f"  closed-set accuracy (weeds)      {acc*100:.1f}%")
    print(f"  mean confidence on NEGATIVES     {out['mean_conf_on_negatives']:.3f}   (on weeds {out['mean_conf_on_weeds']:.3f})")
    for k in scores: print(f"  {k:14s} AUROC={out[k]['auroc']:.3f}  FPR@95={out[k]['fpr95']*100:.1f}%")
    return out

res = {}
src = train_source()
res["A_source_only"] = analyse(src, "A. source-only (no adaptation)")

m_b = train_source(); m_b = shot(m_b, Tw)
res["B_shot_weeds_only"] = analyse(m_b, "B. SHOT adapted on weeds only")

m_c = train_source(); m_c = shot(m_c, Tall)
res["C_shot_full_stream"] = analyse(m_c, "C. SHOT adapted on FULL stream (negatives included)")

print("\n" + "="*72)
print(f"{'condition':34s}{'closed-set':>12s}{'best AUROC':>12s}{'conf on neg':>13s}")
print("-"*72)
for k,v in res.items():
    best = max(v[s]['auroc'] for s in ["MSP","energy","neg-entropy","kNN","Mahalanobis"])
    print(f"{k:34s}{v['closed_set_acc']*100:11.1f}%{best:12.3f}{v['mean_conf_on_negatives']:13.3f}")
json.dump(res, open(f"{OUT}/adaptation_vs_rejection.json","w"), indent=2)
print(f"\nsaved -> {OUT}/adaptation_vs_rejection.json")
