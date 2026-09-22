"""
How much OOD contamination in the unlabelled adaptation stream does it take to break
source-free adaptation?

Condition C showed that adapting on the natural DeepWeeds stream (52% negatives) collapses
rejection to chance. This sweeps the contamination ratio from 0% to the natural 52% to find
where the failure begins, and how each rejection score degrades.

Design:
  adaptation stream = all 8,403 weeds + k negatives, k chosen to hit the target ratio
  evaluation       = ALWAYS the full 17,509 set, so rejection is measured identically throughout
  3 seeds per ratio -> mean +/- std (also the project's first real variance estimate)
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import logsumexp

OUT = "results_novel/realistic"; C = 8; DEV = "cpu"
RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.52]
SEEDS = [0, 1, 2]

Xs = np.load(f"{OUT}/gbif_448.npy"); ys = np.load(f"{OUT}/gbif_labels.npy")
Fa = np.load(f"{OUT}/dw_full_448.npy"); ya = np.load(f"{OUT}/dw_full_labels.npy")
mu, sd = Xs.mean(0), Xs.std(0)+1e-6
Xs_n = (Xs-mu)/sd; Fa_n = (Fa-mu)/sd
is_weed = ya < C
w_idx = np.where(is_weed)[0]; n_idx = np.where(~is_weed)[0]
Ts = torch.tensor(Xs_n).float(); Ys = torch.tensor(ys).long()
Tall = torch.tensor(Fa_n).float(); yw = ya[is_weed]
print(f"weeds={len(w_idx)}  negatives={len(n_idx)}  natural ratio={len(n_idx)/len(ya):.3f}", flush=True)

class Net(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU())
        self.c = nn.Linear(h, C)
    def forward(self,x): return self.c(self.g(x))

def train_source(seed, epochs=60):
    torch.manual_seed(seed)
    m = Net().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Ts))
        for i in range(0,len(Ts),128):
            b = perm[i:i+128]
            loss = F.cross_entropy(m(Ts[b]), Ys[b]); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

def shot(m, Tad, epochs=40, beta=0.3):
    for p in m.c.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(m.g.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        m.eval()
        with torch.no_grad():
            f = m.g(Tad); p = m.c(f).softmax(1)
            cent = (p.T @ f)/(p.sum(0)[:,None]+1e-8)
            pl = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1)).argmin(1)
            for _ in range(2):
                oh = F.one_hot(pl, C).float()
                cent = (oh.T @ f)/(oh.sum(0)[:,None]+1e-8)
                pl = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1)).argmin(1)
        m.train(); perm = torch.randperm(len(Tad))
        for i in range(0,len(Tad),128):
            b = perm[i:i+128]
            if len(b) < 2: continue
            out = m(Tad[b]); pr = out.softmax(1)
            ent = -(pr*torch.log(pr+1e-8)).sum(1).mean()
            mp = pr.mean(0); div = -(mp*torch.log(mp+1e-8)).sum()
            loss = ent - div + beta*F.cross_entropy(out, pl[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

@torch.no_grad()
def analyse(m):
    fs = m.g(Ts).numpy(); fa = m.g(Tall).numpy()
    logits = m.c(torch.tensor(fa).float()).numpy()
    P = torch.tensor(logits).softmax(1).numpy()
    out = {"closed": float(accuracy_score(yw, P[is_weed].argmax(1))),
           "conf_neg": float(P[~is_weed].max(1).mean())}
    S = {"MSP": P.max(1), "energy": logsumexp(logits,axis=1)}
    A = fs/np.linalg.norm(fs,axis=1,keepdims=True); B = fa/np.linalg.norm(fa,axis=1,keepdims=True)
    knn = np.empty(len(B))
    for i in range(0,len(B),4096):
        sim = B[i:i+4096] @ A.T; knn[i:i+4096] = np.partition(-sim,10,axis=1)[:,10]*-1
    S["kNN"] = knn
    mus = np.stack([fs[ys==c].mean(0) for c in range(C)])
    Xc = np.concatenate([fs[ys==c]-mus[c] for c in range(C)])
    Pm = np.linalg.pinv(np.cov(Xc.T)+np.eye(fs.shape[1])*1e-3)
    S["Maha"] = -np.stack([np.einsum('ij,jk,ik->i', fa-mus[c], Pm, fa-mus[c]) for c in range(C)],1).min(1)
    for k,s in S.items(): out[k] = float(roc_auc_score(is_weed.astype(int), s))
    return out

raw = []
for seed in SEEDS:
    src = train_source(seed)
    sd_state = {k:v.clone() for k,v in src.state_dict().items()}
    rng = np.random.default_rng(seed)
    for r in RATIOS:
        k = 0 if r == 0 else int(round(len(w_idx)*r/(1-r)))
        k = min(k, len(n_idx))
        sel = np.concatenate([w_idx, rng.choice(n_idx, k, replace=False)]) if k else w_idx
        Tad = Tall[torch.tensor(sel)]
        m = Net().to(DEV); m.load_state_dict(sd_state)
        m = shot(m, Tad)
        res = analyse(m); res.update(ratio=r, seed=seed, n_neg=k, stream=len(sel))
        raw.append(res)
        print(f"  seed{seed} ratio={r:.2f} (stream {len(sel)}, {k} neg): "
              f"closed={res['closed']*100:.1f}%  MSP={res['MSP']:.3f}  energy={res['energy']:.3f}  "
              f"kNN={res['kNN']:.3f}  Maha={res['Maha']:.3f}  conf_neg={res['conf_neg']:.3f}", flush=True)

agg = {}
print(f"\n{'contam':>8s}{'closed-set':>16s}{'MSP':>14s}{'energy':>14s}{'kNN':>14s}{'Maha':>14s}{'conf_neg':>10s}")
print("-"*92)
for r in RATIOS:
    rows = [x for x in raw if x["ratio"]==r]
    f = lambda k: (np.mean([x[k] for x in rows]), np.std([x[k] for x in rows]))
    agg[str(r)] = {k: dict(mean=round(f(k)[0],4), std=round(f(k)[1],4))
                   for k in ["closed","MSP","energy","kNN","Maha","conf_neg"]}
    c=f("closed"); m=f("MSP"); e=f("energy"); kn=f("kNN"); mh=f("Maha"); cn=f("conf_neg")
    print(f"{r*100:7.0f}%{c[0]*100:11.1f}±{c[1]*100:.1f}{m[0]:10.3f}±{m[1]:.3f}"
          f"{e[0]:10.3f}±{e[1]:.3f}{kn[0]:10.3f}±{kn[1]:.3f}{mh[0]:10.3f}±{mh[1]:.3f}{cn[0]:10.3f}")
json.dump({"raw":raw,"aggregate":agg}, open(f"{OUT}/contamination_sweep.json","w"), indent=2)
print(f"\nsaved -> {OUT}/contamination_sweep.json")
