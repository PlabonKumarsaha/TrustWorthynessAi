"""
Published UDA baselines: CORAL, DANN and SHOT, against our DA / DA+CBST.

All methods use the SAME frozen DINOv2 ViT-B/14 @448 features, the same GBIF source labels,
and are scored on the same full natural DeepWeeds weed set (8,403 images, no subsampling,
no target labels used for training).

SHOT and DANN need a trainable feature transform, so a source-only model with the identical
MLP architecture is included -- otherwise their gain would be confounded with extra capacity.

  CORAL  Sun et al. 2016, Return of Frustratingly Easy Domain Adaptation (closed form)
  DANN   Ganin et al. 2016, gradient reversal + domain discriminator
  SHOT   Liang et al. 2020, frozen source hypothesis + information maximisation + pseudo-labels
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

OUT = "results_novel/realistic"; C = 8
DEV = "cpu"                      # small MLPs; CPU avoids MPS overhead here
torch.manual_seed(0); np.random.seed(0)

def sc(y, p): return dict(acc=round(float(accuracy_score(y,p)),4),
                          macro_f1=round(float(f1_score(y,p,average="macro")),4))

# ---------------- data ----------------
Xs = np.load(f"{OUT}/gbif_448.npy"); ys = np.load(f"{OUT}/gbif_labels.npy")
Fa = np.load(f"{OUT}/dw_full_448.npy"); ya = np.load(f"{OUT}/dw_full_labels.npy")
w = ya < C; Xt, yt = Fa[w], ya[w]
mu, sd = Xs.mean(0), Xs.std(0)+1e-6
Xs_n, Xt_n = (Xs-mu)/sd, (Xt-mu)/sd
print(f"source {Xs.shape}  target {Xt.shape} (natural distribution)")
res = {}

# ---------------- reference: linear probe, no adaptation ----------------
res["source-only (linear)"] = sc(yt, LogisticRegression(max_iter=3000).fit(Xs, ys).predict(Xt))

# ---------------- CORAL ----------------
def coral(Xs, Xt, lam=1.0):
    Cs = np.cov(Xs.T) + lam*np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + lam*np.eye(Xt.shape[1])
    A = np.real(sqrtm(np.linalg.inv(Cs))) @ np.real(sqrtm(Ct))
    return Xs @ A
Xs_c = coral(Xs_n, Xt_n)
res["CORAL"] = sc(yt, LogisticRegression(max_iter=3000).fit(Xs_c, ys).predict(Xt_n))

# ---------------- shared MLP backbone for DANN / SHOT ----------------
class Net(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU())
        self.c = nn.Linear(h, C)
    def forward(self,x): return self.c(self.g(x))

Ts, Tt = torch.tensor(Xs_n).float(), torch.tensor(Xt_n).float()
Ys = torch.tensor(ys).long()

def train_source_mlp(epochs=60):
    m = Net().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Ts))
        for i in range(0,len(Ts),128):
            b = perm[i:i+128]
            loss = F.cross_entropy(m(Ts[b].to(DEV)), Ys[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def pred(m, X):
    m.eval(); return m(X.to(DEV)).argmax(1).cpu().numpy()

src_mlp = train_source_mlp()
res["source-only (MLP, matched arch)"] = sc(yt, pred(src_mlp, Tt))

# ---------------- DANN ----------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l): ctx.l=l; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.l*g, None

def dann(epochs=60):
    m = Net().to(DEV); D = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,2)).to(DEV)
    opt = torch.optim.AdamW(list(m.parameters())+list(D.parameters()), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        m.train(); p = ep/epochs; lam = 2/(1+np.exp(-10*p))-1        # standard schedule
        ps, pt = torch.randperm(len(Ts)), torch.randperm(len(Tt))
        for i in range(0, len(Ts), 128):
            bs = ps[i:i+128]; bt = pt[(i%len(Tt)):(i%len(Tt))+128]
            if len(bt) < 2: continue
            xs, xt = Ts[bs].to(DEV), Tt[bt].to(DEV)
            fs, ft = m.g(xs), m.g(xt)
            loss = F.cross_entropy(m.c(fs), Ys[bs].to(DEV))
            dom = torch.cat([GRL.apply(fs,lam), GRL.apply(ft,lam)])
            dl = torch.cat([torch.zeros(len(fs)), torch.ones(len(ft))]).long().to(DEV)
            loss = loss + F.cross_entropy(D(dom), dl)
            opt.zero_grad(); loss.backward(); opt.step()
    return m
res["DANN"] = sc(yt, pred(dann(), Tt))

# ---------------- SHOT ----------------
def shot(epochs=40, beta=0.3):
    m = Net().to(DEV)
    m.load_state_dict(train_source_mlp().state_dict())
    for p in m.c.parameters(): p.requires_grad_(False)      # freeze source hypothesis
    opt = torch.optim.AdamW(m.g.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        # centroid-based pseudo-labels (self-supervised), refreshed each epoch
        m.eval()
        with torch.no_grad():
            f = m.g(Tt.to(DEV)); p = m.c(f).softmax(1)
            cent = (p.T @ f) / (p.sum(0)[:,None]+1e-8)
            d = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1))
            pl = d.argmin(1)
            for _ in range(2):
                oh = F.one_hot(pl, C).float()
                cent = (oh.T @ f)/(oh.sum(0)[:,None]+1e-8)
                pl = torch.cdist(F.normalize(f,dim=1), F.normalize(cent,dim=1)).argmin(1)
        m.train(); perm = torch.randperm(len(Tt))
        for i in range(0,len(Tt),128):
            b = perm[i:i+128]
            if len(b) < 2: continue
            out = m(Tt[b].to(DEV)); pr = out.softmax(1)
            ent = -(pr*torch.log(pr+1e-8)).sum(1).mean()
            mp = pr.mean(0); div = -(mp*torch.log(mp+1e-8)).sum()
            loss = ent - div + beta*F.cross_entropy(out, pl[b])      # IM + pseudo-label
            opt.zero_grad(); loss.backward(); opt.step()
    return m
res["SHOT"] = sc(yt, pred(shot(), Tt))

# ---------------- ours ----------------
def sinkhorn(P, it=100):
    Q = np.asarray(P,np.float64)+1e-12; t = Q.sum()/C
    for _ in range(it):
        Q = Q/Q.sum(1,keepdims=True); Q = Q/Q.sum(0,keepdims=True)*t
    return Q
def bal(P, frac):
    pr = P.argmax(1); cf = P.max(1); k = np.zeros(len(P),bool)
    for c in range(C):
        idx = np.where(pr==c)[0]
        if len(idx)==0: continue
        n = max(1,int(len(idx)*frac)); k[idx[np.argsort(-cf[idx])[:n]]] = True
    return k, pr
clf = LogisticRegression(max_iter=3000).fit(Xs, ys)
P = clf.predict_proba(Xt)
res["ours: DA"] = sc(yt, sinkhorn(P).argmax(1))
P = sinkhorn(P)
for r in range(3):
    k, pr = bal(P, 0.5+0.15*r)
    c2 = LogisticRegression(max_iter=3000).fit(np.concatenate([Xs,Xt[k]]), np.concatenate([ys,pr[k]]))
    P = sinkhorn(c2.predict_proba(Xt))
res["ours: DA+CBST"] = sc(yt, P.argmax(1))

print(f"\n{'method':36s}{'acc':>8s}{'macroF1':>10s}")
print("-"*54)
for k,v in sorted(res.items(), key=lambda kv:-kv[1]["acc"]):
    print(f"{k:36s}{v['acc']*100:7.1f}%{v['macro_f1']:10.3f}")
json.dump(res, open(f"{OUT}/baselines.json","w"), indent=2)
print(f"\nsaved -> {OUT}/baselines.json")
