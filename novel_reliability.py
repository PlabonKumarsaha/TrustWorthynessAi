"""
Foundational experiment for the proposed technique:
  Conformal, cross-architecture-agreement pseudo-label selection.

Question: under GBIF->DeepWeeds shift, which rule keeps the CLEANEST pseudo-labels at a given
coverage?  We compare, on DeepWeeds (labels used ONLY to score reliability, never to select
except the conformal calibration set which uses the disjoint few-shot pool):

  * single-model confidence (ResNet / CLIP / DINOv2)              -- the usual practice
  * ensemble confidence
  * cross-architecture AGREEMENT gating (our wedge #1)
  * CONFORMAL selection calibrated on a few labelled pool imgs    (our wedge #2)
  * conformal + agreement (the proposed rule)

Three inductively-distinct experts: supervised CNN (ResNet-50), contrastive VLM (CLIP),
self-supervised ViT (DINOv2). DINOv2 gets a source-trained linear head (GBIF labels only).
"""
import os, json, warnings
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

from adapt_common import (load_resnet, load_clip, test_set, pool_set,
                          resnet_probs, clip_probs, DEVICE)
from bench_common import gbif_arrays, NUM, CLASSES

# ---------------- DINOv2 third expert (self-supervised ViT) ----------------
def dinov2_probs():
    """Source-trained linear head on frozen DINOv2 features. Returns (test_probs, pool_probs)."""
    try:
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(DEVICE).eval()
    except Exception as e:
        print("DINOv2 unavailable ->", e); return None, None
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(mean, std)])
    @torch.no_grad()
    def feats(imgs, bs=64):
        xb = torch.stack([tf(Image.fromarray(im)) for im in imgs]); out=[]
        for i in range(0,len(xb),bs): out.append(dino(xb[i:i+bs].to(DEVICE)).cpu())
        return torch.cat(out).numpy()
    Xtr, ytr = gbif_arrays("global_train")
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(feats(Xtr), ytr)
    Xte,_ = test_set(); Xpo,_ = pool_set()
    return clf.predict_proba(feats(Xte)), clf.predict_proba(feats(Xpo))

# ---------------- reliability curve: precision of kept labels vs coverage ----------------
def curve_confidence(prob, y):
    conf = prob.max(1); pred = prob.argmax(1); order = np.argsort(-conf)
    correct = (pred==y).astype(float)[order]
    cov = np.arange(1,len(y)+1)/len(y)
    prec = np.cumsum(correct)/np.arange(1,len(y)+1)
    return cov, prec

def point_at(cov, prec, target=0.5):
    i = np.argmin(np.abs(cov-target)); return float(prec[i])

def conformal(cal_prob, cal_y, test_prob, test_y, alphas):
    """Split-conformal: threshold nonconformity (1 - p[true]) on the labelled pool."""
    s_cal = 1 - cal_prob[np.arange(len(cal_y)), cal_y]
    pred = test_prob.argmax(1); s_test = 1 - test_prob[np.arange(len(test_prob)), pred]
    rows=[]
    for a in alphas:
        q = np.quantile(s_cal, 1-a, method="higher")
        keep = s_test <= q
        cov = float(keep.mean()); prec = float((pred[keep]==test_y[keep]).mean()) if keep.any() else float("nan")
        rows.append(dict(alpha=round(a,3), coverage=round(cov,4), precision=round(prec,4)))
    return rows

def main():
    Xte, yte = test_set(); Xpo, ypo = pool_set()
    rn = load_resnet(); cs = load_clip()
    P = {"ResNet": resnet_probs(rn, Xte), "CLIP": clip_probs(cs, Xte)}
    Ppool = {"ResNet": resnet_probs(rn, Xpo), "CLIP": clip_probs(cs, Xpo)}
    dte, dpo = dinov2_probs()
    if dte is not None: P["DINOv2"] = dte; Ppool["DINOv2"] = dpo
    experts = list(P.keys())
    print("experts:", experts)

    out = {"experts": experts, "n_test": len(yte), "single_acc": {}, "at_coverage_0.5": {},
           "agreement": {}, "conformal": {}}

    # single-model accuracy + confidence reliability at 50% coverage
    for e in experts:
        out["single_acc"][e] = round(float(accuracy_score(yte, P[e].argmax(1))),4)
        cov,prec = curve_confidence(P[e], yte); out["at_coverage_0.5"][f"{e}-confidence"] = round(point_at(cov,prec),4)

    ens = sum(P.values())/len(P)
    cov,prec = curve_confidence(ens, yte); out["at_coverage_0.5"]["ensemble-confidence"] = round(point_at(cov,prec),4)
    ens_pool = sum(Ppool.values())/len(Ppool)

    # ---- WEDGE 1: cross-architecture agreement ----
    preds = {e: P[e].argmax(1) for e in experts}
    all_agree = np.all([preds[e]==preds[experts[0]] for e in experts], axis=0)
    agreed_pred = preds[experts[0]]
    out["agreement"] = {
        "coverage": round(float(all_agree.mean()),4),
        "precision_on_agreed": round(float((agreed_pred[all_agree]==yte[all_agree]).mean()),4),
        "precision_on_disagreed": round(float((ens.argmax(1)[~all_agree]==yte[~all_agree]).mean()),4) if (~all_agree).any() else None,
        "error_concentration": None,
    }
    # what fraction of ALL errors live in the disagreement set?
    ens_err = ens.argmax(1)!=yte
    out["agreement"]["error_concentration"] = round(float((ens_err & ~all_agree).sum()/max(ens_err.sum(),1)),4)

    # ---- WEDGE 2: conformal (calibrated on labelled pool) ----
    alphas = [0.05,0.1,0.2,0.3,0.4,0.5]
    out["conformal"]["ensemble"] = conformal(ens_pool, ypo, ens, yte, alphas)
    # conformal restricted to the agreed set (proposed rule)
    if (~all_agree).any():
        # calibrate on pool-agreed subset
        preds_pool = {e: Ppool[e].argmax(1) for e in experts}
        pool_agree = np.all([preds_pool[e]==preds_pool[experts[0]] for e in experts], axis=0)
        if pool_agree.sum() >= 20:
            out["conformal"]["ensemble+agreement"] = conformal(
                ens_pool[pool_agree], ypo[pool_agree],
                ens[all_agree], yte[all_agree], alphas)

    os.makedirs("results_novel", exist_ok=True)
    json.dump(out, open("results_novel/reliability.json","w"), indent=2)

    print("\n=== single-model accuracy on DeepWeeds ===")
    for e,a in out["single_acc"].items(): print(f"  {e:8s} {a*100:5.1f}%")
    print("\n=== pseudo-label PRECISION at 50% coverage (higher=cleaner) ===")
    for k,v in out["at_coverage_0.5"].items(): print(f"  {k:22s} {v*100:5.1f}%")
    print("\n=== cross-architecture AGREEMENT ===")
    ag=out["agreement"]
    print(f"  agreed on {ag['coverage']*100:.1f}% of images; precision there = {ag['precision_on_agreed']*100:.1f}%")
    print(f"  precision on disagreed = {None if ag['precision_on_disagreed'] is None else round(ag['precision_on_disagreed']*100,1)}%")
    print(f"  {ag['error_concentration']*100:.1f}% of ALL errors fall in the disagreement set")
    print("\n=== CONFORMAL (calibrated on few labelled pool imgs) ===")
    for row in out["conformal"]["ensemble"]:
        print(f"  alpha={row['alpha']}: coverage={row['coverage']*100:5.1f}%  precision={row['precision']*100:5.1f}%")
    if "ensemble+agreement" in out["conformal"]:
        print("  --- conformal + agreement (proposed) ---")
        for row in out["conformal"]["ensemble+agreement"]:
            print(f"  alpha={row['alpha']}: coverage={row['coverage']*100:5.1f}%  precision={row['precision']*100:5.1f}%")
    print("\nsaved -> results_novel/reliability.json")

if __name__ == "__main__":
    main()
