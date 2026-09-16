"""
Train on DeepWeeds (Australian field) -> test on GBIF (web imagery).

Reverse of the earlier GBIF->DeepWeeds study, to test whether the domain gap is symmetric.
Architectures and recipe follow Olsen et al. (2019): ImageNet-pretrained ResNet-50 and
Inception-v3, final layer replaced with an 8-unit layer, binary cross-entropy, Adam 1e-4,
batch 32, LR halved on validation plateau, early stopping.
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from reverse_common import (CLASSES, NUM, DEVICE, train_tf, eval_tf, ImgDS,
                            deepweeds_splits, gbif_all, scores)

MAX_EPOCHS   = 30
BATCH        = 32
LR           = 1e-4
LR_PATIENCE  = 4     # paper halves after 16 epochs w/o improvement (scaled to our budget)
STOP_PATIENCE= 8     # paper aborts after 32 (scaled)

def build(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, NUM)
    elif name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits = False; m.AuxLogits = None          # single head, 224px input
        m.fc = nn.Linear(m.fc.in_features, NUM)
    else:
        raise ValueError(name)
    return m.to(DEVICE)

def bce_loss(logits, y):
    """Paper-faithful: binary cross-entropy over one-hot targets."""
    target = F.one_hot(y, NUM).float()
    return F.binary_cross_entropy_with_logits(logits, target)

@torch.no_grad()
def predict(model, loader):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        ps.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)

@torch.no_grad()
def val_loss(model, loader):
    model.eval(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        tot += bce_loss(model(xb), yb).item() * len(yb); n += len(yb)
    return tot / n

def run(name):
    t0 = time.time()
    (ftr,ytr), (fva,yva), (fte,yte) = deepweeds_splits()
    fg, yg = gbif_all()
    print(f"[{name}] DeepWeeds train {len(ftr)} / val {len(fva)} / test {len(fte)}  |  GBIF test {len(fg)}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, num_workers=4,
                                   persistent_workers=True)
    tr = dl(ImgDS(ftr, ytr, train_tf), True)
    va = dl(ImgDS(fva, yva, eval_tf), False)
    te = dl(ImgDS(fte, yte, eval_tf), False)
    gb = dl(ImgDS(fg,  yg,  eval_tf), False)

    model = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LR_PATIENCE)

    best, best_state, bad, best_ep = float("inf"), None, 0, -1
    for ep in range(1, MAX_EPOCHS+1):
        model.train(); tot, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(yb); n += len(yb)
        vl = val_loss(model, va); sched.step(vl)
        print(f"  epoch {ep:2d}  train_loss={tot/n:.4f}  val_loss={vl:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= STOP_PATIENCE:
                print(f"  early stop at epoch {ep} (best {best_ep})"); break
    model.load_state_dict(best_state)

    y1, p1 = predict(model, te); indom = scores(y1, p1)      # DeepWeeds test (in-domain)
    y2, p2 = predict(model, gb); ood  = scores(y2, p2)       # GBIF (out-of-domain)
    res = dict(model=name, best_epoch=best_ep, best_val_loss=round(best,4),
               in_domain_deepweeds=indom, ood_gbif=ood,
               gap_acc=round(indom["acc"]-ood["acc"], 4),
               seconds=round(time.time()-t0, 1), classes=CLASSES)
    os.makedirs("results_reverse", exist_ok=True)
    json.dump(res, open(f"results_reverse/{name}.json", "w"), indent=2)
    torch.save(best_state, f"models_cache/reverse_{name}.pt")
    print(f"[{name}] DeepWeeds test acc={indom['acc']:.4f} f1={indom['macro_f1']:.4f}  |  "
          f"GBIF acc={ood['acc']:.4f} f1={ood['macro_f1']:.4f}  |  gap={res['gap_acc']:+.4f}")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="resnet50,inception_v3")
    for nm in ap.parse_args().models.split(","):
        run(nm.strip())
