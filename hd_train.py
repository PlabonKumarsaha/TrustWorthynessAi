"""
Stage 1 of the HD(LM)^2D replication: fine-tune each backbone end to end on GBIF, score the
end-to-end model on GBIF test and DeepWeeds, and cache penultimate features for stage 2.

Resumable: a backbone whose results already exist is skipped.
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F

from hd_common import (BACKBONES, CLASSES, DEVICE, OUT, DIRECTION, build, extract, splits,
                       loader, train_tf, eval_tf, metrics)

# 150 epochs matches the paper; early stopping (patience 6) is what actually ends training.
# The first three backbones stopped at epochs 13/20/8, so raising the cap leaves them unchanged.
MAX_EPOCHS, PATIENCE, LR, BATCH = 150, 6, 1e-4, 32
ACCUM = {"regnet_y_32gf": 2}   # 145M params: micro-batch 16 x 2 keeps the effective batch at 32

@torch.no_grad()
def val_loss(model, dl):
    model.eval(); tot = n = 0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        tot += F.cross_entropy(model(xb), yb, reduction="sum").item(); n += len(yb)
    return tot / n

def run(name):
    e2e_path, feat_path = f"{OUT}/e2e/{name}.json", f"{OUT}/feats/{name}.npz"
    if os.path.exists(e2e_path) and os.path.exists(feat_path):
        print(f"[{name}] already done, skipping", flush=True); return
    t0 = time.time()
    (ftr, ytr), (fva, yva), (fte, yte), (fsh, ysh), in_name, sh_name = splits()
    accum = ACCUM.get(name, 1)
    tr = loader(ftr, ytr, train_tf, True, BATCH // accum)
    va = loader(fva, yva, eval_tf, False, 64)
    print(f"[{name}] {in_name} train {len(ftr)} / val {len(fva)} / test {len(fte)} "
          f"| shift {sh_name} {len(fsh)}", flush=True)

    model, head = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        for i, (xb, yb) in enumerate(tr, 1):
            loss = F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE)) / accum
            loss.backward()
            if i % accum == 0 or i == len(tr): opt.step(); opt.zero_grad()
        vl = val_loss(model, va); sched.step(vl)
        print(f"  [{name}] epoch {ep:2d} val_loss={vl:.4f} lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: print(f"  [{name}] early stop (best epoch {best_ep})", flush=True); break
    model.load_state_dict(best_state)

    X, L = {}, {}
    for split, files in [("tr", ftr), ("va", fva), ("te", fte), ("sh", fsh)]:
        X[split], L[split] = extract(model, head, files)
    os.makedirs(f"{OUT}/feats", exist_ok=True); os.makedirs(f"{OUT}/e2e", exist_ok=True)
    np.savez(feat_path, **{f"X{s}": X[s].astype(np.float16) for s in X},
             ytr=ytr, yva=yva, yte=yte, ysh=ysh)

    res = dict(backbone=name, direction=DIRECTION, best_epoch=best_ep, best_val_loss=round(best, 4),
               feat_dim=int(X["tr"].shape[1]), in_domain_name=in_name, shift_name=sh_name,
               in_domain=metrics(yte, L["te"].argmax(1)),
               shift=metrics(ysh, L["sh"].argmax(1)),
               seconds=round(time.time() - t0, 1), classes=CLASSES)
    res["gap_acc"] = round(res["in_domain"]["acc"] - res["shift"]["acc"], 4)
    json.dump(res, open(e2e_path, "w"), indent=2)
    print(f"[{name}] E2E  {in_name} acc={res['in_domain']['acc']:.4f}  "
          f"{sh_name} acc={res['shift']['acc']:.4f}  gap={res['gap_acc']:+.4f}  ({res['seconds']:.0f}s)", flush=True)
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backbones", default=",".join(BACKBONES))
    for nm in ap.parse_args().backbones.split(","):
        run(nm.strip())
