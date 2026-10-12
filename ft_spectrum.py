"""
Fine-tuning spectrum pilot: does constraining *how much* of a backbone is fine-tuned improve
transfer from GBIF to DeepWeeds?

One backbone (ResNet-50), the original GBIF 400/class split (train 1,920 / val 640 / test 640),
shift-tested on all 8,403 DeepWeeds images. Five methods, everything else identical:
  full   full fine-tuning (control; matches hd_train)
  lp     frozen backbone (BatchNorm in eval mode), train only the head
  lpft   LP to convergence, then full fine-tuning at a 10x lower LR (Kumar et al., ICLR 2022)
  lora   LoRA (r=16) on every bottleneck convolution, head trainable
  dora   DoRA (r=16) on the same convolutions
Two seeds each.

Go/no-go rule, fixed before running: extend to all backbones only if a method beats `full`
on DeepWeeds by >= 5 points averaged over seeds, with every seed individually better.
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from hd_common import (build, gbif_splits, deepweeds_all, loader, train_tf, eval_tf, metrics,
                       DEVICE, CLASSES)

BACKBONE, OUT = "resnet50", "results_ftspec"
METHODS, SEEDS = ["full", "lp", "lpft", "lora", "dora"], [42, 43]
MAX_EPOCHS, PATIENCE, BATCH = 150, 6, 32
LORA_TARGET = r".*layer\d\.\d+\.conv\d"          # all 48 bottleneck convs
NOTABLE = 0.05

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

@torch.no_grad()
def val_loss(model, dl):
    model.eval(); tot = n = 0
    for xb, yb in dl:
        tot += F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE), reduction="sum").item(); n += len(yb)
    return tot / n

def fit(model, params, lr, tr, va, max_epochs, bn_frozen=False, tag=""):
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, state, best_ep, bad = float("inf"), None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        if bn_frozen: freeze_bn(model)
        for xb, yb in tr:
            loss = F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        vl = val_loss(model, va); sched.step(vl)
        print(f"    {tag} epoch {ep:3d} val_loss={vl:.4f}", flush=True)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(state)
    return best_ep

@torch.no_grad()
def predict(model, fs, ys):
    model.eval(); out = []
    for xb, _ in loader(fs, ys, eval_tf, False, 64):
        out.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
    return metrics(ys, np.concatenate(out))

def run(method, seed, data, smoke):
    path = f"{OUT}/{method}_s{seed}.json"
    if os.path.exists(path) and not smoke: print(f"[{method} s{seed}] done, skipping"); return
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
    (ftr, ytr), (fva, yva), (fte, yte), (fdw, ydw) = data
    tr = loader(ftr, ytr, train_tf, True, BATCH); va = loader(fva, yva, eval_tf, False, 64)
    epochs = 1 if smoke else MAX_EPOCHS
    model, head = build(BACKBONE)

    if method == "full":
        best_ep = fit(model, model.parameters(), 1e-4, tr, va, epochs, tag=method)
    elif method in ("lp", "lpft"):
        for p in model.parameters(): p.requires_grad_(False)
        for p in head.parameters(): p.requires_grad_(True)
        best_ep = fit(model, head.parameters(), 1e-3, tr, va, epochs, bn_frozen=True, tag=f"{method}/LP")
        if method == "lpft":
            for p in model.parameters(): p.requires_grad_(True)
            best_ep += fit(model, model.parameters(), 1e-5, tr, va, epochs, tag="lpft/FT")
    else:
        cfg = LoraConfig(r=16, lora_alpha=16, target_modules=LORA_TARGET,
                         use_dora=(method == "dora"), modules_to_save=["fc"])
        model = get_peft_model(model, cfg)
        best_ep = fit(model, [p for p in model.parameters() if p.requires_grad], 5e-4, tr, va, epochs, tag=method)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if method not in ("lp",) \
        else sum(p.numel() for p in head.parameters())
    res = dict(backbone=BACKBONE, method=method, seed=seed, best_epoch=best_ep, trainable_params=trainable,
               in_domain=predict(model, fte, yte), shift=predict(model, fdw, ydw),
               seconds=round(time.time() - t0, 1), classes=CLASSES)
    if not smoke: json.dump(res, open(path, "w"), indent=2)
    print(f"[{method} s{seed}] GBIF={res['in_domain']['acc']:.4f}  DeepWeeds={res['shift']['acc']:.4f}"
          f"  trainable={trainable/1e6:.2f}M  ({res['seconds']:.0f}s)", flush=True)

def summary():
    R = {m: [json.load(open(f"{OUT}/{m}_s{s}.json")) for s in SEEDS if os.path.exists(f"{OUT}/{m}_s{s}.json")]
         for m in METHODS}
    base = {r["seed"]: r["shift"]["acc"] for r in R["full"]}
    L = [f"# Fine-tuning spectrum pilot ({BACKBONE}, GBIF 400/class → DeepWeeds)\n",
         "| Method | Trainable params | GBIF test | DeepWeeds | Δ vs full | Every seed better? |",
         "|---|---|---|---|---|---|"]
    go = []
    for m in METHODS:
        if not R[m]: continue
        ind = np.mean([r["in_domain"]["acc"] for r in R[m]]); sh = [r["shift"]["acc"] for r in R[m]]
        d = np.mean(sh) - np.mean(list(base.values())) if base else float("nan")
        every = all(r["shift"]["acc"] > base.get(r["seed"], 1) for r in R[m]) if m != "full" else None
        rng = f" ({min(sh)*100:.1f}–{max(sh)*100:.1f})" if len(sh) > 1 else ""
        L.append(f"| {m} | {R[m][0]['trainable_params']/1e6:.2f}M | {ind*100:.1f}% | {np.mean(sh)*100:.1f}%{rng} | "
                 f"{'—' if m == 'full' else f'{d*100:+.1f}'} | {'—' if every is None else ('yes' if every else 'no')} |")
        if m != "full" and d >= NOTABLE and every: go.append(m)
    L.append(f"\n**Decision (≥{NOTABLE*100:.0f} pts on DeepWeeds, every seed better):** "
             + (f"GO — {', '.join(go)}" if go else "NO-GO — no method clears the bar; do not extend to all backbones."))
    txt = "\n".join(L) + "\n"; open(f"{OUT}/summary.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    (ftr, ytr), (fva, yva), (fte, yte) = gbif_splits(); fdw, ydw = deepweeds_all()
    if a.smoke:
        k = lambda f, y, n: (f[::len(f)//n], y[::len(f)//n])
        data = (k(ftr, ytr, 32), k(fva, yva, 16), k(fte, yte, 16), k(fdw, ydw, 16))
        for m in METHODS: run(m, 42, data, smoke=True)
    else:
        data = ((ftr, ytr), (fva, yva), (fte, yte), (fdw, ydw))
        for s in SEEDS:
            for m in METHODS: run(m, s, data, smoke=False)
        summary()
