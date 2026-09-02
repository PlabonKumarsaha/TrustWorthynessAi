"""
CLIP-guided self-training (SHOT-style, label-free).
Start from the AdaBN-adapted ResNet; form pseudo-labels on the target from the
AdaBN-ResNet + CLIP ensemble; keep only high-confidence samples; fine-tune the ResNet on
those pseudo-labels (+ entropy minimisation). Transductive, uses NO ground-truth target labels.
"""
import os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from adapt_common import (load_resnet, load_clip, test_set, eval_tf, score,
                          resnet_probs, clip_probs, DEVICE)
from adapt_adabn import apply_adabn

CONF = 0.9      # pseudo-label confidence threshold
ROUNDS, EPOCHS, LR = 2, 3, 1e-4

def main():
    Xte, yte = test_set()
    rn = apply_adabn(load_resnet(), Xte); cs = load_clip()
    Xt = torch.stack([eval_tf(Image.fromarray(im)) for im in Xte])

    for rnd in range(ROUNDS):
        p = 0.5*resnet_probs(rn, Xte) + 0.5*clip_probs(cs, Xte)      # ensemble teacher
        pseudo = p.argmax(1); conf = p.max(1)
        mask = conf >= CONF
        kept_acc = float((pseudo[mask]==yte[mask]).mean()) if mask.sum() else 0.0
        print(f"  round {rnd+1}: {int(mask.sum())}/{len(mask)} confident pseudo-labels "
              f"(teacher acc on kept={kept_acc:.3f})")
        idx = np.where(mask)[0]
        yb_all = torch.tensor(pseudo[idx])
        rn.train()
        for m in rn.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()   # freeze BN stats after AdaBN
        opt = torch.optim.AdamW([p_ for p_ in rn.parameters() if p_.requires_grad], lr=LR, weight_decay=1e-4)
        for ep in range(EPOCHS):
            perm = np.random.permutation(len(idx))
            for i in range(0, len(idx), 32):
                bi = idx[perm[i:i+32]]
                xb = Xt[bi].to(DEVICE); yb = torch.tensor(pseudo[bi]).to(DEVICE)
                logits = rn(xb); loss = F.cross_entropy(logits, yb)
                pe = logits.softmax(1); loss = loss + 0.1*(-(pe*torch.log(pe+1e-8)).sum(1).mean())
                opt.zero_grad(); loss.backward(); opt.step()
        rn.eval()

    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    res = {
        "Self-trained ResNet":        score(yte, p_rn.argmax(1)),
        "Self-trained ResNet + CLIP": score(yte, (p_rn + p_cl).argmax(1)),
    }
    os.makedirs("results_adapt", exist_ok=True)
    json.dump({"technique":"CLIP-guided self-training (label-free)","deepweeds":res},
              open("results_adapt/pseudolabel.json","w"), indent=2)
    for k,v in res.items(): print(f"  {k:28s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
