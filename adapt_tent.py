"""
TENT: fully test-time adaptation by entropy minimisation on the ResNet.
Recompute BatchNorm stats on target batches and optimise only the BN affine (scale/shift)
params to minimise prediction entropy. Label-free.  (Wang et al., ICLR 2021)
"""
import os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from adapt_common import (load_resnet, load_clip, test_set, eval_tf, score,
                          resnet_probs, clip_probs, DEVICE)

def configure(model):
    model.train()
    for p in model.parameters(): p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True); m.track_running_stats = False; m.running_mean = None; m.running_var = None
            params += [m.weight, m.bias]
    return params

def tent_adapt(model, imgs, steps=1, bs=64, lr=1e-3):
    params = configure(model)
    opt = torch.optim.Adam(params, lr=lr)
    xb = torch.stack([eval_tf(Image.fromarray(im)) for im in imgs])
    for _ in range(steps):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), bs):
            b = xb[perm[i:i+bs]].to(DEVICE)
            p = model(b).softmax(1)
            ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
            opt.zero_grad(); ent.backward(); opt.step()
    return model

def main():
    Xte, yte = test_set()
    rn = tent_adapt(load_resnet(), Xte, steps=2)
    cs = load_clip()
    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    res = {
        "TENT-ResNet":            score(yte, p_rn.argmax(1)),
        "TENT-ResNet + CLIP ens": score(yte, (p_rn + p_cl).argmax(1)),
    }
    os.makedirs("results_adapt", exist_ok=True)
    json.dump({"technique":"TENT (label-free)","deepweeds":res}, open("results_adapt/tent.json","w"), indent=2)
    for k,v in res.items(): print(f"  {k:26s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
