"""
AdaBN: replace the ResNet's BatchNorm running statistics with statistics estimated on the
(unlabelled) DeepWeeds target. Transductive, label-free, no gradient steps. Also reports the
AdaBN-ResNet ensembled with CLIP.  (Li et al., 2016)
"""
import os, json
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from adapt_common import (load_resnet, load_clip, test_set, eval_tf, score,
                          resnet_probs, clip_probs, DEVICE)

def apply_adabn(model, imgs, bs=64):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None      # cumulative average
    model.train()
    xb = torch.stack([eval_tf(Image.fromarray(im)) for im in imgs])
    with torch.no_grad():
        for i in range(0, len(xb), bs): model(xb[i:i+bs].to(DEVICE))
    return model.eval()

def main():
    Xte, yte = test_set()
    rn = apply_adabn(load_resnet(), Xte)
    cs = load_clip()
    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    res = {
        "AdaBN-ResNet":            score(yte, p_rn.argmax(1)),
        "AdaBN-ResNet + CLIP ens": score(yte, (p_rn + p_cl).argmax(1)),
    }
    os.makedirs("results_adapt", exist_ok=True)
    json.dump({"technique":"AdaBN (label-free)","deepweeds":res}, open("results_adapt/adabn.json","w"), indent=2)
    for k,v in res.items(): print(f"  {k:26s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
