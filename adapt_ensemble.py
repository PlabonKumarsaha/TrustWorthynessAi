"""
Ensemble the two source-trained models on the DeepWeeds target (label-free).
ResNet-50 and CLIP-LoRA make different errors; averaging softmax probabilities usually beats
either alone. Also reports the AdaBN-adapted ResNet ensembled with CLIP.
"""
import os, json
import torch.nn as nn
from adapt_common import (load_resnet, load_clip, test_set, score, resnet_probs, clip_probs)
from adapt_adabn import apply_adabn

def main():
    Xte, yte = test_set()
    rn = load_resnet(); cs = load_clip()
    p_rn = resnet_probs(rn, Xte); p_cl = clip_probs(cs, Xte)
    rn_ada = apply_adabn(load_resnet(), Xte); p_ada = resnet_probs(rn_ada, Xte)
    res = {
        "ResNet only":               score(yte, p_rn.argmax(1)),
        "CLIP only":                 score(yte, p_cl.argmax(1)),
        "ResNet + CLIP":             score(yte, (p_rn + p_cl).argmax(1)),
        "AdaBN-ResNet + CLIP":       score(yte, (p_ada + p_cl).argmax(1)),
    }
    os.makedirs("results_adapt", exist_ok=True)
    json.dump({"technique":"Ensemble (label-free)","deepweeds":res}, open("results_adapt/ensemble.json","w"), indent=2)
    for k,v in res.items(): print(f"  {k:26s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
