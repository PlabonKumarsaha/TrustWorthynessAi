"""
Recover DeepWeeds accuracy WITHOUT training on DeepWeeds labels.

Model stays frozen (no supervised fine-tuning on the target). We test label-free /
zero-touch adaptation methods and report which one recovers the most target accuracy:

  base        : zero-shot CLIP (ensembled prompts)                     [control]
  TTA         : test-time augmentation, average probs over K views     [zero-touch]
  TTA+DA      : + distribution alignment (Sinkhorn to uniform prior)   [label-free*]
  TENT        : entropy-minimising LayerNorm adaptation on target imgs [label-free]
  ViT-L/14    : stronger frozen backbone, zero-shot (+TTA+DA)          [zero-touch model swap]

* DA assumes the target class distribution is uniform (it is, by construction: 200/class).
  This uses the *balance* of the target, never its labels.
"""
import json, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from clip_vlm_experiment import (DEVICE, CLASS_NAMES, READABLE, PROMPT_TEMPLATES,
                                 load_arrays, build_text_classifier)

torch.manual_seed(42); np.random.seed(42)

AU_IMGS = np.load("weeds/_arrays/au_imgs.npy")          # (1600,224,224,3) uint8
AU_LBLS = np.load("weeds/_arrays/au_labels.npy")
N, C = len(AU_LBLS), len(CLASS_NAMES)

def clip_norm(preprocess):
    for t in preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            return t.mean, t.std
    return (0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)

def metrics(pred):
    return dict(acc=round(float(accuracy_score(AU_LBLS, pred)),4),
                macro_f1=round(float(f1_score(AU_LBLS, pred, average="macro")),4))

# ---- probabilities from a frozen model (deterministic center preprocess) ----
@torch.no_grad()
def probs_plain(model, text_w, preprocess, bs=64):
    xs = torch.stack([preprocess(Image.fromarray(a)) for a in AU_IMGS])
    out = []
    for i in range(0, N, bs):
        f = F.normalize(model.encode_image(xs[i:i+bs].to(DEVICE)), dim=-1)
        out.append((100.0 * f @ text_w.T).softmax(-1).cpu())
    return torch.cat(out)

# ---- test-time augmentation: average softmax over K random views ----
@torch.no_grad()
def probs_tta(model, text_w, preprocess, K=12, bs=64):
    mean, std = clip_norm(preprocess)
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65,1.0), ratio=(0.8,1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    acc = torch.zeros(N, C)
    pil = [Image.fromarray(a) for a in AU_IMGS]
    # 1 deterministic view + (K-1) augmented views
    views = [torch.stack([preprocess(p) for p in pil])]
    for _ in range(K-1):
        views.append(torch.stack([aug(p) for p in pil]))
    for v in views:
        for i in range(0, N, bs):
            f = F.normalize(model.encode_image(v[i:i+bs].to(DEVICE)), dim=-1)
            acc[i:i+len(f)] += (100.0 * f @ text_w.T).softmax(-1).cpu()
    return acc / len(views)

# ---- distribution alignment (Sinkhorn) toward a uniform target prior ----
def sinkhorn_align(P, iters=100):
    Q = P.clone().double() + 1e-8
    col_target = Q.sum() / C                      # uniform column mass
    for _ in range(iters):
        Q = Q / Q.sum(1, keepdim=True)            # each row is a distribution
        Q = Q / Q.sum(0, keepdim=True) * col_target
    return Q

# ---- TENT: entropy-minimising adaptation of LayerNorm affine params ----
def tent_adapt(model, text_w, preprocess, epochs=2, bs=64, lr=1e-3):
    xs = torch.stack([preprocess(Image.fromarray(a)) for a in AU_IMGS])
    for p in model.parameters():
        p.requires_grad_(False)
    ln_params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            ln_params += [m.weight, m.bias]
    opt = torch.optim.SGD(ln_params, lr=lr, momentum=0.9)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            f = F.normalize(model.encode_image(xs[idx].to(DEVICE)), dim=-1)
            logits = 100.0 * f @ text_w.T
            p = logits.softmax(-1)
            ent = -(p * torch.log(p + 1e-8)).sum(1).mean()   # no labels used
            opt.zero_grad(); ent.backward(); opt.step()
    model.eval()
    return probs_plain(model, text_w, preprocess)

def run_backbone(name, pretrained, do_tent=False):
    print(f"\n### backbone {name}/{pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(name)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        text_w = build_text_classifier(model, tok, PROMPT_TEMPLATES).to(DEVICE)
    res = {}
    Pbase = probs_plain(model, text_w, preprocess)
    res["base"]     = metrics(Pbase.argmax(1).numpy())
    res["base+DA"]  = metrics(sinkhorn_align(Pbase).argmax(1).numpy())
    Ptta = probs_tta(model, text_w, preprocess)
    res["TTA"]      = metrics(Ptta.argmax(1).numpy())
    res["TTA+DA"]   = metrics(sinkhorn_align(Ptta).argmax(1).numpy())
    if do_tent:
        Ptent = tent_adapt(model, text_w, preprocess)
        res["TENT"]    = metrics(Ptent.argmax(1).numpy())
        res["TENT+DA"] = metrics(sinkhorn_align(Ptent).argmax(1).numpy())
    for k,v in res.items():
        print(f"   {k:10s} acc={v['acc']:.4f}  f1={v['macro_f1']:.4f}")
    return res

def main():
    t0 = time.time()
    print(f"device={DEVICE}  target=DeepWeeds ({N} imgs, {C} balanced classes)  model FROZEN, no target labels")
    all_res = {}
    all_res["ViT-B-32_openai"]  = run_backbone("ViT-B-32", "openai", do_tent=True)
    all_res["ViT-L-14_laion2b"] = run_backbone("ViT-L-14", "laion2b_s32b_b82k", do_tent=False)

    # find the best DeepWeeds accuracy across every method
    best = max(((f"{bb}/{m}", r["acc"]) for bb, mr in all_res.items() for m, r in mr.items()),
               key=lambda x: x[1])
    all_res["_meta"] = dict(seconds=round(time.time()-t0,1), best_method=best[0], best_acc=best[1],
                            zero_shot_baseline=all_res["ViT-B-32_openai"]["base"]["acc"])
    json.dump(all_res, open("clip_testtime_results.json","w"), indent=2)
    print(f"\nBEST DeepWeeds accuracy: {best[1]:.4f}  via  {best[0]}")
    print(f"(zero-shot baseline was {all_res['_meta']['zero_shot_baseline']:.4f})")
    print(f"saved -> clip_testtime_results.json  ({all_res['_meta']['seconds']}s)")

if __name__ == "__main__":
    main()
