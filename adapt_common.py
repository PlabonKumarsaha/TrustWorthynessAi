"""
Shared harness for the domain-adaptation experiments (target = DeepWeeds, 4 species).

Loads FRESH copies of the cached source-trained models (so each adaptation script can mutate
its own model), and exposes probability/scoring helpers plus the held-out few-shot pool.
"""
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
import open_clip
from peft import get_peft_model
from sklearn.metrics import accuracy_score, f1_score

from bench_common import deepweeds_arrays, NUM, READABLE
from prep_common import CACHE, DEVICE, MEAN, STD, _LORA, TEMPLATES, _clip_text

eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def score(y, p): return dict(acc=round(float(accuracy_score(y,p)),4),
                             macro_f1=round(float(f1_score(y,p,average="macro")),4))

# ---- data ----
def test_set():  return deepweeds_arrays()                       # 1200 imgs, 300/class
def pool_set():  return (np.load("weeds/_arrays/au4_pool_imgs.npy"),
                         np.load("weeds/_arrays/au4_pool_labels.npy"))   # 400 imgs, disjoint

def few_shot(k, seed=42):
    Xp, yp = pool_set(); rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(yp==c)[0], k, replace=False) for c in range(NUM)])
    return Xp[idx], yp[idx]

# ---- fresh model loaders ----
def load_resnet():
    m = models.resnet50(); m.fc = nn.Linear(m.fc.in_features, NUM)
    m.load_state_dict(torch.load(f"{CACHE}/resnet50.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_clip():
    cm,_,pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); cm = get_peft_model(cm.to(DEVICE).eval(), _LORA)
    sd = cm.state_dict(); sd.update(torch.load(f"{CACHE}/clip_lora.pt", map_location=DEVICE)); cm.load_state_dict(sd)
    with torch.no_grad(): tw = _clip_text(cm, tok).to(DEVICE)
    return dict(model=cm.eval(), preprocess=pre, text_w=tw)

# ---- probabilities ----
@torch.no_grad()
def resnet_probs(model, imgs, bs=64):
    xb = torch.stack([eval_tf(Image.fromarray(im)) for im in imgs]); out=[]
    for i in range(0,len(xb),bs): out.append(model(xb[i:i+bs].to(DEVICE)).softmax(1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def clip_probs(cs, imgs, bs=64):
    xb = torch.stack([cs["preprocess"](Image.fromarray(im)) for im in imgs]); out=[]
    for i in range(0,len(xb),bs):
        f = F.normalize(cs["model"].encode_image(xb[i:i+bs].to(DEVICE)),dim=-1)
        out.append((100.0*f@cs["text_w"].T).softmax(1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def clip_feats(cs, imgs, bs=64):
    xb = torch.stack([cs["preprocess"](Image.fromarray(im)) for im in imgs]); out=[]
    for i in range(0,len(xb),bs):
        out.append(F.normalize(cs["model"].encode_image(xb[i:i+bs].to(DEVICE)),dim=-1).cpu())
    return torch.cat(out)
