"""
Shared harness for the preprocessing-technique benchmark.

Trains the three models ONCE on GBIF (source) and caches them, then exposes a single
`score_deepweeds(imgs_uint8)` that runs all models on a (possibly preprocessed) DeepWeeds
array. Each prep_<technique>.py applies its own transform to the DeepWeeds test images and
calls score_deepweeds — so every technique is compared on identical frozen models.
"""
import os, json
import numpy as np
from PIL import Image
import joblib
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
import open_clip
from peft import LoraConfig, get_peft_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from bench_common import gbif_files, gbif_arrays, deepweeds_arrays, CLASSES, READABLE, NUM
from bench_ml import extract as ml_extract

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = "models_cache"
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
TEMPLATES = ["a photo of {}.", "a photo of {}, a type of weed.",
             "a close-up photo of a {} plant.", "a field photograph of {}.",
             "an image of the invasive weed {}."]
_eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
_train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
                                transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
_LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=r".*visual\.transformer\.resblocks\.\d+\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)")

def score(y, p): return dict(acc=round(float(accuracy_score(y,p)),4),
                             macro_f1=round(float(f1_score(y,p,average="macro")),4))

# --------------------------------------------------- training (run once, cached)
def _train_rf():
    Xtr, ytr = gbif_arrays("global_train", size=128)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(np.stack([ml_extract(im) for im in Xtr]), ytr)
    joblib.dump(clf, f"{CACHE}/rf.joblib")

class _FileDS(torch.utils.data.Dataset):
    def __init__(self, files, labels, tf): self.f, self.y, self.tf = files, labels, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.tf(Image.open(self.f[i]).convert("RGB")), int(self.y[i])

def _train_resnet():
    from torch.utils.data import DataLoader
    ftr,ytr = gbif_files("global_train"); fva,yva = gbif_files("global_val")
    tr = DataLoader(_FileDS(ftr,ytr,_train_tf), batch_size=32, shuffle=True)
    va = DataLoader(_FileDS(fva,yva,_eval_tf), batch_size=64)
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM); m = m.to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4); crit = nn.CrossEntropyLoss()
    best, best_state = -1, None
    for ep in range(12):
        m.train()
        for xb,yb in tr:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval(); ys,ps=[],[]
        with torch.no_grad():
            for xb,yb in va: ps.append(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); ys.append(yb.numpy())
        acc = accuracy_score(np.concatenate(ys), np.concatenate(ps))
        if acc>best: best, best_state = acc, {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    torch.save(best_state, f"{CACHE}/resnet50.pt")

@torch.no_grad()
def _clip_text(model, tok):
    ws=[]
    for name in READABLE:
        t = tok([tmp.format(name) for tmp in TEMPLATES]).to(DEVICE)
        f = F.normalize(model.encode_text(t), dim=-1).mean(0); ws.append(F.normalize(f,dim=-1))
    return torch.stack(ws)

def _train_clip():
    model,_,preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); model = model.to(DEVICE).eval()
    Xtr,ytr = gbif_arrays("global_train"); Xte,yte = gbif_arrays("global_test")
    model = get_peft_model(model, _LORA)
    with torch.no_grad(): tw = _clip_text(model, tok).to(DEVICE)
    scale = model.logit_scale.exp().detach()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    Xt = torch.stack([preprocess(Image.fromarray(a)) for a in Xtr]); yt = torch.tensor(ytr)
    Xe = torch.stack([preprocess(Image.fromarray(a)) for a in Xte])
    n, best, best_lora = len(Xt), -1, None
    for ep in range(10):
        model.train(); perm=torch.randperm(n)
        for i in range(0,n,32):
            idx=perm[i:i+32]; f=F.normalize(model.encode_image(Xt[idx].to(DEVICE)),dim=-1)
            loss=F.cross_entropy(scale*f@tw.T, yt[idx].to(DEVICE)); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            fe = F.normalize(model.encode_image(Xe.to(DEVICE)),dim=-1); acc=accuracy_score(yte,(fe@tw.T).argmax(1).cpu().numpy())
        if acc>best: best, best_lora = acc, {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora" in k.lower()}
    torch.save(best_lora, f"{CACHE}/clip_lora.pt")

def ensure_models():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(f"{CACHE}/rf.joblib"):      print("training RF ...");      _train_rf()
    if not os.path.exists(f"{CACHE}/resnet50.pt"):    print("training ResNet-50 ..."); _train_resnet()
    if not os.path.exists(f"{CACHE}/clip_lora.pt"):   print("training CLIP-LoRA ..."); _train_clip()

# --------------------------------------------------- loading + scoring
_STATE = {}
def _load():
    if _STATE: return _STATE
    ensure_models()
    _STATE["rf"] = joblib.load(f"{CACHE}/rf.joblib")
    rn = models.resnet50(); rn.fc = nn.Linear(rn.fc.in_features, NUM)
    rn.load_state_dict(torch.load(f"{CACHE}/resnet50.pt", map_location=DEVICE)); _STATE["rn"] = rn.to(DEVICE).eval()
    cm,_,pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); cm = get_peft_model(cm.to(DEVICE).eval(), _LORA)
    sd = cm.state_dict(); sd.update(torch.load(f"{CACHE}/clip_lora.pt", map_location=DEVICE)); cm.load_state_dict(sd)
    with torch.no_grad(): tw = _clip_text(cm, tok).to(DEVICE)
    _STATE.update(clip=cm.eval(), clip_pre=pre, clip_tw=tw); return _STATE

@torch.no_grad()
def score_deepweeds(imgs_uint8, labels=None):
    """imgs_uint8: [N,H,W,3] RGB uint8 (any size). Returns per-model metrics on DeepWeeds."""
    s = _load()
    if labels is None: _, labels = deepweeds_arrays()
    out = {}
    # ML
    F_ = np.stack([ml_extract(cv2.resize(im,(128,128))) for im in imgs_uint8])
    out["ML (RandomForest)"] = score(labels, s["rf"].predict(F_))
    # DL
    xb = torch.stack([_eval_tf(Image.fromarray(im)) for im in imgs_uint8])
    ps=[]
    for i in range(0,len(xb),64): ps.append(s["rn"](xb[i:i+64].to(DEVICE)).argmax(1).cpu().numpy())
    out["DL (ResNet-50)"] = score(labels, np.concatenate(ps))
    # VLM
    xc = torch.stack([s["clip_pre"](Image.fromarray(im)) for im in imgs_uint8]); fs=[]
    for i in range(0,len(xc),64): fs.append(F.normalize(s["clip"].encode_image(xc[i:i+64].to(DEVICE)),dim=-1).cpu())
    out["VLM (CLIP-LoRA)"] = score(labels, (torch.cat(fs) @ s["clip_tw"].cpu().T).argmax(1).numpy())
    return out

if __name__ == "__main__":
    ensure_models()
    imgs, labels = deepweeds_arrays()
    base = score_deepweeds(imgs, labels)
    json.dump(base, open("results_prep/_baseline.json","w") if os.path.isdir("results_prep") else open("baseline_tmp.json","w"), indent=2)
    print("baseline DeepWeeds (no preprocessing):")
    for k,v in base.items(): print(f"  {k:20s} acc={v['acc']:.4f} f1={v['macro_f1']:.4f}")
