"""
Reverse-direction experiment: train on DeepWeeds (Australian field), test on GBIF (web).

Shared data layer + paper-faithful transforms (Olsen et al. 2019):
resize 256 -> random augment (rotation +/-360, scale 0.5-1.0, colour/illumination shift,
perspective, 50% h-flip) -> crop 224. Adam 1e-4, batch 32, binary cross-entropy.
"""
import os, glob, json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLASSES = ["chinee_apple", "lantana", "parkinsonia", "parthenium",
           "prickly_acacia", "rubber_vine", "siam_weed", "snake_weed"]
NUM = len(CLASSES)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DW_DIR   = "weeds/dw8"          # DeepWeeds, 8 classes, negatives dropped
GBIF_DIR = "weeds/global_raw"   # GBIF, 400/class

# ---- paper-faithful transforms ----
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(degrees=360, scale=(0.5, 1.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

class ImgDS(torch.utils.data.Dataset):
    def __init__(self, files, labels, tf):
        self.f, self.y, self.tf = files, labels, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        return self.tf(Image.open(self.f[i]).convert("RGB")), int(self.y[i])

def list_dir(root):
    fs, ys = [], []
    for ci, c in enumerate(CLASSES):
        for fp in sorted(glob.glob(os.path.join(root, c, "*"))):
            fs.append(fp); ys.append(ci)
    return np.array(fs), np.array(ys)

def deepweeds_splits(seed=42, frac=(0.60, 0.20, 0.20)):
    """Stratified 60/20/20 over the exported DeepWeeds 8-class set."""
    fs, ys = list_dir(DW_DIR)
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(NUM):
        idx = np.where(ys == c)[0]; rng.shuffle(idx)
        n = len(idx); a = int(frac[0]*n); b = a + int(frac[1]*n)
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    tr, va, te = map(np.array, (tr, va, te))
    return (fs[tr], ys[tr]), (fs[va], ys[va]), (fs[te], ys[te])

def gbif_all():
    return list_dir(GBIF_DIR)

def scores(y, p):
    return dict(acc=round(float(accuracy_score(y, p)), 4),
                macro_f1=round(float(f1_score(y, p, average="macro")), 4),
                per_class_f1=[round(float(v), 4) for v in
                              f1_score(y, p, average=None, labels=list(range(NUM)))],
                confusion=confusion_matrix(y, p, labels=list(range(NUM))).tolist())
