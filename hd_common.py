"""
Replication of the HD(LM)^2D protocol (Shaukat, Luo & Varadharajan, 2023, EAAI 122:106030)
on weed imagery, plus a distribution-shift test the original paper never ran.

Their pipeline: fine-tune an ImageNet CNN end to end with a new head (dropout -> dense),
take the penultimate ("last fully connected") features, and refit classical classifiers
on them. Here the source domain is GBIF (8 species, 60/20/20) and the shift test is the
full DeepWeeds 8-class set.
"""
import os, glob
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

CLASSES = ["chinee_apple", "lantana", "parkinsonia", "parthenium",
           "prickly_acacia", "rubber_vine", "siam_weed", "snake_weed"]
NUM = len(CLASSES)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GBIF_DIR, DW_DIR, OUT = "weeds/global_raw", "weeds/dw8", "results_hd"

# cheapest first, so partial results arrive early
BACKBONES = ["mobilenet_v2", "resnet50", "densenet201", "vgg16", "vgg19", "regnet_y_32gf"]
WEIGHTS = {
    "mobilenet_v2":  models.MobileNet_V2_Weights.IMAGENET1K_V1,
    "resnet50":      models.ResNet50_Weights.IMAGENET1K_V1,
    "densenet201":   models.DenseNet201_Weights.IMAGENET1K_V1,
    "vgg16":         models.VGG16_Weights.IMAGENET1K_V1,
    "vgg19":         models.VGG19_Weights.IMAGENET1K_V1,
    "regnet_y_32gf": models.RegNet_Y_32GF_Weights.IMAGENET1K_V1,
}

# Paper's augmentation table: rescale, shear 0.2, zoom 0.2, h+v flip, shift 0.1,
# brightness 0.9-1.1, rotation 0. Shear read as 0.2 rad (~11.5 deg); torchvision pads
# with constant fill rather than Keras' "nearest".
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=11.46),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=(0.9, 1.1)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

class ImgDS(Dataset):
    def __init__(self, files, labels, tf): self.f, self.y, self.tf = files, labels, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.tf(Image.open(self.f[i]).convert("RGB")), int(self.y[i])

def loader(files, labels, tf, shuffle, bs=32):
    return DataLoader(ImgDS(files, labels, tf), batch_size=bs, shuffle=shuffle,
                      num_workers=4, persistent_workers=True)

def list_dir(root):
    fs, ys = [], []
    for ci, c in enumerate(CLASSES):
        for fp in sorted(glob.glob(os.path.join(root, c, "*"))):
            fs.append(fp); ys.append(ci)
    return np.array(fs), np.array(ys)

def gbif_splits(seed=42, frac=(0.6, 0.2, 0.2)):
    """Paper-faithful stratified 60/20/20 -> 240/80/80 per class."""
    fs, ys = list_dir(GBIF_DIR); rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in range(NUM):
        idx = np.where(ys == c)[0]; rng.shuffle(idx)
        a = int(frac[0]*len(idx)); b = a + int(frac[1]*len(idx))
        for p, sl in zip(parts, (idx[:a], idx[a:b], idx[b:])): p.extend(sl)
    return [(fs[np.array(p)], ys[np.array(p)]) for p in parts]

def deepweeds_all():
    return list_dir(DW_DIR)

def build(name, p_drop=0.5):
    """ImageNet backbone with its final layer replaced by the paper's head: dropout -> dense(8).
    Returns (model, head); the head's *input* is the penultimate feature vector."""
    m = getattr(models, name)(weights=WEIGHTS[name])
    if name.startswith("vgg"):        parent, key = m.classifier, 6     # keeps fc1/fc2 -> 4096-d
    elif name == "mobilenet_v2":      parent, key = m.classifier, 1     # 1280-d
    elif name == "densenet201":       parent, key = m, "classifier"      # 1920-d
    else:                             parent, key = m, "fc"              # resnet 2048, regnet 3712
    old = parent[key] if isinstance(key, int) else getattr(parent, key)
    head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(old.in_features, NUM))
    if isinstance(key, int): parent[key] = head
    else: setattr(parent, key, head)
    return m.to(DEVICE), head

@torch.no_grad()
def extract(model, head, files, bs=64):
    """Penultimate features (input to the head) and logits for a list of images."""
    buf = {}
    hook = head.register_forward_hook(lambda mod, inp, out: buf.__setitem__("f", inp[0]))
    model.eval(); feats, logits = [], []
    for xb, _ in loader(files, np.zeros(len(files), int), eval_tf, False, bs):
        out = model(xb.to(DEVICE))
        feats.append(buf["f"].float().cpu().numpy()); logits.append(out.float().cpu().numpy())
    hook.remove()
    return np.concatenate(feats), np.concatenate(logits)

def metrics(y, p):
    return dict(acc=round(float(accuracy_score(y, p)), 4),
                macro_f1=round(float(f1_score(y, p, average="macro")), 4),
                bal_acc=round(float(balanced_accuracy_score(y, p)), 4))
