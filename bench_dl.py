"""
DL model: ResNet-50 (ImageNet-pretrained) fine-tuned on GBIF (source).
Benchmark in-domain on GBIF test and out-of-domain on DeepWeeds. Runs on MPS.
"""
import os, json, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score
from bench_common import gbif_files, deepweeds_arrays, CLASSES, NUM

torch.manual_seed(42); np.random.seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class FileDS(Dataset):
    def __init__(self, files, labels, tf): self.f, self.y, self.tf = files, labels, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        return self.tf(Image.open(self.f[i]).convert("RGB")), int(self.y[i])

class ArrDS(Dataset):
    def __init__(self, imgs, labels, tf): self.x, self.y, self.tf = imgs, labels, tf
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.x[i])), int(self.y[i])

def score(y, p): return dict(acc=round(float(accuracy_score(y, p)),4),
                             macro_f1=round(float(f1_score(y, p, average="macro")),4))

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        out = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
        ps.append(out); ys.append(yb.numpy())
    return score(np.concatenate(ys), np.concatenate(ps))

def main():
    t0 = time.time()
    print(f"device={DEVICE}")
    ftr, ytr = gbif_files("global_train"); fva, yva = gbif_files("global_val"); fte, yte = gbif_files("global_test")
    Xau, yau = deepweeds_arrays()
    tr = DataLoader(FileDS(ftr,ytr,train_tf), batch_size=32, shuffle=True)
    va = DataLoader(FileDS(fva,yva,eval_tf), batch_size=64)
    te = DataLoader(FileDS(fte,yte,eval_tf), batch_size=64)
    au = DataLoader(ArrDS(Xau,yau,eval_tf), batch_size=64)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    EPOCHS = 12; best_val, best_state, best_ep = -1, None, -1
    for ep in range(1, EPOCHS+1):
        model.train(); tot=0.0
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*len(xb)
        val = evaluate(model, va)
        print(f"  epoch {ep:2d}  loss={tot/len(ftr):.4f}  val_acc={val['acc']:.3f}")
        if val["acc"] > best_val:
            best_val, best_ep = val["acc"], ep
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  best val_acc={best_val:.3f} @ epoch {best_ep}")

    indom = evaluate(model, te); ood = evaluate(model, au)
    res = dict(model="DL (ResNet-50 fine-tuned)", in_domain_gbif=indom, ood_deepweeds=ood,
               gap_acc=round(indom["acc"]-ood["acc"],4), best_val_acc=round(best_val,4),
               seconds=round(time.time()-t0,1), classes=CLASSES)
    os.makedirs("results_bench", exist_ok=True)
    json.dump(res, open("results_bench/dl.json","w"), indent=2)
    print(f"[DL ] GBIF acc={indom['acc']:.3f} f1={indom['macro_f1']:.3f}  |  "
          f"DeepWeeds acc={ood['acc']:.3f} f1={ood['macro_f1']:.3f}  |  gap={res['gap_acc']:+.3f}")

if __name__ == "__main__":
    main()
