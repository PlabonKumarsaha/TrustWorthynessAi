"""
Feature-space diagnostic: why did more GBIF data (649/class) transfer *worse* to DeepWeeds than
the original 400/class set?

Uses frozen extractors never fine-tuned on either domain (ImageNet ResNet-50, DINOv2 ViT-B/14),
so distances reflect the images, not a trained model. Measures:
  1. distance of each GBIF set to DeepWeeds (Frechet, RBF-MMD, domain-classifier accuracy),
     on class-balanced subsamples of equal size;
  2. herbarium share: GBIF basisOfRecord for the new set (ground truth), and a zero-shot CLIP
     detector for both sets, validated against that ground truth;
  3. whether herbarium images sit farther from DeepWeeds than living-plant photos;
  4. train/test near-duplicate leakage inside each split;
  5. near-duplicate overlap between the old and new GBIF sets.
"""
import os, csv, json, argparse, time
import numpy as np
import requests, torch, torch.nn.functional as F
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from hd_common import list_dir, _stratified, eval_tf, NUM

OUT = "results_shift"
SETS = {"GBIF-400": "weeds/global_raw", "GBIF-649": "weeds/global_900", "DeepWeeds": "weeds/dw8"}
MANIFEST = "weeds/global_900/manifest.csv"
RNG = np.random.default_rng(0)

class Files(Dataset):
    def __init__(self, fs, tf): self.fs, self.tf = fs, tf
    def __len__(self): return len(self.fs)
    def __getitem__(self, i): return self.tf(Image.open(self.fs[i]).convert("RGB"))

# ---------- frozen extractors ----------
def resnet50(device):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
    return m.to(device).eval(), eval_tf

def dinov2(device):
    import timm
    from timm.data import resolve_data_config, create_transform
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    cfg = resolve_data_config({}, model=m); cfg["input_size"] = (3, 224, 224)
    return m.to(device).eval(), create_transform(**cfg, is_training=False)

def clip_scores(device, fs):
    """Zero-shot P(herbarium) per image with CLIP ViT-B/32."""
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); m = m.to(device).eval()
    herb = ["a herbarium specimen", "a pressed dried plant on paper",
            "a scanned herbarium sheet with a label and ruler"]
    live = ["a photo of a living plant", "a photo of a plant growing outdoors",
            "a close-up photo of fresh leaves and flowers"]
    with torch.no_grad():
        t = F.normalize(m.encode_text(tok(herb + live).to(device)), dim=-1)
        out = []
        for xb in DataLoader(Files(fs, pre), batch_size=64, num_workers=4):
            s = (100 * F.normalize(m.encode_image(xb.to(device)), dim=-1) @ t.T).softmax(-1)
            out.append(s[:, :len(herb)].sum(1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def extract(model, tf, fs, device):
    out = []
    for xb in DataLoader(Files(fs, tf), batch_size=64, num_workers=4):
        out.append(model(xb.to(device)).float().cpu())
    return torch.cat(out).numpy()

# ---------- metrics ----------
def frechet(a, b):
    m1, m2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm.real))

def mmd2(x, y, n=1500, reps=5):
    vals = []
    for _ in range(reps):
        a = x[RNG.choice(len(x), min(n, len(x)), replace=False)]
        b = y[RNG.choice(len(y), min(n, len(y)), replace=False)]
        z = np.vstack([a, b]); sq = (z * z).sum(1)
        d = np.maximum(sq[:, None] + sq[None] - 2 * z @ z.T, 0)
        g = 1.0 / np.median(d[np.triu_indices(len(z), 1)])
        k = np.exp(-g * d); na = len(a)
        vals.append(k[:na, :na].mean() + k[na:, na:].mean() - 2 * k[:na, na:].mean())
    return float(np.mean(vals)), float(np.std(vals))

def domain_acc(x, y, n=3000):
    n = min(n, len(x), len(y))
    a = x[RNG.choice(len(x), n, replace=False)]; b = y[RNG.choice(len(y), n, replace=False)]
    X = np.vstack([a, b]); t = np.r_[np.zeros(n), np.ones(n)]
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
                          X, t, cv=5).mean()
    return float(acc), float(2 * (1 - 2 * (1 - acc)))      # accuracy, proxy A-distance

def diversity(x, n=2000):
    a = x[RNG.choice(len(x), min(n, len(x)), replace=False)]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    c = a @ a.T; return float(1 - c[np.triu_indices(len(a), 1)].mean())

def max_cos(q, ref):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); r = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    return np.concatenate([(q[i:i + 512] @ r.T).max(1) for i in range(0, len(q), 512)])

def balanced_idx(ys, per_class):
    return np.concatenate([RNG.choice(np.where(ys == c)[0], per_class, replace=False) for c in range(NUM)])

# ---------- GBIF basisOfRecord ----------
def basis_of_record(keys):
    out = {}
    for i in range(0, len(keys), 100):
        chunk = keys[i:i + 100]
        r = requests.get("https://api.gbif.org/v1/occurrence/search",
                         params={"gbifId": chunk, "limit": 300}, timeout=60).json()
        for o in r["results"]: out[str(o["key"])] = o.get("basisOfRecord", "")
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    device = "cpu" if a.smoke else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True); t0 = time.time()

    data = {k: list_dir(v) for k, v in SETS.items()}
    if a.smoke:
        data = {k: (fs[::max(1, len(fs) // 48)], ys[::max(1, len(fs) // 48)]) for k, (fs, ys) in data.items()}

    feats = {}
    for name, build in [("resnet50", resnet50), ("dinov2", dinov2)]:
        model, tf = build(device)
        for s, (fs, _) in data.items():
            path = f"{OUT}/feats_{name}_{s}.npz"
            if os.path.exists(path) and not a.smoke: feats[(name, s)] = np.load(path)["X"]; continue
            feats[(name, s)] = extract(model, tf, fs, device)
            if not a.smoke: np.savez(path, X=feats[(name, s)])
            print(f"  extracted {name:8s} {s:9s} {feats[(name, s)].shape}", flush=True)
        del model

    R = {"distance_to_deepweeds": {}, "diversity": {}, "gbif_old_vs_new": {}}
    per = 400 if not a.smoke else 4
    for name in ["resnet50", "dinov2"]:
        dw = feats[(name, "DeepWeeds")]
        for s in ["GBIF-400", "GBIF-649"]:
            x = feats[(name, s)][balanced_idx(data[s][1], min(per, np.bincount(data[s][1]).min()))]
            fd = frechet(x, dw) if not a.smoke else float("nan")
            mm, ms = mmd2(x, dw); acc, ad = domain_acc(x, dw)
            R["distance_to_deepweeds"][f"{name}/{s}"] = dict(frechet=fd, mmd2=mm, mmd2_sd=ms,
                                                             domain_acc=acc, a_distance=ad)
            R["diversity"][f"{name}/{s}"] = diversity(feats[(name, s)])
        R["diversity"][f"{name}/DeepWeeds"] = diversity(dw)
        mm, _ = mmd2(feats[(name, "GBIF-400")], feats[(name, "GBIF-649")])
        R["gbif_old_vs_new"][name] = dict(mmd2=mm,
                                          domain_acc=domain_acc(feats[(name, "GBIF-400")], feats[(name, "GBIF-649")])[0])

    # herbarium: ground truth for the new set, CLIP estimate for both
    rows = list(csv.DictReader(open(MANIFEST)))
    if not a.smoke:
        bor = basis_of_record([r["occurrence"] for r in rows])
        file2herb = {os.path.join("weeds/global_900", r["file"]): bor.get(r["occurrence"], "") == "PRESERVED_SPECIMEN"
                     for r in rows}
        new_fs = data["GBIF-649"][0]
        truth = np.array([file2herb.get(f, False) for f in new_fs])
        p_new = clip_scores(device, new_fs); p_old = clip_scores(device, data["GBIF-400"][0])
        pred = p_new > 0.5
        R["herbarium"] = dict(
            new_true_share=float(truth.mean()),
            basis_counts={k: int(v) for k, v in zip(*np.unique(list(bor.values()), return_counts=True))},
            clip_detector_accuracy=float((pred == truth).mean()),
            clip_detector_recall=float(pred[truth].mean()) if truth.any() else None,
            clip_detector_precision=float(truth[pred].mean()) if pred.any() else None,
            clip_est_share_new=float(pred.mean()), clip_est_share_old=float((p_old > 0.5).mean()))
        for name in ["resnet50", "dinov2"]:
            x, dw = feats[(name, "GBIF-649")], feats[(name, "DeepWeeds")]
            # mmd2 draws equal-size random subsamples, so species order in the file list doesn't bias this
            R["herbarium"][f"{name}/mmd2_herbarium_vs_DW"] = mmd2(x[truth], dw, n=min(1500, int(truth.sum())))[0]
            R["herbarium"][f"{name}/mmd2_living_vs_DW"] = mmd2(x[~truth], dw, n=min(1500, int(truth.sum())))[0]

    # leakage: test images with a near-duplicate in their own training split (DINOv2 space)
    R["leakage_test_has_train_neardup"] = {}
    for s in SETS:
        if a.smoke: break
        fs, ys = data[s]; idx = {f: i for i, f in enumerate(fs)}
        (ftr, _), _, (fte, _) = _stratified(SETS[s])
        x = feats[("dinov2", s)]
        mc = max_cos(x[[idx[f] for f in fte]], x[[idx[f] for f in ftr]])
        R["leakage_test_has_train_neardup"][s] = {f">{t}": float((mc > t).mean()) for t in (0.90, 0.95, 0.98)}

    # overlap between old and new GBIF sets
    mc = max_cos(feats[("dinov2", "GBIF-649")], feats[("dinov2", "GBIF-400")])
    R["new_images_with_neardup_in_old"] = {f">{t}": float((mc > t).mean()) for t in (0.95, 0.98)}

    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{OUT}/shift_check{'_smoke' if a.smoke else ''}.json", "w"), indent=2)
    print(json.dumps(R, indent=2))

if __name__ == "__main__":
    main()
