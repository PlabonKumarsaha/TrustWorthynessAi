"""
Step 2 — CLIP ViT-L/14 at 336 px (OpenAI weights), alone and fused with DINOv2-448.

Resolution was the main source of DINOv2's in-domain gain (224 -> 448), so this tests the same
lever for CLIP. As with DINOv2-448, the GBIF images are stored at 224 px, so 336 px means a
finer patch grid on upsampled images rather than more image detail.

Same split, de-duplicated test subset, classifiers and fusion code as fusion.py.
Writes results_clip336/ (features cached for fusion_cv.py).
"""
import os, json, time, warnings
import numpy as np
import torch.nn.functional as F

from fusion import build_features, dedup_info, run_model, cached, model, concat, DEVICE
from hd_common import list_dir, GBIF_DIR, DW_DIR, metrics

warnings.filterwarnings("ignore")
OUT = "results_clip336"
CONFIGS = [(("clip_l14",), k) for k in ("LR", "SVM")] + [(("clip336",), k) for k in ("LR", "SVM")] + \
          [(("dinov2_448", "clip336"), k) for k in ("LR", "SVM", "Stack")] + \
          [(("dinov2_448", "clip_l14"), k) for k in ("SVM", "Stack")]

def extract(fs):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
    m = m.to(DEVICE).eval()
    return run_model(lambda x: F.normalize(m.encode_image(x), dim=-1), pre, fs, 32)

def evaluate(S, configs, ytr, yte, ydw, clean):
    R = {}
    for names, kind in configs:
        Xtr, blocks = concat(S, names, "tr"); Xte, _ = concat(S, names, "te")
        m = model(kind, blocks).fit(Xtr, ytr); p = m.predict(Xte)
        key = f"{'+'.join(names)} | {kind}"
        R[key] = dict(test=metrics(yte, p), test_dedup=metrics(yte[clean], p[clean]))
        if all(S[n]["dw"] is not None for n in names):
            R[key]["deepweeds"] = metrics(ydw, m.predict(concat(S, names, "dw")[0]))
        print(f"  {key:34s} test={R[key]['test']['acc']*100:5.1f}  de-dup={R[key]['test_dedup']['acc']*100:5.1f}"
              f"  DW={R[key].get('deepweeds', {}).get('acc', float('nan'))*100:5.1f}", flush=True)
    return R

def report(R, title, path):
    L = [f"# {title}\n", "| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | DeepWeeds |",
         "|---|---|---|---|---|---|"]
    for k, r in sorted(R.items(), key=lambda kv: -kv[1]["test_dedup"]["acc"]):
        n, c = k.split(" | "); dw = f"{r['deepweeds']['acc']*100:.1f}" if "deepweeds" in r else "—"
        L.append(f"| {n} | {c} | **{r['test']['acc']*100:.1f}** | **{r['test_dedup']['acc']*100:.1f}** | "
                 f"{r['test']['macro_f1']:.3f} | {dw} |")
    txt = "\n".join(L) + "\n"; open(path, "w").write(txt); print(txt)

if __name__ == "__main__":
    os.makedirs(f"{OUT}/feats", exist_ok=True); t0 = time.time()
    S, gys, ytr, yte, ydw, itr, ite = build_features()
    gfs, _ = list_dir(GBIF_DIR); dfs, _ = list_dir(DW_DIR)
    G = cached(f"{OUT}/feats/clip336_gbif.npz", lambda: extract(gfs))
    D = cached(f"{OUT}/feats/clip336_dw.npz", lambda: extract(dfs))
    S["clip336"] = dict(all=G, tr=G[itr], te=G[ite], dw=D)
    _, clean = dedup_info(S["dinov2_224"]["all"], itr, ite)
    R = evaluate(S, CONFIGS, ytr, yte, ydw, clean)
    json.dump(dict(results=R, seconds=round(time.time() - t0, 1)), open(f"{OUT}/results.json", "w"), indent=2)
    report(R, "Step 2 — CLIP ViT-L/14 at 336 px (fixed GBIF-400 split)", f"{OUT}/report.md")
