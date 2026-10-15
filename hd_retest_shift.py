"""
Re-score the DeepWeeds-trained HD(LM)^2D models on a different GBIF set, without retraining.

Loads the fine-tuned checkpoints saved by hd_train.py (HD_DIRECTION=dw2gbif649 ->
models_cache/hd_dw2gbif649_<backbone>.pt), extracts penultimate features for every image of the
new GBIF set, and scores the E2E model plus the 12 classical heads (refit on the cached DeepWeeds
training features, exactly as hd_heads.py does). In-domain (DeepWeeds test) numbers are re-used
from the source run. Writes <out>/{e2e,heads,feats} in the same format as hd_train/hd_heads, so
the report code reads it like any other HD run.

  python hd_retest_shift.py --gbif weeds/gbif_living --name GBIF-Living --out results_hd_rev_living

--extract-only writes feats/ and e2e/ but leaves the heads to hd_heads.py (HD_EXTRA_SHIFT=<out>),
which fits them once and scores both shift sets.
"""
import os, json, time, argparse, warnings
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hd_common import BACKBONES, DEVICE, build, extract, list_dir, metrics
from hd_heads import HEADS

warnings.filterwarnings("ignore")
SRC = "results_hd_rev649"

def run(name, files, ys, sh_name, out, extract_only=False):
    if os.path.exists(f"{out}/heads/{name}.json") or (extract_only and os.path.exists(f"{out}/e2e/{name}.json")):
        print(f"[{name}] done, skipping"); return
    ck = f"models_cache/hd_dw2gbif649_{name}.pt"
    if not os.path.exists(ck): print(f"[{name}] no checkpoint yet, skipping"); return
    t0 = time.time()
    model, head = build(name); model.load_state_dict(torch.load(ck, map_location=DEVICE))
    Xsh, Lsh = extract(model, head, files)
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None
    src = json.load(open(f"{SRC}/e2e/{name}.json")); d = np.load(f"{SRC}/feats/{name}.npz")
    np.savez(f"{out}/feats/{name}.npz", Xsh=Xsh.astype(np.float16), ysh=ys)
    e2e = dict(src, shift_name=sh_name, shift=metrics(ys, Lsh.argmax(1)), source_run=SRC,
               seconds=round(time.time() - t0, 1))
    e2e["gap_acc"] = round(e2e["in_domain"]["acc"] - e2e["shift"]["acc"], 4)
    json.dump(e2e, open(f"{out}/e2e/{name}.json", "w"), indent=2)
    print(f"[{name}] E2E {sh_name}={e2e['shift']['acc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if extract_only: return
    Xtr, Xte = d["Xtr"].astype(np.float32), d["Xte"].astype(np.float32)
    res = {}
    for hn, mk in HEADS.items():
        clf = make_pipeline(StandardScaler(), mk()).fit(Xtr, d["ytr"])
        res[hn] = dict(in_domain=metrics(d["yte"], clf.predict(Xte)), shift=metrics(ys, clf.predict(Xsh.astype(np.float32))))
    json.dump(res, open(f"{out}/heads/{name}.json", "w"), indent=2)
    best = max(res, key=lambda h: res[h]["shift"]["acc"])
    print(f"[{name}] E2E {sh_name}={e2e['shift']['acc']:.4f}  best head {best}={res[best]['shift']['acc']:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbif", default="weeds/gbif_living"); ap.add_argument("--name", default="GBIF-Living")
    ap.add_argument("--out", default="results_hd_rev_living"); ap.add_argument("--extract-only", action="store_true")
    a = ap.parse_args()
    for sub in ("e2e", "heads", "feats"): os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    fs, ys = list_dir(a.gbif)
    for nm in BACKBONES: run(nm, fs, ys, a.name, a.out, a.extract_only)
