"""
Same label-free adaptation stage as hd_adapt.py, applied to the frozen CLIP and BLIP backbones.

Identical heads (SVM, LR, HGB, kNN, RF with hd_heads defaults), identical techniques and settings
(hd_adapt.adapt), identical GBIF-400 split and DeepWeeds test — only the features differ. Writes
results_vlm_adapt/ and a combined leaderboard with the six fine-tuned CNNs from results_hd_adapt/.
"""
import os, json, time, warnings
import numpy as np

from hd_common import BACKBONES, metrics
from hd_adapt import adapt, fit_predict, USE, VARIANTS

warnings.filterwarnings("ignore")
OUT = "results_vlm_adapt"
FEATS = {"clip_vitb32_openai": "results_clip_backbone/feats/clip_vitb32_openai.npz",
         "clip_vitl14_laion2b": "results_clip_backbone/feats/clip_vitl14_laion2b.npz",
         "blip_base_coco": "results_blip_backbone/feats/blip_base_coco.npz",
         "blip_large_coco": "results_blip_backbone/feats/blip_large_coco.npz"}
CFG = {"base": {}, "restd": dict(restd=True), "align": dict(align=True), "st": dict(st=True),
       "all": dict(restd=True, align=True, st=True)}

def run(key):
    path = f"{OUT}/{key}.json"
    if os.path.exists(path): print(f"[{key}] done, skipping"); return json.load(open(path))
    d = np.load(FEATS[key]); Xtr, Xte, Xt = d["Xtr"], d["Xte"], d["Xdw"]; ytr, yte, yt = d["ytr"], d["yte"], d["ydw"]
    R, t0 = {}, time.time()
    for h in USE:
        R[h] = {"in_domain": metrics(yte, fit_predict(h, Xtr, ytr, Xte, restd=False)[0].argmax(1))}
        for v in VARIANTS: R[h][v] = metrics(yt, adapt(h, Xtr, ytr, Xt, **CFG[v]).argmax(1))
        print(f"  [{key}] {h:4s} GBIF={R[h]['in_domain']['acc']*100:5.1f} | DW " +
              "  ".join(f"{v}={R[h][v]['acc']*100:5.1f}" for v in VARIANTS), flush=True)
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(path, "w"), indent=2); return R

def table(name, R):
    L = [f"\n### {name}\n", "| Head | GBIF test | DW: paper pipeline | DW: + restd | DW: + align | "
         "DW: + self-train | **DW: + all three** | macro-F1 |", "|---|---|---|---|---|---|---|---|"]
    for h in USE:
        r = R[h]
        L.append(f"| {h} | {r['in_domain']['acc']*100:.1f} | " +
                 " | ".join((f"**{r[v]['acc']*100:.1f}**" if v == "all" else f"{r[v]['acc']*100:.1f}") for v in VARIANTS)
                 + f" | {r['all']['macro_f1']:.3f} |")
    return L

def report(vlm):
    cnn = {b: json.load(open(f"results_hd_adapt/{b}.json")) for b in BACKBONES if os.path.exists(f"results_hd_adapt/{b}.json")}
    L = ["# Adaptation stage on frozen CLIP / BLIP backbones (GBIF-400 → DeepWeeds)\n",
         "Same heads, techniques and data as hd_adapt.py (fine-tuned CNNs).\n"]
    for k, R in vlm.items(): L += table(k, R)
    rows = [(k, "frozen VLM", h, R[h]["in_domain"]["acc"], R[h]["base"]["acc"], R[h]["all"]["acc"], R[h]["all"]["macro_f1"])
            for k, R in vlm.items() for h in USE]
    rows += [(b, "fine-tuned CNN", h, R[h]["in_domain"]["acc"], R[h]["base"]["acc"], R[h]["all"]["acc"], R[h]["all"]["macro_f1"])
             for b, R in cnn.items() for h in USE]
    rows.sort(key=lambda r: -r[5])
    L += ["\n## Combined leaderboard: 6 fine-tuned CNNs + 4 frozen VLMs, top 15 by DeepWeeds (+ all three)\n",
          "| # | Backbone | Type | Head | GBIF test | DW: paper pipeline | **DW: + all three** | macro-F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for i, (b, t, h, ai, base, al, f) in enumerate(rows[:15], 1):
        L.append(f"| {i} | {b} | {t} | {h} | {ai*100:.1f} | {base*100:.1f} | **{al*100:.1f}** | {f:.3f} |")
    txt = "\n".join(L) + "\n"; open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    report({k: run(k) for k in FEATS})
