"""
Aggregate the preprocessing benchmark: baseline (no preprocessing) vs each technique,
per model, on the DeepWeeds target. Reports accuracy and the change vs baseline.
"""
import os, json, glob
from bench_common import deepweeds_arrays
from prep_common import score_deepweeds

MODELS = ["ML (RandomForest)", "DL (ResNet-50)", "VLM (CLIP-LoRA)"]

def main():
    imgs, labels = deepweeds_arrays()
    base = score_deepweeds(imgs, labels)                      # no preprocessing
    rows = {"Baseline (no preprocessing)": base}
    for fp in sorted(glob.glob("results_prep/*.json")):
        if os.path.basename(fp).startswith("_"): continue
        d = json.load(open(fp))
        if "deepweeds" in d: rows[d["technique"]] = d["deepweeds"]

    agg = {"baseline": base, "techniques": rows}
    json.dump(agg, open("results_prep/benchmark.json","w"), indent=2)

    hdr = f"{'Technique':34s}" + "".join(f"{m.split(' (')[0]:>16s}" for m in MODELS)
    print(hdr); print("-"*len(hdr))
    for name, r in rows.items():
        line = f"{name:34s}"
        for m in MODELS:
            acc = r[m]["acc"]*100
            if name.startswith("Baseline"): line += f"{acc:15.1f}%"
            else:
                d = (r[m]["acc"]-base[m]["acc"])*100
                line += f"{acc:8.1f}% ({d:+.1f})"
        print(line)

if __name__ == "__main__":
    main()
