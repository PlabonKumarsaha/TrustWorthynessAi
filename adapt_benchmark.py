"""Aggregate all adaptation results into one table, flagging what clears the 80% target."""
import json, glob, os

def main():
    rows = []
    # label-free single/hybrid techniques
    for fp in sorted(glob.glob("results_adapt/*.json")):
        name = os.path.basename(fp)
        d = json.load(open(fp))
        if "deepweeds" in d:
            for method, m in d["deepweeds"].items():
                rows.append((f"{method}", m["acc"], "label-free"))
        elif "by_k" in d:
            for k, methods in d["by_k"].items():
                for method, m in methods.items():
                    rows.append((f"{method} ({k})", m["acc"], "few-shot"))
    rows.sort(key=lambda r: -r[1])
    print(f"{'Method':40s}{'DeepWeeds acc':>15s}{'':>4s}{'setting':>12s}")
    print("-"*72)
    for name, acc, kind in rows:
        flag = "  <= 80%+" if acc >= 0.80 else ""
        print(f"{name:40s}{acc*100:13.1f}%   {kind:>10s}{flag}")
    json.dump([{"method":n,"acc":a,"setting":k} for n,a,k in rows],
              open("results_adapt/benchmark.json","w"), indent=2)

if __name__ == "__main__":
    main()
