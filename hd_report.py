"""
Stage 3: tables for the HD(LM)^2D replication, in-domain (GBIF test) and under shift
(DeepWeeds). Mirrors the paper's Table 13 (backbone x head grid) and Table 14 (mean gain of
each head over the end-to-end model), then asks the question the paper did not: do those
conclusions, and in-domain model selection, survive distribution shift?
"""
import os, json
import numpy as np
from scipy.stats import spearmanr
from hd_common import BACKBONES, OUT, DIRECTION
from hd_heads import HEADS

# older forward-direction results stored domain-specific keys
ALIAS = {"in_domain": "gbif_test", "shift": "deepweeds"}

def load():
    e2e, heads = {}, {}
    for b in BACKBONES:
        if os.path.exists(f"{OUT}/e2e/{b}.json"):   e2e[b] = json.load(open(f"{OUT}/e2e/{b}.json"))
        if os.path.exists(f"{OUT}/heads/{b}.json"): heads[b] = json.load(open(f"{OUT}/heads/{b}.json"))
    return e2e, heads

def _get(d, dom):
    return d[dom] if dom in d else d[ALIAS[dom]]

def acc(e2e, heads, b, h, dom):
    return _get(e2e[b] if h == "E2E" else heads[b][h], dom)["acc"]

def main():
    e2e, heads = load()
    bbs = [b for b in BACKBONES if b in e2e and b in heads]
    cols = ["E2E"] + list(HEADS)
    first = e2e[bbs[0]]
    in_name = first.get("in_domain_name", "GBIF"); sh_name = first.get("shift_name", "DeepWeeds")
    DOMAINS = [("in_domain", f"{in_name} test (in-domain)"), ("shift", f"{sh_name} (shift)")]
    L = [f"# HD(LM)²D replication: train on {in_name} → shift-test on {sh_name}\n",
         f"Direction: `{DIRECTION}`. Backbones complete: {len(bbs)}/{len(BACKBONES)} "
         f"({', '.join(bbs)}). 8 classes, chance 12.5%.\n"]

    # 1. end-to-end
    L += ["## 1. End-to-end fine-tuned models (paper Tables 7–9)\n",
          f"| Backbone | {in_name} test | {sh_name} | Gap |", "|---|---|---|---|"]
    for b in bbs:
        L.append(f"| {b} | {_get(e2e[b],'in_domain')['acc']*100:.1f}% | "
                 f"{_get(e2e[b],'shift')['acc']*100:.1f}% | {e2e[b]['gap_acc']*100:.1f} |")

    # 2. grids
    for dom, title in DOMAINS:
        L += [f"\n## 2. Backbone × head accuracy — {title} (paper Table 13)\n",
              "| Backbone | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for b in bbs:
            row = [acc(e2e, heads, b, h, dom) for h in cols]
            best = max(row)
            L.append(f"| {b} | " + " | ".join(
                (f"**{v*100:.1f}**" if v == best else f"{v*100:.1f}") for v in row) + " |")

    # 3. mean gain over E2E, both domains (paper Table 14)
    L += ["\n## 3. Mean gain of each head over the end-to-end model (paper Table 14)\n",
          "| Head | In-domain gain | Improves (in-domain) | Shift gain | Improves (shift) |",
          "|---|---|---|---|---|"]
    rows = []
    for h in HEADS:
        g_in = [acc(e2e, heads, b, h, "in_domain") - acc(e2e, heads, b, "E2E", "in_domain") for b in bbs]
        g_sh = [acc(e2e, heads, b, h, "shift") - acc(e2e, heads, b, "E2E", "shift") for b in bbs]
        rows.append((h, np.mean(g_in), sum(g > 0 for g in g_in), np.mean(g_sh), sum(g > 0 for g in g_sh)))
    for h, gi, ni, gs, ns in sorted(rows, key=lambda r: -r[1]):
        L.append(f"| {h} | {gi*100:+.2f} | {ni}/{len(bbs)} | {gs*100:+.2f} | {ns}/{len(bbs)} |")

    # 4. does in-domain selection survive shift?
    combos = [(b, h) for b in bbs for h in cols]
    a_in = np.array([acc(e2e, heads, b, h, "in_domain") for b, h in combos])
    a_sh = np.array([acc(e2e, heads, b, h, "shift") for b, h in combos])
    rank_sh = (-a_sh).argsort().argsort() + 1
    i_best_in, i_best_sh = int(a_in.argmax()), int(a_sh.argmax())
    rho, p = spearmanr(a_in, a_sh)
    L += ["\n## 4. Does in-domain model selection survive shift?\n",
          f"- Combinations: {len(combos)} ({len(bbs)} backbones × {len(cols)} classifiers incl. E2E)",
          f"- Best in-domain: **{combos[i_best_in][0]} + {combos[i_best_in][1]}** — "
          f"{a_in[i_best_in]*100:.1f}% {in_name} → {a_sh[i_best_in]*100:.1f}% {sh_name} "
          f"(rank {rank_sh[i_best_in]}/{len(combos)} under shift)",
          f"- Best under shift: **{combos[i_best_sh][0]} + {combos[i_best_sh][1]}** — "
          f"{a_sh[i_best_sh]*100:.1f}% {sh_name} ({a_in[i_best_sh]*100:.1f}% {in_name})",
          f"- Cost of selecting on in-domain accuracy: {(a_sh[i_best_sh]-a_sh[i_best_in])*100:.1f} pts on {sh_name}",
          f"- Spearman(in-domain, shift) over all combinations: ρ = {rho:.3f} (p = {p:.2g})",
          f"- Gap ({in_name} − {sh_name}): mean {np.mean(a_in-a_sh)*100:.1f} pts, "
          f"range {np.min(a_in-a_sh)*100:.1f}–{np.max(a_in-a_sh)*100:.1f}"]
    txt = "\n".join(L) + "\n"
    open(f"{OUT}/report.md", "w").write(txt); print(txt)

if __name__ == "__main__":
    main()
