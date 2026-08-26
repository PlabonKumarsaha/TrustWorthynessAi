# DL vs ML vs VLM — domain-shift benchmark (4 species)

**Setup.** Train each model on **GBIF** (globally-sourced, non-Australian imagery); benchmark
in-domain on the GBIF test split; then test out-of-domain on **DeepWeeds** (Australian field
imagery). Same 4 species everywhere. Hardware: Apple M4, MPS.

**Species (chance = 25%):** lantana, parkinsonia, parthenium, rubber_vine
(visually distinct, exact-taxonomic matches, ~1,000+ target images each).

**Data.** GBIF: 280 train / 60 val / 60 test per class. DeepWeeds test: 300 per class (1,200 total).

## Stage 1 + 2 results — accuracy (macro-F1)

| Model | GBIF (in-domain) | DeepWeeds (out-of-domain) | Domain gap |
|---|---|---|---|
| **DL — ResNet-50 fine-tuned** | **94.6%** (0.946) | **65.2%** (0.653) | **−29.3 pts** |
| VLM — CLIP ViT-B/32, LoRA | 98.8% (0.988) | 58.4% (0.569) | −40.3 pts |
| VLM — CLIP ViT-B/32, zero-shot | 80.8% | 28.1% | −52.8 pts |
| ML — Random Forest (colour+texture) | 68.3% (0.680) | 27.7% (0.274) | −40.7 pts |

*(sorted by DeepWeeds accuracy)*

## Key findings

1. **Every model drops hard on Australian data** — the domain shift is real and large for all
   three families (29–53 point gaps). This confirms the core hypothesis.
2. **Best target model = ResNet-50 (DL): 65.2% on DeepWeeds**, and it has the **smallest gap**.
3. **Highest in-domain ≠ best on target.** CLIP+LoRA wins in-domain (98.8%) but *overfits the
   source domain harder* → bigger gap and lower target accuracy (58.4%) than ResNet-50.
   A cautionary, citable result: don't pick a deployment model on in-domain accuracy alone.
4. **Classical ML barely beats chance on the target (27.7% vs 25%)** — hand-crafted colour/texture
   features do not transfer across domains at all.
5. **Zero-shot CLIP is near chance on the target (28.1%)** but the *same* backbone with LoRA
   more than doubles it — adaptation matters even before touching the target.

## Objective going forward
Close the ~29–40 pt gap on DeepWeeds using techniques that **do not train on DeepWeeds labels**
(test-time augmentation, distribution alignment, stronger backbone, retrieval/Tip-Adapter),
and optionally the few-shot / on-target ceiling for comparison.

## Files
- `bench_common.py` — shared 4-species data layer
- `bench_ml.py`, `bench_dl.py`, `bench_vlm.py` — one model family each ("separate pages")
- `results_bench/{ml,dl,vlm,benchmark}.json` — raw numbers
