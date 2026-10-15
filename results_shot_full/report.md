# DINOv2-448 + SHOT with more GBIF source data → DeepWeeds (8,403 weed images)

Same frozen DINOv2-448 features, SHOT settings and target for every row; only the source data changes. 3 seeds (mean ± std).

| Source | Images | Linear probe | Source-only MLP | **SHOT** | SHOT macro-F1 |
|---|---|---|---|---|---|
| original | 2,240 | 64.5% | 66.1 ± 0.7% | **83.6 ± 0.7%** | 0.840 |
| gbif400 | 3,200 | 62.6% | 65.9 ± 0.7% | **83.4 ± 0.4%** | 0.838 |
| combined | 8,133 | 70.7% | 71.5 ± 1.7% | **85.5 ± 1.5%** | 0.859 |

Reference: the earlier run of this setup (original split, single seed) gave 85.7%.
