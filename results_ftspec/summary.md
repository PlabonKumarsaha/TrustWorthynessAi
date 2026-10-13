# Fine-tuning spectrum pilot (resnet50, GBIF 400/class → DeepWeeds)

| Method | Trainable params | GBIF test | DeepWeeds | Δ vs full | Every seed better? |
|---|---|---|---|---|---|
| full | 23.52M | 84.5% | 38.2% (37.4–38.9) | — | — |
| lp | 0.02M | 70.6% | 26.9% (26.1–27.6) | -11.3 | no |
| lpft | 23.52M | 85.1% | 35.4% (34.7–36.1) | -2.8 | no |
| lora | 1.19M | 85.1% | 32.9% (31.2–34.6) | -5.2 | no |
| dora | 1.22M | 84.5% | 32.1% (27.7–36.5) | -6.1 | no |

**Decision (≥5 pts on DeepWeeds, every seed better):** NO-GO — no method clears the bar; do not extend to all backbones.
