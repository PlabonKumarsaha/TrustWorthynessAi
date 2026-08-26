# VLM Domain-Shift Experiment — Weed Species (Global → Australian)

**Task:** Fine-tune a vision-language model (VLM) on a source domain, measure initial
accuracy, test it on a shifted target domain, show the accuracy drop, then apply an
adaptation technique (LoRA) and test whether accuracy recovers.

**Hardware / stack:** Apple M4, 24 GB RAM, PyTorch 2.8 on MPS. VLM = CLIP ViT-B/32
(OpenAI weights) via `open_clip`; adaptation via `peft` LoRA. Total runtime ≈ 6.6 min.

---

## 1. What "source" and "target" mean here

The chat brief mentioned "PlantVillage / a disease" for training and "an Australian dataset
for the same disease" for testing. The actual data on disk is a **weed-species
domain-generalisation** study (8 invasive species), so the brief was mapped as follows:

| Brief term | Actual asset used |
|---|---|
| Source / "PlantVillage" | Globally-sourced GBIF imagery (non-Australian), on disk at `weeds/global_{train,val,test}` — 8 classes |
| Target / "Australian dataset" | **DeepWeeds** (Australian field conditions), materialised from TFDS to `weeds/_arrays/au_*.npy` |
| "VLM fine-tuning" | CLIP ViT-B/32 classified by text prompts, then LoRA-adapted |
| "Evaluation harness" | Accuracy + macro-F1 scoring, source vs target, per stage |

**8 classes:** chinee_apple, lantana, parkinsonia, parthenium, prickly_acacia,
rubber_vine, siam_weed, snake_weed.

- Source test set: 480 images (60/class)
- Australian target set: 1600 images (balanced, 200/class; DeepWeeds "negative" class dropped)
- Chance accuracy for 8 classes = 12.5%

---

## 2. Results — top-1 accuracy (macro-F1 in parentheses)

| Stage | Source — global/GBIF test | Target — Australian/DeepWeeds | Source→Target drop |
|---|---|---|---|
| Zero-shot VLM, single prompt | 35.4% (0.311) | 12.3% (0.069) | −23.1 pts |
| **Zero-shot VLM, ensembled prompts** | **38.8% (0.365)** | **11.7% (0.073)** | **−27.1 pts** |
| **LoRA-adapted (trained on source)** | **88.5% (0.886)** | **35.2% (0.347)** | −53.3 pts |

LoRA config: rank 16, alpha 32, dropout 0.05, applied to the CLIP **visual** tower
(`attn.out_proj`, `mlp.c_fc`, `mlp.c_proj`). Trainable params: **1,769,472 / 153,046,785
(1.16%)**. 10 epochs, AdamW lr 1e-4, batch 32, early-stopped at **epoch 8** (best source
val acc 85.8%). Text tower frozen; class prompts ensembled over 5 templates.

---

## 3. Interpretation (the three things the brief asked for)

1. **Initial accuracy.** Off-the-shelf CLIP scores **38.8%** on the source domain — modest,
   because these 8 weed species are fine-grained and visually similar.

2. **Accuracy falls on the Australian data.** The *same* frozen model scores **11.7%** on
   DeepWeeds — essentially chance (12.5%). A ~27-point collapse. This is the domain-shift
   penalty: global stock/herbarium imagery ≠ Australian field conditions.

3. **The technique recovers accuracy — partially.** LoRA (only 1.16% of params trainable)
   lifts source to **88.5%** and the Australian target from 11.7% → **35.2%** — roughly
   **3× above zero-shot** and clearly above chance. Accuracy increases and the method works,
   but a large domain gap remains (88.5% source vs 35.2% target).

**Headline finding:** LoRA trained *only on the source domain* strongly fits the source and
meaningfully helps the unseen target, but does **not** close the global→Australian gap. That
residual gap is the scientifically interesting result — it motivates target-domain adaptation
(a few Australian shots, or DeepWeeds-side fine-tuning) as the next step.

---

## 4. Honest caveats

- CLIP is a genuine VLM but not a large generative VLM (e.g. LLaVA/Qwen-VL) — those are not
  trainable on a 24 GB M4. CLIP + LoRA is the faithful, runnable version of the request.
- LoRA was trained on **source only**; the target improvement is pure generalisation, not
  target fine-tuning. Training on (or few-shot adapting to) DeepWeeds would raise target
  accuracy further.
- "Evaluation harness" and "agent-in-loop" from the brief: the harness is implemented (the
  scoring in the script); agent-in-loop was not run (LoRA was chosen as the concrete lever).

---

## 5. Reproduce

```bash
# from the project root, with the .venv used for this run
.venv/bin/python clip_vlm_experiment.py
# writes clip_experiment_results.json
```

Key files:
- `clip_vlm_experiment.py` — full pipeline (data load, zero-shot, LoRA, eval)
- `clip_experiment_results.json` — raw numbers
- `weeds/global_{train,val,test}/` — source (GBIF) images
- `weeds/_arrays/au_{imgs,labels}.npy` — Australian (DeepWeeds) target

---

## 6. Raw results JSON

```json
{
  "zeroshot_single": {
    "source_test": { "acc": 0.3542, "macro_f1": 0.3112 },
    "australian":  { "acc": 0.1231, "macro_f1": 0.0691 }
  },
  "zeroshot_ensemble": {
    "source_test": { "acc": 0.3875, "macro_f1": 0.3648 },
    "australian":  { "acc": 0.1169, "macro_f1": 0.0731 }
  },
  "lora_source_adapted": {
    "source_test": { "acc": 0.8854, "macro_f1": 0.8858 },
    "australian":  { "acc": 0.3525, "macro_f1": 0.3471 },
    "best_val_acc": 0.8583,
    "best_epoch": 8,
    "trainable_params": 1769472
  },
  "_meta": {
    "device": "mps",
    "model": "ViT-B-32/openai",
    "seconds": 397.3,
    "n_source_test": 480,
    "n_australian": 1600
  }
}
```
