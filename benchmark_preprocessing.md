# Preprocessing techniques — DeepWeeds gap-closing benchmark (4 species)

**Idea.** Close the GBIF→DeepWeeds domain gap with **image preprocessing only** — no retraining,
no target labels. Each technique transforms the DeepWeeds (target) test images; the three frozen
models (trained on GBIF) are re-scored. One technique per file.

**Models are frozen and cached** in `models_cache/`. Baseline = DeepWeeds accuracy with no
preprocessing. Target: 4 species, 1,200 images, chance = 25%.

## Results — DeepWeeds accuracy (Δ vs baseline)

| Technique | ML (RandomForest) | DL (ResNet-50) | VLM (CLIP-LoRA) |
|---|---|---|---|
| **Baseline (none)** | 27.7% | 66.6% | 67.2% |
| **CLAHE contrast normalisation** | **30.8% (+3.2)** | **68.3% (+1.8)** | 67.2% (+0.1) |
| Histogram matching (to GBIF) | 29.9% (+2.3) | 65.4% (−1.2) | **68.5% (+1.3)** |
| Fourier Domain Adaptation (β=0.01) | 27.0% (−0.7) | 65.3% (−1.2) | **68.5% (+1.3)** |
| Reinhard colour transfer | 26.8% (−0.8) | 46.0% (−20.6) | 37.2% (−30.0) |

## Findings

1. **CLAHE is the only universally-safe win** — best for the CNN (+1.8) and ML (+3.2), neutral for
   CLIP. Normalising local contrast/illumination removes a lighting difference between curated GBIF
   shots and Australian field imagery without destroying content.
2. **Frequency/histogram alignment (FDA, histogram matching) helps the VLM (+1.3) but slightly hurts
   the CNN (−1.2).** Which normalisation helps is **model-dependent** — the CNN likes contrast
   normalisation, CLIP likes spectral/tonal alignment.
3. **Reinhard colour transfer is harmful** — DL −20.6, VLM −30.0. Forcing GBIF colour statistics onto
   the target erases discriminative colour and introduces LAB artefacts. Clear negative result;
   aggressive global colour remapping at inference is dangerous.
4. **Gains are small (a few points).** Pixel-level normalisation only fixes the *photometric* part of
   the shift; most of the gap is background clutter, pose, and morphology, which these methods can't
   touch. Best DL after CLAHE = 68.3% vs 94.6% in-domain → ~26-pt gap remains.

## Recommendation
- Use **CLAHE** as a default, low-risk preprocessing step for the CNN/ML models.
- For the VLM, prefer **histogram matching or FDA**.
- **Do not** use Reinhard-style global colour transfer here.
- Preprocessing is a **small, stackable** contribution — combine with stronger levers
  (BatchNorm adaptation, TTA + distribution alignment, few-shot) to actually close the gap.

## Files
- `prep_common.py` — trains/caches the 3 models, shared `score_deepweeds()`
- `prep_clahe.py`, `prep_histmatch.py`, `prep_fda.py`, `prep_reinhard.py` — one technique each
- `prep_benchmark.py` — aggregates `results_prep/*.json` → `results_prep/benchmark.json`
