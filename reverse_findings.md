# Reverse-direction experiment: train on DeepWeeds → test on GBIF

**Question.** Can a model trained on DeepWeeds (Australian field imagery) reach the same accuracy
on GBIF (web/citizen-science imagery)? This is the reverse of the earlier GBIF→DeepWeeds study,
and tests whether the domain gap is **symmetric**.

## Setup

- **Train/val/test:** DeepWeeds, 8 weed classes, negatives dropped — 5,039 / 1,677 / 1,687
  (stratified 60/20/20), exported at 256×256.
- **Out-of-domain test:** all 3,200 GBIF images (400/class), never used in training.
- **Models:** ImageNet-pretrained **ResNet-50** and **Inception-v3**, fine-tuned end-to-end,
  final layer → 8 units. (Same two architectures as Olsen et al., 2019.)
- **Recipe (paper-faithful):** binary cross-entropy, Adam 1e-4, batch 32, resize 256 → rotation
  ±360°, scale 0.5–1.0, colour/illumination jitter, perspective, 50% h-flip → crop 224;
  LR halved on val plateau, early stopping. Max 30 epochs.
- **Deviations:** PyTorch/MPS instead of Keras/TF (no TF GPU on this machine); 30 epochs with
  patience scaled 4/8 instead of ~100 with 16/32.

## Results

| Model | DeepWeeds test (in-domain) | GBIF (out-of-domain) | Gap |
|---|---|---|---|
| ResNet-50 | **96.4%** (F1 0.964) | **30.4%** (F1 0.283) | **66.0 pts** |
| Inception-v3 | **94.7%** (F1 0.947) | **26.9%** (F1 0.259) | **67.8 pts** |

Chance = 12.5%.

### Reproduction check
Olsen et al. report ResNet-50 95.7% and Inception-v3 95.1%. We get 96.4% and 94.7% — same
ballpark and **the same ordering** (ResNet > Inception). Not directly comparable (ours is 8-class
with negatives dropped; theirs is 9-class including 9,106 negatives), but the reproduction looks
sound.

## Headline: the domain gap is strongly asymmetric

| Direction | Drop |
|---|---|
| GBIF → DeepWeeds (earlier work) | ~30 points |
| **DeepWeeds → GBIF (this work)** | **66–68 points** |

**More than double in the reverse direction**, confirmed independently by two architectures.

### Interpretation: diversity, not difficulty, drives transfer

The intuitive prediction — that training on the harder, cluttered field domain would transfer
*better* to clean web photos — is wrong.

What matters is the **breadth** of the training distribution. DeepWeeds was captured by a single
instrument (WeedLogger) at a fixed 1 m height, across 8 sites in one region, over one 9-month
window: a narrow, homogeneous distribution. A model trained on it can rely on site-specific cues —
soil colour, illumination, scale, background texture — that do not exist in GBIF. GBIF is the
opposite: worldwide, thousands of photographers, many devices, varied framing, close-ups and
herbarium specimens.

**Narrow → broad fails hard; broad → narrow holds up far better.** This is an argument *for*
web/citizen-science imagery as a training source, and supports the original project direction.

### Failure mode: sink-class collapse

Both models funnel predictions into a few classes on GBIF (400 true per class):

| Predicted-class totals | ResNet-50 | Inception-v3 |
|---|---|---|
| lantana | 933 | 636 |
| chinee_apple | 713 | 639 |
| siam_weed | 521 | 845 |
| **prickly_acacia** | **76** | **106** |
| parthenium | 221 | 178 |

Prickly acacia is nearly erased in both (GBIF F1 **0.113** / **0.111**), and it is the worst class
under both architectures — an architecture-independent failure. This is the same sink-class
pathology seen in the forward direction, but far more severe.

## Files
`reverse_common.py` (data + transforms), `reverse_train.py` (training/eval),
`results_reverse/{resnet50,inception_v3}.json` (full metrics, per-class F1, confusion matrices).
DeepWeeds 8-class export in `weeds/dw8/` (git-ignored).
