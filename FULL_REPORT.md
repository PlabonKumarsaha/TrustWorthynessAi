# Cross-domain weed classification: full experimental report

## Aim

The question I set out to answer is whether a weed classifier trained entirely on free,
web-sourced imagery can be deployed on Australian field photographs without any local
annotation. Labelling rangeland imagery is expensive, and there is a large amount of
already-labelled plant photography sitting in biodiversity databases, so if the transfer works
it would remove most of the annotation cost from a weed-mapping pipeline. The obvious risk is
domain shift: web photographs and field photographs of the same species do not look alike.

Everything is single-label image classification — one image in, one species out. It is not
detection or segmentation, which matters for how the result should be read and I return to it
at the end.

All experiments ran on a single M4 MacBook, which shaped several design decisions, in particular
the reliance on frozen features and small trainable heads rather than repeated end-to-end
fine-tuning.

---

## 1. Data

### Source: GBIF

I built the training set by querying GBIF for occurrence photographs of eight weed species,
resolving each species by scientific name rather than common name, and excluding any record from
Australia so that source and target genuinely differ. Only CC0 and CC-BY licensed media were
kept. The download was capped at 400 images per class, giving 3,200 images split 70/15/15 into
280 train, 60 validation and 60 test per class, all 224×224 RGB JPEG.

| Folder | Scientific name |
|---|---|
| chinee_apple | *Ziziphus mauritiana* |
| lantana | *Lantana camara* |
| parkinsonia | *Parkinsonia aculeata* |
| parthenium | *Parthenium hysterophorus* |
| prickly_acacia | *Vachellia nilotica* (syn. *Acacia nilotica*) |
| rubber_vine | *Cryptostegia grandiflora* |
| siam_weed | *Chromolaena odorata* |
| snake_weed | *Stachytarpheta* (genus only) |

Seven resolve to an exact species. Snake weed is the exception — I could only match it at genus
level, so that folder may contain several *Stachytarpheta* species. This turned out to matter,
because snake weed consistently behaves as one of the worst classes and becomes a "sink" that
absorbs other species' predictions.

In character these are deliberate photographs: a plant, a flower, sometimes a herbarium specimen,
usually centred and reasonably lit.

### Target: DeepWeeds

DeepWeeds (Olsen et al., 2019) contains 17,509 photographs taken in situ in northern Australian
rangelands, covering the same eight species plus a ninth "negative" class of vegetation that is
not a target weed. The eight weed classes are close to balanced at 1,009–1,125 images each, but
the negative class is large — 9,106 images, slightly more than half the dataset. The images are
cluttered field scenes with natural lighting, soil and occlusion.

For most of the project I evaluated on subsets: first four species, then all eight, in both cases
with negatives removed and classes balanced by subsampling. Only in the final phase did I
evaluate on the complete set with the natural distribution and negatives included. That decision
turned out to matter and is discussed below.

---

## 2. Related work

The obvious reference point is the DeepWeeds paper itself, reporting 95.7% with ResNet-50 and
95.1% with Inception-v3. Those are in-domain and fully supervised — trained on DeepWeeds — so
they are a ceiling rather than a competitor. Recent in-domain work with latent-diffusion
augmentation reaches about 98.5%.

Domain adaptation for agricultural imagery is active. Computers and Electronics in Agriculture
published a self-training approach for weed segmentation in 2024, and a CVPR workshop paper the
same year used greedy pseudo-labelling. Closest to my setup is "From Web Data to Real Fields"
(2025), which adapts detectors from internet-scraped imagery to unlabelled robot field data using
an attention-based adversarial discriminator, reporting a 7.5% detection improvement. A 2025
review gives the standard taxonomy: discrepancy methods such as CORAL and MMD, adversarial
approaches such as DANN, style transfer, pseudo-labelling, Fourier methods and test-time
adaptation. For context on how badly cross-dataset transfer usually goes, models trained on
PlantVillage typically drop to around 68% on PlantDoc.

On the source-free side the standard methods are TENT (entropy minimisation over BatchNorm affine
parameters), AdaBN (recomputing BatchNorm statistics on the target), SHOT (frozen source
classifier plus information maximisation and pseudo-labels), and class-balanced self-training.

Vision-language work includes Tip-Adapter, CoOp and CLIP-Adapter for few-shot CLIP; ReCLIP, which
uses agreement between CLIP's image and text encoders to select pseudo-labels; Co-learn++ and
PADCLIP, which use CLIP as a pseudo-label teacher; and RCL, which uses multimodal LLMs to guide a
reliability curriculum and reports state-of-the-art results with a 9.4% gain on DomainNet. In
agriculture specifically there is AgriCLIP, CropVLM, WeedCLR and AgroBench, plus a 2025 paper
asking whether VLMs can zero-shot replace supervised classifiers in agriculture — the answer
being not yet for fine-grained species.

One line of work bears directly on a conclusion I initially wanted to draw. A 2025 paper on
class-imbalanced federated source-free adaptation argues explicitly for shifting attention from
improving adaptation methods to improving feature extractors, and another shows large
self-supervised models closing the gap in domain-adaptive object detection; there is a similar
reality-check paper in digital pathology. The claim "the backbone matters more than the adaptation
method" is therefore already published for other settings and I cannot present it as new.

---

## 3. What I tried

Roughly eleven stages. The order matters, because most later decisions were reactions to earlier
results.

### Starting with CLIP

I began with CLIP ViT-B/32 on all eight species. Zero-shot classification with prompt ensembling
gave 38.8% on the GBIF test split but only 11.7% on DeepWeeds — essentially the 12.5% chance
level. LoRA fine-tuning on about 1.2% of parameters lifted GBIF to 88.5% and DeepWeeds to 35.2%,
so adaptation helped in relative terms while the gap widened rather than closed.

I then tried test-time methods on the frozen model. Test-time augmentation did almost nothing.
Sinkhorn distribution alignment helped noticeably. Switching to ViT-L/14 trained on LAION-2B and
combining it with augmentation and alignment reached 27.5% — still poor but more than double the
zero-shot baseline. TENT failed outright, with macro-F1 collapsing to 0.03. In hindsight that is
predictable: entropy minimisation sharpens whatever the model already believes, and when the model
is at chance and confidently wrong, sharpening makes things worse.

### Comparing three model families

To get a cleaner picture I dropped to four visually distinct species with exact taxonomic matches
(lantana, parkinsonia, parthenium, rubber vine) and compared three families trained on GBIF.

| Model | GBIF test | DeepWeeds | Gap |
|---|---|---|---|
| ResNet-50, fine-tuned | 94.6% | 65.3% | 29.3 |
| CLIP ViT-B/32 + LoRA | 98.8% | 58.4% | 40.3 |
| CLIP ViT-B/32, zero-shot | 80.8% | 28.1% | 52.8 |
| Random Forest (HSV + GLCM) | 68.3% | 27.7% | 40.7 |

The result I did not expect is that the best in-domain model was not the best on target. CLIP with
LoRA wins on GBIF by four points and loses to ResNet-50 by seven on DeepWeeds — it fits the source
domain harder and generalises worse. Selecting a deployment model on in-domain accuracy would have
picked the wrong one.

### Preprocessing

Since much of the shift is colour and lighting, I tried four preprocessing techniques applied to
the target images at inference with the models frozen. CLAHE contrast normalisation gave small
consistent gains (+1.8 on ResNet, +3.2 on the Random Forest) and was the only technique that never
hurt. Histogram matching and Fourier Domain Adaptation each gave about +1.3 to CLIP but cost the
CNN about a point. Reinhard colour transfer was a disaster, costing ResNet-50 twenty points and
CLIP thirty — forcing source colour statistics onto the target destroys the colour information the
classifier relies on.

The useful conclusion is that the best preprocessing is model-dependent, and that aggressive
global colour remapping should be avoided.

### Adaptation methods

Still on four species: AdaBN on its own hurt, dropping ResNet from 66.6% to 63.5%. TENT gave about
a point. Ensembling ResNet with CLIP was the largest easy gain, taking either model from around
67% to 73.3%, presumably because they make different mistakes.

The best result in this phase came from self-training: using the ResNet–CLIP ensemble as a teacher,
keeping predictions above 0.9 confidence, and fine-tuning ResNet on those pseudo-labels. The
confident subset was 98% correct, and two rounds took ResNet to 82.8% and the ensemble to 84.1%
without a single DeepWeeds label. For comparison, ten real labels per class and a linear probe
reached 89.2%. Tip-Adapter, which I expected to do well, barely moved — around 68% regardless of
shot count, because frozen CLIP features are not discriminative enough on these species.

### A selection rule of my own

Two observations suggested something. Single-model confidence is untrustworthy under this shift —
CLIP was confidently wrong. But when ResNet and CLIP agreed they were right 87.6% of the time,
and when they disagreed only 53.3%; in fact 72.8% of all errors fell in the disagreement set.

So I built a pseudo-label selection rule keeping a sample only if the two architectures agree and
it passes a conformal threshold calibrated on a small held-out pool, and compared it against plain
confidence thresholding in an otherwise identical pipeline. It kept more data (51% vs 35% coverage)
at equal or better precision and improved the final result from 75.6% to 80.3% — a 4.7 point gain
from the selection rule alone. Checking the literature afterwards, ReCLIP, Co-learn++ and CLIP-OT
already occupy much of this space, so the idea was less novel than it seemed when I built it.

### Changing the backbone

The largest single improvement in the project came from replacing the backbone. A linear probe on
frozen DINOv2 ViT-B/14 features, trained only on GBIF labels, reached 78.2% on the four-species
subset at 224 pixels, 88.2% at 336 and 90.8% at 448. Adding label-free self-training took it to
94.2% — more than twenty points above the fine-tuned ResNet, with no adaptation machinery beyond
self-training.

Resolution mattered far more than expected: ten points between 224 and 336 alone. Two things
failed here. Fusing multiple resolutions was worse than using the best one, and ensembling DINOv2
with the weaker ResNet and CLIP dropped it from 88.2% to 85.2%. Averaging models of unequal quality
drags the good one down, which contradicts the conclusion I had drawn earlier when the two models
being ensembled were evenly matched.

### Scaling to eight species

At this point I had 94.2% and was fairly pleased, so the next step was checking it held on all
eight species. It did not: 57.8% at 336 pixels, 63.8% at 448, rising only to 68.8% with
self-training. Going from four classes to eight cost roughly twenty-five points, so the earlier
number had been substantially an artefact of an easier task.

The per-class breakdown showed why. Parthenium and snake weed had high recall and low precision —
0.96 recall at 0.44 precision for parthenium — while prickly acacia collapsed to 0.32 recall
despite 0.69 precision. Two classes were acting as sinks. Worse, self-training made this
progressively worse: pseudo-label precision fell 81.3% → 77.2% → 74.9% across rounds as the sinks
swallowed more of the pseudo-labelled set.

Sinkhorn distribution alignment plus class-balanced selection fixed most of it:

| Method (8 species) | DeepWeeds |
|---|---|
| DINOv2 @448, no adaptation | 63.8% |
| plus naive self-training | 68.8% |
| plus distribution alignment only | 73.2% |
| plus class-balanced self-training only | 71.8% |
| plus both | 80.6% |

Per-class F1 evened from a 0.43–0.90 spread to 0.71–0.90, and — the part I find most interesting —
pseudo-label precision stopped decaying, holding at 89.6%, 89.0%, 87.6% instead of falling.
Alignment does not merely add accuracy; it stops self-training degrading itself.

### A factorial comparison

To see how much came from each ingredient I ran a factorial over three backbones, three
resolutions and five adaptation settings, using an identical frozen-feature and linear-probe
protocol so the classifier was held constant.

| Backbone at resolution | in-domain | none | DA | ST | CBST | DA+CBST |
|---|---|---|---|---|---|---|
| ResNet-50 @224 | 79.4 | 29.8 | 33.2 | 29.1 | 29.8 | 36.0 |
| ResNet-50 @336 | 81.9 | 33.2 | 38.0 | 32.6 | 33.9 | 41.2 |
| ResNet-50 @448 | 84.0 | 37.6 | 41.8 | 37.5 | 37.8 | 45.6 |
| CLIP ViT-B/32 @224 | 85.8 | 34.2 | 35.4 | 32.8 | 35.6 | 38.6 |
| DINOv2 @224 | 93.1 | 54.6 | 64.2 | 56.0 | 55.6 | 72.9 |
| DINOv2 @336 | 94.2 | 57.8 | 69.1 | 60.8 | 62.8 | 79.8 |
| DINOv2 @448 | 95.2 | 63.8 | 73.2 | 68.8 | 71.8 | 80.6 |

Spread attributable to the backbone is 30.8 points, against 11.9 for the adaptation method and 9.8
for resolution. My hypothesis going in was that the ranking of adaptation methods would break down
once the backbone was strong enough — that conclusions drawn on ResNet-50 at 224 pixels would not
transfer. That was wrong: Spearman correlation between rankings from different backbones is 0.80
to 0.90, so the ordering is essentially stable.

What the data does support is closer to the opposite. The benefit of adaptation *grows* with the
quality of the representation: alignment plus class-balanced self-training adds 6.2 points on
ResNet-50 at 224 but 16.8 on DINOv2 at 448. Backbone quality and adaptation are complementary
rather than substitutes, which is a more specific claim than the 2025 papers arguing that
foundation models should replace adaptation methods.

### Testing under realistic conditions

Everything to this point used balanced subsets with negatives removed, and both choices flatter the
result. The next stage removed them: all 17,509 images, the natural class distribution, and the
9,106 negatives included.

The closed-set result held. On the natural unbalanced weed set the unadapted probe gives 64.9%,
alignment takes it to 72.9%, and alignment plus class-balanced self-training reaches 80.1% with
macro-F1 0.801, against 80.6% on the balanced subset. The subsampling had not been inflating
anything, for the straightforward reason that DeepWeeds' weed classes are already close to
balanced.

Two things did not hold.

The first is the class prior. Alignment needs the target class distribution, and I had been
assuming uniform, which happened to be roughly right. Estimating it from unlabelled data with EM
gave an L1 error of 0.577, and using that estimate dropped accuracy to 61.3% — worse than not
adapting at all. EM inherits the model's existing bias toward the sink classes and amplifies it.
So the alignment gain rests on prior knowledge a practitioner may not have.

The second is open-set rejection. A deployed system must reject vegetation that is not a target
weed, and over half of DeepWeeds is exactly that. I compared eleven label-free rejection scores.

| Score | AUROC | FPR at 95% TPR |
|---|---|---|
| kNN in feature space | 0.808 | 63.8% |
| Energy | 0.772 | 76.4% |
| MaxLogit | 0.770 | 75.6% |
| Mahalanobis | 0.769 | 76.3% |
| Entropy | 0.733 | 78.8% |
| Max softmax | 0.718 | 83.7% |
| Relative Mahalanobis | 0.677 | 91.9% |
| CLIP text-defined negatives | 0.634 | 85.3% |
| Cross-architecture disagreement | 0.545 | 94.4% |

Only kNN improved on the baseline, and modestly. Two ideas of my own failed. Describing non-target
vegetation in language — grass, bare soil, gravel, native bushland — and rejecting anything
matching those prompts gave 0.634, below simple energy scoring. And cross-architecture
disagreement, which predicted classification errors so well earlier, is essentially at chance
(0.545) for detecting whether an image is out of distribution; those are different quantities and
I had conflated them. Fusing scores also made things worse, the third time in this project that
combining signals of unequal quality has hurt.

The operational number is the last column. To catch 95% of real weeds the best method still
accepts 64% of ordinary vegetation. An AUROC of 0.81 sounds respectable and is useless for a
sprayer.

### Comparing against published baselines

Last, I implemented CORAL, DANN and SHOT on the same frozen DINOv2 features and scored them on the
same full natural weed set. Since SHOT and DANN need a trainable feature transform I also trained a
source-only model with the identical MLP architecture, so their gains could not be confounded with
extra capacity.

| Method | Accuracy | macro-F1 |
|---|---|---|
| SHOT | 85.7% | 0.860 |
| DA + CBST (mine) | 79.9% | 0.799 |
| DA only (mine) | 72.9% | 0.730 |
| source-only, MLP (matched architecture) | 67.4% | 0.677 |
| source-only, linear probe | 64.9% | 0.653 |
| DANN | 63.9% | 0.633 |
| CORAL | 52.6% | 0.520 |

SHOT beats my approach by 5.8 points, and the matched-architecture control shows this is not a
capacity effect: SHOT gains 18.3 points over its own source-only baseline against 15.0 for mine
over the linear probe. Its adaptation is simply better, and I cannot present my combination as a
contribution.

Two of the three baselines were worse than doing nothing. DANN came in below its own source-only
model, which I attribute to adversarial training being unstable with only 2,240 source samples;
it can also align the domains while mismatching classes. CORAL was worst by a wide margin, which
is unsurprising in hindsight, since whiten-and-recolour requires estimating a 768×768 covariance
matrix from 2,240 points.

There is a secondary observation supporting the factorial result: SHOT was developed on ResNet-era
backbones, and on DINOv2 features it yields a larger absolute gain than on its original
benchmarks. That is independent corroboration, from a method I did not design, that adaptation and
representation quality compound.

---

## 4. How the methods work

**Sinkhorn distribution alignment.** Predicted probabilities form an N×8 matrix. Rows should sum to
one, and if classes are balanced the columns should each carry about N/8 of the mass. A biased
model satisfies the first and violates the second, so alternately normalising rows and rescaling
columns pushes mass off over-predicted classes onto starved ones. It only rescales outputs; the
features never change, and it needs the target prior.

**Class-balanced self-training.** Ordinary self-training keeps the globally most confident
predictions, which are dominated by the sink classes, so retraining amplifies the bias. CBST
instead takes the top-k most confident samples per predicted class so every class contributes,
retrains including those pseudo-labelled target samples, and repeats with a growing fraction.

**SHOT.** Inverts the usual approach: rather than moving decision boundaries to fit the target, it
freezes the classifier — the "source hypothesis" — and moves the features to fit the existing
boundaries. Only the bottleneck is trained, on unlabelled target data, with an information
maximisation loss that minimises the entropy of individual predictions while maximising the
entropy of the average prediction. That second term is an implicit anti-collapse constraint doing
what my Sinkhorn does explicitly, but without needing a prior. It adds self-supervised
pseudo-labels from class centroids in feature space, refreshed each epoch.

**DANN.** Trains a feature extractor, a label classifier and a domain discriminator together. A
gradient reversal layer between extractor and discriminator multiplies gradients by −λ during
backpropagation, so the extractor learns to fool the discriminator and produce domain-invariant
features.

**CORAL.** Closed-form and training-free. Whiten the source features by their own covariance, then
recolour them with the target covariance, so the two domains share second-order statistics; train
the classifier on the transformed source features.

The ranking follows the level of intervention almost exactly. Output-level methods recovered 26%
of the gap, data-level methods 50%, and the method that reshapes the representation itself 69%.
The two feature-level methods that failed did so for the same reason — too little source data to
estimate what they needed.

---

## 5. The final model

```
image 448×448×3
   → DINOv2 ViT-B/14, frozen            86,314,752 params
     (patch 14 → 32×32 = 1024 patches + CLS; 768-d output)
   → standardise features (μ, σ from source)
   → bottleneck: Linear(768→256) + BatchNorm + ReLU   197,376 params
   → classifier: Linear(256→8)                          2,056 params
   → 8 logits → softmax → species
```

86.5M parameters in total, of which **197,376 (0.23%) are trained during adaptation**. Training has
two stages: supervised training of bottleneck and classifier on the 2,240 GBIF images, then SHOT
adaptation on unlabelled DeepWeeds with the classifier frozen and only the bottleneck updated,
using entropy minus diversity plus 0.3 times a centroid pseudo-label loss.

This ran comfortably on a laptop because the expensive 86M-parameter forward pass happens once per
image, offline; everything in the training loop operates on cached 768-d vectors.

---

## 6. How much of the domain shift is recovered

| | Accuracy |
|---|---|
| In-domain (GBIF test, same model) | 95.2% |
| Out-of-domain, no adaptation | 64.9% |
| **Domain gap** | **30.3 points** |

| Method | DeepWeeds | Points recovered | Share of gap |
|---|---|---|---|
| DA only | 72.9% | +8.0 | 26% |
| DA + CBST | 79.9% | +15.0 | 50% |
| SHOT | 85.7% | +20.8 | 69% |

The best method recovers about two-thirds of the shift, leaving roughly ten points against
in-domain performance. DeepWeeds' published fully-supervised benchmark is 95.7%, almost the same
ceiling, so the residual gap is about ten points by either reference.

This accounting is closed-set only: it excludes the 9,106 negatives, which the system cannot
reliably reject. Deployable performance is therefore materially worse than 85.7% suggests.

---

## 7. Where things stand

Training only on free GBIF imagery with no Australian labels, the best configuration reaches 85.7%
accuracy and 0.860 macro-F1 on the full DeepWeeds weed set. That configuration is SHOT on frozen
DINOv2 features, not my own method, which reaches 79.9%. Both are well above the 64.9% unadapted
probe and the 11.7% zero-shot CLIP result I started from.

The findings I am reasonably confident in: backbone choice dominates everything else and was worth
more than all the adaptation methods combined; input resolution is a first-class variable and is
under-reported in this literature; adaptation and representation quality are complementary, with
adaptation gains growing rather than shrinking as the backbone improves, now corroborated by SHOT
as well as my own methods; class-bias correction is unnecessary at four classes and essential at
eight; distribution alignment stops self-training degrading itself; and evaluating on a species
subset inflated my result by roughly twenty-five points.

The failures are worth recording since several cost time. Label-free prior estimation does not work
here and actively backfires. Open-set rejection is not deployable at the accuracy currently
reachable. Reinhard colour transfer is harmful. AdaBN alone hurts and TENT fails on a weak base
model. Tip-Adapter is no better than zero-shot on this task. Language-defined negatives and
cross-architecture disagreement both fail as OOD signals. DANN and CORAL both underperform doing
nothing at this sample size. And combining signals of unequal quality has hurt every time.

---

## 8. Limitations

My own combination of alignment and class-balanced self-training is not competitive with SHOT, so
the contribution lies in the empirical study and the characterised failure modes rather than in a
new algorithm.

On variance: the factorial ran three seeds per cell but the standard deviations came out exactly
zero, because a linear probe on fixed data with a deterministic selection rule has nothing to vary.
I therefore have no variance estimate at all, and obtaining one requires resampling the splits
rather than changing a seed.

The frozen linear-probe protocol used in the factorial also handicaps ResNet-50, which scores 29.8%
frozen against 66.6% fine-tuned, so part of the 30.8-point backbone spread reflects that design
choice rather than the backbones themselves.

Beyond that: adaptation is transductive, assuming the unlabelled target batch is available at
adaptation time; there is only one target dataset; there are no cross-validation folds or
significance tests; and snake weed is matched only at genus level.

Finally, DeepWeeds is a classification benchmark because its images are pre-cropped to roughly one
plant. A real weed-control robot sees a wide field of view and would need detection or segmentation
first, then classification on each crop. This pipeline is one component of a deployment system,
not the whole of one, which is also why much of the closest literature works on segmentation.

---

## 9. Next steps

In priority order: obtain a real variance estimate by resampling the splits; check whether my
alignment stacks on top of SHOT, since SHOT has no explicit class-prior correction and the
sink-class bias may persist in its output; add a second target dataset such as CWFID so the
findings are not tied to DeepWeeds; and treat open-set rejection as the main open problem, since
it is what actually blocks deployment.

On publication, this is an application and benchmarking contribution rather than a novel method,
and the baseline comparison confirms it. What remains defensible is the interaction result that
adaptation and representation quality compound; the demonstration that annotation-free transfer
from web imagery reaches 85.7% on real field data; and a fairly thorough set of characterised
failure modes. Computers and Electronics in Agriculture or Frontiers in Plant Science seem the
realistic targets, framed around annotation-free deployment with an honestly characterised open
problem.

---

## Code

The four-species comparison is in `bench_*.py`, preprocessing in `prep_*.py`, adaptation methods in
`adapt_*.py`, the conformal selection rule in `novel_*.py`, the DINOv2 and resolution work in
`accuracy_push*.py`, the factorial in `qi_*.py`, the realistic and open-set evaluations in
`realistic_*.py` and `openset.py`, and the published baselines in `baselines.py`. Results are in
the corresponding `results_*` directories. Datasets, model weights and feature caches are excluded
from version control.

---

## Appendix: adaptation on a contaminated target stream

Every adaptation result above assumes the unlabelled target stream contains only the eight weed
classes. A deployed system does not get that: over half of DeepWeeds is non-target vegetation. I
therefore repeated SHOT adaptation while varying how much of the unlabelled stream is
out-of-distribution, from 0% up to the natural 52%, evaluating throughout on the full 17,509
images. Three seeds per point, which is also the only genuine variance estimate in this project.

| Contamination | Closed-set | MSP | Energy | kNN | Mahalanobis | Confidence on negatives |
|---|---|---|---|---|---|---|
| 0% | 84.3 ± 0.2 | 0.830 ± 0.016 | 0.800 ± 0.014 | 0.810 ± 0.021 | 0.788 ± 0.023 | 0.925 |
| 10% | 82.7 ± 0.7 | 0.672 ± 0.018 | 0.610 ± 0.024 | 0.704 ± 0.024 | 0.804 ± 0.045 | 0.961 |
| 20% | 80.9 ± 1.8 | 0.640 ± 0.016 | 0.583 ± 0.016 | 0.731 ± 0.029 | 0.799 ± 0.039 | 0.969 |
| 30% | 78.0 ± 2.2 | 0.600 ± 0.017 | 0.540 ± 0.019 | 0.672 ± 0.032 | 0.781 ± 0.023 | 0.979 |
| 40% | 76.1 ± 2.3 | 0.578 ± 0.007 | 0.520 ± 0.015 | 0.665 ± 0.032 | 0.794 ± 0.013 | 0.987 |
| 52% (natural) | 71.8 ± 3.6 | 0.559 ± 0.014 | 0.514 ± 0.012 | 0.650 ± 0.017 | 0.775 ± 0.005 | 0.996 |

The mechanism is straightforward. SHOT minimises prediction entropy over every image in the
adaptation stream, so any out-of-distribution image in that stream is explicitly trained toward a
confident assignment to one of the eight weed classes. Mean confidence on negatives rises
monotonically from 0.925 to 0.996, at which point it is indistinguishable from the 0.997 the model
assigns to real weeds, and the separation that rejection depends on has been erased.

Four observations. The cliff arrives early: most of the damage is done by 10% contamination, where
max-softmax AUROC has already fallen from 0.830 to 0.672, so this is not a gradual trade-off that
can be budgeted for. Rejection is much more fragile than classification, degrading to chance while
closed-set accuracy falls only gracefully. Feature-space Mahalanobis is almost unaffected across
the whole range, 0.788 to 0.775, while logit-based scores collapse — so if adaptation must run on a
real stream, rejection should be computed from feature geometry rather than classifier outputs.
And contamination destabilises adaptation, with the closed-set standard deviation growing from
±0.2 to ±3.6.

This is a characterisation rather than a discovery. The underlying phenomenon is established in
the test-time adaptation literature: SoTTA (NeurIPS 2023) reports comparable collapses on noisy
streams, and work on the ID–OOD trade-off in open-set test-time adaptation analyses the conflict
between entropy minimisation and OOD detection directly. There is an active subfield around
open-world and open-set test-time adaptation. What the table adds is the magnitude of the effect
in the web-to-field agricultural setting, the location of the cliff, and the finding that the
choice of rejection score determines whether the failure occurs at all.
