# CLIK as-is on the PrePostOrthodontic dataset — results

Pretrained CLIK (crown-only), **no fine-tuning of any component**, run on the full
official test split (250 subjects) of Wang et al. 2024, comparing the predicted
post-orthodontic dentition against the real post-treatment scan.

## Numbers

Per-tooth, mean ± std over **246 subjects / 6223 teeth**. "No movement" is the
error obtained by leaving every tooth exactly where it is — the reference any
useful prediction must beat.

| Metric | CLIK as-is | No movement | Paper (landmark detector retrained) |
|---|---|---|---|
| Rotation (°) | 11.00 ± 8.56 | 11.05 ± 10.27 | 6.69 ± 2.56 |
| Translation | 2.18 ± 1.56 | 2.01 ± 1.79 | 1.20 ± 0.44 |
| Point cloud (mm) | 3.10 ± 2.18 | 2.15 ± 2.06 | 1.30 ± 0.68 |

Medians and win rate: rotation 8.99 vs 8.59 (CLIK better on 48% of teeth),
translation 1.81 vs 1.58 (43%), point cloud 2.47 vs 1.53 (24%).

**Reading.** On rotation CLIK as-is essentially ties with doing nothing; on
position it is clearly worse. The gap to the published numbers is consistent with
the one methodological difference: the authors retrained the landmark detector on
this dataset, we did not. The error is largest exactly where the real treatment
moved teeth least — the model applies a textbook correction where the clinical
plan asked for very little.

## Where the error comes from

Because the ground-truth transforms are exact, the pipeline can be measured stage
by stage rather than only end to end.

| Stage | Measurement | Result |
|---|---|---|
| 1 — landmark detection | Landmarks found on `final` vs landmarks found on `ori` carried through the true motion (40 subjects, 8912 landmark pairs) | **0.10 mm median**, 86% within 1 mm — *below* the 0.31 mm noise floor of the random sampling |
| 2 — diffusion | Predicted target landmarks vs the ideal ones, i.e. the detected landmarks under the true motion (40 subjects) | **2.92 mm median** (mean 3.36 ± 2.07) |
| 3 — rigid fit | Part of the predicted landmark cloud that is not a rigid motion, so cannot be represented | **0.71 mm median** |

**The error is concentrated in the diffusion stage.** Stage 1 is repeatable: presented with the same tooth in a different pose it selects
the same points, and the small residual is fully accounted for by the random start
of the farthest-point sampling (the pose figure is *lower* than the noise floor).
Stage 3 loses little: the predicted landmark cloud is very nearly a rigid motion of
the input — just the wrong one. What Stage 2 predicts sits about 3 mm away from
where the teeth actually had to go.

This is corroborated by an independent coincidence: the diffusion error in landmark
space (2.92 mm) matches the end-to-end point-cloud error measured on the meshes
(3.10 mm), and the two are computed by different routes.

Repeatability alone would only prove *consistency*, so the landmarks were also
checked for *anatomical correctness* against the annotations shipped with the
dataset (70 test subjects, 6164 landmarks). For each annotated point, the distance
to the nearest CLIK landmark, against a random-surface-point baseline:

| Class | CLIK | Random baseline | Skill |
|---|---|---|---|
| Pt6 | 0.45 mm | 1.66 mm | 0.73 |
| Pt0 | 0.57 mm | 1.66 mm | 0.65 |
| Pt1 | 0.71 mm | 1.66 mm | 0.57 |
| Pt2 | 0.84 mm | 1.66 mm | 0.49 |
| Pt3 | 0.94 mm | 1.66 mm | 0.43 |
| Pt7 | 1.17 mm | 1.66 mm | 0.30 |
| **Overall** | **0.75 mm** | **1.66 mm** | **0.55** |

Sub-millimetre and clearly above chance on every class, so CLIK's landmarks are
not merely repeatable, they sit on anatomically meaningful points of *this*
dataset. The same check on the post-treatment crowns gives 0.68 mm against a
1.60 mm baseline (skill 0.58, 5953 landmarks), so the detector behaves identically
once the teeth have been repositioned. **The diffusion therefore receives sound
input and errs on its own** — which points the fine-tuning effort at the diffusion
rather than the detector.

*Remaining uncertainty:* this is a coverage measure (the two landmark schemes
differ), so it shows CLIK places *some* landmark on each true anatomical point. It
does not fully prove that each CLIK landmark id sits where the diffusion was
trained to expect that id; a systematic shift in that assignment would still
originate in Stage 1 and propagate.

Raw numbers in `data/`: `landmark_repeatability.json`, `diffusion_error.json`, `landmark_accuracy*.json`, `metrics_detail.json`, `seed_variability.json`.

## Collision, significance, tooth type

The paper's fourth metric, penetration between neighbouring teeth, was measured on
40 subjects for the initial, predicted and ground-truth dentitions:

| Dentition | Penetration depth | Penetrating points |
|---|---|---|
| Pre-treatment | 0.003 mm | 0.24% |
| **CLIK prediction** | **0.383 mm** | **4.30%** |
| Ground truth | 0.154 mm | 0.98% |

Some penetration is expected — treatment brings teeth into tight contact, which is
why the ground truth is not zero. The prediction, however, is about two and a half times
deeper and affects four times more points, quantifying what the figures show.

The gap to the "no movement" reference is statistically significant on all three
headline metrics (paired t-test, confirmed by Wilcoxon): p = 6e-3 for rotation,
6e-10 for translation, 3.6e-33 for point cloud.

Split by tooth type (median, with the baseline in brackets): incisor 10.81° (10.97)
and 2.79 mm (2.19); canine 10.06° (11.16) and 2.71 mm (2.00); premolar 9.19° (8.91)
and 2.15 mm (1.33); molar 6.91° (5.41) and 2.39 mm (1.02). Molars look best in
absolute terms but are the worst relative to the baseline, because they barely move;
CLIK holds up best on the anterior teeth, the ones treatment actually repositions.

## Is the error just noise?

The diffusion is generative, so the evaluation was repeated with five different
seeds on 40 subjects. The per-subject spread is 0.075 mm (point cloud) and 0.69°
(rotation), while the distance from the no-movement reference is 1.147 mm — about
**15 times the spread**. The medians of the five runs are 2.92–3.05 mm. The error is
therefore systematic, and evaluating with a single seed was sound.

## Teeth left out of the metrics

152 third molars are never predicted, because CLIK's scheme covers 28 teeth. 335
teeth were extracted during treatment, so they are absent from the ground truth;
CLIK does not know this and still moves them, by a median of 2.68 mm. A further 93
teeth were re-tessellated between the two stages, leaving no point correspondence.

## Figures

| File | What it shows |
|---|---|
| `clik_vs_baseline.png` | Per-subject error, CLIK vs no movement. Points above the diagonal = worse than doing nothing (56% for rotation, 83% for position). |
| `0791_comparison.png` | Best case: initial / prediction / ground truth, upper and lower arch. |
| `0327_comparison.png` | Median case — the prediction opens gaps and breaks arch continuity. |
| `0944_comparison.png` | Worst case. |
| `clik_as_is_results.xlsx` | Every result in one workbook (11 sheets), regenerated at each run. |

## What was done to the data, and why

The dataset itself was never modified; everything runs on a converted copy. The
conversion cuts the per-tooth meshes out of the arch scan using the segmentation
supplied with the dataset, renumbers teeth from FDI to the Universal scheme CLIK
uses, and applies one rigid transform (recentring + rotation) to place the
dentition in CLIK's canonical jaw frame.

That last step is necessary because **CLIK is not orientation-invariant**: its
diffusion model learned dental-arch shape in a fixed frame (anterior −x, patient
right +y, upper arch +z). This dataset is not consistently oriented — 186 of 247
subjects have the upper arch *below* the lower one relative to the rest. Using a
single fixed rotation derived from one subject made results far worse (13.25° vs a
9.41° baseline) simply because most dentitions were fed in upside down. The frame
is therefore estimated **per subject** from anatomy: incisors vs molars, upper vs
lower arch, right vs left teeth, orthonormalised and mapped onto CLIK's axes, with
a determinant check so the anatomy is never mirrored.

The same transform is applied to the pre- and the post-treatment stage, so the
movement being measured is untouched: per-tooth rotations and displacements are
identical before and after conversion (verified). The model and its weights were
not modified.

## Validity checks

- CLIK's `transformation.json` reproduces its own exported meshes to 1e-6 mm, so
  the predicted transforms are read correctly.
- Ground truth is exact: `ori` and `final` contain the same crown meshes rigidly
  repositioned (Kabsch residual ~0), so per-tooth GT transforms are recovered
  analytically rather than estimated.
- Excluded from the metrics: 310 teeth extracted during treatment (absent from the
  post-treatment scan) and 93 re-tessellated between stages (no point correspondence).

## Reproducing

```bash
conda run -n clik-tooth python prepost_pipeline/prepost_to_clik.py \
    -i ../PrePostOrthodontic/Orthodontic_dental_dataset -o ./Data_prepost
for s in $(ls Data_prepost); do
  conda run -n clik-tooth python Code/infer_crown.py -i "./Data_prepost/$s" -o ./Output_prepost
done
conda run -n clik-tooth python prepost_pipeline/evaluate_alignment.py \
    --converted ./Data_prepost --output ./Output_prepost
conda run -n clik-tooth python prepost_pipeline/render_cases.py \
    --converted ./Data_prepost --output ./Output_prepost --out-dir ./report_prepost
```
