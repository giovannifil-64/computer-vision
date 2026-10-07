# Step 3: fine-tuning CLIK on intra-oral data

Step 2 established that CLIK, applied as-is to the intra-oral scans of PrePostOrthodontic, does no better than leaving the teeth where they are. Here the diffusion model alone is retrained, not the landmark detector, to see whether that makes it usable.

It makes it useful but not decisive, and the search for the remaining error turned up something about the published method that matters more than the fine-tuning itself.

## Result

Compared over **246 test subjects**, paired: both arms share the same landmark detection, so the only thing that differs is the diffusion weights.

| metric | as-is | fine-tuned | no movement |
|---|---|---|---|
| rotation | 9.28° | **7.38°** | 9.49° |
| translation | 1.84 | **1.32** | 1.76 |
| point cloud | 2.56 mm | **1.58 mm** | 1.72 mm |

Medians alone flatter the model, so the honest figure is how often it helps a given patient rather than the cohort:

| beats leaving the teeth alone in | rotation | translation | point cloud |
|---|---|---|---|
| as-is | 44% of cases | 33% | 17% |
| fine-tuned | **70%** | **69%** | **55%** |

That is a large gain and a real one, with p on the order of 1e-32, but on point cloud accuracy it is still close to a coin toss: for 110 of 246 cases, applying the prediction is worse than not intervening.

A harder reference moves every tooth by the average motion of its position over the 703 training cases (`scripts/mean_motion_baseline.py`). It already beats not intervening (8.58°, 1.68, 1.65 mm), and the fine-tuned model beats it clearly on rotation and translation (72% of subjects, p < 1e-15) but not on point cloud (55%, p = 0.28): in position the model is not yet better than a prediction that ignores the patient.

The dataset has no patient identifiers, so the split cannot be checked for patients on both sides. `scripts/split_overlap.py` checks for reused scans instead, comparing pose-independent crown descriptors: 4 test subjects share most of their crowns with a training subject (one is a full duplicate), and dropping them changes the medians by at most 0.01.

### Collision, the paper's fourth metric

How far neighbouring teeth end up inside one another, over the first 40 test subjects, with all four dentitions measured on the same subjects. The target row is what the orthodontist-approved plan achieves, and it is the level to aim at rather than zero: treatment does bring teeth into tight contact.

| dentition | median penetration | points inside a neighbour |
|---|---|---|
| before treatment | 0.003 mm | 0.24% |
| CLIK as-is | 0.397 mm | 4.26% |
| CLIK fine-tuned | 0.325 mm | 3.24% |
| simulated target | 0.141 mm | 1.00% |

**Fine-tuning helps less here than elsewhere.** Depth falls by 18% and the affected points by 24%, leaving the depth more than twice and the share of affected points three times what the target plan produces. It fits the rest of the picture: the contact constraint made no measurable difference, and the collision term of Eq. 8 was never implemented, so nothing in the training ever asked the model to keep teeth apart.

## The diffusion is decorative

Looking for the missing accuracy turned up this. Feed the network eight very different noisy inputs at the same noise level and its prediction moves by four ten-thousandths of the variation it was given, at every level from pure noise to almost-the-answer. The network does not read its own iterate.

The consequence is testable and it holds: **one denoising step gives the same answer as all 2000.** On a model fine-tuned for 100 epochs, 7.35° against 7.40°, all three metrics within 1% and none of it significant, and the final model gives 7.36° against 7.35°. On the **released weights**, 9.27° against 9.28°: this is a property of the published model, not something the fine-tuning caused. Sampling drops from 16.2 s per subject to nothing.

It explains three things that had looked unrelated: why changing the random seed moves the predicted landmarks by a median of 38 microns, why 200 steps were already enough, and why the model is timid.

## The model is timid, and that is not a training deficiency

Over all 246 test subjects it predicts translations at 0.86 of the target size and rotations at **0.64**, while correlating with them at 0.54 and 0.56. It has found the direction and holds back, which is what a squared-error loss does when several treatment plans are plausible: the loss-optimal answer becomes their average, and an average of motions is smaller than the motions.

Two attempts to fix it both failed, informatively.

| attempt | rotation | translation | verdict |
|---|---|---|---|
| continuous noise sampling, as Palette does | +2.4% | +2.5% | worse (p < 0.01) |
| three times the epochs | −0.4% | −2.6% | calibrates translation only |

Tripling the budget moved the translation ratio from 0.83 to 0.86 and left the rotation ratio at 0.64, with both correlations changing by about 0.01. The continuous sampling being worse is a negative result about that variant only: the training loop is not in the release and had to be rebuilt from the paper, and this comparison does not by itself show whether the rebuilt loop is correct.

## Clinical constraints: one of the three transfers

Ablation over 100 subjects, each constraint against the reconstruction loss alone.

| constraint (paper's weight) | rotation | outcome |
|---|---|---|
| dental arch (0.1) | 8.04° → 15.04° | harmful, 0/100 subjects improved |
| inter-tooth contact (0.001) | 8.04° → 7.93° | no measurable effect |
| tooth orientation (0.01) | 8.04° → **7.38°** | helps, p = 4e-4 |

The arch constraint is still harmful at a hundredth of the published weight (9.77°, +21%). That does not rule out a scale problem: in the first epoch the weighted term is about two thousand times the reconstruction loss at the published weight and still about thirty times at the reduced one, so a much smaller weight is the experiment that remains. The likely mechanism, written up in `scripts/losses.py`, is that it compares fourth-order polynomial coefficients, which are badly conditioned. It is not the noise: the term is computed on the clean landmarks the network predicts, and since that prediction ignores the noisy input, the term sees almost the same finished dentition at every timestep.

Why the orientation constraint helps is not settled. The logged training loss includes the constraint itself while validation measures the reconstruction term alone, so the train/validation ratios of the two runs are not comparable and do not show a regularising effect.

## Layout

Four stages, one per file, because their needs differ: two have to sit next to the meshes, two are pure neural-network work that runs anywhere.

| stage | file | what it does | where |
|---|---|---|---|
| A | `scripts/stage_a_prepare.py` | meshes → tensors (detection, descriptors, targets) | local, needs the dataset |
| B | `scripts/stage_b_train.py` | tensors → checkpoint | any device |
| C | `scripts/stage_c_predict.py` | tensors + checkpoint → landmarks | any device |
| D | `scripts/stage_d_transform.py` | landmarks → aligned dentition | local, float64 |

The rest: `scripts/losses.py` implements the three clinical constraints, `scripts/mean_motion_baseline.py` builds the average-motion reference, `scripts/split_overlap.py` looks for scans shared between the training and the test split, `scripts/dataset.py` the split, `scripts/compare_runs.py` produces the tables above, `scripts/diagnose.py` answers what kind of mistake a model is making rather than how big, `scripts/landmark_consistency.py` checks whether the detector's points mean the same thing on different patients, and `scripts/evaluate_checkpoint.py` is the independent check that goes through the authors' own `infer_crown.py`. `main.py` orchestrates the four stages.

## Usage

```bash
python main.py --stages a --split train              # training tensors
python main.py --stages b --run my_variant --individual 0.01 --epochs 300
python main.py --stages c,d,evaluate --run my_variant
python scripts/compare_runs.py --runs "as-is=output/asis_split" "mine=output/my_variant/inference"
python scripts/diagnose.py --predicted output/my_variant/predicted_landmarks.npz \
    --scored output/my_variant/inference --tensors data/test_tensors
```

Sampling takes `--steps`. The default is the full 2000; `--steps 200` is ten times faster and `--steps 1` around two thousand times, and all three give the same answer, for the reason above.

## Figures

| File | What it shows |
|---|---|
| `clik_vs_baseline.pdf`, `tuned_vs_baseline.pdf` | Every test subject against leaving the teeth alone, one model per chart. The fine-tuned cloud crosses below the diagonal. |
| `landmark_spread.pdf` | How far each landmark moves between patients, one row per tooth position. |
| `calibration.pdf` | Predicted motion against the target motion, per tooth, over all 246 test subjects. Translation nearly calibrated at 0.86, rotation held back at 0.64. |
| `ablation.pdf` | Median error per clinical constraint, against the base loss and against no movement. |
| `training.pdf` | Training and validation loss of the reconstruction-only run and of the final 300-epoch model. Validation is flat after ten to twenty epochs; the training curve of the final model includes the orientation constraint, so its two curves are not on the same footing. |
| `<id>_comparison.png` | One dentition in four columns: before, as-is, fine-tuned, and the simulated target. Best, median and worst case. |

```bash
python scripts/reporting/make_charts.py --as-is output/asis_split \
    --tuned output/exp_long_s200 --ablation output \
    --predicted output/exp_long_s200_pred.npz --tensors data/test_tensors \
    --consistency data/train_tensors --out-dir report
python scripts/reporting/render_cases.py --converted ../step2_alignment/data/Data_prepost \
    --as-is output/_render/asis --tuned output/_render/final --out-dir report
```

Charts are written as PDF, sized for the width of a report page. The renders need the aligned meshes, which are not kept: regenerate a handful with `stage_d_transform.py --subjects <ids>` rather than all 246.

## Splits

810 subjects in the training list and 250 in the test list, **with no overlap**, decided upstream rather than carved out around the results. In practice: 703 training, 100 validation, 246 test. The eleven missing subjects are the ones where conversion or tensor preparation fails; they are 1% of the total, but not a random 1%, so the figures may be very slightly optimistic.

## What was checked, and how

| link | check | result |
|---|---|---|
| training targets | target centroid against the simulated-target mesh | 0.004 mm worst case |
| noise conditioning | training against inference | identical |
| splits | train / validation / test overlap | none |
| checkpoint loading | tensors differing from the released weights | 332 of 338 |
| the split pipeline | against `infer_crown.py` | 0.0004° |
| convergence | validation flat from epoch 10 | reached |

## A limitation of the input: the landmarks are not all landmarks

Everything here is fine-tuned on top of a detector that was left frozen, so it is worth asking whether its output means the same thing on every patient. It does not, and `scripts/landmark_consistency.py` measures it without needing annotations: superimpose one tooth's landmark set across 250 patients, removing position, orientation and size, and see how far each point still moves. The centroid is the thing to read that against, because CLIK does not predict it: it is the mean of the mesh vertices, so its spread says how much the crown itself varies.

It is a reference and not a bound. Sixty one of the 228 predicted landmarks are steadier than their own centroid, which is what a cusp tip should be, being a sharper feature than the average of a crown whose extent depends on where the segmentation cut it. The median landmark sits at 1.17 times its centroid, so typically about level with it.

The tail is what matters. Thirty one landmarks, 14% of them, scatter more than twice their centroid, thirteen of them on the premolars. On an incisor of radius 3 mm two landmarks move 3.8 and 4.2 mm from one patient to the next, further than the tooth's own radius, while on molars the worst stays under 1.5 mm.

It agrees with step 1, where CLIK's landmarks were compared against 3DTeethLand annotations on 85 unseen intra-oral scans: four of the six annotated classes land within 0.5 to 0.7 mm, while the outer point and the facial axis point sit at 2.5 and 2.9 mm and score no better than chance. Part of what the diffusion model is asked to work from is therefore noise with a fixed pattern, which is a ceiling on what fine-tuning the diffusion alone can reach.

## Two traps found along the way

**CLIK's detector is stochastic at inference.** It draws its own random point samples inside every encoder stage, and how many draws it takes depends on the batch size. Two consequences: the teeth cannot be processed as a batch, and the order of seeding versus model construction has to match `infer_crown.py` exactly: seeding after loading the detectors shifts landmarks by millimetres.

**`load_diffusion` loads with `strict=False`.** Hand it a training checkpoint and it loads no weights at all, returning a randomly initialised network, with no error. Use `stage_c_predict.load_network`, which unwraps the usual wrappers and refuses a load that did not take.
