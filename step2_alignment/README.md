# Step 2: does CLIK align teeth as released?

Step 1 showed the landmark detector transfers reasonably to intra-oral scans. This step asks the question that actually matters: given a pre-treatment dentition, does CLIK move the teeth where an orthodontist moved them?

PrePostOrthodontic gives pairs of scans of the same patient, before and after treatment. The two are the same meshes rigidly moved, so the true motion of every tooth is exactly recoverable, which makes both the ground truth and the error decomposition exact rather than approximate.

Nothing is retrained here. This is CLIK exactly as the authors published it.

## Result

Per tooth, mean and standard deviation over 246 subjects and 6223 teeth. "No movement" is the error you get by leaving every tooth exactly where it is, and it is the reference any useful prediction has to beat.

| metric | CLIK as released | no movement | the paper, with a retrained detector |
|---|---|---|---|
| rotation | 11.00° ± 8.56 | 11.05° ± 10.27 | 6.69° ± 2.56 |
| translation | 2.18 ± 1.56 | 2.01 ± 1.79 | 1.20 ± 0.44 |
| point cloud | 3.10 mm ± 2.18 | 2.15 mm ± 2.06 | 1.30 ± 0.68 |

By median and win rate: rotation 8.99 against 8.59, with CLIK better on 48% of teeth; translation 1.81 against 1.58, 43%; point cloud 2.47 against 1.53, 24%.

On rotation it ties with doing nothing, on position it is clearly worse. The error is largest exactly where the real treatment moved teeth least, which says the model applies a textbook correction where the clinical plan asked for very little.

## Where the error comes from

Because the true transforms are exact, the three stages can be measured separately instead of only end to end.

| stage | what is compared | result |
|---|---|---|
| 1, landmark detection | landmarks found on the post scan against landmarks found on the pre scan carried through the true motion | 0.10 mm median, below the 0.31 mm noise floor of the random sampling |
| 2, diffusion | predicted target landmarks against the ideal ones, the detected landmarks under the true motion | 2.92 mm median |
| 3, rigid fit | the part of the predicted landmark cloud that is not a rigid motion, so cannot be represented | 0.71 mm median |

The error is concentrated in the diffusion. Stage 1 is repeatable: shown the same tooth in a different pose it picks the same points, and the residual is fully explained by the random start of the farthest-point sampling. Stage 3 loses little, because the predicted cloud is very nearly a rigid motion of the input, just the wrong one. What stage 2 predicts sits about 3 mm from where the teeth had to go.

An independent check agrees: the diffusion error in landmark space, 2.92 mm, matches the end-to-end point-cloud error measured on the meshes, 3.10 mm, and the two are computed by different routes. This is what pointed the fine-tuning in step 3 at the diffusion model alone.

## Is it just noise?

The diffusion is generative, so the whole evaluation was repeated with five seeds on 40 subjects. The per-subject spread is 0.075 mm on point cloud and 0.69° on rotation, while the distance from the no-movement reference is 1.147 mm, about fifteen times the spread. The error is systematic and one seed is enough.

The gap to the reference is significant on all three metrics, by paired t-test and confirmed by Wilcoxon: p = 6e-3 for rotation, 6e-10 for translation, 3.6e-33 for point cloud.

## Collision and tooth type

The paper's fourth metric, how far neighbouring teeth end up inside one another, on 40 subjects:

| dentition | penetration depth | penetrating points |
|---|---|---|
| pre-treatment | 0.003 mm | 0.24% |
| CLIK prediction | 0.383 mm | 4.30% |
| ground truth | 0.154 mm | 0.98% |

Some penetration is expected, since treatment brings teeth into tight contact, which is why the ground truth is not zero. The prediction is two and a half times deeper and affects four times more points.

By tooth type, median with the baseline in brackets: incisor 10.81° (10.97) and 2.79 mm (2.19); canine 10.06° (11.16) and 2.71 mm (2.00); premolar 9.19° (8.91) and 2.15 mm (1.33); molar 6.91° (5.41) and 2.39 mm (1.02). Molars look best in absolute terms but are the worst against the baseline, because they barely move. CLIK holds up best on the anterior teeth, the ones treatment actually repositions.

## What was done to the data, and why

The dataset is never modified, everything runs on a converted copy. The conversion cuts per-tooth meshes out of the arch using the supplied segmentation, renumbers from FDI to the Universal scheme CLIK expects, and applies one rigid transform to place the dentition in CLIK's canonical jaw frame.

That last step is not cosmetic, because **CLIK is not orientation invariant**: its diffusion model learned arch shape in a fixed frame, anterior along -x, patient right along +y, upper arch along +z. This dataset is not consistently oriented, and 186 of 247 subjects have the upper arch below the lower one. Using a single fixed rotation taken from one subject made the results much worse, 13.25° against a 9.41° baseline, simply because most dentitions were fed in upside down. The frame is therefore estimated per subject from anatomy, incisors against molars, upper against lower, right against left, orthonormalised onto CLIK's axes with a determinant check so the anatomy is never mirrored.

The same transform is applied to both stages, so the movement being measured is untouched: per-tooth rotations and displacements are identical before and after conversion.

## Teeth left out of the metrics

152 third molars are never predicted, because CLIK's scheme covers 28 teeth. 335 teeth were extracted during treatment and are absent from the post-treatment scan; CLIK does not know that and moves them anyway, by a median of 2.68 mm. Another 93 teeth were re-tessellated between the two scans, leaving no point correspondence to measure.

## Usage

```bash
python main.py                                          # everything
python main.py --stages convert,infer,evaluate,figures  # the usual run
python main.py --stages infer --limit 10                # a quick check
python main.py --with-seeds --seeds 2 3 4 5             # the seed study
```

Inference is the slow part. Everything else runs in minutes.

## Layout

| file | what it does |
|---|---|
| `scripts/prepost_to_clik.py` | conversion, including the per-subject frame estimate |
| `scripts/evaluate_alignment.py` | the three headline metrics against the no-movement reference |
| `scripts/evaluate_detail.py` | collision, significance, breakdown by tooth type, excluded teeth |
| `scripts/diffusion_error.py` | the stage-by-stage decomposition |
| `scripts/landmark_repeatability.py` | is stage 1 repeatable under a change of pose |
| `scripts/landmark_accuracy.py` | stage 1 against the annotations, where they exist |
| `scripts/seed_variability.py` | the same evaluation over several seeds |
| `scripts/reporting/render_cases.py` | best, median and worst case, rendered side by side |
| `scripts/reporting/make_chart.py` | per-subject error against the no-movement reference |

## Figures

| file | what it shows |
|---|---|
| `report/figures/clik_vs_baseline.png` | every subject as a point against the no-movement reference. Above the diagonal means worse than doing nothing: 56% for rotation, 83% for position |
| `report/figures/0791_comparison.png` | best case, initial and prediction and ground truth, both arches |
| `report/figures/0327_comparison.png` | median case, where the prediction opens gaps and breaks the arch |
| `report/figures/0944_comparison.png` | worst case |
