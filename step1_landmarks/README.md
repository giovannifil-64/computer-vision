# Step 1: are CLIK's landmarks any good on intra-oral scans?

CLIK was trained on crowns segmented from CBCT. Teeth3DS is a different thing entirely, an optical surface scan of the mouth, so the first question is whether CLIK's landmark detector still puts its points where it should. CLIK itself is never modified here: everything in this folder is preprocessing, orchestration, evaluation and rendering.

The reference is 3DTeethLand, which annotates six landmark classes on Teeth3DS by hand.

## Result

Over 85 scans that have annotations, distance from each annotated point to the nearest landmark CLIK produced. The skill score compares that against points sampled at random on the same crowns: zero means no better than chance, one means perfect, negative means the model actively avoids the right region.

| annotated class | CLIK | skill |
|---|---|---|
| inner point | 0.72 mm | 0.68 |
| mesial contact | 0.48 mm | 0.67 |
| cusp | 0.55 mm | 0.66 |
| distal contact | 0.52 mm | 0.66 |
| outer point | 2.53 mm | -0.05 |
| facial axis point | 2.94 mm | -0.54 |
| overall | 0.70 mm | 0.61 |

Four of the six transfer well. The other two do not, and the reason is not that the detector is bad at them: CLIK's own landmark scheme has no equivalent of the outer point or the facial axis point, so it never had a reason to place anything there. The negative skill on the facial axis says its points systematically land elsewhere.

The detector also behaves sensibly in the basic sense: it produces the expected number of landmarks per tooth type, they lie on the surface, and they spread over the crown rather than collapsing onto one spot, which is the usual failure when a model is pushed outside its training distribution.

## How it works

1. **Conversion.** CLIK works on individual teeth, not on a whole arch, so the scan is split first. The converter reads the per-vertex FDI labels and gives a face to a tooth when at least two of its three vertices carry that label, which is robust at the boundaries between neighbouring teeth and gingiva. Each tooth is then renumbered from FDI to the Universal scheme, the arches are merged and the dentition is recentred. The renumbering matters because CLIK uses the Universal id both to choose the right detector for the tooth type and to organise the dentition internally, so FDI numbers would misclassify every tooth.

2. **Inference.** CLIK runs in crown-only mode through its own `infer_crown.py`, launched as a subprocess so its checkpoint paths resolve exactly as in the original repository. Only the Stage 1 landmarks are used; the alignment stages are the subject of step 2.

3. **The metric.** For each annotated landmark, the distance to the nearest CLIK landmark of any type. This is a coverage measure and deliberately not a per-class match, because the two schemes are different and CLIK cannot be blamed for not using someone else's definitions. Repeating the same measurement with random surface points gives the chance level, and the skill score is one minus the ratio of the two medians.

4. **Hybrid completion.** `hybrid.py` turns CLIK's output into the full six-class scheme, in two parts. For the four classes CLIK covers it relabels CLIK's own landmarks, using a map from CLIK id to class learned on a subject that has annotations; the map transfers because it is defined per tooth type, not per patient. For the two classes CLIK has no equivalent of, it ignores CLIK and computes them from the crown geometry: the outer point as the most buccal vertex on the gingival margin, the facial axis point as the centre of the facial surface.

On the one subject where the hybrid was scored class by class, the geometric half is the clear success: the facial axis point goes from 2.75 mm with CLIK to 0.72 mm when derived from the surface. That is worth stating plainly, because it means these landmarks are geometric definitions and do not need a network at all.

## Usage

```bash
python main.py                  # everything
python main.py --stages gather  # only collect the scans that have annotations
python main.py --limit 5        # a quick run
```

The driver works in two passes. It converts every patient and runs inference, learns the id-to-class map from a subject that has annotations, then renders everyone and scores the ones it can. Results land in `output/`, with a summary in `evaluation_summary.json`.

## Layout

| file | what it does |
|---|---|
| `scripts/setup_input.py` | collect the Teeth3DS scans that have annotations |
| `scripts/teeth3ds_to_clik.py` | one arch into CLIK's per-tooth folders, FDI to Universal |
| `scripts/common.py` | paths, landmark loading, coordinate frames, the shared palette |
| `scripts/evaluate.py` | the coverage metric, the random baseline and the skill score |
| `scripts/hybrid.py` | relabel four classes, derive the other two from geometry |
| `scripts/visualize.py` | per-arch renders, occlusal view, landmarks as coloured spheres |
| `scripts/run_all.py` | the two-pass driver |

Each patient gets up to three images sharing one camera and one colour legend, so they can be read side by side: `_CLIK` for the raw output, `_HYBRID` for the completed six classes, `_GT` for the annotations where they exist.

## Limitations

The comparison is a coverage proxy. CLIK and 3DTeethLand define landmarks differently, and the map bridging them is learned from a single annotated subject, so the per-class hybrid figures are indicative rather than solid. The aggregate CLIK numbers over 85 scans are the reliable part.

More fundamentally, this measures CLIK outside its intended task. It was built to align teeth, not to reproduce someone else's landmark scheme, and its alignment stages cannot be judged here at all because Teeth3DS has no post-treatment scan to compare against and the two arches are not registered into occlusion. That is what step 2 is for.
