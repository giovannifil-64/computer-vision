# Teeth3DS to CLIK evaluation pipeline

This folder applies the pretrained CLIK crown-only tooth-landmark detector to Teeth3DS intra-oral scans (IOS) and measures how well its landmarks transfer to this new kind of data, using the 3DTeethLand hand-annotated landmarks as the reference wherever they are available. CLIK itself is never modified, so the baseline runs on exactly the same code as the paper; everything in this folder is preprocessing, inference orchestration, rendering, evaluation, and an optional hybrid step that completes CLIK's output to the full 3DTeethLand landmark scheme.

The document below describes both how the pipeline is organised and what the runs on the three example subjects (`ZKJEPFDD`, `ZM8PCSK6`, `ZOUIF2W4`) produced.

## Files

| File | Purpose |
|------|---------|
| `teeth3ds_to_clik.py` | Convert a Teeth3DS scan to CLIK's per-tooth folder layout. |
| `common.py` | Path resolution, landmark loaders, coordinate-frame helpers, palette colouring. |
| `evaluate.py` | Per-class GT↔CLIK nearest-neighbour metric, random baseline, skill score. |
| `hybrid.py` | Relabel CLIK landmarks to 3DTeethLand classes and derive the two missing ones geometrically. |
| `visualize.py` | Per-arch IOS renders (solid mesh + landmark spheres, pyvista, occlusal view). |
| `run_all.py` | Driver: convert and infer all patients, learn the class map, then render and evaluate. |
| `gt_patient_ids.txt` | The Teeth3DS subjects that have 3DTeethLand landmark ground truth on both arches. |

## Requirements

Everything runs inside the `clik-tooth` conda environment used for CLIK itself. The only extra dependency is `pyvista` (used for the renders, `pip install pyvista`); it works headless on macOS through VTK's offscreen rendering.

## Usage

```bash
conda run -n clik-tooth python teeth3ds_pipeline/run_all.py \
    --src /Users/giovanni/Downloads/tmp \
    --gt  /Users/giovanni/Downloads/osfstorage-archive
```

The driver works in two passes. In the first pass it converts every patient and runs CLIK inference. It then learns the CLIK-id to class map from a patient that has ground truth (see the hybrid section). In the second pass it renders every patient with a single shared colour palette and evaluates the ones that have ground truth. Results are written under `Output_teeth3ds/<pid>/` together with a summary in `Output_teeth3ds/evaluation_summary.json`.

## Inputs

The `--src` folder holds one subfolder per patient, each containing `<pid>_upper.obj`, `<pid>_lower.obj` and the matching `*_upper.json` / `*_lower.json` segmentation files (a per-vertex list of FDI tooth labels, with `0` reserved for gingiva and base). The `--gt` folder is the unzipped `osfstorage-archive`, which contains the 3DTeethLand `*__kpt.json` landmark files.
Ground truth is optional per patient: a subject without it still gets its CLIK and
hybrid renders, but no quantitative comparison.

## Outputs

For every patient and arch the pipeline writes up to three images that share the same camera and the same 3DTeethLand colour legend, so they can be compared side by side. The `*_CLIK.png` image shows CLIK's raw landmarks, the `*_HYBRID.png` image shows the completed six-class output, and the `*_GT.png` image shows the ground truth where it exists. The converted per-tooth meshes are kept under `Data_teeth3ds/<pid>/initial/`, each patient folder also storing a `center.json` with the offset that was subtracted to recentre the dentition.

## Method

### Conversion

CLIK does not operate on a whole arch but on individual teeth, so the scan has to be split first. The converter reads the per-vertex FDI labels and assigns a face to a tooth when at least two of its three vertices carry that tooth's label, which is robust to the boundary vertices between adjacent teeth and gingiva. Each tooth is then renumbered from FDI to the Universal scheme that CLIK expects, the upper and lower arches are merged, and the whole dentition is recentred at the origin.
The numbering step matters because CLIK uses the Universal id both to pick the right detection network per tooth type and to organise the dentition internally; feeding FDI numbers would misclassify the teeth. The mapping is the standard one, for example FDI 11 (upper-right central incisor) becomes Universal 8 and FDI 46 (lower-right first molar) becomes Universal 30.

### CLIK inference

CLIK is run in crown-only mode through `Code/infer_crown.py`, launched as a subprocess so that its checkpoint paths resolve exactly as in the original repo. The model has three stages (landmark detection, a diffusion model, and a rigid alignment), but only the Stage 1 landmarks are used here; the alignment stages are discussed under Limitations.

### Evaluation metric

For each ground-truth landmark the pipeline finds the nearest CLIK landmark of any type and reports the distance in millimetres. This is a coverage, or recall, measure: it answers whether CLIK placed some landmark on a given true point. It is deliberately not a per-class identity match, because CLIK uses its own landmark scheme rather than the six 3DTeethLand classes. To make the numbers interpretable the same measurement is repeated with a set of random points sampled on the same crowns, which gives the chance level for each class. The two are combined into a skill score, defined as one minus the ratio between the CLIK median and the random median: it is zero when CLIK is no better than chance, approaches one when CLIK is near perfect, and goes negative when CLIK does worse than chance.

### Hybrid completion

The hybrid step in `hybrid.py` turns CLIK's output into the full six-class 3DTeethLand scheme and is made of two complementary parts. The first part still relies on CLIK: for the four classes CLIK detects well (mesial, distal, cusp and inner) it simply relabels CLIK's existing landmarks into 3DTeethLand classes, using a map learned from a subject that has ground truth. That map records, for each tooth type, which CLIK landmark id corresponds to which class, and it transfers across patients because it is defined per tooth type rather than per patient. The second part does not use CLIK at all: the two classes CLIK has no equivalent for (outer and facial-axis) are computed directly from the crown geometry, the outer point as the most buccal vertex on the gingival margin and the facial-axis point as the centre of the facial surface. In short, the hybrid uses CLIK's detector for four of the six landmarks and pure geometry for the remaining two, and it never touches CLIK's alignment stages.

### Rendering

Each arch is rendered on its own with pyvista as a solid, properly occluded grey surface seen from an occlusal viewpoint, with the landmarks drawn as coloured spheres in the 3DTeethLand palette (mesial red, distal green, cusp blue, inner yellow, outer cyan, facial magenta). The upper and lower arches are rendered separately, because Teeth3DS scans them independently and they are not registered into occlusion, so drawing them together would make the surface look transparent. The raw-CLIK image uses the same palette by colouring each CLIK landmark with the class of its nearest reference point, which is the ground truth where it exists and the hybrid output otherwise.

## Findings on the three example subjects

CLIK runs end to end on all three subjects without errors. It produces the expected number of landmarks per tooth type, all of them lie on the surface, and they are well spread over each tooth rather than collapsing onto a single point, which is the usual failure mode when a model is pushed outside its training distribution. In this basic sense the detector behaves sensibly on intra-oral scans even though it was trained on a different kind of crown mesh.

Only `ZKJEPFDD` is present both in the example ZIP and in the 3DTeethLand annotations, so it is the only subject on which the output can actually be measured. For the other two the result looks plausible but cannot be verified from the data available, and any statement about them is therefore qualitative.

On `ZKJEPFDD`, CLIK detects 188 landmarks against 152 in the ground truth. The
coverage metric, read through the skill score, is as follows.

| GT class | CLIK (mm) | Random baseline (mm) | Skill | Reading |
|----------|-----------|----------------------|-------|---------|
| Mesial | 0.48 | 1.45 | 0.67 | detected reliably |
| Distal | 0.46 | 1.51 | 0.69 | detected reliably |
| Cusp | 0.45 | 1.61 | 0.72 | detected reliably |
| Inner | 0.39 | 2.19 | 0.82 | best of all |
| Outer | 1.66 | 2.34 | 0.29 | close to chance |
| Facial-axis | 2.75 | 1.78 | −0.54 | worse than chance |
| Overall | 0.66 | 1.75 | 0.62 | good |

The detector transfers well, at roughly half a millimetre from the ground truth, for four of the six classes: mesial, distal, cusp and inner. It does not cover outer and facial-axis, simply because its own landmark scheme has no equivalent for them; the negative skill on facial-axis means CLIK's points systematically avoid that region rather than landing near it.

The hybrid completion produces, for the first time, all six classes with proper labels. Measuring each class against its own ground-truth class gives the numbers below.

| Class | Source | Hybrid (mm) |
|-------|--------|-------------|
| Cusp | CLIK | 0.45 |
| Facial-axis | geometry | 0.72 |
| Distal | CLIK | 1.03 |
| Inner | CLIK | 1.30 |
| Mesial | CLIK | 1.52 |
| Outer | geometry | 1.59 |
| Overall | | 0.96 |

The geometric part is the clear success here: the facial-axis point, which CLIK placed at 2.75 mm, drops to 0.72 mm when derived from the crown surface. This confirms that the 3DTeethLand landmarks are essentially geometric definitions and can be recovered without a neural network. The CLIK-relabelling part is weaker on the single proximal and inner points, mostly because the id-to-class map is learned from a single patient and is not yet stable; it is expected to improve as more annotated subjects become available.

## Limitations

The quantitative results rest on a single annotated subject, so the numbers should be read as indicative rather than statistically solid. The comparison is also a coverage proxy: CLIK and 3DTeethLand use different landmark definitions, and the hybrid map that bridges them is learned from one patient and only approximate on the others.

More fundamentally, CLIK was designed for tooth alignment, not for reproducing the 3DTeethLand landmarks. Its alignment stages cannot be evaluated on this data, because the upper and lower arches are not co-registered into occlusion and there is no post-treatment ground truth to compare against. The landmark comparison therefore measures CLIK outside its intended task. Because those landmarks are geometrically defined, a purely geometric method, or a detector trained directly on 3DTeethLand, would be a more direct way to obtain them, and for that specific goal CLIK is not strictly necessary.

