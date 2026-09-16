# Make and body-type transfer baselines

## Status and ownership

The first local make/body version is trained and verified. It extends Yuchen's shared dataset, `AttributePredictor`, pipeline, hash-based pilot split and colour transfer recipe. Nicholas's committed manifests remain the source of labels and official split membership.

| Task | Fit / validation images | Classes | Validation accuracy | Macro F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| Make | 4,416 / 1,120 | 75 | 54.64% | 0.5890 | 1.79% |
| Body type | 3,112 / 782 | 12 | 67.01% | 0.6961 | 12.79% |

Both runs selected epoch 8 and passed fresh-process CPU reload and independent metric reconstruction. These are capped validation pilots from one seed. Official test images were checked for data integrity but were not evaluated by the models.

Aggregate results, figures and checkpoint verification summaries are included in [the v0.1 evidence package](evidence/web_transfer_2026-09-16/README.md). Raw images, saved image demos and trained weights remain local.

The code and this guide were developed with AI assistance. They remain working material for Yuxiang's review; no personal review or submission declaration is implied.

## Open the local demo

The workspace output directory is `09_Model_Outputs/web_v01`. Open `demo/index.html` in a browser for five saved examples, including correct and incorrect predictions. Each example includes the annotated image, structured JSON and checkpoint provenance. The pages work offline. The gallery displays saved inference; use the command below to analyse a new image.

From this repository root with the environment activated:

```powershell
$workspace = (Resolve-Path '../..').Path
$outputRoot = Join-Path $workspace '09_Model_Outputs/web_v01'
$localData = Join-Path $workspace '08_Local_Data/CompCars'
python scripts/demo_vehicle_models.py `
  --image data/samples/blocked_car.jpg `
  --make "$outputRoot/make_run/best.pt" `
  --body-type "$outputRoot/body_run/best.pt" `
  --detector "$localData/yolo26n.pt" `
  --output "$outputRoot/new_demo"
```

Choose a new output directory for each run. Open its `demo.html` afterward. The same command accepts another local image path. `report/MODEL_CARD.md` contains results, error analysis and limitations; each task's run directory contains the frozen config, source snapshots, predictions and verification evidence.

The live detector path is local YOLO → vehicle crops → make/body predictors → the existing `VehicleProfile`. Colour, fine-grained vehicle model, damage and accessories remain `not_assessed` when no predictor is supplied. A colour checkpoint can be added with `--colour` once Yuchen's trusted trained file is available.

## Environment

The checked local environment is Windows, Python 3.12.14, PyTorch 2.5.1+cu121 and TorchVision 0.20.1+cu121. The complete package record is `requirements-model-windows-cu121.txt`. This is a separate environment recipe; the team's existing requirements file is preserved.

From the repository root, with the isolated environment activated:

```powershell
python -m pip install -r requirements-model-windows-cu121.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-deps -e .
python -m pip check
python -m unittest discover -s tests -v
```

The RTX 4070 Ti (12 GB) completed both sequential training runs: make took 218.7 seconds with peak reserved CUDA memory of 480 MiB; body type took 147.7 seconds with 490 MiB. These training timings exclude acquisition, audits and fresh-process verification. The earlier full-network FP32 capacity check used 1,310 MiB reserved memory at batch 32. No external compute was used.

## Data preparation

Acquire CompCars from the [official dataset page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) and [instructions](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt). Keep archives, images, checkpoints and per-image evidence local under the permitted research-use conditions.

The existing audit scripts expect this layout under the supplied archive root:

```text
data.zip, data.z01 ... data.z22
sv_data.zip, sv_data.z01 ... sv_data.z03
data/data/image/...
data/data/label/...
data/data/misc/...
sv_data/sv_data/image/...
```

For complete official archives, run the full extraction CRC audit, then the manifest/image audit. These audit both web and surveillance archives by default. CRC comparison detects disagreement with the archive; it is not a publisher signature. The manifest audit decodes images and checks official labels and split membership. It does not run model inference on the official test set.

```powershell
python scripts/compcars_archive_integrity.py --archives-root $dataRoot --output $integrityDir
python scripts/audit_local_compcars.py --archives-root $dataRoot --integrity $integrityDir --output $auditDir
```

Here `$dataRoot`, `$integrityDir` and `$auditDir` are local directories chosen for the run. Audit output directories must be new. Preserve the resulting JSON and CSV evidence with the experiment.

### Data used for this version

The official folder download was incomplete because of download limits. The web archive was recovered from [JorgeLlorente/CompCars-Repository](https://huggingface.co/datasets/JorgeLlorente/CompCars-Repository/tree/0a3d1f7c802614171e8517e511f2b27e5fb8ad0d), pinned to revision `0a3d1f7c802614171e8517e511f2b27e5fb8ad0d`. The 16,533,899,407-byte ZIP has SHA-256 `142298e260f1246572c7f564d0b069749163f066cf35c1298c56072c0ec03d76`, matching its LFS digest. This third-party provenance remains distinct from an official-host download.

All 301,095 extracted files passed archive CRC checks. A bounded sample of 100 images matched independently decoded official bytes; this is not full publisher authentication. All 30,925 manifest rows passed image decoding, official label and split checks. The audit identified two exact-duplicate groups, including one crossing official train/test, and 721 unavailable body labels (369 train, 352 test). The split preparation handles these exclusions explicitly. Surveillance data remain incomplete and are not used in these make/body runs.

A later coverage check found that the official classification lists contain 30,955 images, while the source CSV contains 30,925. Thirteen official-train and seventeen official-test images with an `unknown` year are omitted by the notebook's numeric-year path pattern. All thirty files exist locally. The original audit verifies included-row consistency; it does not establish complete official-list coverage. These v0.1 runs preserve the recorded source CSV and splits. The coverage correction belongs to a versioned follow-up before larger training or final benchmark claims; see [the aggregate coverage record](evidence/web_transfer_2026-09-16/source_coverage.json).

The local `$dataRoot` for this run is `08_Local_Data/CompCars/mirror` under the workspace, with images at `data/data/image`. To reproduce its web-only audit, use new evidence directories:

```powershell
python scripts/audit_web_mirror.py --archive "$dataRoot/Compcars_Data.zip" --root $dataRoot --official-first-volume $officialFirstVolume --output $integrityDir
python scripts/audit_local_compcars.py --archives-root $dataRoot --integrity $integrityDir --output $auditDir --kinds web
```

`$officialFirstVolume` is the acquired official `data.z01`. The completed evidence is retained in the workspace's `99_Working/implementation_2026-09-16/mirror_integrity` and `web_audit` directories. Nicholas's repository `data` directory supplies manifests and sample assets; it does not contain the full raw training-image collection.

Official body type `0` means unavailable. A blank manifest target is correct for those rows; labelling them convertible is incorrect. The body task excludes these images before splitting. Their make targets remain usable. Official 1-based xyxy boxes become zero-based half-open boxes only in the dataset loader view; persisted manifests and split files retain the original coordinates.

## Frozen experiment

`prepare` checks the completed audit and freezes a class-stratified split from official train only. The existing `pilot_split` excludes exact-byte copies of official test images and conflicting-label content, then selects one representative per remaining SHA-256 group. Twenty percent of each selected class supplies validation. The default cap is 100 images per class before splitting; specify the intended cap explicitly.

The first bounded recipes use make cap 100 and body type cap 500. Small classes retain all available representatives. These capped validation results do not estimate the natural class distribution or physical-vehicle-disjoint performance. Near duplicates and vehicle identity remain unresolved limitations. Official test metrics are reserved for a later frozen final evaluation.

```powershell
python scripts/train_web_transfer.py prepare --task make --cap 100 --root $dataRoot --audit $auditDir --prepared $makeSplit
python scripts/train_web_transfer.py prepare --task body_type --cap 500 --root $dataRoot --audit $auditDir --prepared $bodySplit
python scripts/train_web_transfer.py train --root $dataRoot --audit $auditDir --prepared $makeSplit --weights $resnetWeights --output $makeRun --device cuda
python scripts/train_web_transfer.py train --root $dataRoot --audit $auditDir --prepared $bodySplit --weights $resnetWeights --output $bodyRun --device cuda
```

The split and run variables refer to separate new directories. `$resnetWeights` is the local official `resnet18-f37072fd.pth` file. Its full SHA-256 must be `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

The recipe uses ImageNet initialization, 224×224 square crops, ImageNet normalization and no augmentation. Eight epochs comprise two head-only epochs followed by six epochs updating layer4 and the head. BatchNorm statistics stay frozen. AdamW uses head learning rate 0.001, layer4 learning rate 0.0001 and weight decay 0.01. Seed 36127 and batch size 32 are fixed. Validation macro F1 selects the checkpoint; the earliest epoch wins ties. The recipe follows the existing colour transfer implementation, but its results are a separate experiment on different labels and images.

Training starts only after selected image hashes and frozen split contents are rechecked. Each run retains config, class mapping, source snapshots, history and a best checkpoint. A fresh Python process reloads that checkpoint on CPU, compares validation predictions with the selected epoch and independently reconstructs metrics from the saved prediction CSV. A training result with `verification_pending: true` is incomplete.

## Shared integration contract

| Item | Contract |
|---|---|
| Predictor key | `make`, `body_type` or `colour`; it must match checkpoint `config.task` even with no detections. |
| Checkpoint | `state_dict`, ordered `classes`, `config`, `synthetic`, `manifest_sha256`, `best_epoch`. |
| Class order | Derived from training labels; reused unchanged by validation and inference. |
| Input | PIL RGB crop; OpenCV square resize; ImageNet normalization. No second bbox conversion in inference. |
| Detection box | Zero-based xyxy; bounded to the image by the pipeline. |
| Output | Existing `VehicleProfile` fields preserved. `model`, damage, accessories and unavailable predictors stay `not_assessed`. |
| Score | Uncalibrated softmax score, not estimated accuracy. Pilot weights report `pilot_prediction`. |
| JSON | Serialize `vehicles` and `confidence_semantics`; save the returned PIL `annotated_image` separately. |

```python
import json
from vehicle_id.attributes.modelling import AttributePredictor
from vehicle_id.pipeline import analyse_vehicle_image

predictors = {
    'make': AttributePredictor(make_checkpoint),
    'body_type': AttributePredictor(body_checkpoint),
}
result = analyse_vehicle_image(image_path, predictors, detector_weights=local_yolo_weights)
result['annotated_image'].save(annotated_path)
json_path.write_text(json.dumps({
    'vehicles': result['vehicles'],
    'confidence_semantics': result['confidence_semantics'],
}, indent=2), encoding='utf-8')
```

Colour can be added when its trusted local checkpoint is available. Synthetic interface tests do not reproduce Yuchen's colour result. The older generic `evaluate()` expects a different manifest-root configuration; use this runner's `verify` mode for these validation checkpoints. Official-test evaluation needs a separate declared final-evaluation step.

## Verification coverage

The 25-test suite passes and covers the existing colour workflow, model heads, paths, split exclusions and archive CRC fixture, plus missing body-type handling, training/inference crop equality and wrong-task rejection with no detections. Both real checkpoints passed separate reload verification. Seven actual-checkpoint integration checks cover a single vehicle, multiple supplied boxes, no detections, unavailable predictors, invalid boxes, task mismatch and invalid confidence. Five actual YOLO examples produced 1, 3, 1, 12 and 5 detections; they are demonstrations, not a new accuracy sample.

Near-duplicate review, physical-vehicle-disjoint evaluation, calibrated abstention and official-test evaluation remain future work. No claim is made that this prototype is ready for operational use.
