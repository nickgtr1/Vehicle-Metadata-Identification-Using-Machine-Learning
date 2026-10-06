# 36127 Stage 2 Baseline Integration

The trained make/body version and its local demo are documented in [Make and body-type transfer baselines](docs/WEB_TRANSFER.md). The historical HSV baseline below remains available as a separate interface example.

This contribution connects persisted Stage 1 vehicle boxes to a Stage 2
attribute-output pipeline without downloading data or model weights.

## Current runnable component

- Reads the draft Stage 1 localisation JSON contract.
- Crops every valid vehicle box from the source image.
- Runs a deterministic central-region HSV colour baseline.
- Writes a structured JSON file, a flat CSV file, vehicle crops and an
  annotated output image.
- Marks make, model, body type, damage and accessories as `not_assessed` until
  appropriate labelled data and model weights are available.

## Smoke-test command

From the repository root, using the project environment:

```powershell
python scripts/run_stage2_baseline.py `
  --image data/samples/blocked_car.jpg `
  --localisation-json data/manifests/blocked_car_yolo26n_notebook_saved_output.json `
  --output-dir outputs/stage2_baseline_smoke_test
```

The detection manifest was transcribed from persisted output in Nicholas's
localisation notebook. The detector was not rerun and no weights were used by
this Stage 2 smoke test.

## Evidence boundary

The test proves that the proposed interface and output generation are runnable
on a supplied sample image. The colour result is an unvalidated image-processing
baseline, not a trained classifier and not an accuracy result. It does not prove
make/model, body type, damage, accessory or client-domain performance. No NSW
Police data is included or used.

The reported colour confidence is only the dominant classified-pixel share. If
most detections receive the same label, the JSON includes a warning about scene
colour cast, background contamination or baseline bias.
