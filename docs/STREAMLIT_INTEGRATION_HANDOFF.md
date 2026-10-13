# Streamlit integration handoff

## Behaviour

Compare retains the original local attribute profile in `attribute_profile` and stores the independent VLM response in `vlm_profile`. The existing `vehicle_profile` still supports fallback values for missing attributes. A failed VLM request preserves the local result and allows later vehicles to continue. Fine-tuned VLM colour remains unassessed.

CV uses the detection crop. VLM adds 15% padding on each side, so the comparison is a demonstration rather than a matched-crop benchmark. JSON exports include the separate profiles when VLM is enabled.

CSV and JSON download buttons avoid rerunning the analysis. Missing or incomplete VLM evaluation files display an availability message rather than breaking the findings page.

## Local CV demonstration

Use an existing project environment with the dependencies in `pyproject.toml`, plus Streamlit, Matplotlib and Ultralytics. From the repository root:

```sh
python -m streamlit run app/streamlit_app.py
```

Load the detector and the separately shared attribute checkpoints using the sidebar. The sidebar takes one checkpoint per attribute: `make`, `model`, `body_type` and `colour`. The four-checkpoint package shared on 12 October contains:

- `model_best_seed36128.pt` (431 classes)
- `make_best_seed36127.pt` (75 classes)
- `body_type_best_seed36128.pt` (12 classes)
- `colour_best_seed36128.pt` (10 classes)

Keep VLM disabled for the local CV demonstration. Check the annotated detections, corresponding vehicle cards/table rows, and CSV/JSON downloads using a repository sample such as `data/samples/high_angle_cars.jpg`. Confirm that downloads leave the displayed results available. Without a detector or attribute checkpoint, use the sidebar diagnostics rather than interpreting missing outputs as model predictions.

Weights and datasets are distributed separately. The make/body checkpoint `pilot_only` metadata still needs confirmation from the package owner. The car-model checkpoint is selected on validation macro F1 and scored once on the declared 14,922-row official test subset; its evidence pack stays local and the numbers are self-audited. No training or evaluation results are changed by this handoff.

When a car-model checkpoint is loaded, the VLM fill leaves `model` untouched and only fills the attributes that have no trained predictor; the VLM's own `model` answer is still recorded in `vlm_profile` for the comparison view.

## Validation

```sh
python -m unittest discover -s tests -p 'test_vlm_comparison.py' -v
python -m unittest discover -s tests -v
```

The comparison tests mock VLM responses and require no cloud credentials or paid inference. Coverage includes independent answers, retained local values, fallback fields, partial batch failure, temporary-file cleanup, unavailable clients, disabled/empty input and zero-shot colour. Historical tests requiring unavailable fixtures may skip.

Real checkpoint loading, multi-vehicle cards/tables and actual downloads were checked locally on 10 October. The mock checks establish application behaviour, not VLM prediction quality. The original VLM evaluation CSVs remain a separate handoff from the VLM workstream.
