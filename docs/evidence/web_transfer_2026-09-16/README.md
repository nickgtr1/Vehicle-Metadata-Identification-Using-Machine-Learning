# Make/body v0.1 — aggregate evidence

16 September 2026 · CompCars web validation pilots

These experiments extend Yuchen's dataset, predictor, pipeline and colour transfer recipe with make and body-type checkpoints. Nicholas's source manifests remain unchanged. The model card, metrics and figures here are the shareable aggregate evidence; raw images, weights, private split assignments and image-based demo exports remain local.

## Results

| Task | Fit / validation | Classes | Accuracy | Macro F1 | Majority accuracy | Majority macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Make | 4,416 / 1,120 | 75 | 54.64% | 0.5890 | 1.79% | 0.000468 |
| Body type | 3,112 / 782 | 12 | 67.01% | 0.6961 | 12.79% | 0.018896 |

Both checkpoints were selected at epoch 8 by validation macro F1 and passed fresh-process CPU reload plus independent metric reconstruction from saved predictions. These are annotated-crop validation results, not final-test or end-to-end detector accuracy. The majority reference predicts one most frequent fit label for every validation image: Audi for make and MPV for body. Class caps create ties; these labels are not estimates of real-world prevalence.

The make/body split assignments were created for these runs using Yuchen's `pilot_split`, seed 36127 and the rows marked official train in Nicholas's manifest. They were not pre-existing team validation sets. Each task has separate eligibility, class mapping and cap (make 100; body 500 before an approximately 80/20 allocation). Counts below each cap remain limited by available data. Official test membership was retained and no official-test model evaluation was performed.

## Reading the evidence

- [Model card and figures](report/MODEL_CARD.md): baseline comparison, learning history, error analysis and limitations.
- [Make metrics](make/validation_metrics.json) and [body metrics](body_type/validation_metrics.json): class order, confusion matrix and full scores.
- Task directories also contain config, class counts, history, recall, source hashes, training timing and reload verification.
- [Integration summary](integration_summary.json): seven actual-checkpoint checks and five live detector examples. This is not a detector benchmark.
- [Source coverage](source_coverage.json): the later-discovered thirty-row official-list omission.
- [Additional test-content overlap check](omitted_test_overlap_review.json): the seventeen omitted official-test files have no exact-hash overlap with either frozen fit/validation split. Near duplicates and physical identity remain unchecked.
- [Copy provenance](copy_provenance.json): hashes for byte-identical copies and the named derived summaries. The local commands and user-specific paths are omitted from environment summaries.

## Known limits

The original source CSV contains 30,925 classification rows; official lists contain 30,955. The notebook's numeric-year path rule omits thirty existing images with an `unknown` year (13 train, 17 test). Included-row label/decode checks passed, but those checks did not establish exhaustive source-list coverage. These results preserve the smaller frozen manifest; a versioned coverage correction must precede the next larger-data experiment or complete official benchmark claim.

Nineteen make classes have fewer than ten validation images. Wealeak has only two, so one changed prediction moves its recall by fifty percentage points. No universal Macro F1 pass mark is assumed. Macro F1 0.5890/0.6961 supports a useful first baseline, while per-class failures, small supports, one-seed selection and uncalibrated scores limit stronger claims.

The same images have not been shown to represent independent physical vehicles. Data came from a documented mirror; all extracted files passed archive CRC checks and a bounded set of 100 images matched decoded official bytes. These are distinct checks with limited provenance and identity coverage.

Colour requires Yuchen's trusted trained checkpoint. Vehicle model, damage and accessories remain unassessed. The local demo keeps incorrect predictions visible. The [run guide](../../WEB_TRANSFER.md) describes new-image inference and reproduction with locally available data and weights.

AI assistance was used for implementation, verification scripts and documentation. This is a development contribution for review; it does not assert that Yuxiang has already completed personal coursework review.
