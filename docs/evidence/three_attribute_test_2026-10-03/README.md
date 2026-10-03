# Aggregate results package

3 October 2026 · **published** · this revision follows the 3 October review of the
post-audit response

## Publication status

| Item | Value |
|---|---|
| Repository | `nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning` |
| Path | `docs/evidence/three_attribute_test_2026-10-03/` |
| Pull request | https://github.com/nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning/pull/9 |
| Branch | `yuchen-three-attribute-results` |
| First published commit | `c75d9e3dbecbac99752ccd81093070ff8acb2f8e` (3 October 2026, +7,234 lines, 7 files) |
| Remote-versus-local check | every published file is byte-identical to the reviewed local package; receipt archived with the full commit SHA, per-file blob hashes and the comparison output |

An earlier version of this README said that publication had not been requested. That
was accurate when it was written and is superseded by this revision. Hashing is not a
reason to keep an obsolete status statement in a living document; the earlier
commits and the original local evidence are preserved unchanged.

## Audit status, stated precisely

* **Numerical and binding checks: pass.** An independent audit recomputed accuracy,
  macro F1 and weighted F1 from the saved confusion matrices (maximum difference
  1.1 × 10⁻¹⁶) and verified that the checkpoints, protocol records, membership
  sources and membership tables match their recorded hashes.
* **Approval and protocol provenance: qualified.** The make and body-type test
  subsets were scored twice: once with the capped pilots, published here as an
  explicitly labelled secondary comparison, and once formally with the uncapped
  models. The project owner ratified that as an approved protocol amendment
  **retrospectively, after both runs had happened**; the primary dated messages have
  not yet been exported into the evidence set, so the position is properly described
  as "the owner reports approval" rather than as independently established
  pre-registration. The owner has confirmed that the approvals given in the working
  thread are the authoritative instruction and has declined to provide a separate
  export, so this wording is not provisional; the thread is the accessible original
  reference.
* The audit that recommended preparing this package **did not pre-approve this
  package**. The package was prepared afterwards and reviewed separately.

## What this package contains

| File | Content |
|---|---|
| `results.json` | per-task counts, accuracy, macro and weighted F1, macro precision and recall, per-class precision/recall/F1/support/predicted, the confusion matrix in the declared class order, and the internal consistency checks |
| `validation_seeds.json` | every seed's validation result, with the note that make and body type share one holdout while the colour seeds drew different partitions |
| `secondary_pilot_level.json` | the capped-pilot test scores, labelled as secondary comparisons |
| `populations.json` | what each number is measured on, and every exclusion |
| `hashes.json` | evaluation entry point, protocol records, checkpoints, membership sources and tables |
| `commands.md` | intake, draft emission, freezing and the declared run, as single-line commands with a per-task value table |

## What this package excludes

Model weights; images; image identifiers; row-level predictions; absolute filesystem
paths; environment records; approval records, chat transcripts and other private
correspondence.

## Populations and shortfalls

| Attribute | Measured population | Test shortfall | Training shortfall |
|---|---|---|---|
| colour | route-1 subset of the official CompCars **surveillance** test split | 13,323 of 13,333 keys | 31,118 of 31,148 official train keys |
| make | route-1 subset of the official CompCars **web** classification test split | 14,922 of 14,939 keys | 16,001 fit and validation rows from 16,003 manifest train rows, and the official train list holds 16,016 keys, so **13 official train keys are absent** |
| body type | as make, minus unavailable labels | 14,570 rows after the 352 rows whose official body type is 0 | 15,631 rows from the same 16,003 manifest rows, with the same 13-key official-train shortfall |

These are declared subsets, not complete official test sets. The make and
body-type training figures reconcile as: 16,003 manifest train rows, minus 1 row
excluded as a duplicate of test content or as a conflicting label, minus 1 duplicate
representative of an identical-byte group, giving 16,001; and for body type, 16,003
minus 369 rows whose label is unavailable, minus 3 exclusions, giving 15,631.

## Authorship

Colour: Yuchen's dataset, predictor, pipeline, split logic and colour-transfer
recipe, with the uncapped colour runs and the evaluation gating produced by this
workstream. Make and body type: Yuxiang's runner, training recipe and capped pilots
from PR #5, with the uncapped training runs and two documented local adaptations
(a seed option and per-epoch CPU validation) produced by this workstream. Protocol
design, evidence gating and this package: the pre-test workstream.

## Disclosures that must be published with the numbers

1. **Two exposures for make and body type.** Those test subsets were scored once
   with the capped pilots and once formally with the uncapped models. The formal
   results are not the first or only exposure, the amendment was retrospective, and
   the selection rule was fixed before both runs.
2. **Colour has been independently audited; make and body type have been reviewed
   as part of the same hand-back but have not had a separate independent audit.**
3. Declared subsets, not complete official test sets, as tabulated above.
4. Results are **image level**, not vehicle-disjoint: near-duplicate images and
   images of the same physical vehicle were not identified.
5. Scores are **uncalibrated**; no abstention threshold has been validated.
6. The data are CompCars imagery, **not NSW Police material**, and this is not an
   end-to-end detector-to-attributes measurement.
7. There is **no combined score**, and none should be constructed.

## Suggested result sentence

> The validation-selected attribute checkpoints achieved 91.58% accuracy and macro
> F1 of 0.8459 for colour (13,323 images), 72.00% accuracy and macro F1 of 0.6796 for
> make (14,922 images) and 85.72% accuracy and macro F1 of 0.7260 for body type
> (14,570 images), each on the predeclared route-1 subset of the corresponding
> official CompCars test split, with model selection using validation data only. The
> make and body-type subsets were also scored once with the earlier capped pilots, a
> pilot-level comparison retained as secondary evidence. Results are image-level and
> do not establish end-to-end or NSW Police performance.
