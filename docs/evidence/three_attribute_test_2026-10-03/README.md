# Aggregate results package — draft, not published

3 October 2026 · prepared at the request of the 3 October three-attribute audit ·
**publication requires separate authorisation and has not been requested**

## What this package contains

| File | Content |
|---|---|
| `results.json` | the three formal results: counts, accuracy, macro and weighted F1, macro precision/recall, per-class precision/recall/F1/support/predicted, the confusion matrix in the declared class order, and the two internal consistency checks |
| `validation_seeds.json` | validation results for every seed, with the note that make/body share one holdout while the colour seeds drew different partitions |
| `secondary_pilot_level.json` | the two capped-pilot test scores, explicitly labelled as secondary comparisons, not the formal result |
| `populations.json` | what each number is measured on, and the exclusions, including the route-1 shortfalls and the unavailable body labels |
| `hashes.json` | evaluation entry point, freeze records, checkpoints, membership sources and membership tables |
| `commands.md` | portable command templates for intake, protocol emission, freezing and the declared run |

## What this package deliberately excludes

Model weights; images; image identifiers; row-level predictions; absolute
filesystem paths; local environment records; approval records, chat transcripts and
other private correspondence. Nothing here identifies an individual image.

## Authorship

Colour: Yuchen's dataset, predictor, pipeline, split logic and colour-transfer
recipe, with the uncapped colour runs and the evaluation gating produced by this
workstream. Make and body type: Yuxiang's runner, training recipe and capped pilots
from PR #5, with the uncapped training runs and two documented local adaptations
(a seed option and per-epoch CPU validation) produced by this workstream. Evidence
gating, protocol design and this package: the pre-test workstream.

## Disclosures that must be published with the numbers

1. **Two exposures for make and body type.** Those test subsets were scored once
   with the capped pilots (pre-declared as a pilot-level comparison) and once
   formally with the uncapped models. The formal results must not be described as
   the first or only test exposure; the pilot-level scores above are secondary.
   The selection rule was fixed before both exposures.
2. **Colour is independently audited; make and body type are not yet.**
3. **Declared subsets, not complete official test sets**: colour 13,323 of 13,333;
   make 14,922 of 14,939; body type 14,570 after 17 absent keys and 352 unavailable
   labels. Training manifest shortfalls are listed in `populations.json`.
4. **No combined score.** The attributes use different datasets and class counts;
   any three-attribute average would be meaningless and none is provided.
5. Results are **image level**, not vehicle-disjoint; near-duplicate and
   same-physical-vehicle images were not identified.
6. Scores are **uncalibrated**; no abstention threshold has been validated.
7. The data are CompCars imagery, **not NSW Police material**, and this is not an
   end-to-end detector-to-attributes measurement.

## Suggested result sentence

> The validation-selected attribute checkpoints achieved 91.58% accuracy and macro
> F1 of 0.8459 for colour (13,323 images), 72.00% accuracy and macro F1 of 0.6796
> for make (14,922 images) and 85.72% accuracy and macro F1 of 0.7260 for body type
> (14,570 images), each on the predeclared route-1 subset of the corresponding
> official CompCars test split, with model selection using validation data only.
> The make and body-type subsets were also scored once with the earlier capped
> pilots, a pilot-level comparison retained as secondary evidence. Results are
> image-level and do not establish end-to-end or NSW Police performance.

## 中文摘要

这是按 10 月 3 日审计要求准备的**聚合包草稿**（尚未发布、也未申请发布授权）。包含：
三属性的正式指标与逐类支持数、混淆矩阵；各种子的验证结果；两个截断 pilot 的**次级**
对照分数；人群口径与排除项；哈希清单；以及可移植的命令模板。

**不含**：权重、图像、图像标识、逐行预测、本机绝对路径、环境记录、审批与聊天等私人
材料。**必须随数字披露**：① make/body 的 test 子集被计分两次（pilot 级在先、正式在后，
不得称为"首次/唯一一次"）；② colour 已独立审计、make/body 尚未；③ 是 route-1 声明子集
而非完整官方测试集；④ 不提供三属性合并总分；⑤ 图像级、非车辆级不重复；⑥ 分数未校准；
⑦ 非警务数据、非端到端测量。
