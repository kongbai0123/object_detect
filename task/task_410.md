# Task 2026-04-10

## Context

### Goal
- Rebuild downstream data from the updated `3_processed`.
- Verify whether `imgsz=832` improves `close` recall.
- Extract effective `close FN` samples for the next incremental loop.

### Scope
- Data rebuild
- FN analysis
- Stable training
- Feedback candidate packaging

## Inputs

### Source Data
- Gold pool: `data/3_processed`
- Refreshed dataset yaml: [dataset_refresh_20260410.yaml](/c:/antigravity/data/dataset_refresh_20260410.yaml)

### Training Config
- Config file: [train_config_stable_long_832.yaml](/c:/antigravity/configs/train_config_stable_long_832.yaml)
- Key params:
  - `imgsz=832`
  - `epochs=60`
  - `batch=16`
  - `lr0=0.001`
  - `patience=30`
  - `cache=disk`

### Base Model
- Initialization weights: [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

## Work Performed

### Data Rebuild
- Re-split data from `data/3_processed` into refresh paths to avoid stale `6_augmented` content and file-lock conflicts.
- Split result:
  - Train: `238` images, `open_boxes=91`, `close_boxes=225`
  - Val: `60` images, `open_boxes=22`, `close_boxes=67`

### Pre-Train FN Baseline
- Ran `close`-only FN analysis on `val_refresh`.
- Analysis condition:
  - `conf=0.4`
  - `imgsz=768`
  - `IoU match=0.5`
- Output:
  - [close_fn_pre832_20260410_summary.json](/c:/antigravity/data/reports/close_fn_pre832_20260410_summary.json)
  - [close_fn_pre832_20260410.csv](/c:/antigravity/data/reports/close_fn_pre832_20260410.csv)

### Train Data Preparation
- Balanced refresh train source:
  - Output: `data/6_augmented/train_src_balanced_refresh`
  - Result: `open_boxes=91`, `close_boxes=87`
- Augmented balanced train source:
  - Output: `data/6_augmented/train_refresh`
  - Result: `411` train images

### Stable 832 Training
- Experiment directory: [exp_v072_base3](/c:/antigravity/data/7_experiments/exp_v072_base3)
- Best weights: [best.pt](/c:/antigravity/data/7_experiments/exp_v072_base3/weights/best.pt)
- Runtime config evidence:
  - [args.yaml](/c:/antigravity/data/7_experiments/exp_v072_base3/args.yaml)

### Post-Train FN Analysis
- Re-ran `close` FN analysis using the new best model.
- Analysis condition:
  - `conf=0.4`
  - `imgsz=832`
  - `IoU match=0.5`
- Output:
  - [close_fn_post832_20260410_summary.json](/c:/antigravity/data/reports/close_fn_post832_20260410_summary.json)
  - [close_fn_post832_20260410.csv](/c:/antigravity/data/reports/close_fn_post832_20260410.csv)

### Feedback Candidate Packaging
- Packaged effective `close FN` candidates for the next incremental loop.
- Output folder:
  - [close_fn_effective_20260410](/c:/antigravity/data/fn_feedback/close_fn_effective_20260410)

## Results

### Training Metrics
- `mAP50 = 0.9064`
- `mAP50-95 = 0.6295`
- `Precision = 0.9510`
- `Recall = 0.8497`

### Class Metrics
- `open`: `P=1.000`, `R=0.864`, `AP50=0.932`, `AP50-95=0.737`
- `close`: `P=0.902`, `R=0.836`, `AP50=0.881`, `AP50-95=0.522`

### Promotion Result
- Promoted to new [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- Promotion evidence:
  - [global_best_info.json](/c:/antigravity/data/7_experiments/global_best_info.json)
- Fitness:
  - `0.6499 -> 0.6572`

## Evidence and Comparison

### Close FN Before 832
- `total_close_gt=67`
- `matched_close_gt=36`
- `fn_close_gt=31`
- `close_recall=0.5373`
- `small_fn=19`

### Close FN After 832
- `total_close_gt=67`
- `matched_close_gt=56`
- `fn_close_gt=11`
- `close_recall=0.8358`
- `small_fn=9`
- `effective_fn_gt=6`

### Interpretation
- `imgsz=832` improved `close` recall materially on the refreshed validation split.
- Remaining FN cases are still dominated by small `close` objects.
- The current bottleneck remains weak visual signal for `close`, not class confusion.

## Decisions

### Accepted
- Use refreshed data derived from `3_processed` as the current working baseline.
- Keep `imgsz=832` as the current preferred setting for `close`-sensitive training.
- Use `exp_v072_base3` / `global_best.pt` as the base model for the next incremental round.

### Rejected
- Do not use stale `6_augmented` splits as the source of truth.
- Do not treat all FN as feedback candidates.
- Do not change multiple major variables at once in the next incremental run.

## Next Actions

### Immediate
- Review `close_fn_effective_20260410`
- Merge approved feedback samples into the next incremental intake
- Add a small batch of new-distribution raw samples

### Next Training Rule
- Start from current `global_best.pt`
- Use incremental mode
- Keep `imgsz` fixed
- Keep validation threshold fixed for comparison
- Limit each incremental round to a small, traceable batch

## Logging Standard

### Section Order
- `Context`
- `Inputs`
- `Work Performed`
- `Results`
- `Evidence and Comparison`
- `Decisions`
- `Next Actions`

### Purpose
- Make daily files readable without replaying chat history
- Separate evidence from conclusions
- Keep execution and follow-up decisions traceable

## Incremental Round Scaffold

### Round ID
- `2026-04-10_r1`

### Assets Created
- Incremental config:
  - [train_config_incremental_832.yaml](/c:/antigravity/configs/train_config_incremental_832.yaml)
- Round workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/README.md)
- Review manifest:
  - [close_fn_review_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/manifests/close_fn_review_manifest.csv)

### Current Status
- Review candidates have been copied into:
  - [images](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/review/close_fn_effective/images)
  - [labels](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/review/close_fn_effective/labels)
- User approved all 5 effective `close FN` feedback samples.
- Approved bucket:
  - [images](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/approved/images)
  - [labels](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/approved/labels)
- `1_raw` is treated as an image-only source for new-distribution intake. Existing stray `.txt` files are ignored for this round.
- Auto-selected 20 new-distribution candidates from `1_raw` using current `global_best.pt` predictions with provisional labels:
  - [images](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/pending_new_distribution/images)
  - [labels](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/pending_new_distribution/labels)
  - [new_distribution_review_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/manifests/new_distribution_review_manifest.csv)
- Current blocker:
  - manual review is still required for the 20 new-distribution candidates before merge into the next incremental intake.

## Incremental Round 1 Execution

### Round Dataset
- Round dataset yaml:
  - [dataset_incremental_r1.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_r1/dataset_incremental_r1.yaml)
- Intake batch:
  - `25` images
  - source = `5` approved feedback + `20` new-distribution candidates
- Round validation:
  - `55` images
  - `5` feedback images removed to avoid leakage

### Training
- Experiment:
  - [exp_v072_inc11](/c:/antigravity/data/7_experiments/exp_v072_inc11)
- Config:
  - [train_config_incremental_832.yaml](/c:/antigravity/configs/train_config_incremental_832.yaml)
- Mode:
  - incremental
- Base weights:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

### Incremental Result
- `mAP50 = 0.9465`
- `Precision = 0.9405`
- `Recall = 0.8594`
- class metrics:
  - `open`: `P=1.000`, `R=0.886`, `AP50=0.974`, `AP50-95=0.724`
  - `close`: `P=0.881`, `R=0.833`, `AP50=0.919`, `AP50-95=0.531`
- Promotion:
  - promoted to new [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

## Incremental Round 1 Analysis

### Close FN Report
- Summary:
  - [close_fn_inc11_20260410_summary.json](/c:/antigravity/data/reports/close_fn_inc11_20260410_summary.json)
- Detail:
  - [close_fn_inc11_20260410.csv](/c:/antigravity/data/reports/close_fn_inc11_20260410.csv)

### Close FN Result
- `total_close_gt = 54`
- `matched_close_gt = 51`
- `fn_close_gt = 3`
- `close_recall = 0.9444`
- `small_fn = 3`
- `effective_fn_gt = 1`

### Round 2 Feedback Candidate
- Feedback pack:
  - [close_fn_effective_20260410_r2](/c:/antigravity/data/fn_feedback/close_fn_effective_20260410_r2)
- current effective image:
  - `img_0068.jpg`

## Incremental Round 2 Candidate Refresh

### Rule Applied
- Rebuild `new_scene_distribution` candidates for round 2.
- Exclude any image already used in:
  - `data/incremental_rounds/2026-04-10_r1/intake`
- Exclude any image already present in:
  - `data/3_processed`

### Output
- Round 2 workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_r2/README.md)
- Round 2 candidate manifest:
  - [new_distribution_review_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_r2/manifests/new_distribution_review_manifest.csv)
- Round 2 candidate images:
  - [images](/c:/antigravity/data/incremental_rounds/2026-04-10_r2/pending_new_distribution/images)
- Round 2 provisional labels:
  - [labels](/c:/antigravity/data/incremental_rounds/2026-04-10_r2/pending_new_distribution/labels)

## Incremental Round 2 Execution

### Round Dataset
- Round dataset yaml:
  - [dataset_incremental_r2.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_r2/dataset_incremental_r2.yaml)
- Intake batch:
  - `21` images
  - source = `1` feedback carry-over + `20` new-distribution samples
- Round validation:
  - `54` images
  - feedback image removed to avoid leakage

### Training
- Experiment:
  - [exp_v072_inc12](/c:/antigravity/data/7_experiments/exp_v072_inc12)
- Config:
  - [train_config_incremental_832.yaml](/c:/antigravity/configs/train_config_incremental_832.yaml)
- Mode:
  - incremental
- Base weights:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

### Incremental Result
- `mAP50 = 0.9483`
- `Precision = 0.9225`
- `Recall = 0.8910`
- class metrics:
  - `open`: `P=1.000`, `R=0.917`, `AP50=0.976`, `AP50-95=0.715`
  - `close`: `P=0.845`, `R=0.865`, `AP50=0.921`, `AP50-95=0.544`
- Promotion:
  - promoted to new [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

## Final Round Preparation

### Objective
- Run one last non-incremental consolidation round after the user refreshed `data/1_raw`.
- Keep `imgsz=832`, switch to explicit `AdamW`, and allow a longer schedule than the incremental path.

### Final Intake Build
- Final workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_final/README.md)
- New-distribution manifest:
  - [final_new_distribution_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_final/manifests/final_new_distribution_manifest.csv)
- Final intake manifest:
  - [final_intake_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_final/manifests/final_intake_manifest.csv)
- Final intake composition:
  - `61` unique images
  - `25` from round 1 intake
  - `21` from round 2 intake
  - `15` from final new-distribution intake

### Final Training Inputs
- Dataset yaml:
  - [dataset_final_20260410.yaml](/c:/antigravity/data/dataset_final_20260410.yaml)
- Train config:
  - [train_config_final_832_adamw.yaml](/c:/antigravity/configs/train_config_final_832_adamw.yaml)
- Base weights:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

## Final Round Execution

### Training
- Experiment:
  - [exp_v072_base5](/c:/antigravity/data/7_experiments/exp_v072_base5)
- Mode:
  - non-incremental
- Key config:
  - `imgsz = 832`
  - `optimizer = AdamW`
  - `epochs = 70`
  - `lr0 = 0.0008`
  - `patience = 25`
  - `mosaic = 0.15`

### Final Result
- Early stop at epoch `63`, best epoch = `38`
- strict validation summary from pipeline:
  - `mAP50 = 0.9038`
  - `Precision = 0.8590`
  - `Recall = 0.8625`
- validation block on saved best:
  - `all`: `P=0.824`, `R=0.913`, `AP50=0.930`, `AP50-95=0.595`
  - `open`: `P=0.952`, `R=0.902`, `AP50=0.971`, `AP50-95=0.702`
  - `close`: `P=0.695`, `R=0.923`, `AP50=0.890`, `AP50-95=0.489`

### Promotion Decision
- Result:
  - rejected by safety gate
- Reason:
  - `Fitness (0.6313)` did not exceed historical best `0.6614`
- Consequence:
  - weights kept in [latest_best.pt](/c:/antigravity/data/7_experiments/weight/latest_best.pt)
  - current [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt) remains unchanged

## Next Incremental Prep: `inc_base5_1`

### Objective
- Prepare the next repair round from [exp_v072_base5](/c:/antigravity/data/7_experiments/exp_v072_base5), focusing on:
  - `open FN`
  - `close FP background`
  - `new scene distribution`

### Workspace
- Round workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_inc_base5_1/README.md)
- Summary:
  - [base5_error_buckets_20260410_summary.json](/c:/antigravity/data/reports/base5_error_buckets_20260410_summary.json)

### Error Buckets
- `open FN` report:
  - [open_fn_base5_20260410.csv](/c:/antigravity/data/reports/open_fn_base5_20260410.csv)
- `close FP background` report:
  - [close_fp_background_base5_20260410.csv](/c:/antigravity/data/reports/close_fp_background_base5_20260410.csv)
- current bucket sizes:
  - `open FN`: `3` images
  - `close FP background`: `12` images

### New Distribution Refresh
- Pending new-distribution manifest:
  - [new_distribution_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_inc_base5_1/manifests/new_distribution_manifest.csv)
- current pending size:
  - `15` images

### Current State
- Automated preparation is complete.
- The next blocking step is manual review of the three buckets before building the intake for `inc_base5_1`.

## `inc_base5_1` Execution

### User Review Decision
- `open FN`: all approved
- `close FP background`: all approved
- `new_distribution`: all approved

### Round Build
- Dataset yaml:
  - [dataset_incremental_inc_base5_1.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_inc_base5_1/dataset_incremental_inc_base5_1.yaml)
- Config:
  - [train_config_inc_base5_1_832.yaml](/c:/antigravity/configs/train_config_inc_base5_1_832.yaml)
- Intake manifest:
  - [intake_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_inc_base5_1/manifests/intake_manifest.csv)
- Val manifest:
  - [val_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_inc_base5_1/manifests/val_manifest.csv)
- Intake composition:
  - `30` images total
  - `3` from `open FN`
  - `12` from `close FP background`
  - `15` from `new_distribution`
- Validation split:
  - `39` images
  - `15` leakage candidates removed from prior val

### Training
- Base weights:
  - [best.pt](/c:/antigravity/data/7_experiments/exp_v072_base5/weights/best.pt)
- Experiment:
  - [exp_v072_inc13](/c:/antigravity/data/7_experiments/exp_v072_inc13)
- Mode:
  - incremental
- Key config:
  - `imgsz = 832`
  - `optimizer = AdamW`
  - `epochs = 24`
  - `lr0 = 0.0005`
  - `mosaic = 0.15`

### Result
- strict validation summary from pipeline:
  - `mAP50 = 0.9221`
  - `Precision = 0.9261`
  - `Recall = 0.8927`
- validation block on saved best:
  - `all`: `P=0.911`, `R=0.910`, `AP50=0.943`, `AP50-95=0.617`
  - `open`: `P=0.942`, `R=0.955`, `AP50=0.992`, `AP50-95=0.747`
  - `close`: `P=0.880`, `R=0.865`, `AP50=0.894`, `AP50-95=0.487`

### Promotion Decision
- Result:
  - rejected by safety gate
- Reason:
  - `Fitness (0.6574)` did not exceed historical best `0.6614`
- Consequence:
  - weights kept in [latest_best.pt](/c:/antigravity/data/7_experiments/weight/latest_best.pt)
  - current [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt) remains unchanged

## Box Refinement Prep

### Objective
- Build the next intake around localization quality instead of sample volume.
- Primary target:
  - improve `mAP50-95`
  - improve promotion `fitness`

### Workspace
- Round workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_box_refine_1/README.md)
- Summary:
  - [box_refine_base5_20260410_summary.json](/c:/antigravity/data/reports/box_refine_base5_20260410_summary.json)

### Review Buckets
- `low IoU TP` report:
  - [low_iou_tp_base5_20260410.csv](/c:/antigravity/data/reports/low_iou_tp_base5_20260410.csv)
- `close FP background` report:
  - [close_fp_background_base5_20260410.csv](/c:/antigravity/data/reports/close_fp_background_base5_20260410.csv)
- `open boundary hard` report:
  - [open_boundary_hard_base5_20260410.csv](/c:/antigravity/data/reports/open_boundary_hard_base5_20260410.csv)

### Current Bucket Sizes
- `low_iou_tp`: `26` rows / `26` images
- `close_fp_background`: `13` rows / `12` images
- `open_boundary_hard`: `3` rows / `3` images

### Current State
- Automated extraction is complete.
- The next blocking step is manual review of the three refinement buckets.

## Residual Analysis Triage

### Manual Corrections
- `img_042.jpg`
  - originally appeared in `close_fp_background`
  - user flagged it as `open`-related
  - GT label actually contains both `open` and `close`
  - redirected to mixed bucket instead of keeping it in `close_fp_background`
- `img_039.jpg`
  - originally appeared in `close_fn`
  - GT label contains both `open` and `close`
  - redirected to mixed bucket instead of keeping it in `close_fn`

### Outputs
- mixed bucket:
  - [images](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_analysis/mixed_open_close/images)
- note csv:
  - [residual_mixed_open_close_20260410.csv](/c:/antigravity/data/reports/residual_mixed_open_close_20260410.csv)
- filtered close-fp report:
  - [close_fp_background_global_best_20260410_filtered.csv](/c:/antigravity/data/reports/close_fp_background_global_best_20260410_filtered.csv)
- filtered close-fn report:
  - [close_fn_global_best_20260410_filtered.csv](/c:/antigravity/data/reports/close_fn_global_best_20260410_filtered.csv)

## Residual Priority Shortlist

### Objective
- shrink the residual error pool into a reviewable next-step shortlist
- avoid using the full residual dump as the next intake

### Workspace
- shortlist workspace:
  - [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_refine_1/README.md)
- summary:
  - [residual_refine_priority_20260410_summary.json](/c:/antigravity/data/reports/residual_refine_priority_20260410_summary.json)
- manifest:
  - [priority_review_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_refine_1/manifests/priority_review_manifest.csv)

### Shortlist Composition
- `low_iou_tp`: `19` images
- `close_fp_background`: `10` images
- `close_fn`: `8` images
- total: `37` images

## `residual_refine_1` Execution

### Assumption Used
- user replied `ok下一步`
- interpreted as approval to proceed with the full priority shortlist

### Round Build
- Dataset yaml:
  - [dataset_incremental_residual_refine_1.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_refine_1/dataset_incremental_residual_refine_1.yaml)
- Config:
  - [train_config_residual_refine_1_832.yaml](/c:/antigravity/configs/train_config_residual_refine_1_832.yaml)
- Intake manifest:
  - [intake_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_refine_1/manifests/intake_manifest.csv)
- Val manifest:
  - [val_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_residual_refine_1/manifests/val_manifest.csv)
- Intake composition after dedupe:
  - `29` images total
  - `19` from `low_iou_tp`
  - `5` from `close_fp_background`
  - `5` from `close_fn`
- Validation split:
  - `25` images
  - `29` leakage candidates removed from prior val

### Training
- Base weights:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- Experiment:
  - [exp_v072_inc15](/c:/antigravity/data/7_experiments/exp_v072_inc15)
- Mode:
  - incremental
- Key config:
  - `imgsz = 832`
  - `optimizer = AdamW`
  - `epochs = 20`
  - `lr0 = 0.0004`
  - `mosaic = 0.1`

### Result
- strict validation summary from pipeline:
  - `mAP50 = 0.9242`
  - `Precision = 0.9719`
  - `Recall = 0.8233`
- validation block on saved best:
  - `all`: `P=0.945`, `R=0.859`, `AP50=0.955`, `AP50-95=0.641`
  - `open`: `P=1.000`, `R=0.862`, `AP50=0.974`, `AP50-95=0.681`
  - `close`: `P=0.890`, `R=0.857`, `AP50=0.937`, `AP50-95=0.601`

### Promotion Decision
- Result:
  - rejected by safety gate
- Reason:
  - `Fitness (0.6950)` did not exceed historical best `0.7744`
- Consequence:
  - weights kept in [latest_best.pt](/c:/antigravity/data/7_experiments/weight/latest_best.pt)
  - current [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt) remains unchanged

## Residual Taxonomy And Merge Planning

### Decision
- stop same-type residual incremental training
- convert residual samples into structured assets for the next full rebuild

### Outputs
- full residual taxonomy:
  - [residual_taxonomy_20260410.csv](/c:/antigravity/data/residual_registry/residual_taxonomy_20260410.csv)
- merge shortlist:
  - [residual_merge_shortlist_20260410.csv](/c:/antigravity/data/residual_registry/residual_merge_shortlist_20260410.csv)
- summary:
  - [residual_taxonomy_summary_20260410.json](/c:/antigravity/data/residual_registry/residual_taxonomy_summary_20260410.json)

### Current Counts
- full registry:
  - `46` rows
- current merge shortlist:
  - `41` rows
- taxonomy breakdown:
  - `box_refinement`: `17`
  - `background_suppression`: `14`
  - `miss_recovery`: `13`
  - `mixed_state_review`: `2`

## FiftyOne Residual Review And Merge

### Review Dataset
- FiftyOne review dataset:
  - [residual_merge_shortlist_20260410](/c:/antigravity/data/fiftyone_review/residual_merge_shortlist_20260410)
- Edited export:
  - [residual_merge_shortlist_20260410_edited](/c:/antigravity/data/fiftyone_review/residual_merge_shortlist_20260410_edited)

### Validation Of Edited Output
- total reviewed labels:
  - `41`
- labels changed by FiftyOne editing:
  - `33`
- unchanged labels:
  - `8`
- missing images in `3_processed`:
  - `0`

### Merge Back To Gold Pool
- merged labels into:
  - [labels](/c:/antigravity/data/3_processed/labels)
- merged count:
  - `41`
- overwritten existing labels:
  - `41`
- backup of original labels:
  - [residual_merge_shortlist_labels_20260410_114158](/c:/antigravity/data/backup/residual_merge_shortlist_labels_20260410_114158)
- merge report:
  - [residual_merge_shortlist_merge_report_20260410_114158.csv](/c:/antigravity/data/backup/residual_merge_shortlist_merge_report_20260410_114158.csv)

### Current State
- FiftyOne-reviewed residual labels have been merged back into `3_processed`.
- The next full rebuild can now directly absorb these corrected samples from the gold pool.

## Full Rebuild After Residual Merge

### Goal
- run one full rebuild after merging the FiftyOne-reviewed residual labels back into `3_processed`

### Split And Data Build
- initial `split_dataset.py` result was rejected for training use because `val` had `0` open boxes
- rebuilt a refined split (`v3`) with finer granularity for oversized groups so `val` retained both classes

### Final Rebuild Dataset
- dataset yaml:
  - [dataset_full_rebuild_20260410_v3.yaml](/c:/antigravity/data/dataset_full_rebuild_20260410_v3.yaml)
- split outputs:
  - [train_src_rebuild_v3](/c:/antigravity/data/6_augmented/train_src_rebuild_v3)
  - [val_rebuild_v3](/c:/antigravity/data/6_augmented/val_rebuild_v3)
- balanced train:
  - [train_src_balanced_rebuild_v3](/c:/antigravity/data/6_augmented/train_src_balanced_rebuild_v3)
- augmented train:
  - [train_rebuild_v3](/c:/antigravity/data/6_augmented/train_rebuild_v3)

### Rebuild Data Stats
- split stats:
  - train images: `179`
  - val images: `119`
  - train boxes: `open=91`, `close=167`
  - val boxes: `open=22`, `close=126`
- balanced train:
  - images: `116`
  - boxes: `open=91`, `close=60`
- augmented train:
  - images: `340`

### Training
- experiment:
  - [exp_v072_base6](/c:/antigravity/data/7_experiments/exp_v072_base6)
- base weights:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- config:
  - [train_config_final_832_adamw.yaml](/c:/antigravity/configs/train_config_final_832_adamw.yaml)

### Result
- early stop at epoch `44`, best epoch = `19`
- strict validation summary from pipeline:
  - `mAP50 = 0.8649`
  - `Precision = 0.8060`
  - `Recall = 0.8610`
- validation block on saved best:
  - `all`: `P=0.784`, `R=0.871`, `AP50=0.846`, `AP50-95=0.605`
  - `open`: `P=0.929`, `R=0.909`, `AP50=0.908`, `AP50-95=0.750`
  - `close`: `P=0.640`, `R=0.833`, `AP50=0.785`, `AP50-95=0.461`

### Promotion Decision
- result:
  - rejected by safety gate
- reason:
  - `Fitness (0.6592)` did not exceed historical best `0.7744`
- consequence:
  - weights kept in [latest_best.pt](/c:/antigravity/data/7_experiments/weight/latest_best.pt)
  - current [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt) remains unchanged

## Log Update: `exp_v072_base6`

### Status
- `exp_v072_base6` has been completed and recorded as today's full rebuild attempt

### Key References
- experiment:
  - [exp_v072_base6](/c:/antigravity/data/7_experiments/exp_v072_base6)
- dataset:
  - [dataset_full_rebuild_20260410_v3.yaml](/c:/antigravity/data/dataset_full_rebuild_20260410_v3.yaml)
- config:
  - [train_config_final_832_adamw.yaml](/c:/antigravity/configs/train_config_final_832_adamw.yaml)

### Final Decision
- this run is archived as a completed but rejected rebuild candidate
- deployment baseline remains:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)

## YOLOv8s Initial Baseline

### Goal
- start a fresh baseline using pretrained `yolov8s.pt`
- do not use any self-trained weights as the training initializer

### Data Build
- split source:
  - [3_processed](/c:/antigravity/data/3_processed)
- frozen val:
  - [val_frozen_v1](/c:/antigravity/data/val_frozen_v1)
- outputs:
  - [train_src_y8s_init_20260410](/c:/antigravity/data/6_augmented/train_src_y8s_init_20260410)
  - [train_src_balanced_y8s_init_20260410](/c:/antigravity/data/6_augmented/train_src_balanced_y8s_init_20260410)
  - [train_y8s_init_20260410](/c:/antigravity/data/6_augmented/train_y8s_init_20260410)
  - [val_y8s_init_20260410](/c:/antigravity/data/6_augmented/val_y8s_init_20260410)

### Data Stats
- split:
  - train images: `298`
  - val images: `112`
  - train boxes: `open=113`, `close=293`
- balanced train:
  - images: `173`
  - boxes: `open=113`, `close=105`
- augmented train:
  - images: `507`

### Training
- experiment:
  - [exp_v072_base8](/c:/antigravity/data/7_experiments/exp_v072_base8)
- weights:
  - [yolov8s.pt](/c:/antigravity/yolov8s.pt)
- dataset yaml:
  - [dataset_y8s_initial_20260410.yaml](/c:/antigravity/data/dataset_y8s_initial_20260410.yaml)
- config:
  - [train_config_y8s_initial_20260410.yaml](/c:/antigravity/configs/train_config_y8s_initial_20260410.yaml)

### Result
- strict pipeline eval:
  - `mAP50 = 0.9117`
  - `Precision = 0.9298`
  - `Recall = 0.8969`
- validation block on best weights:
  - `all`: `P=0.929`, `R=0.897`, `AP50=0.928`, `AP50-95=0.847`
  - `open`: `P=0.998`, `R=1.000`, `AP50=0.995`, `AP50-95=0.958`
  - `close`: `P=0.860`, `R=0.794`, `AP50=0.860`, `AP50-95=0.736`

### Promotion Decision
- result:
  - rejected by safety gate
- reason:
  - `Fitness (0.8555)` did not exceed historical best `0.8965`
- consequence:
  - weights kept in [latest_best.pt](/c:/antigravity/data/7_experiments/weight/latest_best.pt)
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt) remains unchanged

### Formal Assessment
- role:
  - this run is recorded as the `YOLOv8s clean baseline`
- interpretation:
  - first full retrain from pretrained `yolov8s.pt`
  - no residual incremental repair
  - no initialization from prior self-trained best weights
- judgment:
  - successful baseline
  - not a promoted best model
  - suitable as the clean starting point for targeted repair rounds
- caveats:
  - `open` metrics are currently less trustworthy due to known label ambiguity
  - `close` remains the main optimization target
  - main residual pattern is `close` miss / close background FP / close low-IoU localization

## YOLOv8s Close-Focused Repair 1

### Goal
- start the first targeted repair round from `exp_v072_base8`
- focus only on:
  - `close_fp_background`
  - `close_fn`
  - `low_iou_tp_close`

### Workspace
- [README.md](/c:/antigravity/data/incremental_rounds/2026-04-10_y8s_close_repair_1/README.md)
- [dataset_incremental_y8s_close_repair_1.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_y8s_close_repair_1/dataset_incremental_y8s_close_repair_1.yaml)
- [train_config_y8s_close_repair_1.yaml](/c:/antigravity/configs/train_config_y8s_close_repair_1.yaml)

### Error Bucket Summary
- [y8s_close_repair_base8_20260410_summary.json](/c:/antigravity/data/reports/y8s_close_repair_base8_20260410_summary.json)
- `close_fp_background`: `13` images
- `close_fn`: `7` images
- `low_iou_tp_close`: `3` images
- `intake`: `20` images
- `remaining val`: `36` images

### Reports
- [close_fp_background_base8_20260410.csv](/c:/antigravity/data/reports/close_fp_background_base8_20260410.csv)
- [close_fn_base8_20260410.csv](/c:/antigravity/data/reports/close_fn_base8_20260410.csv)
- [low_iou_tp_close_base8_20260410.csv](/c:/antigravity/data/reports/low_iou_tp_close_base8_20260410.csv)
- [intake_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_y8s_close_repair_1/manifests/intake_manifest.csv)
- [val_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_y8s_close_repair_1/manifests/val_manifest.csv)

### Training
- base weights:
  - [best.pt](/c:/antigravity/data/7_experiments/exp_v072_base8/weights/best.pt)
- experiment:
  - [exp_v072_inc18](/c:/antigravity/data/7_experiments/exp_v072_inc18)
- mode:
  - incremental
- config:
  - [train_config_y8s_close_repair_1.yaml](/c:/antigravity/configs/train_config_y8s_close_repair_1.yaml)

### Result
- early stop at epoch `15`, best epoch = `3`
- strict pipeline eval:
  - `mAP50 = 0.9950`
  - `Precision = 1.0000`
  - `Recall = 1.0000`
- validation block on best weights:
  - `all`: `P=0.998`, `R=1.000`, `AP50=0.995`, `AP50-95=0.968`
  - `open`: `P=0.998`, `R=1.000`, `AP50=0.995`, `AP50-95=0.972`
  - `close`: `P=0.998`, `R=1.000`, `AP50=0.995`, `AP50-95=0.964`

### Promotion Decision
- result:
  - promoted to new [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- reason:
  - `Fitness` improved from `0.8965` to `0.9710`
- caution:
  - this round trained on a very small targeted intake (`20` images) and validated on the remaining `36` images from the same frozen-val source family
  - result is strong on this split, but should still be checked against video behavior and broader residual patterns before treating it as a stable deployment endpoint

## Edge False-Positive Suppression

### Goal
- define an edge-side suppression strategy so detector output is not used directly as a control signal
- reduce `open` flicker false positives in deployment scenarios

### Spec Document
- [edge_state_machine_spec.md](/c:/antigravity/task/edge_state_machine_spec.md)

### Included In Spec
- system layers:
  - detector
  - post-filter
  - temporal validator
  - state machine
- state diagram:
  - `UNKNOWN`
  - `CLOSED`
  - `OPENING_CANDIDATE`
  - `OPEN`
  - `CLOSING_CANDIDATE`
- parameter table
- input/output field definitions
- transition rules
- pseudocode for implementation

### Recommended Initial Deployment Policy
- `open` threshold higher than `close`
- ROI filtering enabled for fixed-camera scenes
- `3-of-5` persistence validation
- cooldown and timeout logic
- control layer consumes only stable state output, not raw detector boxes

## object_detect Edge Runtime Integration

### Goal
- implement the `edge_state_machine_spec` expectations inside `object_detect`
- keep the runtime practical for edge deployment instead of using raw per-frame detector output

### Implementation
- added state-machine runtime:
  - [state_machine.py](/c:/antigravity/object_detect/branchs_runtime/state_machine.py)
- updated cascade pipeline to emit stable edge outputs in addition to behavior events:
  - [cascade.py](/c:/antigravity/object_detect/branchs_runtime/cascade.py)
- updated runtime exports:
  - [__init__.py](/c:/antigravity/object_detect/branchs_runtime/__init__.py)
- rewrote edge entry CLI:
  - [detect.py](/c:/antigravity/object_detect/detect.py)

### Edge Runtime Features
- class-specific thresholds:
  - `open_conf = 0.80`
  - `close_conf = 0.55`
- geometry filtering:
  - minimum area ratio
  - aspect-ratio validation
- temporal validation:
  - `3-of-5` persistence
- stable state machine:
  - `UNKNOWN`
  - `CLOSED`
  - `OPENING_CANDIDATE`
  - `OPEN`
  - `CLOSING_CANDIDATE`
- cooldown and timeout:
  - `state_cooldown_ms = 2000`
  - `state_timeout_ms = 3000`
- output contract:
  - emits `edge_outputs` with `state`, `confidence`, `stable_frames`, `bbox_xyxy`, `cooldown_active`, `source_class`

### Validation
- CLI help verified:
  - [detect.py](/c:/antigravity/object_detect/detect.py)
- runtime import verified:
  - `BranchsPipeline`
  - `EdgeDoorStateMachine`
- synthetic frame smoke test:
  - pipeline completed without runtime error
  - emitted `edge_outputs = 0` on an empty black frame as expected

### Current Limitation
- [videos](/c:/antigravity/data/1_raw/videos) is currently empty
- no real video smoke test was run from `object_detect/detect.py` because there was no local source video available at execution time

## Edge Model Size Check

### Files Checked
- current promoted detector:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- object-detect classifier:
  - [best.pt](/c:/antigravity/object_detect/best.pt)

### Sizes
- `global_best.pt = 22,551,203 bytes` (`~21.5 MiB`)
- `object_detect/best.pt = 6,328,035 bytes` (`~6.0 MiB`)

### Deployment Assessment
- from storage perspective, both models easily fit on an `8 GB` edge module
- from RAM perspective, an `8 GB` edge module is still sufficient for this class of model under normal inference conditions
- deployment risk is more likely to come from:
  - runtime backend
  - preprocessing overhead
  - concurrent services
  - video resolution / fps target
  - not from raw `.pt` size alone

## Quantization Decision

### Current Decision
- quantization is **not required for capacity reasons**
- the current model already fits comfortably inside an `8 GB` edge target

### When Quantization Still Helps
- lower latency
- lower memory bandwidth
- lower power draw
- higher edge throughput

### Recommended Path
- first choice:
  - export to `TensorRT FP16` or `ONNX FP16`
- second choice:
  - `INT8` only if a calibration set is available and accuracy regression is acceptable

### Practical Note
- for this project, quantization should be treated as a deployment optimization step
- it should not be the first fix for current `open` false positives

## object_detect Edge Runtime Test

### Source
- video:
  - [1_close-call-cyclist-almost-hit-by-car-door-ytshorts.savetube.vip.mp4](/c:/antigravity/data/1_raw/videos/1_close-call-cyclist-almost-hit-by-car-door-ytshorts.savetube.vip.mp4)

### Runtime Entry
- [detect.py](/c:/antigravity/object_detect/detect.py)

### Models Used
- stage 1 detector:
  - [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- stage 2 classifier:
  - [best.pt](/c:/antigravity/object_detect/best.pt)

### Outputs
- annotated video:
  - [1_close-call_edge_runtime.mp4](/c:/antigravity/object_detect/outputs/1_close-call_edge_runtime.mp4)
- behavior events:
  - [1_close-call_edge_events.jsonl](/c:/antigravity/object_detect/outputs/1_close-call_edge_events.jsonl)
- stable edge states:
  - [1_close-call_edge_states.jsonl](/c:/antigravity/object_detect/outputs/1_close-call_edge_states.jsonl)

### Run Result
- completed frames:
  - `1744`
- runtime completed successfully
- per-frame state output was generated
- edge runtime exercised:
  - stabilization mode switching
  - detector + classifier cascade
  - temporal state machine output

## `box_refine_1` Execution

### User Review Decision
- `low_iou_tp`: all approved
- `close_fp_background`: approved, excluding `img_042.jpg` and `img_047.jpg` from the negative bucket
- `open_boundary_hard`: all approved

### Round Build
- Dataset yaml:
  - [dataset_incremental_box_refine_1.yaml](/c:/antigravity/data/incremental_rounds/2026-04-10_box_refine_1/dataset_incremental_box_refine_1.yaml)
- Config:
  - [train_config_box_refine_1_832.yaml](/c:/antigravity/configs/train_config_box_refine_1_832.yaml)
- Intake manifest:
  - [intake_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_box_refine_1/manifests/intake_manifest.csv)
- Val manifest:
  - [val_manifest.csv](/c:/antigravity/data/incremental_rounds/2026-04-10_box_refine_1/manifests/val_manifest.csv)
- Intake composition after dedupe:
  - `34` images total
  - `26` from `low_iou_tp`
  - `5` from `close_fp_background`
  - `3` from `open_boundary_hard`
- Validation split:
  - `20` images
  - `34` leakage candidates removed from prior val

### Training
- Base weights:
  - [best.pt](/c:/antigravity/data/7_experiments/exp_v072_base5/weights/best.pt)
- Experiment:
  - [exp_v072_inc14](/c:/antigravity/data/7_experiments/exp_v072_inc14)
- Mode:
  - incremental
- Key config:
  - `imgsz = 832`
  - `optimizer = AdamW`
  - `epochs = 24`
  - `lr0 = 0.0005`
  - `mosaic = 0.15`

### Result
- early stop at epoch `20`, best epoch = `8`
- strict validation summary from pipeline:
  - `mAP50 = 0.9675`
  - `Precision = 0.9877`
  - `Recall = 0.9500`
- validation block on saved best:
  - `all`: `P=0.986`, `R=0.950`, `AP50=0.983`, `AP50-95=0.755`
  - `open`: `P=0.995`, `R=1.000`, `AP50=0.995`, `AP50-95=0.818`
  - `close`: `P=0.976`, `R=0.900`, `AP50=0.972`, `AP50-95=0.691`

### Promotion Decision
- Result:
  - promoted to new [global_best.pt](/c:/antigravity/data/7_experiments/weight/global_best.pt)
- Reason:
  - `Fitness` improved from `0.6614` to `0.7744`
  - `Ghost` background rejection maintained or improved
