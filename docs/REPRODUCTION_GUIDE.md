# ML Project 2 – Reproduction Guide (Amazon SageMaker Studio)

**Project:** E-Commerce Delivery Risk Management (Olist) · **Author:** Ajay Uppu (GSU, MS Data Science)  
**AWS:** account 044450716277, region us-east-1 · domain `machinelearningproj2ajay-ecommerce` (d-hfxnlufab81e) · user profile `ajay-uppu` · JupyterLab space `mlproj2-ecommerce-space` (ml.m5.xlarge, 20 GB EBS) · kernel **Python (ml-project2-v2)**

This guide tells you exactly which notebook and which cell to run, in which order, what each cell does, and the output you should see. All expected values below are copied from the real execution inside the SageMaker Studio space (2026-11-15, 21:48–21:54 ET). If your numbers match, your run is a faithful reproduction.

## 1. What is already set up for you

| Resource | Value |
|---|---|
| Studio domain | machinelearningproj2ajay-ecommerce (IAM auth, default VPC, public internet access) |
| Domain URL | https://d-hfxnlufab81e.studio.us-east-1.sagemaker.aws |
| User profile | ajay-uppu |
| JupyterLab space | mlproj2-ecommerce-space – private, ml.m5.xlarge, 20 GB storage, no idle shutdown configured |
| Execution role | ml-project2-olist-sagemaker-exec (SageMaker full access + read/write on the project bucket) |
| Lifecycle config | mlproj2-ecommerce-sync-and-run-v2 – copies the project from S3 on first start; runs the pipeline only if logs/RUN_DONE is missing |
| Project folder in Studio | /home/sagemaker-user/ml-project2-v2 (shown as ml-project2-v2 in the file browser) |
| Data on S3 | s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/ (9 Olist CSVs) |
| Run results on S3 | s3://ml-project2-olist-044450716277/ml-project2-v2/studio-run/ |
| Deliverables on S3 | s3://ml-project2-olist-044450716277/ml-project2-v2/deliverables/ |

> The space is left running on purpose so you can browse the code. It costs about USD 0.23 per hour while running (ml.m5.xlarge, us-east-1). Stop it from the Studio JupyterLab page (Stop space) when you no longer need it; your files on the 20 GB volume are kept.

## 2. Open the project in Studio (5 clicks)

1. Sign in to the AWS console (account 044450716277) and switch the region (top right) to **US East (N. Virginia) us-east-1**.
1. Search for **Amazon SageMaker AI** → left menu **Admin configurations → Domains** → click `machinelearningproj2ajay-ecommerce`.
1. Tab **User profiles** → row `ajay-uppu` → **Launch** ▸ **Studio**. Studio opens in a new browser tab.
1. In Studio's left bar click **JupyterLab** → you will see the space `mlproj2-ecommerce-space`. If its status is *Stopped*, click **Run space** and wait about 1–2 minutes; then click **Open**.
1. In JupyterLab's file browser (left) double-click `ml-project2-v2` → `notebooks`. The seven notebooks already contain the outputs of the SageMaker run.

## 3. Prepare the environment (only needed for a fresh space)

The kernel **Python (ml-project2-v2)** and its virtual environment were created by the bootstrap script and live on the space's persistent volume, so normally you can go straight to section 4. On a new space or after deleting `.venv`, open **File → New → Terminal** and run:

```bash
cd ~/ml-project2-v2
bash aws/studio_bootstrap.sh --no-run   # venv + pinned packages + kernel + raw data from S3 (~2 min)
```

Expected last lines: `... environment ready` followed by `environment only` (the kernel is registered and the 9 CSV files are in `data/raw`). Then reload the browser tab so JupyterLab shows the new kernel.

If the project folder itself is missing (new space), copy it from S3 first:

```bash
aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/studio/ ~/ml-project2-v2/
```

## 4. Run order

Run the notebooks strictly in this order; each one saves files used by the next:

| # | Notebook | Purpose | Reads | Writes | Runtime* |
|---|---|---|---|---|---|
| 00 | `00_setup_environment.ipynb` | Environment setup | data/raw (S3 if missing) | – | <10 s |
| 01 | `01_data_audit.ipynb` | Raw data audit | data/raw | tables/01_*, figures/01_* | ~25 s |
| 02 | `02_labels_and_eligibility.ipynb` | Target definition and eligibility | data/raw | data/processed/labels.pkl, tables/02_* | ~15 s |
| 03 | `03_feature_engineering.ipynb` | Approval-time features and leakage checks | raw + labels.pkl | data/processed/features.pkl, tables/03_* | ~60 s |
| 04 | `04_modeling.ipynb` | Rules, models and validation-based selection | features.pkl | models/*.joblib, experiment_configs.json, tables/04_* | ~2–3 min |
| 05 | `05_daily_policy_evaluation.ipynb` | Final test: daily top-10 simulation | features.pkl, models | daily_prioritization_test.csv, tables/05_* | ~20 s |
| 06 | `06_sensitivity_diagnostics.ipynb` | Sensitivity analysis and diagnostics | features.pkl, models | tables/06_*, figures/06_* | ~60 s |

\*Measured on ml.m5.large in the original run (whole pipeline 283 s); ml.m5.xlarge is a bit faster.

For every notebook: open it → check the kernel name at the top right says **Python (ml-project2-v2)** (click it to change) → **Kernel → Restart Kernel and Clear Outputs of All Cells** (optional, for a clean run) → run the cells one by one with **Shift + Enter** and compare with the expected output below. **Run → Run All Cells** also works.

Shortcut – run all seven notebooks in one go from a terminal (the notebooks are overwritten with fresh outputs):

```bash
cd ~/ml-project2-v2 && .venv/bin/python run_all.py
```

## 5. Cell-by-cell guide

### Notebook 00 – Environment setup (`00_setup_environment.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/00_setup_environment__cell01.png`

**Cell 2 – Python and library versions actually used by this kernel**

- *What it does:* Prints the Python version and compares every library pinned in requirements.txt with the version installed in the kernel.
- *Why:* Results are only reproducible with the same library versions (e.g. scikit-learn 1.6.1, LightGBM 4.6.0).
- *Expected output:* python 3.12.x and a table where every row has ok = True. Any False means the kernel is not 'Python (ml-project2-v2)'.
- *Screenshot:* `screenshots/00_setup_environment__cell02.png`

**Cell 3 – install the pinned versions only if something is missing/different**

- *What it does:* Installs the pinned versions only when something differs.
- *Why:* Safety net for a fresh kernel; it never touches packages that are already correct.
- *Expected output:* 'All pinned packages already installed - nothing to do.' If it installs, restart the kernel and run the notebook again.
- *Screenshot:* `screenshots/00_setup_environment__cell03.png`

**Cell 4 – make sure the raw CSVs are present (download from S3 if missing)**

- *What it does:* Checks that the 9 raw CSV files are in data/raw/ and downloads any missing file from s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/.
- *Why:* All later notebooks read the raw files from this folder; the dataset itself is never stored in the code package.
- *Expected output:* 'missing before download: []' (the files were already copied by the bootstrap). On a fresh machine you will see 'downloaded …' lines.
- *Screenshot:* `screenshots/00_setup_environment__cell04.png`

**Cell 5 – verify the files: size and number of data rows**

- *What it does:* Counts rows and file sizes of the 9 tables and asserts the official Olist row counts.
- *Why:* Proves you are using the unmodified dataset (orders 99,441; items 112,650; payments 103,886; geolocation 1,000,163).
- *Expected output:* A 9-row table; the reviews line count (104,719) is larger than its 99,224 records because review comments contain line breaks.
- *Screenshot:* `screenshots/00_setup_environment__cell05.png`

**Cell 6 – key configuration used by all notebooks**

- *What it does:* Prints the configuration used by every notebook: seed, split dates, label-maturity window, daily capacity, budgets.
- *Why:* These are the key design decisions; changing them here (in src/olist_risk/config.py) changes the whole experiment.
- *Expected output:* SEED = 42, TRAIN_START = 2017-01-01, VAL_START = 2018-03-01, TEST_START = 2018-06-01, TEST_END = 2018-09-01, LABEL_MATURITY_DAYS = 60, DAILY_CAPACITY = 10.
- *Screenshot:* `screenshots/00_setup_environment__cell06.png`

### Notebook 01 – Raw data audit (`01_data_audit.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/01_data_audit__cell01.png`

**Cell 2 – load all raw tables (dates parsed, zip prefixes kept as 5-character strings)**

- *What it does:* Loads all 9 CSVs with dates parsed and zip prefixes kept as 5-character strings.
- *Why:* Zip prefixes like 01001 would lose the leading zero if read as numbers and would then not match the geolocation table.
- *Expected output:* orders (99441, 8), items (112650, 7), customers (99441, 5), sellers (3095, 4), products (32951, 9), payments (103886, 5), reviews (99224, 7), geolocation (1000163, 5), translation (71, 2).
- *Screenshot:* `screenshots/01_data_audit__cell02.png`

**Cell 3 – table-level audit: primary keys, duplicates, missing cells**

- *What it does:* Table-level audit: primary-key uniqueness, exact duplicate rows and missing cells per table.
- *Why:* Tells us which keys are safe to join on and where missing values must be handled.
- *Expected output:* All primary keys unique; geolocation has 261,831 exact duplicate rows and no key; orders has 4,908 missing cells (approval/delivery dates).
- *Screenshot:* `screenshots/01_data_audit__cell03.png`

**Cell 4 – relationships: which joins are 1:1 and which would multiply orders**

- *What it does:* Relationship audit: how many rows per order each table has.
- *Why:* Joining a one-to-many table directly would duplicate orders and silently over-weight multi-item orders.
- *Expected output:* 9,803 orders have >1 item row (max 21), 2,961 have >1 payment row, 1,278 have >1 seller, geolocation median 29 rows per zip prefix.
- *Screenshot:* `screenshots/01_data_audit__cell04.png`

**Cell 5 – order lifecycle timestamps: missing values and inconsistencies**

- *What it does:* Checks the order lifecycle timestamps for missing and inconsistent values.
- *Why:* Defines which outcomes are usable and shows that post-approval dates (carrier pickup) are noisy.
- *Expected output:* 160 orders without approval, 2,965 without customer delivery, 0 promised dates with a time of day (all midnight), approvals from 2016-09-15 to 2018-09-03.
- *Screenshot:* `screenshots/01_data_audit__cell05.png`

**Cell 6 – order status distribution (final status at data export - an OUTCOME, never a feature)**

- *What it does:* Distribution of the final order status.
- *Why:* The status is the END of the lifecycle, i.e. an outcome – it can never be a predictor.
- *Expected output:* delivered 96,478 (97.0%), shipped 1,107, canceled 625, unavailable 609, invoiced 314, processing 301.
- *Screenshot:* `screenshots/01_data_audit__cell06.png`

**Cell 7 – monthly volume of approved orders and the share delivered late (preview of the target)**

- *What it does:* Plots approved orders and the late rate per approval month (Figure 01) with the split boundaries.
- *Why:* Shows the biggest risk for modelling: the late rate drifts (≈3% in 2017, 19% in Mar 2018, 1% in Jun 2018).
- *Expected output:* The bar+line chart and a table; late_rate_% 12.23 in 2017-11, 13.63 in 2018-02, 19.2 in 2018-03, 1.17 in 2018-06, 6.27 in 2018-08.
- *Screenshot:* `screenshots/01_data_audit__cell07.png`

**Cell 8 – basket structure: items, sellers and payments per order (why we aggregate before joining)**

- *What it does:* Describes items, sellers and payments per order and demonstrates the naive join.
- *Why:* Evidence for the 'aggregate before join' rule required by the assignment.
- *Expected output:* Naive orders×items×payments join: 117,601 rows for 98,665 orders → 18,936 duplicated order rows.
- *Screenshot:* `screenshots/01_data_audit__cell08.png`

**Cell 9 – promised delivery window and actual delivery time (days) for delivered orders**

- *What it does:* Histogram of promised days vs actual delivery days.
- *Why:* Promises are generous on average (median 23 days promised vs 10 actual), so lateness is a tail event.
- *Expected output:* Figure 02; promised mean 23.9 days, actual mean 12.0 days.
- *Screenshot:* `screenshots/01_data_audit__cell09.png`

**Cell 10 – reviews are written after delivery: proof that they cannot be approval-time features**

- *What it does:* Compares review creation dates with approval dates.
- *Why:* Proves reviews are post-outcome information; using them would be leakage.
- *Expected output:* 33 reviews before approval vs 99,035 after; median 0 days after delivery → reviews are never joined.
- *Screenshot:* `screenshots/01_data_audit__cell10.png`

### Notebook 02 – Target definition and eligibility (`02_labels_and_eligibility.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell01.png`

**Cell 2 – build labels and outcome status for every order**

- *What it does:* Builds is_late, outcome_status (late / on_time / undetermined), days_late and outcome_known_at for every order, plus split and eligibility flags.
- *Why:* This is the label construction step of the assignment; undelivered orders get no label instead of being guessed.
- *Expected output:* on_time 89,941, late 6,535, undetermined 2,965; late rate among delivered orders 0.0677.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell02.png`

**Cell 3 – why calendar dates and not raw timestamps?**

- *What it does:* Compares the timestamp-based and calendar-date-based late definitions.
- *Why:* The promised date is stored at 00:00, so a delivery at 15:00 on the promised day would wrongly count as late.
- *Expected output:* timestamp late rate 0.0811 vs calendar-date late rate 0.0677; 1,292 orders delivered on the promised day.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell03.png`

**Cell 4 – outcome status by final order status**

- *What it does:* Cross-tab of final status vs outcome status.
- *Why:* Shows which statuses have usable outcomes (6 canceled orders actually have a delivery date).
- *Expected output:* delivered → 6,534 late / 89,936 on time / 8 undetermined; shipped, invoiced, processing, unavailable are all undetermined.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell04.png`

**Cell 5 – chronological splits on the approval timestamp + eligibility funnel**

- *What it does:* Assigns chronological splits by approval date and prints the eligibility funnel.
- *Why:* Documents exactly how many orders are candidates, labelled, late, and usable for training.
- *Expected output:* train 58,866 / validation 21,132 / test 18,960 approved orders; 43,276 rows usable for fitting; 64,109 for the final refit.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell05.png`

**Cell 6 – label maturity: how long after approval do outcomes become known?**

- *What it does:* Measures how long after approval outcomes become known.
- *Why:* Justifies the 60-day label-maturity window (late outcomes take longer to be observed).
- *Expected output:* After 30 days 96% of all but only 51% of late outcomes are known; after 60 days 99.7% / 95.7%.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell06.png`

**Cell 7 – censoring bias illustrated: late rate of train orders by month, with and without the maturity rule**

- *What it does:* Shows the censoring bias near the cutoff by month.
- *Why:* Without the maturity rule, February 2018 training orders would look almost never late.
- *Expected output:* 2018-02: all delivered late rate 0.1363 vs 0.0047 for outcomes known by the cutoff; Jan–Feb 2018 not used for fitting.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell07.png`

**Cell 8 – split summary (the 'modeling examples and outcomes' evidence table)**

- *What it does:* Split summary table (modeling examples and outcomes).
- *Why:* Required supporting evidence: per split, dates, days, candidates, late/on-time/undetermined, late rate.
- *Expected output:* train 420 days, late rate 0.0657; validation 89 days, 0.1026; test 88 days (2018-06-01 to 2018-08-29), 0.0360.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell08.png`

**Cell 9 – distribution of days late (late orders only)**

- *What it does:* Histogram of how many days late the late orders were.
- *Why:* Context for the business: a typical late order is about a week late.
- *Expected output:* median 7 days, mean 10.6 days, max 188.
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell09.png`

**Cell 10 – save labels for the next notebooks**

- *What it does:* Saves the labels to data/processed/labels.pkl.
- *Why:* Notebook 03 builds features on top of these labels.
- *Expected output:* saved …/data/processed/labels.pkl (99441, 19).
- *Screenshot:* `screenshots/02_labels_and_eligibility__cell10.png`

### Notebook 03 – Approval-time features and leakage checks (`03_feature_engineering.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/03_feature_engineering__cell01.png`

**Cell 2 – load raw tables and labels**

- *What it does:* Loads the raw tables and the labels from notebook 02.
- *Why:* Features are always built from raw data so the logic is traceable.
- *Expected output:* (99441, 19).
- *Screenshot:* `screenshots/03_feature_engineering__cell02.png`

**Cell 3 – build the order-level feature table (one row per order)**

- *What it does:* Builds the order-level feature table with build_feature_table() and asserts one row per order.
- *Why:* Implements aggregation-before-join and the leakage-safe historical aggregates.
- *Expected output:* feature table (99441, 73); 'one row per order_id: OK | predictors: 49 (45 numeric, 4 categorical)'.
- *Screenshot:* `screenshots/03_feature_engineering__cell03.png`

**Cell 4 – feature-availability audit (also saved for the report)**

- *What it does:* Writes the feature-availability audit table.
- *Why:* Required deliverable: for every field, its source, when it is observed and whether it is used.
- *Expected output:* A table listing timing, basket, payment, geography and history features as used, and order_status, carrier date, reviews as excluded.
- *Screenshot:* `screenshots/03_feature_engineering__cell04.png`

**Cell 5 – leakage unit test: recompute seller/customer history by brute force for 300 random orders**

- *What it does:* Leakage unit test: recomputes seller and customer history for 300 random orders by brute force.
- *Why:* Independent proof that history features use only outcomes delivered before approval and never the order's own outcome.
- *Expected output:* All four checks = 1.0 (100%).
- *Screenshot:* `screenshots/03_feature_engineering__cell05.png`

**Cell 6 – missing values per predictor (handled by median/constant imputation + missing indicators)**

- *What it does:* Missing-value rates per predictor.
- *Why:* Shows that missingness is small (≤1.1%) and handled by imputation + indicators.
- *Expected output:* distance_km 1.12%, seller coordinates 0.84%, basket features 0.63% (orders without items).
- *Screenshot:* `screenshots/03_feature_engineering__cell06.png`

**Cell 7 – univariate signal: late rate by decile of selected features (training rows only)**

- *What it does:* Late rate by decile for six features on training rows (Figure 03).
- *Why:* Quick sanity check that features carry signal before modelling.
- *Expected output:* Six small line charts; late rate rises with distance and route/seller late rate, falls with promised days.
- *Screenshot:* `screenshots/03_feature_engineering__cell07.png`

**Cell 8 – late rate by customer state (top 12 by volume, training rows)**

- *What it does:* Late rate by customer state in training data.
- *Why:* Geography matters strongly: RJ ≈10%, BA ≈10% vs SP ≈3.8%, MG ≈2.7%.
- *Expected output:* A two-row table (orders, late_rate) for the 12 largest states.
- *Screenshot:* `screenshots/03_feature_engineering__cell08.png`

**Cell 9 – descriptive statistics of the numeric predictors in the daily pool**

- *What it does:* Descriptive statistics of all numeric predictors.
- *Why:* Documents ranges and outliers (e.g. a few negative promised windows due to late approvals).
- *Expected output:* A 45-row describe() table.
- *Screenshot:* `screenshots/03_feature_engineering__cell09.png`

**Cell 10 – save the modeling table**

- *What it does:* Saves data/processed/features.pkl.
- *Why:* Notebooks 04–06 read this file.
- *Expected output:* saved …/data/processed/features.pkl (99441, 73).
- *Screenshot:* `screenshots/03_feature_engineering__cell10.png`

### Notebook 04 – Rules, models and validation-based selection (`04_modeling.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/04_modeling__cell01.png`

**Cell 2 – load features, define training and validation sets**

- *What it does:* Loads the features and defines the training set (mature labels before 2018-03-01) and the validation pool.
- *Why:* Training rows must have outcomes known at the cutoff; the validation pool contains all approved orders, like real operations.
- *Expected output:* train rows 43,276 (late rate 0.0562) approved 2017-01-05 .. 2017-12-30; validation pool 21,132 orders on 89 days; labelled 20,662 (late rate 0.1026).
- *Screenshot:* `screenshots/04_modeling__cell02.png`

**Cell 3 – business rules and a random reference on validation (same daily top-10 policy)**

- *What it does:* Evaluates the four business rules and random selection with the daily top-10 policy on validation.
- *Why:* Non-ML baselines required by the assignment; the best one becomes the reference rule.
- *Expected output:* longest distance 202 late identified, worst seller history 123, freight ratio 119, shortest promise 46, random 87.8 → best rule: rule_longest_distance.
- *Screenshot:* `screenshots/04_modeling__cell03.png`

**Cell 4 – hyper-parameter and feature-set search for both learned models (fit on train, score validation)**

- *What it does:* Grid search: 2 feature sets × (6 logistic-regression + 12 LightGBM configurations), each fit on train and scored on validation.
- *Why:* Model, hyper-parameter and feature-set selection use validation only; the test period is untouched.
- *Expected output:* '36 configurations evaluated' and the top-10 table led by logreg no_market C=0.01 (256 late identified). Takes 2–3 minutes.
- *Screenshot:* `screenshots/04_modeling__cell04.png`

**Cell 5 – best configuration per model family (selected on validation only)**

- *What it does:* Picks the best configuration per model family.
- *Why:* Selection criterion = late orders captured by the daily top-10 (the business metric), tie-break PR-AUC.
- *Expected output:* logreg (no_market, C=0.01) 256 / PR-AUC 0.2008; lgbm (all, 7 leaves, 200 trees, min_child_samples 400) 227 / 0.1956 → proposed model: logreg.
- *Screenshot:* `screenshots/04_modeling__cell05.png`

**Cell 6 – validation comparison table: rules vs learned models under the same daily cap**

- *What it does:* Validation comparison table of rules and learned models.
- *Why:* Main validation evidence for the report (Table 6).
- *Expected output:* logreg 256 > lgbm 227 > longest distance 202 > … > random 87.8 > shortest promise 46.
- *Screenshot:* `screenshots/04_modeling__cell06.png`

**Cell 7 – final refit of the selected configurations on train+validation (labels mature before TEST_START)**

- *What it does:* Refits the selected configurations on train+validation rows with labels mature before 2018-06-01 and saves the models.
- *Why:* Uses the most recent data available at the start of the test period without touching test outcomes.
- *Expected output:* refit rows 64,109 (late rate 0.0796), approved up to 2018-04-01; four .joblib files saved.
- *Screenshot:* `screenshots/04_modeling__cell07.png`

**Cell 8 – experiment configuration record (traceability)**

- *What it does:* Writes experiment_configs.json (splits, grids, selected parameters, library versions).
- *Why:* Required supporting evidence: experiment configuration and traceability.
- *Expected output:* The selected block for logreg and lgbm printed as JSON.
- *Screenshot:* `screenshots/04_modeling__cell08.png`

### Notebook 05 – Final test: daily top-10 simulation (`05_daily_policy_evaluation.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell01.png`

**Cell 2 – load features, configuration and final models**

- *What it does:* Loads features, the experiment configuration and the two final models; builds the test pool.
- *Why:* This is the first and only time the test period is used.
- *Expected output:* test pool: 18,960 orders, 88 days, labelled 18,645, late 672, undetermined 315; proposed model: logreg | reference rule: rule_longest_distance.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell02.png`

**Cell 3 – score the test orders with every method**

- *What it does:* Scores every test order with both models and all rules.
- *Why:* All methods are scored with approval-time features only.
- *Expected output:* First rows of order_id, approval_date, score_logreg, score_lgbm, promised_days, distance_km.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell03.png`

**Cell 4 – operational results under the 10-per-day cap**

- *What it does:* Runs the daily top-10 simulation for every method and random (200 draws).
- *Why:* Core operational result: identified late orders, unnecessary investigations, undetermined, missed.
- *Expected output:* logreg 70 identified / 780 unnecessary / 19 undetermined / 602 missed; lgbm 58; shortest promise 80; longest distance 34; random 26.45; 869 investigations each; perfect foresight 437.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell04.png`

**Cell 5 – predictive metrics on the labelled test orders**

- *What it does:* Predictive metrics on labelled test orders.
- *Why:* PR-AUC/ROC-AUC are threshold-free and suited to the 3.6% positive rate.
- *Expected output:* logreg PR-AUC 0.0675, ROC-AUC 0.6244; lgbm 0.0522 / 0.6234; shortest-promise rule 0.1309 / 0.7459.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell05.png`

**Cell 6 – headline: proposed model vs reference rule vs random**

- *What it does:* Headline table: proposed model vs reference rule vs random.
- *Why:* The comparison requested by the assignment under the same capacity.
- *Expected output:* 'logreg: 70 late orders identified vs 34 for rule_longest_distance and 26.4 expected at random, using 869 investigations.'
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell06.png`

**Cell 7 – paired daily comparison: on how many days did the model beat / tie / lose to the rule?**

- *What it does:* Paired daily comparison with a day-level bootstrap.
- *Why:* Quantifies uncertainty of the model-vs-rule difference.
- *Expected output:* model better on 30 days, equal on 52, worse on 6; +36 late orders (95% CI 20 .. 55).
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell07.png`

**Cell 8 – cumulative late orders identified over the test period**

- *What it does:* Cumulative late orders identified over the test period (Figure 05a).
- *Why:* Visual comparison with random and the perfect-foresight ceiling.
- *Expected output:* Line chart; the shortest-promise rule ends highest, logreg second, random and longest distance lowest.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell08.png`

**Cell 9 – precision-recall curves on labelled test orders**

- *What it does:* Precision–recall curves (Figure 05b).
- *Why:* Shows ranking quality across all thresholds.
- *Expected output:* Three curves with AP values and the base-rate line at 0.036.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell09.png`

**Cell 10 – save the daily prioritisation list for the test period (traceable to order_id + approval date)**

- *What it does:* Saves outputs/daily_prioritization_test.csv (and the rule's list).
- *Why:* Required deliverable: every prioritised record traceable to order_id and approval date.
- *Expected output:* (869, 18) rows saved; first rows for 2018-06-01 ranks 1–10.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell10.png`

**Cell 11 – how were days with fewer than 10 orders and undetermined outcomes handled?**

- *What it does:* Reports how days with <10 orders and undetermined outcomes were handled.
- *Why:* Required by the assignment (section 4D).
- *Expected output:* 88 days, 2 days with <10 approved orders, 869 of 880 possible investigations, 19 selected orders undetermined.
- *Screenshot:* `screenshots/05_daily_policy_evaluation__cell11.png`

### Notebook 06 – Sensitivity analysis and diagnostics (`06_sensitivity_diagnostics.ipynb`)

**Cell 1 – make the project package importable and define folders**

- *What it does:* Adds the project's src/ folder to Python's import path, sets PROJECT_ROOT, imports pandas/numpy/matplotlib and creates the outputs/ folders (tables, figures, models) if they do not exist.
- *Why:* Every notebook starts with this same cell so the notebooks can be run independently and always write to the same place.
- *Expected output:* Two lines: PROJECT_ROOT = /home/sagemaker-user/ml-project2-v2 and outputs -> .../outputs. If you see ModuleNotFoundError: olist_risk, you are not running the notebook from inside ml-project2-v2/notebooks.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell01.png`

**Cell 2 – reload test pool and scores**

- *What it does:* Reloads the test pool and recomputes all scores.
- *Why:* Keeps notebook 06 independent of notebook 05's memory.
- *Expected output:* logreg rule_longest_distance (18960, 79).
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell02.png`

**Cell 3 – investigation budget sensitivity**

- *What it does:* Budget sensitivity for k = 5, 10, 20, 50 investigations per day (Figure 06a).
- *Why:* Shows the trade-off between capacity and precision.
- *Expected output:* logreg 45 / 70 / 105 / 181 late identified; precision 0.103 / 0.081 / 0.061 / 0.042.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell03.png`

**Cell 4 – performance by month of the test period (10/day)**

- *What it does:* Performance by test month.
- *Why:* Detects time periods where the model fails.
- *Expected output:* June lift 3.8, July 4.4, August 0.99 (ROC-AUC 0.51) for logreg.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell04.png`

**Cell 5 – failure groups: where are late orders missed? (customer state)**

- *What it does:* Failure groups by customer state.
- *Why:* Shows where late orders are missed and where false alarms concentrate.
- *Expected output:* SP: 405 late, 11 identified (2.7%); RJ: 381 selections, 352 false alarms; MA: 13 of 14 identified.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell05.png`

**Cell 6 – failure groups by product category (top 15 by late orders)**

- *What it does:* Failure groups by product category.
- *Why:* Second grouping for error analysis.
- *Expected output:* health_beauty 32 of 121 identified (26%); housewares 1 of 47.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell06.png`

**Cell 7 – what do missed late orders look like compared with caught ones?**

- *What it does:* Profiles caught, missed and false-alarm orders.
- *Why:* Explains why misses happen: short routes with good history look safe at approval.
- *Expected output:* median distance 1,670 km (hits) vs 109 km (missed) vs 808 km (false alarms).
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell07.png`

**Cell 8 – model drivers: logistic-regression coefficients and LightGBM gain importance**

- *What it does:* Model drivers: logistic-regression coefficients and LightGBM gain importance (Figure 06b).
- *Why:* Interpretability and a reason why LightGBM generalised worse (relies on market volume and state).
- *Expected output:* Two bar charts.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell08.png`

**Cell 9 – calibration of the predicted probabilities (labelled test orders)**

- *What it does:* Calibration by decile (Figure 06c).
- *Why:* Checks whether scores can be read as probabilities.
- *Expected output:* Mean predicted 0.189 (logreg) vs observed 0.036 – over-predicted in the low-risk test period; ranking still works.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell09.png`

**Cell 10 – refit on train+validation vs the validation-period model (no refit)**

- *What it does:* Refit on train+validation vs model trained on train only.
- *Why:* Justifies the final refit.
- *Expected output:* logreg 62 → 70; lgbm 57 → 58 late identified.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell10.png`

**Cell 11 – feature-group ablation for the proposed model family (refit on train+validation, test)**

- *What it does:* Feature-group ablation (diagnostic only).
- *Why:* Which feature groups carry the signal.
- *Expected output:* all 67, no history 69, no market 70, history only 31.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell11.png`

**Cell 12 – investigate fewer than 10 when risk is low? precision of the daily rank positions**

- *What it does:* Precision by daily rank position.
- *Why:* Should the team investigate fewer than 10 on some days?
- *Expected output:* rank 1 late rate 13.6%, rank 2 12.5%, ranks 6–10 ≈5–7%, ranks 11–20 mostly ≤6%.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell12.png`

**Cell 13 – post-hoc: slack features (validation first, then test)**

- *What it does:* Post-hoc slack features (clearly labelled as test-informed).
- *Why:* Tests the hypothesis suggested by the strong shortest-promise rule without changing the reported results.
- *Expected output:* logreg + slack 251 (val) / 72 (test) vs 256 / 70; slack rule 116 (val) / 73 (test).
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell13.png`

**Cell 14 – summary of all saved outputs**

- *What it does:* Lists every saved output file with its size.
- *Why:* Final check that all deliverables were produced.
- *Expected output:* daily_prioritization_test.csv, experiment_configs.json, figures, models, tables.
- *Screenshot:* `screenshots/06_sensitivity_diagnostics__cell14.png`

## 6. Checklist of key numbers

| Check | Expected |
|---|---|
| Orders / approved / delivered status | 99,441 / 99,281 / 96,478 |
| Late rate among delivered (calendar dates) | 6.77% (8.11% with raw timestamps) |
| Train / validation / test candidate orders | 58,866 / 21,132 / 18,960 |
| Rows used for fitting (train) / final refit | 43,276 / 64,109 |
| Best rule on validation | longest distance – 202 late orders identified |
| Selected model | logistic regression, C = 0.01, no class weights, feature set no_market (256 on validation) |
| Test – logistic regression (10/day) | 70 late identified, 780 unnecessary, 19 undetermined, 602 missed (869 investigations) |
| Test – LightGBM / longest-distance rule / random | 58 / 34 / 26.45 |
| Test – shortest-promise rule (worst on validation) | 80 |
| Perfect-foresight maximum | 437 |
| Rows in daily_prioritization_test.csv | 869 |

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'olist_risk'` | Notebook not opened from ml-project2-v2/notebooks, or wrong kernel | Open the notebook from that folder; select kernel Python (ml-project2-v2); re-run Cell 1. |
| Kernel 'Python (ml-project2-v2)' not in the list | Bootstrap not run on this space | Run `bash ~/ml-project2-v2/aws/studio_bootstrap.sh --no-run`, then reload the browser tab. |
| Cell 2 of notebook 00 shows ok = False | Notebook uses the default Python 3 kernel with other versions | Switch kernel; or run Cell 3, restart kernel, run again. |
| `FileNotFoundError: data/raw/olist_orders_dataset.csv` | Raw data not downloaded | Run notebook 00 Cell 4, or `aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/ ~/ml-project2-v2/data/raw/`. |
| `AccessDenied` from S3 | Space not using the execution role ml-project2-olist-sagemaker-exec | Check the domain/user execution role in the console; the role needs s3:GetObject/PutObject/ListBucket on the bucket. |
| `FileNotFoundError: data/processed/features.pkl` in notebook 04–06 | Notebooks run out of order | Run notebooks 02 and 03 first. |
| Kernel dies / MemoryError in notebook 03 | Very small instance (e.g. ml.t3.medium, 4 GB) | Use ml.m5.large or larger (the space uses ml.m5.xlarge, 16 GB). |
| Notebook 04 seems stuck | 36-configuration grid search runs ~2–3 minutes | Wait; the cell prints when finished. |
| Slightly different 4th-decimal values | Floating-point differences between CPUs/BLAS builds | Counts (late identified etc.) should still match exactly; small metric differences are expected. |
| Space will not start: ResourceLimitExceeded | Instance quota for Studio JupyterLab apps | Choose ml.t3.xlarge in the space settings, or request a quota increase in Service Quotas. |
| Lifecycle script re-runs the pipeline on start | logs/RUN_DONE was deleted | Harmless (results are rewritten); recreate the file to prevent it: `date > ~/ml-project2-v2/logs/RUN_DONE`. |

## 8. Studio screenshots you can add yourself

The per-cell screenshots in `screenshots/` were rendered from the notebooks executed in the space. Screens of the AWS console/Studio UI need your console sign-in: (1) Domains list showing machinelearningproj2ajay-ecommerce; (2) the domain's User profiles tab with ajay-uppu and Launch ▸ Studio; (3) Studio's JupyterLab page with mlproj2-ecommerce-space (Running, ml.m5.xlarge); (4) the JupyterLab file browser showing ml-project2-v2/notebooks; (5) a notebook open with the kernel Python (ml-project2-v2); (6) a Terminal running the bootstrap.