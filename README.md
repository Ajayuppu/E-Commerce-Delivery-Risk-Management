# ML Project 2 – E-Commerce Delivery Risk Management (Olist)

**Author:** Ajay Uppu (GSU, MS Data Science) · **Environment:** Amazon SageMaker Studio, domain
`machinelearningproj2ajay-ecommerce` (us-east-1), user profile `ajay-uppu`, JupyterLab space
`mlproj2-ecommerce-space`.

The project turns the business requirement *"identify newly approved orders that are at risk of late
delivery; the logistics team can investigate at most 10 orders per day"* into a leakage-safe ML task,
trains a logistic-regression baseline and a LightGBM model, compares them with non-ML business rules
under the same 10-per-day capacity, and recommends an operating policy.

## Key results (test period Jun–Aug 2018, 88 days, 18,960 approved orders, 672 late)
Each method investigates the top 10 orders per day (869 investigations; 2 days had fewer than 10 orders).

| Method | Late identified | Unnecessary | Undetermined selected | Late missed | Precision |
|---|---|---|---|---|---|
| **Logistic regression (proposed, chosen on validation)** | **70** | 780 | 19 | 602 | 8.1% |
| LightGBM | 58 | 796 | 15 | 614 | 6.7% |
| Rule: longest distance (best rule on validation) | 34 | 823 | 12 | 638 | 3.9% |
| Rule: shortest promised window | 80 | 763 | 26 | 592 | 9.2% |
| Random selection (mean of 200 draws) | 26.45 | 828.05 | 14.5 | 645.55 | 3.0% |

* Validation (Mar–May 2018): logistic regression 256 late orders identified, LightGBM 227, longest-distance rule 202, random 87.8, shortest-promise rule 46.
* The model found 2.27× more late orders than random and +36 vs the reference rule (95% day-bootstrap CI 20–55). Perfect foresight could have found 437.
* Honest caveat: the shortest-promise rule (worst on validation) was best on test – a sign of drift (late rate 10.3% in validation vs 3.6% in test). It was not swapped in after the fact; see the report's diagnostics.
* Test PR-AUC / ROC-AUC: logistic regression 0.068 / 0.624 (base rate 0.036).

## Problem formulation (short)
| Item | Decision |
|---|---|
| Unit of prediction | one order (`order_id`) |
| Decision time | `order_approved_at` |
| Target `is_late` | 1 if the calendar date of `order_delivered_customer_date` > calendar date of `order_estimated_delivery_date` |
| Undetermined outcome | no customer-delivery date (canceled, unavailable, still in transit): never trained on, kept in the daily candidate pool, reported separately |
| Features | only information observable at approval; history aggregates use outcomes **delivered before** approval |
| Splits (approval date) | train 2017-01-01→2018-02-28 · validation 2018-03-01→2018-05-31 · test 2018-06-01→2018-08-31 |
| Label maturity | training rows must be approved ≥60 days before the cutoff and have their outcome observed before it |
| Policy | each day, rank that day's approved orders by risk and investigate the top 10 (all of them if fewer) |

## Repository layout
```
E-Commerce-Delivery-Risk-Management/
├── README.md               this file
├── requirements.txt        pinned library versions (verified on the box and in SageMaker Studio)
├── run_all.py              executes notebooks 00→06 in order with papermill
├── src/olist_risk/         the pipeline as an importable package
│   ├── config.py           paths, split dates, capacity, seeds
│   ├── data.py             raw loading + table/relationship/date audits
│   ├── labels.py           target, outcome status, eligibility, splits
│   ├── features.py         approval-time features, leakage-safe history, audit table, brute-force leakage test
│   ├── models.py           logistic regression + LightGBM pipelines and grids
│   ├── policy.py           business rules, daily top-k selection, operational metrics
│   └── metrics.py          PR-AUC, ROC-AUC, Brier, precision/recall@top-x%
├── notebooks/              00_setup_environment … 06_sensitivity_diagnostics (run in order)
├── aws/                    Studio lifecycle config + bootstrap script used for the SageMaker run
├── tools/                  notebook/report/guide builders, per-cell screenshot renderer
├── docs/                   report, reproduction guide, tutorial, execution guide
├── screenshots/            one PNG per executed notebook cell
├── data/raw/               the 9 Olist CSVs (NOT in this repo – get them from Kaggle or S3)
├── data/processed/         labels.pkl, features.pkl (created by notebooks 02/03)
└── outputs/                tables/*.csv, figures/*.png, models/*.joblib,
                            experiment_configs.json, daily_prioritization_test.csv
```

## How to reproduce
### A. In SageMaker Studio (recommended – this is how the submitted results were produced)
1. AWS console → **Amazon SageMaker AI** → **Domains** → `machinelearningproj2ajay-ecommerce` → user profile
   `ajay-uppu` → **Launch → Studio**.
2. In Studio choose **JupyterLab** → open the space `mlproj2-ecommerce-space` (ml.m5.xlarge) → **Open JupyterLab**
   (click **Run space** first if it is stopped).
3. In the file browser open `ml-project2-v2/`. The notebooks there already contain the outputs of the Studio run.
4. To re-run: open a Terminal (File → New → Terminal) and run
   `bash ~/ml-project2-v2/aws/studio_bootstrap.sh --no-run` (creates the `.venv`, installs `requirements.txt`,
   registers the kernel **Python (ml-project2-v2)** and downloads the raw CSVs from S3).
   If the project folder is missing in a new space, copy the code bundle from S3 first:
   `aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/studio/ ~/ml-project2-v2/`.
   To copy the finished documentation (report, guides, tutorial) into the Studio space:
   `aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/deliverables/docs/ ~/ml-project2-v2/docs/`.
   Alternatively clone this repository in a Studio terminal: `git clone https://github.com/Ajayuppu/E-Commerce-Delivery-Risk-Management.git ~/ml-project2-v2`
   and put the 9 Olist CSVs into `data/raw/`.
5. Open each notebook in order (00 → 06), make sure the kernel at the top right is **Python (ml-project2-v2)**,
   and use **Run → Run All Cells**. Or run everything at once from the terminal:
   `cd ~/ml-project2-v2 && .venv/bin/python run_all.py` (≈5 min; the submitted run took 283 s for all notebooks on ml.m5.large, the space was afterwards switched to ml.m5.xlarge).

### B. Anywhere else (laptop / other notebook)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name mlproj2 --display-name "Python (ml-project2-v2)"
# put the 9 Olist CSV files into data/raw/
python run_all.py
```
`NB_KERNEL=python3 python run_all.py` uses a different kernel name if needed.

## Documentation
* [Final report (PDF)](docs/ML_Project2_Final_Report.pdf) · [DOCX](docs/ML_Project2_Final_Report.docx)
* [Reproduction guide](docs/REPRODUCTION_GUIDE.md) · [Medium-style tutorial](docs/MEDIUM_TUTORIAL.md)
* [Step-by-step execution guide with per-cell screenshots (DOCX)](docs/ML_Project2_Step_by_Step_Execution_Guide.docx)
* `docs/ML_Project2_Final_Report.docx/.pdf` – final report (main text + appendices incl. compliance checklist).
* `docs/REPRODUCTION_GUIDE.md/.docx` – cell-by-cell run guide with expected outputs and troubleshooting.
* `docs/MEDIUM_TUTORIAL.md/.docx` – hands-on tutorial write-up.
* `screenshots/` – one PNG per executed notebook cell (code + real output from the SageMaker run).

## Main outputs
* `outputs/daily_prioritization_test.csv` – the 10-per-day investigation list for the test period
  (policy, approval_date, daily_rank, order_id, approval time, risk score, context, eventual outcome).
* `outputs/experiment_configs.json` – splits, grids, selected hyper-parameters, library versions.
* `outputs/tables/03_feature_availability_audit.csv` – feature-availability audit.
* `outputs/tables/02_split_summary.csv`, `02_eligibility_funnel.csv` – modeling examples and outcomes.
* `outputs/tables/04_validation_*.csv`, `05_test_*.csv`, `06_*.csv` – comparisons and diagnostics.

## Data
The raw Olist CSVs are **not** included (see `.gitignore`). Download the *Brazilian E-Commerce Public Dataset by Olist*
from Kaggle and place the 9 CSV files in `data/raw/`.

## Attribution
Dataset: Olist, *Brazilian E-Commerce Public Dataset by Olist* (Kaggle, CC BY-NC-SA 4.0). Libraries: pandas,
NumPy, scikit-learn, LightGBM, matplotlib, papermill. Code was written for this project with the help of an
AI assistant (Grok Bot); all results were produced by running this code.
