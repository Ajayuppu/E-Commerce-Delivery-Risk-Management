# Ten Orders a Day: Predicting Late Deliveries at Approval Time with Amazon SageMaker Studio

*How I turned "which orders will arrive late?" into a leakage-safe ranking problem, tested it under a real 10-per-day investigation budget, and discovered that a one-line business rule can beat a model when the world changes.*

**By Ajay Uppu** · MS Data Science, Georgia State University

---

## The 10-slot problem

Imagine you run logistics for a Brazilian marketplace. Every day a couple of hundred orders are approved. A few of them will arrive after the date the customer was promised – and each of those is a potential bad review, a support ticket, a refund. Your team can call a seller, check a shipment or talk to a carrier, but only for **ten orders per day**.

So the question is not "can we classify every order?" It is: **which ten orders approved today deserve those slots?**

That reframing is the whole project. In this post I walk you through how I built it end to end in Amazon SageMaker Studio using the public Olist dataset – from raw CSVs to a daily investigation list – and what I learned when the final test didn't go the way validation said it would.

## What you will build

- A **leakage-safe order-level dataset**: one row per order, using only information available at the moment the order is approved.
- A **target** that respects how promised dates are stored, with an explicit policy for canceled and undelivered orders.
- **Two learned models** (logistic regression and LightGBM) and **four non-ML business rules**, all ranked under the same 10-per-day budget.
- A **chronological train / validation / test design** in which the test months are opened only once.
- A **daily top-10 simulation** that reports late orders caught, unnecessary investigations and late orders missed.
- Everything running in a **SageMaker Studio JupyterLab space**, reproducible with one command.

## The building blocks

**Amazon SageMaker Studio** is AWS's browser-based IDE for ML. A *domain* holds users and settings; a *user profile* is your identity inside it; a *JupyterLab space* is your persistent workspace (a storage volume plus a compute instance you can start and stop).

**Lifecycle configurations** are shell scripts that run when a space starts. I used one to copy my project from S3 and execute all notebooks automatically on the first start.

**Olist** is a real Brazilian e-commerce dataset with about 100k orders (2016–2018) spread over nine tables: orders, items, payments, customers, sellers, products, reviews, geolocation and a category translation.

**Leakage** means accidentally giving the model information it would not have at decision time. In delivery prediction it hides everywhere: the final order status, carrier pickup dates, reviews, and "seller late rate" computed over the whole dataset.

## Architecture

```
            S3 bucket (ml-project2-olist-044450716277)
            ├── ml-project2-v2/data/raw/        9 Olist CSVs
            ├── ml-project2-v2/studio/          code bundle (src, notebooks, aws scripts)
            └── ml-project2-v2/studio-run/      executed notebooks, outputs, logs
                         ▲              │
             sync results│              │ sync on first start (lifecycle config)
                         │              ▼
 SageMaker Studio domain: machinelearningproj2ajay-ecommerce
   └── user profile: ajay-uppu
         └── JupyterLab space: mlproj2-ecommerce-space (ml.m5.xlarge, 20 GB)
               ├── .venv + kernel "Python (ml-project2-v2)"
               ├── src/olist_risk/  (data, labels, features, models, policy, metrics)
               └── notebooks 00 → 06  (papermill: run_all.py)
```

## Prerequisites

- An AWS account with permission to create a SageMaker domain, plus an S3 bucket.
- An IAM execution role for SageMaker with read/write access to that bucket (I reused `ml-project2-olist-sagemaker-exec`).
- The Olist CSVs from Kaggle (CC BY-NC-SA 4.0), uploaded to `s3://<bucket>/ml-project2-v2/data/raw/`.
- Basic Python, pandas and scikit-learn.

## The steps at a glance

1. Create the Studio domain, the user profile and the JupyterLab space.
2. Audit the raw tables (keys, joins, timestamps).
3. Define the target and eligibility, and handle canceled orders.
4. Engineer approval-time features and leakage-safe history.
5. Build the baseline rules, train the models and select on validation.
6. Run the daily top-10 test simulation.
7. Run the sensitivity analysis and diagnostics.

---

## Step 1 – Create the Studio environment

I created a new domain called `machinelearningproj2ajay-ecommerce` with IAM authentication in the default VPC, a user profile `ajay-uppu` and a private JupyterLab space. You can click through the console (SageMaker AI → Domains → Create domain → *Set up for organizations*), or script it with boto3:

```python
import boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
sm.create_domain(
    DomainName="machinelearningproj2ajay-ecommerce", AuthMode="IAM",
    VpcId=vpc_id, SubnetIds=subnet_ids, AppNetworkAccessType="PublicInternetOnly",
    DefaultUserSettings={"ExecutionRole": role_arn})
sm.create_user_profile(DomainId=domain_id, UserProfileName="ajay-uppu")
sm.create_space(DomainId=domain_id, SpaceName="mlproj2-ecommerce-space",
    OwnershipSettings={"OwnerUserProfileName": "ajay-uppu"},
    SpaceSharingSettings={"SharingType": "Private"},
    SpaceSettings={"AppType": "JupyterLab",
                   "JupyterLabAppSettings": {"DefaultResourceSpec": {"InstanceType": "ml.m5.xlarge"}},
                   "SpaceStorageSettings": {"EbsStorageSettings": {"EbsVolumeSizeInGb": 20}}})
```

The lifecycle configuration copies the project from S3 the first time the space starts and launches a bootstrap script in the background. The script creates a virtual environment with pinned versions, registers a Jupyter kernel and runs every notebook with papermill:

```bash
if [ ! -f "$PROJ/.synced_from_s3" ]; then
  aws s3 sync "s3://$BUCKET/$PREFIX/studio/" "$PROJ/" --only-show-errors
  date > "$PROJ/.synced_from_s3"
fi
[ -f "$PROJ/logs/RUN_DONE" ] || nohup bash "$PROJ/aws/studio_bootstrap.sh" > ~/ml-project2-v2-bootstrap.out 2>&1 &
```

> **Expected vs. observed:** the space started, the bootstrap ran all seven notebooks in **283 seconds** and wrote `RUN_DONE` with exit code 0. Opening JupyterLab showed `ml-project2-v2/notebooks` with every output already in place.

*[Screenshot placeholder: Studio JupyterLab page showing mlproj2-ecommerce-space running]*

## Step 2 – Audit before you join

The first thing I learned is that joining Olist naively is a trap. 9,803 orders have several item rows (up to 21), 2,961 have several payment rows, and the geolocation table has a median of 29 rows per zip prefix. Join orders × items × payments directly and you get **117,601 rows for 98,665 orders**.

So the rule is to **aggregate every child table to one row per order first**, then join, then assert:

```python
items_agg = items.groupby("order_id").agg(n_items=("order_item_id", "size"),
                                          total_price=("price", "sum"),
                                          total_freight=("freight_value", "sum"), ...)
X = orders.merge(items_agg, on="order_id", how="left").merge(pay_agg, on="order_id", how="left")
assert X["order_id"].is_unique
```

![Monthly approved orders and late rate](outputs/figures/01_monthly_volume_late_rate.png)

That chart turned out to be the most important picture in the project. The late rate sits around 3–5% for most of 2017, spikes to 12% in November 2017 (the Black Friday month), reaches **19% in March 2018** and then collapses to **1% in June 2018**. Whatever the model learns, the world it gets tested on will look different.

## Step 3 – Define "late" carefully

My first definition compared timestamps, and it gave a late rate of 8.1%. Then I noticed that **every promised date is stored at 00:00**. An order delivered at 3 pm on the promised day is *on time* for the customer, but "late" by timestamp. Comparing calendar dates fixed 1,292 mislabelled orders:

```python
is_late = (delivered_customer.dt.normalize() > estimated_delivery.dt.normalize()).astype(int)
```

**Canceled and undelivered orders** (2,965) have no delivery date, so their outcome is unknown, not "on time". I never train on them, but I **keep them in the daily candidate pool** because on the approval day nobody knows they will be canceled. When the policy picks one, it counts separately as *undetermined*.

The subtle bug I nearly shipped was **censoring**. Late orders take longer to resolve, so the newest training orders before a cutoff look almost never late: February 2018 had a true late rate of 13.6%, but only **0.5%** of its orders were known to be late by the cutoff. I added a **60-day label-maturity rule**: a training row must be approved at least 60 days before the cutoff and have its outcome observed before it. After 60 days, 99.7% of all outcomes and 95.7% of late outcomes are known.

## Step 4 – Approval-time features without time travel

I grouped the features by when they become known:

- **Timing:** promised days (estimated date − approval), approval delay, hour and weekday, days to the seller's shipping limit.
- **Basket:** items, sellers, price, freight, freight ratio, weight, volume, category.
- **Payment:** type, installments, voucher use.
- **Geography:** customer and seller state, same state or city, and seller–customer distance using one median coordinate per zip prefix.
- **History:** seller, route, customer-state and category late rates; seller workload; market-wide overdue orders.

The history features are where leakage hides. I compute them with an "as-of" cumulative trick: events (deliveries) and queries (approvals) are sorted together, and an outcome counts only if it was **delivered before** the approval timestamp.

```python
def _cum_before(event_keys, event_times, event_values, query_keys, query_times):
    # sort events and queries together; queries first on ties -> strictly "before"
    ...
seller_late_rate_sm = (late_before + m * market_rate_before) / (outcomes_before + m)   # m = 20
```

Then I wrote a test I recommend to everyone: for **300 random orders**, I recomputed the history by brute force with plain filtering and compared. Every check matched (100%), and no order ever saw its own outcome.

## Step 5 – Rules first, then models

Before training anything, I asked what a sensible operations team would do without ML:

| Rule | Ranking |
|---|---|
| Shortest promised window | fewest days between approval and promised date |
| Longest distance | seller–customer km |
| Highest freight ratio | freight / price |
| Worst seller history | smoothed seller late rate (only past outcomes) |

Every method uses the same policy: each day, rank that day's approved orders and investigate the top 10. The models were a regularized **logistic regression** (log-transformed, scaled numerics plus one-hot categories) and **LightGBM** (`n_jobs=-1`, shallow trees). Model, hyper-parameter and feature-set selection (36 configurations) used **validation only (Mar–May 2018)**, and the criterion was the business metric: late orders caught in the daily top 10.

> **Expected vs. observed (validation, 89 days):** logistic regression caught **256** late orders, LightGBM 227, the best rule (longest distance) 202 and random selection 88. The shortest-promise rule was the worst at **46**, even below random.

The selected logistic regression was then refit on all mature train+validation rows (64,109 orders) before the test.

## Step 6 – Opening the test set once

The test covers June–August 2018: 88 days, 18,960 approved orders and only **672 late ones (3.6%)**.

| Method (10/day) | Late caught | Unnecessary | Undetermined | Missed |
|---|---|---|---|---|
| **Logistic regression (proposed)** | **70** | 780 | 19 | 602 |
| LightGBM | 58 | 796 | 15 | 614 |
| Longest distance (reference rule) | 34 | 823 | 12 | 638 |
| Shortest promised window | 80 | 763 | 26 | 592 |
| Random (mean of 200 draws) | 26.5 | 828 | 14.5 | 646 |

![Cumulative late orders identified](outputs/figures/05_cumulative_hits_test.png)

The model caught **2.7× more late orders than random** and twice as many as the pre-specified reference rule (+36; 95% day-bootstrap CI 20–55). That's the good news.

The honest news is that **the shortest-promise rule, the worst method on validation, came out best on test** with 80. I did not swap it in after the fact, because that would mean using the test set to choose the method. Instead I report it and treat it as a finding.

## Step 7 – Why did that happen? Diagnostics

- **Drift by month:** the model's lift was 3.8 in June and 4.4 in July but **1.0 in August** (ROC-AUC 0.51). Most missed late orders were short routes (median 109 km vs 1,670 km for the ones caught), largely inside São Paulo state – exactly what history-based features call "safe".
- **Failure groups:** São Paulo had 405 late orders and the model caught 2.7% of them, while Rio de Janeiro received 381 of the 869 investigations.
- **Budget sensitivity:** at 5, 10, 20 and 50 investigations per day the model caught 45, 70, 105 and 181 late orders. Precision falls from 10% to 4%, and the top two ranks of each day are late about 13% of the time.
- **Calibration:** scores averaged 0.19 against an observed rate of 0.036. They are useful for **ranking**, not as probabilities, until they are recalibrated on recent data.
- **Post-hoc check (clearly labelled as test-informed):** adding "slack versus typical route time" features nudged the test result from 70 to 72. That's a hypothesis for the next version, not a reported result.

![Budget sensitivity](outputs/figures/06_budget_sensitivity.png)

## Troubleshooting I hit along the way

- **`CreateApp` failed without an image:** pass the SageMaker Distribution image ARN (`.../image/sagemaker-distribution-cpu`) in `ResourceSpec`.
- **`get_feature_names_out` failed** on a `FunctionTransformer`: set `feature_names_out="one-to-one"`.
- **Negative "open orders" counts:** a few orders have inconsistent timestamps (delivered before carrier pickup), so intervals with a missing or non-positive length are ignored.
- **Brier score on rule scores:** rules aren't probabilities, so I compute only ranking metrics for them.

## Lessons learned

1. **Frame the metric as the decision.** "Late orders caught in 10 slots per day" made every choice easier than "accuracy" or even ROC-AUC would have.
2. **Time is the hardest feature.** Calendar-date labels, as-of history, label maturity and chronological splits each changed the results materially.
3. **Always ship a rule baseline.** It keeps the model honest, and here it exposed drift that a single test number would have hidden.
4. **Write a leakage unit test.** A brute-force check on a few hundred rows is cheap insurance.
5. **Never re-pick on test.** Report the uncomfortable result and make it the next experiment.

## Conclusion

Under a strict 10-per-day budget, a simple, regularized logistic regression with approval-time features caught 70 late orders over three months, compared with 26 at random and 34 for the best validation rule. My recommendation is to pilot it **alongside** the shortest-promise rule (for example 5 + 5 slots), monitor weekly lift, and retrain monthly with the 60-day maturity rule, because in this marketplace the definition of "risky" moves with the season. None of this shows that investigating an order *prevents* lateness; that needs a controlled pilot.

All the code, notebooks and outputs are reproducible in the Studio space with one command: `.venv/bin/python run_all.py`.

---

*Ajay Uppu is an MS Data Science student at Georgia State University interested in ML engineering on AWS. Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0). Code was written with the help of an AI assistant; all numbers come from running it.*
