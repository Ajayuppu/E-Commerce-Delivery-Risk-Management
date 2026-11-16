"""Generates the numbered notebooks in notebooks/ (run: python tools/build_notebooks.py)."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)

BOOT = '''# Cell 1 - make the project package importable and define folders
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
from olist_risk import config
config.ensure_dirs()
print("PROJECT_ROOT =", PROJECT_ROOT)
print("outputs ->", config.OUTPUT_DIR)'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "mlproj2", "display_name": "Python (ml-project2-v2)", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(n, NB / name)


# ============================================================================ 00
nb("00_setup_environment.ipynb", [
("md", """# 00 - Environment setup
**ML Project 2 - E-Commerce Delivery Risk Management (Olist)** - Ajay Uppu

Run this notebook first, top to bottom. It
1. checks the Python version and the pinned libraries (installs them if they differ),
2. makes sure the 9 raw Olist CSV files are in `data/raw/` (downloads them from the project S3 bucket if they are missing),
3. verifies the files (row counts and sizes) so every later notebook starts from identical data.

> If cell 3 reports that it installed packages, use **Kernel -> Restart Kernel** and run this notebook again."""),
("code", BOOT),
("code", '''# Cell 2 - Python and library versions actually used by this kernel
import platform, importlib
print("python", platform.python_version())
pinned = {l.split("==")[0]: l.split("==")[1].strip() for l in open(PROJECT_ROOT / "requirements.txt")
          if "==" in l and not l.startswith("#")}
status = []
for pkg, want in pinned.items():
    mod = {"scikit-learn": "sklearn"}.get(pkg, pkg)
    try:
        have = importlib.import_module(mod).__version__
    except Exception:
        have = None
    status.append({"package": pkg, "required": want, "installed": have, "ok": have == want})
status = pd.DataFrame(status); display(status)'''),
("code", '''# Cell 3 - install the pinned versions only if something is missing/different
if not status.ok.all():
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements.txt")], check=True)
    print("Installed pinned requirements -> restart the kernel and re-run this notebook.")
else:
    print("All pinned packages already installed - nothing to do.")'''),
("code", '''# Cell 4 - make sure the raw CSVs are present (download from S3 if missing)
S3_BUCKET = os.environ.get("PROJECT_BUCKET", "ml-project2-olist-044450716277")
S3_RAW_PREFIX = os.environ.get("RAW_PREFIX", "ml-project2-v2/data/raw/")
missing = [f for f in config.RAW_FILES.values() if not (config.RAW_DIR / f).exists()]
print("missing before download:", missing)
if missing:
    import boto3
    s3 = boto3.client("s3")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in missing:
        s3.download_file(S3_BUCKET, S3_RAW_PREFIX + f, str(config.RAW_DIR / f))
        print("downloaded", f)
print("raw dir:", config.RAW_DIR)'''),
("code", '''# Cell 5 - verify the files: size and number of data rows
inv = []
for name, f in config.RAW_FILES.items():
    p = config.RAW_DIR / f
    with open(p, "rb") as fh:
        n_lines = sum(1 for _ in fh) - 1
    inv.append({"table": name, "file": f, "size_MB": round(p.stat().st_size / 1e6, 2), "rows": n_lines})
inv = pd.DataFrame(inv); display(inv)
expected_rows = {"orders": 99441, "items": 112650, "customers": 99441, "sellers": 3095, "products": 32951,
                 "payments": 103886, "geolocation": 1000163, "translation": 71}
for t_, n_ in expected_rows.items():
    assert int(inv.loc[inv.table == t_, "rows"].iloc[0]) == n_, f"{t_}: unexpected row count"
print("Row counts match the published Olist dataset (reviews has embedded newlines, so its line count is larger than its record count).")
inv.to_csv(config.TABLE_DIR / "00_raw_file_inventory.csv", index=False)'''),
("code", '''# Cell 6 - key configuration used by all notebooks
cfg = {k: getattr(config, k) for k in ["SEED", "TRAIN_START", "VAL_START", "TEST_START", "TEST_END",
                                        "LABEL_MATURITY_DAYS", "DAILY_CAPACITY", "BUDGETS", "SMOOTH_M", "OPEN_CAP_DAYS"]}
for k, v in cfg.items():
    print(f"{k:20s} = {v}")'''),
])

# ============================================================================ 01
nb("01_data_audit.ipynb", [
("md", """# 01 - Raw data audit
Goal: understand the 9 tables, their keys and relationships **before** joining anything, and find the
data-quality issues that drive later decisions (how to aggregate, what is missing, what is inconsistent)."""),
("code", BOOT),
("code", '''# Cell 2 - load all raw tables (dates parsed, zip prefixes kept as 5-character strings)
from olist_risk import data
t = data.load_raw()
for k, v in t.items():
    print(f"{k:12s} {v.shape}")'''),
("code", '''# Cell 3 - table-level audit: primary keys, duplicates, missing cells
tab = data.audit_tables(t)
tab.to_csv(config.TABLE_DIR / "01_table_audit.csv", index=False)
display(tab)'''),
("code", '''# Cell 4 - relationships: which joins are 1:1 and which would multiply orders
rel = data.audit_relationships(t)
rel.to_csv(config.TABLE_DIR / "01_relationship_audit.csv", index=False)
display(rel)'''),
("code", '''# Cell 5 - order lifecycle timestamps: missing values and inconsistencies
dates = data.audit_order_dates(t["orders"])
dates.to_csv(config.TABLE_DIR / "01_order_date_audit.csv", index=False)
display(dates)'''),
("code", '''# Cell 6 - order status distribution (final status at data export - an OUTCOME, never a feature)
st = t["orders"].order_status.value_counts().rename_axis("order_status").reset_index(name="orders")
st["share_%"] = (100 * st.orders / st.orders.sum()).round(2)
st["has_customer_delivery_date"] = st.order_status.map(
    t["orders"].groupby("order_status").order_delivered_customer_date.apply(lambda s: int(s.notna().sum())))
st.to_csv(config.TABLE_DIR / "01_order_status.csv", index=False)
display(st)'''),
("code", '''# Cell 7 - monthly volume of approved orders and the share delivered late (preview of the target)
o = t["orders"].copy()
o["month"] = o.order_approved_at.dt.to_period("M")
dl = o[o.order_delivered_customer_date.notna()]
late = dl.order_delivered_customer_date.dt.normalize() > dl.order_estimated_delivery_date.dt.normalize()
monthly = pd.DataFrame({"approved_orders": o.groupby("month").size(),
                        "delivered": dl.groupby("month").size(),
                        "late_rate_%": (100 * late.groupby(dl.month).mean()).round(2)}).dropna(subset=["approved_orders"])
monthly.to_csv(config.TABLE_DIR / "01_monthly_volume_late_rate.csv")
fig, ax = plt.subplots(figsize=(11, 4))
monthly.approved_orders.plot(kind="bar", ax=ax, color="#9ecae1", label="approved orders")
ax2 = ax.twinx(); ax2.plot(range(len(monthly)), monthly["late_rate_%"], color="#d62728", marker="o", label="late rate %")
ax.set_ylabel("approved orders"); ax2.set_ylabel("late rate (%)"); ax.set_title("Approved orders and late-delivery rate by approval month")
for lbl, x0 in [("train", "2017-01"), ("val", "2018-03"), ("test", "2018-06")]:
    if pd.Period(x0) in monthly.index:
        i = list(monthly.index).index(pd.Period(x0)); ax.axvline(i - 0.5, color="k", ls="--"); ax.text(i - 0.4, ax.get_ylim()[1] * 0.92, lbl)
fig.tight_layout(); fig.savefig(config.FIG_DIR / "01_monthly_volume_late_rate.png", dpi=130); plt.close(fig)
display(Image(str(config.FIG_DIR / "01_monthly_volume_late_rate.png")))
display(monthly.T)'''),
("code", '''# Cell 8 - basket structure: items, sellers and payments per order (why we aggregate before joining)
it = t["items"]; p = t["payments"]
basket = pd.DataFrame({
    "items_per_order": it.groupby("order_id").size().describe(),
    "sellers_per_order": it.groupby("order_id").seller_id.nunique().describe(),
    "payment_rows_per_order": p.groupby("order_id").size().describe()}).round(3)
display(basket)
naive = t["orders"].merge(it, on="order_id").merge(p, on="order_id")
print(f"Naive orders x items x payments join: {len(naive):,} rows for {naive.order_id.nunique():,} orders "
      f"-> {len(naive) - naive.order_id.nunique():,} duplicated order rows. We aggregate each table per order first.")'''),
("code", '''# Cell 9 - promised delivery window and actual delivery time (days) for delivered orders
dl = t["orders"].dropna(subset=["order_delivered_customer_date", "order_approved_at"])
promised = (dl.order_estimated_delivery_date - dl.order_approved_at.dt.normalize()).dt.days
actual = (dl.order_delivered_customer_date.dt.normalize() - dl.order_approved_at.dt.normalize()).dt.days
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(promised.clip(0, 70), bins=70, alpha=.6, label="promised days (approval -> estimated date)")
ax.hist(actual.clip(0, 70), bins=70, alpha=.6, label="actual days (approval -> delivery)")
ax.legend(); ax.set_xlabel("days (clipped at 70)"); ax.set_title("Promised vs actual delivery time")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "01_promised_vs_actual_days.png", dpi=130); plt.close(fig)
display(Image(str(config.FIG_DIR / "01_promised_vs_actual_days.png")))
print(pd.DataFrame({"promised_days": promised.describe(), "actual_days": actual.describe()}).round(1).T)'''),
("code", '''# Cell 10 - reviews are written after delivery: proof that they cannot be approval-time features
rv = t["reviews"].merge(t["orders"][["order_id", "order_approved_at", "order_delivered_customer_date"]], on="order_id")
print("reviews created BEFORE approval :", int((rv.review_creation_date < rv.order_approved_at).sum()))
print("reviews created AFTER approval  :", int((rv.review_creation_date >= rv.order_approved_at).sum()))
print("median days review created after delivery:",
      (rv.review_creation_date - rv.order_delivered_customer_date).dt.days.median())
print("-> the reviews table is never joined into the feature table.")'''),
])

# ============================================================================ 02
nb("02_labels_and_eligibility.ipynb", [
("md", """# 02 - Target definition and eligibility
* **Unit:** one order. **Decision time:** `order_approved_at`.
* **Target `is_late` = 1** if the *calendar date* of `order_delivered_customer_date` is after the *calendar date* of
  `order_estimated_delivery_date` (the promise is stored at 00:00, so a delivery on the promised day is on time).
* Orders without a customer-delivery date have an **undetermined** outcome: never used for training or predictive
  metrics, but they stay in the daily candidate pool (at approval nobody knows they will be canceled)."""),
("code", BOOT),
("code", '''# Cell 2 - build labels and outcome status for every order
from olist_risk import data, labels
t = data.load_raw()
lab = labels.eligibility_flags(labels.build_labels(t["orders"]))
display(lab.outcome_status.value_counts().rename("orders").to_frame())
print("late rate among delivered orders: %.4f" % lab.loc[lab.outcome_known, "is_late"].mean())'''),
("code", '''# Cell 3 - why calendar dates and not raw timestamps?
d = lab[lab.outcome_known]
ts_late = (d.order_delivered_customer_date > d.order_estimated_delivery_date).mean()
same_day = (d.order_delivered_customer_date.dt.normalize() == d.order_estimated_delivery_date).sum()
print(f"timestamp comparison late rate : {ts_late:.4f}")
print(f"calendar-date late rate (used) : {d.is_late.mean():.4f}")
print(f"orders delivered ON the promised day (on time for the customer, 'late' by raw timestamp): {same_day:,}")'''),
("code", '''# Cell 4 - outcome status by final order status
ct = pd.crosstab(lab.order_status, lab.outcome_status, margins=True)
ct.to_csv(config.TABLE_DIR / "02_status_vs_outcome.csv"); display(ct)'''),
("code", '''# Cell 5 - chronological splits on the approval timestamp + eligibility funnel
funnel = labels.eligibility_funnel(lab)
funnel.to_csv(config.TABLE_DIR / "02_eligibility_funnel.csv", index=False)
display(funnel)'''),
("code", '''# Cell 6 - label maturity: how long after approval do outcomes become known?
d = lab[lab.outcome_known & lab.order_approved_at.notna()]
lag = (d.outcome_known_at - d.order_approved_at).dt.days
rows = []
for G in [15, 30, 45, 60, 90]:
    rows.append({"days_after_approval": G, "share_all_outcomes_known": round((lag <= G).mean(), 4),
                 "share_late_outcomes_known": round((lag[d.is_late == 1] <= G).mean(), 4)})
mat = pd.DataFrame(rows); mat.to_csv(config.TABLE_DIR / "02_label_maturity.csv", index=False); display(mat)
print(f"Chosen LABEL_MATURITY_DAYS = {config.LABEL_MATURITY_DAYS}: training examples must be approved at least this long "
      "before the cutoff, otherwise the newest training orders would contain only the fast (mostly on-time) deliveries.")'''),
("code", '''# Cell 7 - censoring bias illustrated: late rate of train orders by month, with and without the maturity rule
tr = lab[lab.split == "train"].copy(); tr["month"] = tr.order_approved_at.dt.to_period("M")
naive = tr[tr.outcome_known & (tr.outcome_known_at < pd.Timestamp(config.VAL_START))]
cmp_ = pd.DataFrame({"all_delivered_late_rate": tr[tr.outcome_known].groupby("month").is_late.mean(),
                     "known_by_cutoff_late_rate": naive.groupby("month").is_late.mean(),
                     "used_for_training_rows": tr[tr.train_eligible].groupby("month").size()}).round(4)
cmp_.to_csv(config.TABLE_DIR / "02_censoring_by_month.csv"); display(cmp_.tail(6))'''),
("code", '''# Cell 8 - split summary (the 'modeling examples and outcomes' evidence table)
rows = []
for s in ["train", "validation", "test"]:
    m = lab.split == s
    rows.append({"split": s, "approval_from": lab.loc[m, "order_approved_at"].min(), "approval_to": lab.loc[m, "order_approved_at"].max(),
                 "days": lab.loc[m, "approval_date"].nunique(), "orders_in_daily_pool": int(m.sum()),
                 "labelled_orders": int((m & lab.outcome_known).sum()), "late": int((m & (lab.is_late == 1)).sum()),
                 "on_time": int((m & (lab.is_late == 0)).sum()), "undetermined": int((m & ~lab.outcome_known).sum()),
                 "late_rate_labelled": round(lab.loc[m & lab.outcome_known, "is_late"].mean(), 4),
                 "fit_rows_of_model_scored_on_next_split": int(lab.loc[m, "train_eligible"].sum()) if s == "train" else (int(lab.trainval_eligible.sum()) if s == "validation" else 0)})
split = pd.DataFrame(rows); split.to_csv(config.TABLE_DIR / "02_split_summary.csv", index=False); display(split)
print("fit_rows_of_model_scored_on_next_split: train row = rows that fit the models scored on VALIDATION (selection, cutoff 2018-03-01); "
      "validation row = train+validation rows that fit the final models scored on TEST (cutoff 2018-06-01); test row = 0 (never used for fitting).")'''),
("code", '''# Cell 9 - distribution of days late (late orders only)
dl = lab[lab.is_late == 1].days_late
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.hist(dl.clip(upper=60), bins=60, color="#d62728")
ax.set_xlabel("days after promised date (clipped at 60)"); ax.set_ylabel("orders"); ax.set_title("How late are late orders?")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "02_days_late_hist.png", dpi=130); plt.close(fig)
display(Image(str(config.FIG_DIR / "02_days_late_hist.png"))); print(dl.describe().round(2))'''),
("code", '''# Cell 10 - save labels for the next notebooks
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
lab.to_pickle(config.PROCESSED_DIR / "labels.pkl")
print("saved", config.PROCESSED_DIR / "labels.pkl", lab.shape)'''),
])

# ============================================================================ 03
nb("03_feature_engineering.ipynb", [
("md", """# 03 - Approval-time features and leakage checks
Every feature must be observable when the order is approved. Multi-row tables are aggregated per order before
joining. Historical aggregates use only orders whose **outcome was known (delivered) before** the scored order's
approval time - not merely orders approved earlier."""),
("code", BOOT),
("code", '''# Cell 2 - load raw tables and labels
from olist_risk import data, features
t = data.load_raw()
lab = pd.read_pickle(config.PROCESSED_DIR / "labels.pkl")
print(lab.shape)'''),
("code", '''# Cell 3 - build the order-level feature table (one row per order)
import time
t0 = time.time()
ft = features.build_feature_table(t, lab)
print(f"feature table {ft.shape} built in {time.time() - t0:.1f}s")
assert ft.order_id.is_unique and len(ft) == len(t["orders"]), "join multiplied orders!"
print("one row per order_id: OK  |  predictors:", len(features.FEATURES),
      f"({len(features.NUM_FEATURES)} numeric, {len(features.CAT_FEATURES)} categorical)")'''),
("code", '''# Cell 4 - feature-availability audit (also saved for the report)
audit = features.feature_availability_audit()
audit.to_csv(config.TABLE_DIR / "03_feature_availability_audit.csv", index=False)
display(audit[["feature_or_field", "source", "observed_at", "used_as_predictor"]])'''),
("code", '''# Cell 5 - leakage unit test: brute-force recompute of seller/customer history for 300 random orders
#          + every order whose delivery timestamp precedes its own approval timestamp (own-label edge case)
chk = features.verify_history_bruteforce(ft, lab, n=300)
summary = chk[["match_outcomes", "match_orders", "match_customer", "no_future_outcome"]].mean()
print(f"orders checked: {len(chk)} (300 random + {int(chk.delivered_before_own_approval.sum())} delivered-before-approval edge cases)")
display(summary.to_frame("share_true"))
assert summary.min() == 1.0 and not chk.own_outcome_used.any()
chk.to_csv(config.TABLE_DIR / "03_leakage_bruteforce_check.csv", index=False)
print("All sampled history features use only outcomes observed before approval and never the order's own outcome.")'''),
("code", '''# Cell 6 - missing values per predictor (handled by median/constant imputation + missing indicators)
miss = ft.loc[ft.in_daily_pool, features.FEATURES].isna().mean().mul(100).round(2)
miss = miss[miss > 0].sort_values(ascending=False).rename("missing_%").to_frame()
miss.to_csv(config.TABLE_DIR / "03_feature_missingness.csv"); display(miss)'''),
("code", '''# Cell 7 - univariate signal: late rate by decile of selected features (training rows only)
tr = ft[ft.train_eligible]
cols = ["promised_days", "distance_km", "freight_ratio", "seller_late_rate_sm", "route_late_rate_sm", "seller_overdue_open"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, c in zip(axes.ravel(), cols):
    q = pd.qcut(tr[c].rank(method="first"), 10, labels=False)
    tr.groupby(q).is_late.mean().plot(ax=ax, marker="o"); ax.set_title(c); ax.set_xlabel("decile"); ax.set_ylabel("late rate")
fig.suptitle("Late rate by feature decile (training rows)"); fig.tight_layout()
fig.savefig(config.FIG_DIR / "03_late_rate_by_decile.png", dpi=120); plt.close(fig)
display(Image(str(config.FIG_DIR / "03_late_rate_by_decile.png")))'''),
("code", '''# Cell 8 - late rate by customer state (top 12 by volume, training rows)
g = tr.groupby("customer_state").agg(orders=("order_id", "size"), late_rate=("is_late", "mean")).sort_values("orders", ascending=False)
g.round(4).to_csv(config.TABLE_DIR / "03_late_rate_by_customer_state_train.csv"); display(g.head(12).round(4).T)'''),
("code", '''# Cell 9 - descriptive statistics of the numeric predictors in the daily pool
desc = ft.loc[ft.in_daily_pool, features.NUM_FEATURES].describe().T.round(3)
desc.to_csv(config.TABLE_DIR / "03_feature_describe.csv"); display(desc.head(46))'''),
("code", '''# Cell 10 - save the modeling table
ft.to_pickle(config.PROCESSED_DIR / "features.pkl")
print("saved", config.PROCESSED_DIR / "features.pkl", ft.shape)'''),
])

# ============================================================================ 04
nb("04_modeling.ipynb", [
("md", """# 04 - Rules, models and selection on the validation period
* **Business rules (non-ML):** shortest promised window, longest distance, highest freight ratio, worst seller history.
* **Learned models:** logistic regression (simple, interpretable baseline) and LightGBM (gradient-boosted trees).
* **Selection criterion (validation, Mar-May 2018):** number of late orders captured by the daily top-10 policy
  (= daily precision@10); ties broken by PR-AUC. The test period is not touched in this notebook."""),
("code", BOOT),
("code", '''# Cell 2 - load features, define training and validation sets
import json, time, joblib
from olist_risk import models, policy, metrics, features
ft = pd.read_pickle(config.PROCESSED_DIR / "features.pkl")
train = ft[ft.train_eligible].copy(); y_train = train.is_late.astype(int)
val_pool = ft[ft.split == "validation"].copy()            # daily candidate pool (all approved orders)
val_lab = val_pool[val_pool.label_available]               # labelled subset for predictive metrics
print(f"train rows {len(train):,} (late rate {y_train.mean():.4f}) approved {train.order_approved_at.min().date()} .. {train.order_approved_at.max().date()}")
print(f"validation pool {len(val_pool):,} orders on {val_pool.approval_date.nunique()} days; labelled {len(val_lab):,} (late rate {val_lab.is_late.mean():.4f})")'''),
("code", '''# Cell 3 - business rules and a random reference on validation (same daily top-10 policy)
rows = []
for r, f in policy.RULES.items():
    val_pool[r] = f(val_pool).values
    e = policy.evaluate_policy(val_pool, r)
    pm = metrics.predictive_metrics(val_lab.is_late, f(val_lab), probabilistic=False)
    rows.append({"method": r, "family": "rule", **{k: e[k] for k in ["late_identified", "unnecessary_investigations", "undetermined_selected", "late_missed", "precision"]},
                 "pr_auc": pm["pr_auc"], "roc_auc": pm["roc_auc"]})
rnd = policy.random_policy_expectation(val_pool, n_rep=200)
rows.append({"method": "random (mean of 200 draws)", "family": "reference", "late_identified": rnd["late_identified"],
             "unnecessary_investigations": rnd["unnecessary_investigations"], "undetermined_selected": rnd["undetermined_selected"],
             "late_missed": rnd["late_missed"], "precision": rnd["precision"], "pr_auc": val_lab.is_late.mean(), "roc_auc": 0.5})
rules_val = pd.DataFrame(rows); display(rules_val.round(4))
best_rule = rules_val[rules_val.family == "rule"].sort_values(["late_identified", "pr_auc"], ascending=False).iloc[0].method
print("best rule on validation ->", best_rule)'''),
("code", '''# Cell 4 - hyper-parameter and feature-set search for both learned models (fit on train, score validation)
grid_rows = []
for fs_name, fs in models.FEATURE_SETS.items():
    for kind, grid in [("logreg", models.LOGREG_GRID), ("lgbm", models.LGBM_GRID)]:
        for params in grid:
            t0 = time.time()
            m = models.FittedModel(kind, params, num_features=fs).fit(train, y_train)
            val_pool["_s"] = m.score(val_pool)
            e = policy.evaluate_policy(val_pool, "_s")
            pm = metrics.predictive_metrics(val_lab.is_late, m.score(val_lab))
            grid_rows.append({"model": kind, "feature_set": fs_name, "params": json.dumps(params), "late_identified": e["late_identified"],
                              "unnecessary_investigations": e["unnecessary_investigations"], "precision_at_10_daily": e["precision"],
                              "pr_auc": pm["pr_auc"], "roc_auc": pm["roc_auc"], "brier": pm["brier"], "fit_seconds": round(time.time() - t0, 1)})
grid = pd.DataFrame(grid_rows)
grid.to_csv(config.TABLE_DIR / "04_validation_grid.csv", index=False)
print(len(grid), "configurations evaluated")
display(grid.sort_values(["late_identified", "pr_auc"], ascending=False).head(10).round(4))'''),
("code", '''# Cell 5 - best configuration per model family (selected on validation only)
best = (grid.sort_values(["late_identified", "pr_auc"], ascending=False).groupby("model").head(1).set_index("model"))
display(best.round(4))
selected_model = best.sort_values(["late_identified", "pr_auc"], ascending=False).index[0]
print("proposed model (best on validation):", selected_model)'''),
("code", '''# Cell 6 - validation comparison table: rules vs learned models under the same daily cap
val_cmp = pd.concat([rules_val, pd.DataFrame([{"method": f"{k} (best config)", "family": "learned", "late_identified": r.late_identified,
          "unnecessary_investigations": r.unnecessary_investigations, "precision": r.precision_at_10_daily, "pr_auc": r.pr_auc,
          "roc_auc": r.roc_auc} for k, r in best.iterrows()])], ignore_index=True)
val_cmp.to_csv(config.TABLE_DIR / "04_validation_comparison.csv", index=False)
display(val_cmp.round(4))'''),
("code", '''# Cell 7 - final refit of the selected configurations on train+validation (labels mature before TEST_START)
trainval = ft[ft.trainval_eligible].copy(); y_tv = trainval.is_late.astype(int)
print(f"refit rows {len(trainval):,} (late rate {y_tv.mean():.4f}), approved up to {trainval.order_approved_at.max().date()}")
final = {}
for kind, r in best.iterrows():
    fs = models.FEATURE_SETS[r.feature_set]
    final[kind] = models.FittedModel(kind, json.loads(r.params), num_features=fs).fit(trainval, y_tv)
    joblib.dump(final[kind], config.MODEL_DIR / f"{kind}_final.joblib")
    # validation-period model kept for the 'refit vs no refit' diagnostic in notebook 06
    joblib.dump(models.FittedModel(kind, json.loads(r.params), num_features=fs).fit(train, y_train), config.MODEL_DIR / f"{kind}_trainonly.joblib")
print("saved:", sorted(p.name for p in config.MODEL_DIR.glob("*.joblib")))'''),
("code", '''# Cell 8 - experiment configuration record (traceability)
exp = {
    "seed": config.SEED, "splits": {"train": [config.TRAIN_START, config.VAL_START], "validation": [config.VAL_START, config.TEST_START],
                                    "test": [config.TEST_START, config.TEST_END]},
    "label_definition": "is_late = date(order_delivered_customer_date) > date(order_estimated_delivery_date); undelivered = undetermined",
    "label_maturity_days": config.LABEL_MATURITY_DAYS, "daily_capacity": config.DAILY_CAPACITY,
    "train_rows_for_selection": int(len(train)), "trainval_rows_for_final_refit": int(len(trainval)),
    "selection_criterion": "validation late orders captured by daily top-10, tie-break PR-AUC",
    "logreg_grid": models.LOGREG_GRID, "lgbm_grid": models.LGBM_GRID, "feature_sets": {k: v for k, v in models.FEATURE_SETS.items()},
    "categorical_features": features.CAT_FEATURES,
    "selected": {k: {"params": json.loads(r.params), "feature_set": r.feature_set, "val_late_identified": int(r.late_identified),
                     "val_pr_auc": round(r.pr_auc, 4)} for k, r in best.iterrows()},
    "proposed_model": selected_model, "reference_rule": best_rule,
    "libraries": {m.__name__: m.__version__ for m in [np, pd, __import__("sklearn"), __import__("lightgbm")]},
}
json.dump(exp, open(config.OUTPUT_DIR / "experiment_configs.json", "w"), indent=2, default=str)
print(json.dumps({k: exp[k] for k in ["selected", "proposed_model", "reference_rule"]}, indent=2))'''),
])

# ============================================================================ 05
nb("05_daily_policy_evaluation.ipynb", [
("md", """# 05 - Final test: daily top-10 simulation (Jun-Aug 2018)
First and only use of the test period. For every calendar day, all orders approved that day are scored with
approval-time information, the 10 highest-risk orders are selected (all of them on days with fewer than 10),
and the selections are compared with eventual outcomes. Every method uses exactly the same rules."""),
("code", BOOT),
("code", '''# Cell 2 - load features, configuration and final models
import json, joblib
from olist_risk import policy, metrics
ft = pd.read_pickle(config.PROCESSED_DIR / "features.pkl")
exp = json.load(open(config.OUTPUT_DIR / "experiment_configs.json"))
final = {k: joblib.load(config.MODEL_DIR / f"{k}_final.joblib") for k in ["logreg", "lgbm"]}
test_pool = ft[ft.split == "test"].copy(); test_lab = test_pool[test_pool.label_available]
print(f"test pool: {len(test_pool):,} orders, {test_pool.approval_date.nunique()} days, labelled {len(test_lab):,}, "
      f"late {int((test_pool.outcome_status=='late').sum())}, undetermined {int((test_pool.outcome_status=='undetermined').sum())}")
print("proposed model:", exp["proposed_model"], "| reference rule:", exp["reference_rule"])'''),
("code", '''# Cell 3 - score the test orders with every method
for k, m in final.items():
    test_pool[f"score_{k}"] = m.score(test_pool)
for r, f in policy.RULES.items():
    test_pool[r] = f(test_pool).values
methods = {"logreg": "score_logreg", "lgbm": "score_lgbm", **{r: r for r in policy.RULES}}
display(test_pool[["order_id", "approval_date", "score_logreg", "score_lgbm", "promised_days", "distance_km"]].head())'''),
("code", '''# Cell 4 - operational results under the 10-per-day cap
rows = []
for name, col in methods.items():
    rows.append({"method": name, **policy.evaluate_policy(test_pool, col)})
rnd = policy.random_policy_expectation(test_pool, n_rep=200)
rows.append({"method": "random (mean of 200 draws)", **rnd})
ops = pd.DataFrame(rows).set_index("method")
cols = ["days", "investigations", "late_identified", "unnecessary_investigations", "undetermined_selected", "late_missed",
        "pool_late", "precision", "precision_resolved", "recall_of_late", "lift_vs_random", "oracle_late_identified", "share_of_oracle", "days_with_zero_hits"]
ops[cols].to_csv(config.TABLE_DIR / "05_test_operational_comparison.csv")
display(ops[cols].round(4))'''),
("code", '''# Cell 5 - predictive metrics on the labelled test orders
pm_rows = []
for name, col in methods.items():
    pm_rows.append({"method": name, **metrics.predictive_metrics(test_lab.is_late, test_pool.loc[test_lab.index, col], probabilistic=name in final)})
pm = pd.DataFrame(pm_rows).set_index("method")
pm.to_csv(config.TABLE_DIR / "05_test_predictive_metrics.csv"); display(pm.round(4))'''),
("code", '''# Cell 6 - headline: proposed model vs reference rule vs random
P, R = exp["proposed_model"], exp["reference_rule"]
head = ops.loc[[P, R, "random (mean of 200 draws)"], ["investigations", "late_identified", "unnecessary_investigations",
                                                      "undetermined_selected", "late_missed", "precision", "recall_of_late"]]
head.to_csv(config.TABLE_DIR / "05_headline.csv"); display(head.round(3))
print(f"{P}: {int(ops.loc[P,'late_identified'])} late orders identified vs {int(ops.loc[R,'late_identified'])} for {R} "
      f"and {ops.loc['random (mean of 200 draws)','late_identified']:.1f} expected at random, using {int(ops.loc[P,'investigations'])} investigations.")'''),
("code", '''# Cell 7 - paired daily comparison: on how many days did the model beat / tie / lose to the rule?
dP = policy.daily_breakdown(test_pool, methods[P]).set_index("approval_date")
dR = policy.daily_breakdown(test_pool, methods[R]).set_index("approval_date")
diff = dP.late_identified - dR.late_identified
print(f"days model > rule: {(diff>0).sum()}, equal: {(diff==0).sum()}, model < rule: {(diff<0).sum()}")
rng = np.random.default_rng(config.SEED)
boots = [rng.choice(diff.values, len(diff)).sum() for _ in range(5000)]
ci = np.percentile(boots, [2.5, 97.5])
print(f"total extra late orders identified by the model: {diff.sum()} (95% day-bootstrap CI {ci[0]:.0f} .. {ci[1]:.0f})")
pd.DataFrame({"model_hits": dP.late_identified, "rule_hits": dR.late_identified, "diff": diff}).to_csv(config.TABLE_DIR / "05_daily_model_vs_rule.csv")'''),
("code", '''# Cell 8 - cumulative late orders identified over the test period
fig, ax = plt.subplots(figsize=(10, 4))
for name in [P, "lgbm" if P == "logreg" else "logreg", R, "rule_shortest_promise"]:
    d = policy.daily_breakdown(test_pool, methods[name]).set_index("approval_date")
    ax.plot(d.index, d.late_identified.cumsum(), label=name)
d = policy.daily_breakdown(test_pool, methods[P]).set_index("approval_date")
ax.plot(d.index, (d.selected * d.late_in_pool / d.orders_approved).cumsum(), "k--", label="random (expected)")
ax.plot(d.index, np.minimum(d.late_in_pool, config.DAILY_CAPACITY).cumsum(), ":", color="grey", label="perfect foresight")
ax.set_title("Cumulative late orders identified (10 investigations/day, test period)"); ax.legend(); ax.set_ylabel("late orders identified")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "05_cumulative_hits_test.png", dpi=130); plt.close(fig)
display(Image(str(config.FIG_DIR / "05_cumulative_hits_test.png")))'''),
("code", '''# Cell 9 - precision-recall curves on labelled test orders
from sklearn.metrics import precision_recall_curve
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name in ["logreg", "lgbm", R]:
    pr, rc, _ = precision_recall_curve(test_lab.is_late, test_pool.loc[test_lab.index, methods[name]])
    ax.plot(rc, pr, label=f"{name} (AP={pm.loc[name,'pr_auc']:.3f})")
ax.axhline(test_lab.is_late.mean(), color="grey", ls="--", label="base rate")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-recall, test period"); ax.legend()
fig.tight_layout(); fig.savefig(config.FIG_DIR / "05_pr_curves_test.png", dpi=130); plt.close(fig)
display(Image(str(config.FIG_DIR / "05_pr_curves_test.png")))'''),
("code", '''# Cell 10 - save the daily prioritisation list for the test period (traceable to order_id + approval date)
sel = policy.select_daily_topk(test_pool, methods[P])
out = sel[sel.selected][["approval_date", "daily_rank", "order_id", "order_approved_at", methods[P], "customer_id", "main_seller_id",
                          "customer_state", "seller_state", "main_category", "promised_days", "distance_km", "order_estimated_delivery_date",
                          "outcome_status", "is_late", "days_late", "order_status"]].rename(columns={methods[P]: "risk_score"})
out.insert(0, "policy", f"{P}_top{config.DAILY_CAPACITY}")
out.to_csv(config.OUTPUT_DIR / "daily_prioritization_test.csv", index=False)
selR = policy.select_daily_topk(test_pool, methods[R])
selR[selR.selected][["approval_date", "daily_rank", "order_id", "order_approved_at", R, "outcome_status", "is_late"]].to_csv(
    config.OUTPUT_DIR / "daily_prioritization_test_rule.csv", index=False)
print(out.shape, "rows saved to outputs/daily_prioritization_test.csv"); display(out.head(12))'''),
("code", '''# Cell 11 - how were days with fewer than 10 orders and undetermined outcomes handled?
db = policy.daily_breakdown(test_pool, methods[P]); db.to_csv(config.TABLE_DIR / "05_daily_breakdown_proposed.csv", index=False)
print("days in test:", len(db), "| days with < 10 approved orders:", int((db.orders_approved < 10).sum()),
      "| investigations used:", int(db.selected.sum()), "of", 10 * len(db), "possible")
print("selected orders with undetermined outcome (canceled/unavailable/not delivered):", int(db.undetermined_selected.sum()))
display(db.describe().round(2))
# Sensitivity for incomplete outcomes (right-censoring at the data export, 2018-10-17): an undelivered order that was not
# canceled/unavailable (still open: shipped/invoiced/processing/approved/created) and was already past its promised date at export is at least overdue -> count it as a delivery problem.
export_ts = ft.order_purchase_timestamp.max()
adj_status, open_overdue = policy.censoring_adjusted_status(test_pool, export_ts)
print(f"data export: {export_ts} | test orders undetermined: {int((test_pool.outcome_status == 'undetermined').sum())}, "
      f"of which still open (shipped/invoiced/processing/approved/created) and past the promised date at export: {int(open_overdue.sum())}")
tp_adj = test_pool.assign(outcome_status=adj_status)
cs = []
for name, col in methods.items():
    e0, e1 = policy.evaluate_policy(test_pool, col), policy.evaluate_policy(tp_adj, col)
    cs.append({"method": name, "late_identified_main": e0["late_identified"], "problems_identified_censoring_adjusted": e1["late_identified"],
               "precision_main": e0["precision"], "precision_censoring_adjusted": e1["precision"],
               "problems_in_pool_adjusted": e1["pool_late"], "undetermined_selected_adjusted": e1["undetermined_selected"]})
cs = pd.DataFrame(cs).set_index("method"); cs.to_csv(config.TABLE_DIR / "05_censoring_sensitivity.csv"); display(cs.round(4))'''),
])

# ============================================================================ 06
nb("06_sensitivity_diagnostics.ipynb", [
("md", """# 06 - Sensitivity analysis and diagnostics
Budget sensitivity (k = 5/10/20/50), performance by month, failure groups (state, category), drivers of the
model, calibration, refit vs no refit, and feature-set ablation. All on the test period with the same policy rules."""),
("code", BOOT),
("code", '''# Cell 2 - reload test pool and scores
import json, joblib
from olist_risk import policy, metrics, models, features
ft = pd.read_pickle(config.PROCESSED_DIR / "features.pkl")
exp = json.load(open(config.OUTPUT_DIR / "experiment_configs.json")); P, R = exp["proposed_model"], exp["reference_rule"]
final = {k: joblib.load(config.MODEL_DIR / f"{k}_final.joblib") for k in ["logreg", "lgbm"]}
tp = ft[ft.split == "test"].copy()
for k, m in final.items(): tp[f"score_{k}"] = m.score(tp)
for r, f in policy.RULES.items(): tp[r] = f(tp).values
methods = {"logreg": "score_logreg", "lgbm": "score_lgbm", **{r: r for r in policy.RULES}}
print(P, R, tp.shape)'''),
("code", '''# Cell 3 - investigation budget sensitivity
rows = []
for k in config.BUDGETS:
    for name in ["logreg", "lgbm", R, "rule_shortest_promise"]:
        e = policy.evaluate_policy(tp, methods[name], k); rows.append({"k_per_day": k, "method": name, **e})
    rnd = policy.random_policy_expectation(tp, k, n_rep=50); rows.append({"k_per_day": k, "method": "random", **rnd})
bud = pd.DataFrame(rows)
bt = bud.pivot_table(index="k_per_day", columns="method", values="late_identified").round(1)
bp = bud.pivot_table(index="k_per_day", columns="method", values="precision").round(3)
bud.to_csv(config.TABLE_DIR / "06_budget_sensitivity.csv", index=False)
print("late orders identified"); display(bt); print("precision of investigations"); display(bp)
fig, ax = plt.subplots(figsize=(7, 4))
for c in bt.columns: ax.plot(bt.index, bt[c], marker="o", label=c)
ax.set_xlabel("investigations per day (k)"); ax.set_ylabel("late orders identified (test)"); ax.legend(); ax.set_title("Budget sensitivity")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "06_budget_sensitivity.png", dpi=130); plt.close(fig); display(Image(str(config.FIG_DIR / "06_budget_sensitivity.png")))'''),
("code", '''# Cell 4 - performance by month of the test period (10/day)
tp["month"] = tp.order_approved_at.dt.to_period("M").astype(str)
rows = []
for mth, g in tp.groupby("month"):
    for name in [P, R]:
        e = policy.evaluate_policy(g, methods[name]); rows.append({"month": mth, "method": name, **{k: e[k] for k in
              ["pool_orders", "pool_late", "base_rate", "late_identified", "unnecessary_investigations", "precision", "recall_of_late", "lift_vs_random"]}})
    pm = metrics.predictive_metrics(g[g.label_available].is_late, g[g.label_available][methods[P]])
    rows[-2]["pr_auc"] = pm["pr_auc"]; rows[-2]["roc_auc"] = pm["roc_auc"]
mon = pd.DataFrame(rows); mon.to_csv(config.TABLE_DIR / "06_by_month.csv", index=False); display(mon.round(3))'''),
("code", '''# Cell 5 - failure groups: where are late orders missed? (customer state)
s = policy.select_daily_topk(tp, methods[P])
s["hit"] = s.selected & (s.outcome_status == "late"); s["miss"] = ~s.selected & (s.outcome_status == "late")
s["false_alarm"] = s.selected & (s.outcome_status == "on_time")
grp = s.groupby("customer_state").agg(orders=("order_id", "size"), late=("miss", "sum"), hits=("hit", "sum"),
                                      selected=("selected", "sum"), false_alarms=("false_alarm", "sum"))
grp["late"] = grp.late + grp.hits; grp = grp.rename(columns={"late": "late_orders"})
grp["capture_rate"] = (grp.hits / grp.late_orders.replace(0, np.nan)).round(3)
grp["late_rate"] = (grp.late_orders / grp.orders).round(4)
grp = grp.sort_values("late_orders", ascending=False); grp.to_csv(config.TABLE_DIR / "06_failure_by_customer_state.csv")
display(grp.head(15))'''),
("code", '''# Cell 6 - failure groups by product category (top 15 by late orders)
cat = s.groupby("main_category").agg(orders=("order_id", "size"), hits=("hit", "sum"), missed=("miss", "sum"), false_alarms=("false_alarm", "sum"))
cat["late_orders"] = cat.hits + cat.missed; cat["capture_rate"] = (cat.hits / cat.late_orders.replace(0, np.nan)).round(3)
cat = cat.sort_values("late_orders", ascending=False); cat.to_csv(config.TABLE_DIR / "06_failure_by_category.csv"); display(cat.head(15))'''),
("code", '''# Cell 7 - what do missed late orders look like compared with caught ones?
cmpc = ["promised_days", "distance_km", "seller_late_rate_sm", "route_late_rate_sm", "market_overdue_open", "days_late", "total_price"]
prof = pd.DataFrame({"caught (hits)": s[s.hit][cmpc].median(), "missed late": s[s.miss][cmpc].median(),
                     "false alarms": s[s.false_alarm][cmpc].median()}).round(2)
prof.to_csv(config.TABLE_DIR / "06_error_profile.csv"); display(prof)'''),
("code", '''# Cell 8 - model drivers: logistic-regression coefficients and LightGBM gain importance
lr = final["logreg"].model
names = lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=names).sort_values()
coef.to_csv(config.TABLE_DIR / "06_logreg_coefficients.csv")
gb = final["lgbm"].model.booster_
imp = pd.Series(gb.feature_importance("gain"), index=gb.feature_name()).sort_values(ascending=False)
imp.to_csv(config.TABLE_DIR / "06_lgbm_gain_importance.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pd.concat([coef.head(8), coef.tail(8)]).plot(kind="barh", ax=axes[0], color="#3182bd"); axes[0].set_title("LogReg standardized coefficients (top +/-)")
imp.head(15)[::-1].plot(kind="barh", ax=axes[1], color="#31a354"); axes[1].set_title("LightGBM gain importance (top 15)")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "06_model_drivers.png", dpi=120); plt.close(fig); display(Image(str(config.FIG_DIR / "06_model_drivers.png")))'''),
("code", '''# Cell 9 - calibration of the predicted probabilities (labelled test orders)
tl = tp[tp.label_available]
fig, ax = plt.subplots(figsize=(5.5, 4.5)); rows = []
for k in ["logreg", "lgbm"]:
    q = pd.qcut(tl[f"score_{k}"].rank(method="first"), 10, labels=False)
    c = tl.groupby(q).agg(pred=(f"score_{k}", "mean"), obs=("is_late", "mean")); ax.plot(c.pred, c.obs, marker="o", label=k)
    rows.append(c.assign(model=k))
ax.plot([0, .3], [0, .3], "k--"); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed late rate"); ax.legend(); ax.set_title("Calibration by decile (test)")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "06_calibration.png", dpi=130); plt.close(fig); display(Image(str(config.FIG_DIR / "06_calibration.png")))
pd.concat(rows).to_csv(config.TABLE_DIR / "06_calibration.csv")
print("Mean predicted vs observed:", {k: (round(tl[f'score_{k}'].mean(), 4), round(tl.is_late.mean(), 4)) for k in ["logreg", "lgbm"]})'''),
("code", '''# Cell 10 - refit on train+validation vs the validation-period model (no refit)
rows = []
for k in ["logreg", "lgbm"]:
    m0 = joblib.load(config.MODEL_DIR / f"{k}_trainonly.joblib"); tp["_s0"] = m0.score(tp)
    for lbl, col in [("trained on train only", "_s0"), ("refit on train+validation (used)", f"score_{k}")]:
        e = policy.evaluate_policy(tp, col); pm = metrics.predictive_metrics(tl.is_late, tp.loc[tl.index, col])
        rows.append({"model": k, "version": lbl, "late_identified": e["late_identified"], "precision": e["precision"], "pr_auc": pm["pr_auc"], "roc_auc": pm["roc_auc"]})
rf = pd.DataFrame(rows); rf.to_csv(config.TABLE_DIR / "06_refit_vs_trainonly.csv", index=False); display(rf.round(4))'''),
("code", '''# Cell 11 - feature-group ablation for the proposed model family (refit on train+validation, test)
trainval = ft[ft.trainval_eligible]; ytv = trainval.is_late.astype(int)
params = exp["selected"][P]["params"]
groups = {"all features": features.NUM_FEATURES,
          "no history features": [c for c in features.NUM_FEATURES if c not in features.HISTORY_FEATURES],
          "no market-level features": [c for c in features.NUM_FEATURES if not c.startswith("market_")],
          "history features only": features.HISTORY_FEATURES}
rows = []
for g, cols in groups.items():
    m = models.FittedModel(P, params, num_features=cols).fit(trainval, ytv); tp["_a"] = m.score(tp)
    e = policy.evaluate_policy(tp, "_a"); pm = metrics.predictive_metrics(tl.is_late, tp.loc[tl.index, "_a"])
    rows.append({"feature_group": g, "n_numeric": len(cols), "late_identified": e["late_identified"], "precision": e["precision"], "pr_auc": pm["pr_auc"], "roc_auc": pm["roc_auc"]})
abl = pd.DataFrame(rows); abl.to_csv(config.TABLE_DIR / "06_ablation.csv", index=False); display(abl.round(4))
print("(diagnostic only - the test set was NOT used to choose the model)")'''),
("code", '''# Cell 12 - investigate fewer than 10 when risk is low? precision of the daily rank positions
s = policy.select_daily_topk(tp, methods[P])
byrank = s[s.daily_rank <= 20].groupby("daily_rank").apply(lambda g: pd.Series({"orders": len(g), "late_rate": (g.outcome_status == "late").mean(),
                                                     "mean_score": g[methods[P]].mean()}), include_groups=False).round(4)
byrank.to_csv(config.TABLE_DIR / "06_precision_by_daily_rank.csv"); display(byrank.T)'''),
("md", """## Post-hoc analysis (added AFTER the final test evaluation - not a valid test estimate)
The final test showed that the rule chosen on validation (longest distance) collapsed, while the shortest-promise rule
(worst on validation) did best. That suggested a feature the pre-specified model lacked: **delivery slack** = promised days
minus the typical delivery time on the same route (learned only from orders delivered before approval). The cell below
checks this idea on validation first and then on test. Because the idea came from looking at test results, these numbers
are exploratory and are reported separately from the main results."""),
("code", '''# Cell 13 - post-hoc: slack features (validation first, then test)
train = ft[ft.train_eligible]; ytr = train.is_late.astype(int)
vp = ft[ft.split == "validation"].copy()
rows = []
for kind in ["logreg", "lgbm"]:
    prm = exp["selected"][kind]["params"]
    base_cols = models.FEATURE_SETS[exp["selected"][kind]["feature_set"]]
    for lbl, cols in [("selected feature set", base_cols), ("selected feature set + slack features (post-hoc)", base_cols + features.POSTHOC_FEATURES)]:
        mv = models.FittedModel(kind, prm, num_features=cols).fit(train, ytr); vp["_p"] = mv.score(vp)
        mt = models.FittedModel(kind, prm, num_features=cols).fit(trainval, ytv); tp["_p"] = mt.score(tp)
        rows.append({"model": kind, "features": lbl, "val_late_identified": policy.evaluate_policy(vp, "_p")["late_identified"],
                     "test_late_identified": policy.evaluate_policy(tp, "_p")["late_identified"],
                     "test_pr_auc": metrics.predictive_metrics(tl.is_late, tp.loc[tl.index, "_p"])["pr_auc"]})
for df_, nm in [(vp, "val"), (tp, "test")]:
    df_["_slack"] = -df_["slack_vs_route"].fillna(df_["slack_vs_route"].median())
rows.append({"model": "rule_smallest_route_slack", "features": "non-ML rule (post-hoc)", "val_late_identified": policy.evaluate_policy(vp, "_slack")["late_identified"],
             "test_late_identified": policy.evaluate_policy(tp, "_slack")["late_identified"],
             "test_pr_auc": metrics.predictive_metrics(tl.is_late, tp.loc[tl.index, "_slack"], False)["pr_auc"]})
ph = pd.DataFrame(rows); ph.to_csv(config.TABLE_DIR / "06_posthoc_slack_features.csv", index=False); display(ph.round(4))
print("POST-HOC / test-informed: use as a hypothesis for the next model version, not as the reported test result.")'''),
("code", '''# Cell 14 - summary of all saved outputs
for p in sorted(config.OUTPUT_DIR.rglob("*")):
    if p.is_file(): print(f"{str(p.relative_to(config.OUTPUT_DIR)):55s} {p.stat().st_size/1024:8.1f} KB")'''),
])
print("notebooks written to", NB)
