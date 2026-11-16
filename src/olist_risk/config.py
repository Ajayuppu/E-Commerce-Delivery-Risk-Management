"""Central configuration: paths, chronological split boundaries, capacity, seeds."""
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = Path(os.environ.get("RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

SEED = 42

# Chronological design (all boundaries are on order_approved_at, Olist local time).
# Orders approved before TRAIN_START (Sep-Dec 2016, ~320 orders) are used only as
# history for the aggregate features ("warm-up"), never as training examples.
TRAIN_START = "2017-01-01"
VAL_START = "2018-03-01"      # validation = 2018-03-01 .. 2018-05-31
TEST_START = "2018-06-01"     # test       = 2018-06-01 .. 2018-08-31
TEST_END = "2018-09-01"       # exclusive

# Label-maturity window: a training example must have been approved at least this many days
# before the cutoff (and its outcome observed before the cutoff). Without it, the last weeks
# before a cutoff keep only the orders that were delivered quickly (mostly on time) -> the
# training labels would be biased toward "on time". 60 days covers ~99.7% of all and ~96% of
# late outcomes (measured in notebook 02).
LABEL_MATURITY_DAYS = 60

DAILY_CAPACITY = 10           # max investigations per calendar day (business rule)
BUDGETS = [5, 10, 20, 50]     # sensitivity analysis

# Smoothing strength (pseudo-orders) for historical late-rate features
SMOOTH_M = 20
# A never-delivered order is treated as "open" until promised date + this many days
OPEN_CAP_DAYS = 60

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}


def ensure_dirs():
    for d in (PROCESSED_DIR, OUTPUT_DIR, TABLE_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
