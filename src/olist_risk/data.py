"""Loading the raw Olist CSVs and auditing them (keys, duplicates, missingness, ranges)."""
import pandas as pd
from . import config

DATE_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}
PRIMARY_KEYS = {
    "orders": ["order_id"], "items": ["order_id", "order_item_id"], "customers": ["customer_id"],
    "sellers": ["seller_id"], "products": ["product_id"],
    "payments": ["order_id", "payment_sequential"], "reviews": ["review_id", "order_id"],
    "geolocation": None, "translation": ["product_category_name"],
}


def load_raw(raw_dir=None):
    """Read all nine CSVs. Zip-code prefixes are kept as zero-padded strings."""
    raw_dir = raw_dir or config.RAW_DIR
    t = {}
    for name, fn in config.RAW_FILES.items():
        dtype = {}
        if name in ("customers", "sellers", "geolocation"):
            col = {"customers": "customer_zip_code_prefix", "sellers": "seller_zip_code_prefix",
                   "geolocation": "geolocation_zip_code_prefix"}[name]
            dtype[col] = str
        df = pd.read_csv(raw_dir / fn, dtype=dtype, parse_dates=DATE_COLS.get(name, []))
        for c in dtype:
            df[c] = df[c].str.zfill(5)
        t[name] = df
    return t


def audit_tables(t):
    """One row per table: shape, primary-key uniqueness, missing cells."""
    rows = []
    for name, df in t.items():
        pk = PRIMARY_KEYS[name]
        rows.append({
            "table": name, "rows": len(df), "columns": df.shape[1],
            "primary_key": "+".join(pk) if pk else "(none)",
            "pk_unique": bool(not df.duplicated(pk).any()) if pk else None,
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "cells_missing": int(df.isna().sum().sum()),
            "columns_with_missing": ", ".join(f"{c}={n}" for c, n in df.isna().sum().items() if n > 0)[:200],
        })
    return pd.DataFrame(rows)


def audit_relationships(t):
    """Cardinality / referential checks that decide how tables may be joined."""
    o, it, p, rv = t["orders"], t["items"], t["payments"], t["reviews"]
    per_order_items = it.groupby("order_id").size()
    per_order_sellers = it.groupby("order_id").seller_id.nunique()
    per_order_pay = p.groupby("order_id").size()
    per_order_rev = rv.groupby("order_id").size()
    geo_zip_dupes = t["geolocation"].groupby("geolocation_zip_code_prefix").size()
    checks = {
        "orders: unique order_id": o.order_id.is_unique,
        "orders: customer_id unique per order (1:1)": o.customer_id.is_unique,
        "orders without any item row": int((~o.order_id.isin(it.order_id)).sum()),
        "orders without any payment row": int((~o.order_id.isin(p.order_id)).sum()),
        "orders with >1 item row (join would multiply)": int((per_order_items > 1).sum()),
        "max item rows per order": int(per_order_items.max()),
        "orders with >1 seller": int((per_order_sellers > 1).sum()),
        "orders with >1 payment row": int((per_order_pay > 1).sum()),
        "orders with >1 review row": int((per_order_rev > 1).sum()),
        "geolocation rows per zip prefix (median)": float(geo_zip_dupes.median()),
        "geolocation rows per zip prefix (max)": int(geo_zip_dupes.max()),
        "customer zips missing from geolocation": int((~t["customers"].customer_zip_code_prefix.isin(t["geolocation"].geolocation_zip_code_prefix)).sum()),
        "seller zips missing from geolocation": int((~t["sellers"].seller_zip_code_prefix.isin(t["geolocation"].geolocation_zip_code_prefix)).sum()),
        "products without category": int(t["products"].product_category_name.isna().sum()),
        "categories missing from translation": int((~t["products"].product_category_name.dropna().isin(t["translation"].product_category_name)).sum()),
        "items whose product_id not in products": int((~it.product_id.isin(t["products"].product_id)).sum()),
        "items whose seller_id not in sellers": int((~it.seller_id.isin(t["sellers"].seller_id)).sum()),
    }
    return pd.DataFrame({"check": list(checks), "value": [str(v) for v in checks.values()]})


def audit_order_dates(o):
    """Inconsistencies in the order lifecycle timestamps."""
    checks = {
        "approved_at missing": int(o.order_approved_at.isna().sum()),
        "approved before purchase": int((o.order_approved_at < o.order_purchase_timestamp).sum()),
        "delivered_customer missing": int(o.order_delivered_customer_date.isna().sum()),
        "status=delivered but delivered_customer missing": int(((o.order_status == "delivered") & o.order_delivered_customer_date.isna()).sum()),
        "status!=delivered but delivered_customer present": int(((o.order_status != "delivered") & o.order_delivered_customer_date.notna()).sum()),
        "delivered to customer before carrier pickup": int((o.order_delivered_customer_date < o.order_delivered_carrier_date).sum()),
        "carrier pickup before approval": int((o.order_delivered_carrier_date < o.order_approved_at).sum()),
        "delivered to customer before approval": int((o.order_delivered_customer_date < o.order_approved_at).sum()),
        "estimated delivery has a time-of-day (non-midnight)": int((o.order_estimated_delivery_date.dt.hour != 0).sum()),
        "purchase range": f"{o.order_purchase_timestamp.min()} .. {o.order_purchase_timestamp.max()}",
        "approval range": f"{o.order_approved_at.min()} .. {o.order_approved_at.max()}",
        "customer delivery range": f"{o.order_delivered_customer_date.min()} .. {o.order_delivered_customer_date.max()}",
    }
    return pd.DataFrame({"check": list(checks), "value": [str(v) for v in checks.values()]})
