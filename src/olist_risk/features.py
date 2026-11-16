"""Approval-time feature engineering with explicit leakage control.

Rules
-----
* Every table with several rows per order (items, payments) is AGGREGATED to one row per
  order BEFORE joining, and the result is asserted to have exactly one row per order_id.
* Geolocation has many rows per zip prefix -> collapsed to one median coordinate per prefix.
* Reviews, order_status, carrier/customer delivery dates are NEVER used as features.
* Historical aggregates (seller/customer/route/category/market history) use only records
  that were observable strictly before the scored order's approval time:
    - outcome-based aggregates (late rates) use only orders whose OUTCOME was known
      (order_delivered_customer_date) before the approval time, not merely orders approved
      before it;
    - activity aggregates (prior orders, open orders, orders last 7 days) use approval
      timestamps strictly before the scored approval time.
"""
import numpy as np
import pandas as pd
from . import config

CAT_FEATURES = ["customer_state", "seller_state", "main_category", "payment_type_main"]

NUM_FEATURES = [
    # timing / promise
    "approve_delay_hours", "promised_days", "promised_days_from_purchase", "days_to_shipping_limit",
    "approval_hour", "approval_dow", "purchase_hour", "is_weekend_approval",
    # basket
    "n_items", "n_products", "n_sellers", "total_price", "total_freight", "freight_ratio",
    "max_item_price", "total_weight_g", "total_volume_cm3", "max_weight_g", "category_missing",
    # payment
    "n_payment_rows", "max_installments", "payment_value", "used_voucher",
    # geography
    "distance_km", "same_state", "same_city", "customer_lat", "customer_lng", "seller_lat", "seller_lng",
    "geo_missing",
    # leakage-safe history
    "seller_prior_orders", "seller_orders_7d", "seller_days_active", "seller_prior_outcomes",
    "seller_late_rate_sm", "seller_open_orders", "seller_overdue_open",
    "customer_prior_orders", "cust_state_late_rate_sm", "route_late_rate_sm", "category_late_rate_sm",
    "market_late_rate_30d", "market_overdue_open", "market_orders_7d",
]
HISTORY_FEATURES = NUM_FEATURES[NUM_FEATURES.index("seller_prior_orders"):]
FEATURES = NUM_FEATURES + CAT_FEATURES
# Computed with the same leakage rules but NOT part of the pre-specified model. They are used only in
# the clearly labelled post-hoc analysis of notebook 06 (added after the final test evaluation).
POSTHOC_FEATURES = ["route_mean_delivery_days", "seller_mean_handling_days", "market_mean_delivery_days_30d",
                    "slack_vs_route", "slack_vs_market"]


# ----------------------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _cum_before(ev_key, ev_time, ev_val, q_key, q_time):
    """For each query (key, t): sum of ev_val over events with the same key and ev_time < t.
    Vectorised: sort events and queries together; on ties queries go first (strict <)."""
    ev = pd.DataFrame({"k": np.asarray(ev_key), "t": np.asarray(ev_time, dtype="datetime64[ns]"),
                       "v": np.asarray(ev_val, dtype=float), "q": 1, "i": -1})
    qq = pd.DataFrame({"k": np.asarray(q_key), "t": np.asarray(q_time, dtype="datetime64[ns]"),
                       "v": 0.0, "q": 0, "i": np.arange(len(q_key))})
    ev = ev[ev.t.notna() & pd.notna(ev.k)]
    allr = pd.concat([ev, qq], ignore_index=True).sort_values(["k", "t", "q"], kind="mergesort")
    allr["cs"] = allr.groupby("k", sort=False).v.cumsum()
    out = np.full(len(q_key), np.nan)
    qs = allr[allr.q == 0]
    out[qs.i.values] = qs.cs.values
    return out


def _active_at(st_key, start, end, q_key, q_time):
    """Number of intervals [start, end) with the same key that contain q_time,
    counting only intervals that started strictly before q_time. Intervals with a missing
    start or with end <= start (inconsistent timestamps) are ignored."""
    start, end = pd.Series(start).reset_index(drop=True), pd.Series(end).reset_index(drop=True)
    valid = start.notna() & end.notna() & (end > start)
    start, end = start.where(valid), end.where(valid)
    started = _cum_before(st_key, start, np.ones(len(start)), q_key, q_time)
    # an interval is finished at q_time if end <= q_time  <=> end < q_time + 1ns
    ended = _cum_before(st_key, end, np.ones(len(end)), q_key, pd.Series(q_time) + pd.Timedelta(1, "ns"))
    started, ended = np.nan_to_num(started), np.nan_to_num(ended)
    return started - ended


def _smoothed_rate(late, n, prior, m=config.SMOOTH_M):
    return (late + prior * m) / (n + m)


# ----------------------------------------------------------------------------- static
def build_static_features(t, lab):
    """Order-level, approval-time basket/payment/geo/timing features. One row per order."""
    o = lab[["order_id", "customer_id", "order_purchase_timestamp", "order_approved_at",
             "order_estimated_delivery_date"]].copy()

    # ---- items + products (aggregate first: 1 row per order)
    pr = t["products"].merge(t["translation"], on="product_category_name", how="left")
    pr["category_en"] = pr.product_category_name_english.fillna(pr.product_category_name).fillna("unknown")
    pr["volume_cm3"] = pr.product_length_cm * pr.product_height_cm * pr.product_width_cm
    it = t["items"].merge(pr[["product_id", "category_en", "product_weight_g", "volume_cm3"]], on="product_id", how="left")
    assert len(it) == len(t["items"]), "product join multiplied item rows"
    it = it.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
    main = it.drop_duplicates("order_id")[["order_id", "seller_id", "category_en"]].rename(
        columns={"seller_id": "main_seller_id", "category_en": "main_category"})
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "size"), n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"), total_price=("price", "sum"), total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"), total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"), total_volume_cm3=("volume_cm3", "sum"),
        shipping_limit=("shipping_limit_date", "max"),
        category_missing=("category_en", lambda s: float((s == "unknown").any())))
    agg = agg.reset_index().merge(main, on="order_id", how="left")

    # ---- payments (aggregate first)
    p = t["payments"].sort_values(["order_id", "payment_value"], ascending=[True, False])
    pmain = p.drop_duplicates("order_id")[["order_id", "payment_type"]].rename(columns={"payment_type": "payment_type_main"})
    pagg = p.groupby("order_id").agg(n_payment_rows=("payment_sequential", "size"),
                                     max_installments=("payment_installments", "max"),
                                     payment_value=("payment_value", "sum"),
                                     used_voucher=("payment_type", lambda s: float((s == "voucher").any())))
    pagg = pagg.reset_index().merge(pmain, on="order_id", how="left")

    # ---- geolocation: one median coordinate per zip prefix, inside Brazil's bounding box
    g = t["geolocation"]
    g = g[g.geolocation_lat.between(-34, 6) & g.geolocation_lng.between(-74, -34)]
    gz = g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    c = t["customers"].merge(gz, left_on="customer_zip_code_prefix", right_index=True, how="left").rename(
        columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    s = t["sellers"].merge(gz, left_on="seller_zip_code_prefix", right_index=True, how="left").rename(
        columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})

    df = o.merge(agg, on="order_id", how="left").merge(pagg, on="order_id", how="left")
    df = df.merge(c[["customer_id", "customer_unique_id", "customer_state", "customer_city", "customer_lat", "customer_lng"]],
                  on="customer_id", how="left")
    df = df.merge(s[["seller_id", "seller_state", "seller_city", "seller_lat", "seller_lng"]],
                  left_on="main_seller_id", right_on="seller_id", how="left").drop(columns="seller_id")
    assert len(df) == len(o) and df.order_id.is_unique, "join multiplied orders"

    a = df.order_approved_at
    df["approve_delay_hours"] = (a - df.order_purchase_timestamp).dt.total_seconds() / 3600
    df["promised_days"] = (df.order_estimated_delivery_date - a.dt.normalize()).dt.days
    df["promised_days_from_purchase"] = (df.order_estimated_delivery_date - df.order_purchase_timestamp.dt.normalize()).dt.days
    df["days_to_shipping_limit"] = (df.shipping_limit - a).dt.total_seconds() / 86400
    df["approval_hour"] = a.dt.hour
    df["approval_dow"] = a.dt.dayofweek
    df["purchase_hour"] = df.order_purchase_timestamp.dt.hour
    df["is_weekend_approval"] = (a.dt.dayofweek >= 5).astype(float)
    df["freight_ratio"] = df.total_freight / (df.total_price + df.total_freight)
    df["distance_km"] = haversine_km(df.customer_lat, df.customer_lng, df.seller_lat, df.seller_lng)
    df["geo_missing"] = df.distance_km.isna().astype(float)
    df["same_state"] = (df.customer_state == df.seller_state).astype(float)
    df["same_city"] = (df.customer_city == df.seller_city).astype(float)
    return df


# ----------------------------------------------------------------------------- history
def add_history_features(df, lab):
    """Leakage-safe historical aggregates. `lab` must contain ALL orders (incl. warm-up)."""
    d = df.merge(lab[["order_id", "is_late", "outcome_known_at", "order_delivered_customer_date",
                      "order_delivered_carrier_date"]], on="order_id", how="left")
    a = d.order_approved_at
    ok = d.outcome_known_at  # NaT when outcome never observed
    late = d.is_late.fillna(0).values
    has_out = d.outcome_known_at.notna().astype(float).values
    one = np.ones(len(d))
    appr_mask = a.notna().values

    # outcome events (only orders whose delivery was observed, at the delivery timestamp)
    def outcome_rate(key):
        n = np.nan_to_num(_cum_before(d[key], ok, has_out, d[key], a))
        l = np.nan_to_num(_cum_before(d[key], ok, late, d[key], a))
        return n, l

    # market-wide prior (global late rate among outcomes known before approval)
    gkey = pd.Series("all", index=d.index)
    gn = np.nan_to_num(_cum_before(gkey, ok, has_out, gkey, a))
    gl = np.nan_to_num(_cum_before(gkey, ok, late, gkey, a))
    prior = np.where(gn > 0, gl / np.maximum(gn, 1), 0.07)

    n, l = outcome_rate("main_seller_id")
    d["seller_prior_outcomes"] = n
    d["seller_late_rate_sm"] = _smoothed_rate(l, n, prior)
    for key, col in [("customer_state", "cust_state_late_rate_sm"), ("main_category", "category_late_rate_sm")]:
        n, l = outcome_rate(key)
        d[col] = _smoothed_rate(l, n, prior)
    d["route"] = d.seller_state.fillna("NA") + ">" + d.customer_state.fillna("NA")
    n, l = outcome_rate("route")
    d["route_late_rate_sm"] = _smoothed_rate(l, n, prior)

    # market late rate over outcomes observed in the last 30 days
    a30 = a - pd.Timedelta(days=30)
    n30 = gn - np.nan_to_num(_cum_before(gkey, ok, has_out, gkey, a30))
    l30 = gl - np.nan_to_num(_cum_before(gkey, ok, late, gkey, a30))
    d["market_late_rate_30d"] = np.where(n30 > 0, l30 / np.maximum(n30, 1), prior)

    # activity (approval-time) aggregates
    akey = d.main_seller_id
    appr_val = appr_mask.astype(float)
    d["seller_prior_orders"] = np.nan_to_num(_cum_before(akey, a, appr_val, akey, a))
    d["seller_orders_7d"] = d.seller_prior_orders - np.nan_to_num(_cum_before(akey, a, appr_val, akey, a - pd.Timedelta(days=7)))
    first = d[appr_mask].groupby("main_seller_id").order_approved_at.min()
    d["seller_days_active"] = (a - d.main_seller_id.map(first)).dt.total_seconds() / 86400
    d["customer_prior_orders"] = np.nan_to_num(_cum_before(d.customer_unique_id, a, appr_val, d.customer_unique_id, a))
    d["market_orders_7d"] = np.nan_to_num(_cum_before(gkey, a, appr_val, gkey, a)) - np.nan_to_num(
        _cum_before(gkey, a, appr_val, gkey, a - pd.Timedelta(days=7)))

    # open / overdue backlog observable at approval time
    cap = d.order_estimated_delivery_date + pd.Timedelta(days=config.OPEN_CAP_DAYS)
    end_open = d.order_delivered_customer_date.fillna(cap)
    st = a.where(appr_mask)
    d["seller_open_orders"] = _active_at(akey, st, end_open, akey, a)
    od_start = d.order_estimated_delivery_date + pd.Timedelta(days=1)       # overdue from the day after the promise
    od_valid = (end_open > od_start) & appr_mask
    od_s, od_e = od_start.where(od_valid), end_open.where(od_valid)
    d["seller_overdue_open"] = _active_at(akey, od_s, od_e, akey, a)
    d["market_overdue_open"] = _active_at(gkey, od_s, od_e, gkey, a)
    # ---- post-hoc features: typical delivery / handling time learned from outcomes known before approval
    deliv_days = ((d.order_delivered_customer_date.dt.normalize() - a.dt.normalize()).dt.days).where(d.outcome_known_at.notna()).fillna(0).values
    handl_days = ((d.order_delivered_carrier_date - a).dt.total_seconds() / 86400).where(
        d.outcome_known_at.notna() & d.order_delivered_carrier_date.notna()).values
    has_h = (~np.isnan(handl_days)).astype(float) * has_out
    handl_days = np.nan_to_num(handl_days)
    rn = np.nan_to_num(_cum_before(d.route, ok, has_out, d.route, a))
    rs = np.nan_to_num(_cum_before(d.route, ok, deliv_days, d.route, a))
    gs = np.nan_to_num(_cum_before(gkey, ok, deliv_days, gkey, a))
    g_mean = np.where(gn > 0, gs / np.maximum(gn, 1), 12.0)
    d["route_mean_delivery_days"] = (rs + g_mean * config.SMOOTH_M) / (rn + config.SMOOTH_M)
    sn = np.nan_to_num(_cum_before(akey, ok, has_h, akey, a))
    ss = np.nan_to_num(_cum_before(akey, ok, handl_days, akey, a))
    gh_n = np.nan_to_num(_cum_before(gkey, ok, has_h, gkey, a)); gh_s = np.nan_to_num(_cum_before(gkey, ok, handl_days, gkey, a))
    gh = np.where(gh_n > 0, gh_s / np.maximum(gh_n, 1), 3.0)
    d["seller_mean_handling_days"] = (ss + gh * config.SMOOTH_M) / (sn + config.SMOOTH_M)
    s30 = gs - np.nan_to_num(_cum_before(gkey, ok, deliv_days, gkey, a30))
    d["market_mean_delivery_days_30d"] = np.where(n30 > 0, s30 / np.maximum(n30, 1), g_mean)
    d["slack_vs_route"] = d.promised_days - d.route_mean_delivery_days
    d["slack_vs_market"] = d.promised_days - d.market_mean_delivery_days_30d
    return d.drop(columns=["is_late", "outcome_known_at", "order_delivered_customer_date", "order_delivered_carrier_date", "route"])


def build_feature_table(t, lab):
    df = build_static_features(t, lab)
    df = add_history_features(df, lab)
    keep = ["order_id", "customer_id", "customer_unique_id", "main_seller_id", "order_purchase_timestamp",
            "order_approved_at", "order_estimated_delivery_date"] + FEATURES + POSTHOC_FEATURES
    df = df[keep].merge(lab[["order_id", "order_status", "approval_date", "split", "is_late", "outcome_status",
                             "outcome_known", "outcome_known_at", "days_late", "in_daily_pool", "label_available",
                             "train_eligible", "trainval_eligible"]], on="order_id", how="left")
    assert df.order_id.is_unique and len(df) == len(lab)
    return df


# ----------------------------------------------------------------------------- audit table
def feature_availability_audit():
    """Feature-availability audit: every candidate field, its source, when it is observed,
    and whether it is used as a predictor."""
    R = []
    def add(name, source, observed, used, note):
        R.append({"feature_or_field": name, "source": source, "observed_at": observed,
                  "available_at_approval": None,
                  "used_as_predictor": used, "how_computed_or_why_excluded": note})
    A = "at/before approval"
    add("approve_delay_hours", "orders.purchase, orders.approved_at", A, "yes", "approved_at - purchase timestamp (h)")
    add("promised_days", "orders.estimated_delivery_date, approved_at", A + " (promise shown at purchase)", "yes", "promised date - approval date (days)")
    add("promised_days_from_purchase", "orders.estimated_delivery_date, purchase", A, "yes", "promised date - purchase date (days)")
    add("days_to_shipping_limit", "order_items.shipping_limit_date", A + " (seller hand-over deadline set when order is placed/approved)", "yes", "max shipping_limit - approved_at (days); assumption documented")
    add("approval_hour / approval_dow / is_weekend_approval / purchase_hour", "orders timestamps", A, "yes", "calendar parts")
    add("n_items, n_products, n_sellers, total_price, total_freight, freight_ratio, max_item_price", "order_items (aggregated per order)", A, "yes", "sum/count/max per order BEFORE join")
    add("total_weight_g, max_weight_g, total_volume_cm3, category_missing, main_category", "products + translation via items", A + " (catalogue attributes)", "yes", "main category = category of highest-priced item")
    add("payment_type_main, n_payment_rows, max_installments, payment_value, used_voucher", "order_payments (aggregated)", A + " (payment is what gets approved)", "yes", "aggregated per order BEFORE join")
    add("customer_state, seller_state, same_state, same_city", "customers, sellers (main seller)", A, "yes", "static location")
    add("customer/seller lat-lng, distance_km, geo_missing", "geolocation (median per zip prefix)", A, "yes", "zip-prefix medians, haversine distance")
    add("seller_prior_orders, seller_orders_7d, seller_days_active", "orders+items history", "approvals strictly before this approval", "yes", "cumulative counts by main seller")
    add("seller_prior_outcomes, seller_late_rate_sm", "orders history", "outcomes DELIVERED strictly before this approval", "yes", f"smoothed late rate, m={config.SMOOTH_M}, prior = market rate known at that time")
    add("seller_open_orders, seller_overdue_open, market_overdue_open", "orders history", "status as of approval time", "yes", "orders approved earlier and not yet delivered (overdue = past promised date)")
    add("customer_prior_orders", "orders + customers.customer_unique_id", "approvals strictly before", "yes", "repeat-customer count")
    add("cust_state_late_rate_sm, route_late_rate_sm, category_late_rate_sm, market_late_rate_30d", "orders history", "outcomes delivered strictly before approval", "yes", "smoothed historical late rates")
    add("route_mean_delivery_days, seller_mean_handling_days, market_mean_delivery_days_30d, slack_vs_route, slack_vs_market",
        "orders history (carrier/customer dates of orders DELIVERED before approval)", "outcomes delivered strictly before approval",
        "post-hoc only", "added after the final test run; used only in the labelled post-hoc analysis (notebook 06)")
    add("order_status", "orders", "end of lifecycle (snapshot at data export)", "no", "final status is a future outcome -> leakage")
    add("order_delivered_carrier_date", "orders", "after approval (carrier pickup)", "no", "post-approval event -> leakage")
    add("order_delivered_customer_date", "orders", "after approval (delivery)", "label only", "used ONLY to build is_late and outcome_known_at")
    add("is_late / days_late", "derived", "after delivery", "label only", "target")
    add("review_score, review_comment_*, review_creation_date, review_answer_timestamp", "order_reviews", "after delivery (or after problems)", "no", "post-outcome feedback -> leakage; not joined at all")
    add("customer_id (raw id), order_id, seller_id (raw id)", "ids", A, "no", "identifiers kept only for traceability/history keys")
    for r in R:
        r["available_at_approval"] = "no" if r["used_as_predictor"] in ("no", "label only") and "ids" not in r["source"] else "yes"
    return pd.DataFrame(R)


# ----------------------------------------------------------------------------- leakage test
def verify_history_bruteforce(ft, lab, n=300, seed=config.SEED):
    """Recompute seller history for a random sample with plain boolean filters and compare
    with the vectorised features. Returns a DataFrame; all `match_*` columns must be True."""
    rng = np.random.default_rng(seed)
    pool = ft[ft.in_daily_pool].reset_index(drop=True)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    full = ft[["order_id", "main_seller_id", "order_approved_at", "customer_unique_id"]].merge(
        lab[["order_id", "is_late", "outcome_known_at"]], on="order_id")
    rows = []
    for i in idx:
        r = pool.iloc[i]
        same = full[full.main_seller_id == r.main_seller_id]
        known = same[same.outcome_known_at < r.order_approved_at]           # outcome observed before approval
        approved_before = same[same.order_approved_at < r.order_approved_at]
        cust = full[(full.customer_unique_id == r.customer_unique_id) & (full.order_approved_at < r.order_approved_at)]
        rows.append({
            "order_id": r.order_id, "approved_at": r.order_approved_at,
            "bf_seller_prior_outcomes": len(known), "ft_seller_prior_outcomes": r.seller_prior_outcomes,
            "bf_seller_prior_orders": len(approved_before), "ft_seller_prior_orders": r.seller_prior_orders,
            "bf_customer_prior_orders": len(cust), "ft_customer_prior_orders": r.customer_prior_orders,
            "own_outcome_used": bool((known.order_id == r.order_id).any()),
            "latest_outcome_used": known.outcome_known_at.max(),
        })
    out = pd.DataFrame(rows)
    out["match_outcomes"] = out.bf_seller_prior_outcomes == out.ft_seller_prior_outcomes
    out["match_orders"] = out.bf_seller_prior_orders == out.ft_seller_prior_orders
    out["match_customer"] = out.bf_customer_prior_orders == out.ft_customer_prior_orders
    out["no_future_outcome"] = out.latest_outcome_used.isna() | (out.latest_outcome_used < out.approved_at)
    return out
