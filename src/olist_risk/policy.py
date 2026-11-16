"""Daily capacity-constrained prioritisation and its operational evaluation.

Daily decision process (identical for every model and rule):
  1. Candidate pool for calendar day D = every order whose order_approved_at falls on D
     (all statuses: orders later canceled are included because that is unknown at approval).
  2. Each order receives a score computed only from its approval-time features.
  3. Select the top-k orders (k = 10). Days with fewer than k approved orders: select all of
     them (k is a cap, not a quota). Ties are broken by the earlier approval timestamp, then order_id.
  4. Compare with eventual outcomes: late -> hit (correctly flagged delivery problem),
     on_time -> unnecessary investigation, undetermined (no delivery record: canceled,
     unavailable, still in transit at export) -> reported separately and excluded from
     precision-among-resolved; late orders not selected -> missed.
"""
import numpy as np
import pandas as pd
from . import config

# Non-ML business rules: higher value = higher priority
RULES = {
    "rule_shortest_promise": lambda df: -df["promised_days"].fillna(df["promised_days"].median()),
    "rule_longest_distance": lambda df: df["distance_km"].fillna(-1),
    "rule_highest_freight_ratio": lambda df: df["freight_ratio"].fillna(-1),
    "rule_worst_seller_history": lambda df: df["seller_late_rate_sm"].fillna(0),
}
RULE_DESCRIPTIONS = {
    "rule_shortest_promise": "Investigate the orders with the fewest days between approval and the promised delivery date",
    "rule_longest_distance": "Investigate the orders with the longest seller-to-customer distance",
    "rule_highest_freight_ratio": "Investigate the orders where freight is the largest share of the order value",
    "rule_worst_seller_history": "Investigate orders from sellers with the worst (smoothed) past late-delivery rate",
}


def select_daily_topk(pool, score_col, k=config.DAILY_CAPACITY):
    p = pool.sort_values(["approval_date", score_col, "order_approved_at", "order_id"],
                         ascending=[True, False, True, True], kind="mergesort")
    p = p.assign(daily_rank=p.groupby("approval_date").cumcount() + 1)
    return p.assign(selected=p.daily_rank <= k)


def evaluate_policy(pool, score_col, k=config.DAILY_CAPACITY):
    s = select_daily_topk(pool, score_col, k)
    sel = s[s.selected]
    late = s.outcome_status == "late"
    hits = int((sel.outcome_status == "late").sum())
    unnecessary = int((sel.outcome_status == "on_time").sum())
    undetermined = int((sel.outcome_status == "undetermined").sum())
    total_late = int(late.sum())
    # best achievable with perfect foresight under the same cap
    per_day_late = s.groupby("approval_date").apply(lambda g: (g.outcome_status == "late").sum(), include_groups=False)
    oracle = int(np.minimum(per_day_late, k).sum())
    days = s.approval_date.nunique()
    return {
        "days": days, "days_with_fewer_than_k_orders": int((s.groupby("approval_date").size() < k).sum()),
        "pool_orders": len(s), "pool_late": total_late, "investigations": len(sel),
        "late_identified": hits, "unnecessary_investigations": unnecessary,
        "undetermined_selected": undetermined, "late_missed": total_late - hits,
        "precision": hits / max(len(sel), 1),
        "precision_resolved": hits / max(hits + unnecessary, 1),
        "recall_of_late": hits / max(total_late, 1),
        "oracle_late_identified": oracle, "share_of_oracle": hits / max(oracle, 1),
        "base_rate": total_late / max(len(s), 1),
        "lift_vs_random": (hits / max(len(sel), 1)) / max(total_late / max(len(s), 1), 1e-9),
        "days_with_zero_hits": int((sel.groupby("approval_date").apply(lambda g: (g.outcome_status == "late").sum(), include_groups=False) == 0).sum()),
    }


def random_policy_expectation(pool, k=config.DAILY_CAPACITY, n_rep=200, seed=config.SEED):
    """Average over n_rep random daily selections (the 'no prioritisation' reference)."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_rep):
        p = pool.assign(_r=rng.random(len(pool)))
        res.append(evaluate_policy(p, "_r", k))
    df = pd.DataFrame(res)
    out = df.mean(numeric_only=True).to_dict()
    out["late_identified_sd"] = float(df.late_identified.std())
    return out


def daily_breakdown(pool, score_col, k=config.DAILY_CAPACITY):
    s = select_daily_topk(pool, score_col, k)
    g = s.groupby("approval_date")
    return pd.DataFrame({
        "orders_approved": g.size(),
        "late_in_pool": g.apply(lambda x: (x.outcome_status == "late").sum(), include_groups=False),
        "selected": g.selected.sum(),
        "late_identified": g.apply(lambda x: ((x.outcome_status == "late") & x.selected).sum(), include_groups=False),
        "unnecessary": g.apply(lambda x: ((x.outcome_status == "on_time") & x.selected).sum(), include_groups=False),
        "undetermined_selected": g.apply(lambda x: ((x.outcome_status == "undetermined") & x.selected).sum(), include_groups=False),
    }).reset_index()
