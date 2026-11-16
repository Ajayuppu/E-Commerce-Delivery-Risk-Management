"""Target definition, outcome status and eligibility.

Unit of prediction : one order (order_id).
Decision time      : order_approved_at (payment approved -> order enters fulfilment).
Target (is_late)   : 1 if the calendar date of order_delivered_customer_date is AFTER the
                     calendar date of order_estimated_delivery_date (the promised date is stored
                     at midnight, so a same-day delivery is on time), else 0.
Outcome status     : 'late' / 'on_time' when a customer delivery date exists;
                     'undetermined' otherwise (canceled, unavailable, shipped/invoiced/processing
                     with no delivery date) -> no label; never used for training or for predictive
                     metrics, but these orders ARE part of the daily candidate pool because at
                     approval time nobody knows they will be canceled.
"""
import numpy as np
import pandas as pd
from . import config


def build_labels(orders):
    o = orders.copy()
    delivered = o.order_delivered_customer_date.notna()
    o["outcome_known"] = delivered
    late = o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date.dt.normalize()
    o["is_late"] = np.where(delivered, late.astype(float), np.nan)
    o["outcome_status"] = np.select([delivered & late, delivered & ~late], ["late", "on_time"], "undetermined")
    o["days_late"] = (o.order_delivered_customer_date.dt.normalize() - o.order_estimated_delivery_date.dt.normalize()).dt.days
    # time at which the outcome becomes observable to the company
    o["outcome_known_at"] = o.order_delivered_customer_date
    o["approval_date"] = o.order_approved_at.dt.normalize()
    return o


def assign_split(approved_at):
    a = approved_at
    return np.select(
        [a.isna(), a < pd.Timestamp(config.TRAIN_START), a < pd.Timestamp(config.VAL_START),
         a < pd.Timestamp(config.TEST_START), a < pd.Timestamp(config.TEST_END)],
        ["not_approved", "warmup_history_only", "train", "validation", "test"], "after_test")


def eligibility_flags(lab):
    """Eligibility rules (documented in the report):
    E1  order has an approval timestamp (otherwise there is no decision point)      -> scored
    E2  approved inside train/validation/test window                               -> scored
    E3  for TRAINING/predictive metrics: outcome known (delivered to customer)
    E4  for TRAINING only: label mature at the cutoff, i.e. approved >= LABEL_MATURITY_DAYS
        before the cutoff AND outcome observed before the cutoff (cutoff = VAL_START for the
        model selected on validation, TEST_START for the final refit on train+validation).
    """
    lab = lab.copy()
    lab["split"] = assign_split(lab.order_approved_at)
    lab["in_daily_pool"] = lab.split.isin(["train", "validation", "test"])
    lab["label_available"] = lab.in_daily_pool & lab.outcome_known
    G = pd.Timedelta(days=config.LABEL_MATURITY_DAYS)
    v0, t0 = pd.Timestamp(config.VAL_START), pd.Timestamp(config.TEST_START)
    lab["train_eligible"] = ((lab.split == "train") & lab.outcome_known & (lab.outcome_known_at < v0)
                             & (lab.order_approved_at < v0 - G))
    lab["trainval_eligible"] = (lab.split.isin(["train", "validation"]) & lab.outcome_known
                                & (lab.outcome_known_at < t0) & (lab.order_approved_at < t0 - G))
    return lab


def eligibility_funnel(lab):
    rows = [("all orders in orders table", len(lab))]
    rows.append(("approved (has order_approved_at)", int(lab.order_approved_at.notna().sum())))
    for s in ["warmup_history_only", "train", "validation", "test", "after_test"]:
        rows.append((f"approved in split={s}", int((lab.split == s).sum())))
    for s in ["train", "validation", "test"]:
        m = lab.split == s
        rows.append((f"{s}: outcome known (delivered)", int((m & lab.outcome_known).sum())))
        rows.append((f"{s}: outcome undetermined", int((m & ~lab.outcome_known).sum())))
        rows.append((f"{s}: late", int((m & (lab.is_late == 1)).sum())))
    G = config.LABEL_MATURITY_DAYS
    rows.append((f"train rows usable for fitting (approved >= {G}d before VAL_START, outcome known before it)", int(lab.train_eligible.sum())))
    rows.append(("train rows with outcome but not mature at VAL_START (not used for fitting)", int(((lab.split == "train") & lab.outcome_known & ~lab.train_eligible).sum())))
    rows.append((f"train+val rows usable for final refit (approved >= {G}d before TEST_START, outcome known before it)", int(lab.trainval_eligible.sum())))
    return pd.DataFrame(rows, columns=["step", "orders"])
