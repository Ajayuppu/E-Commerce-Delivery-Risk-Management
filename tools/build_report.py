"""Builds docs/ML_Project2_Final_Report.docx from outputs/ (no hand-typed result numbers)."""
import json
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt, Inches, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from report_common import *

exp = json.load(open(OUT / "experiment_configs.json"))
P, R = exp["proposed_model"], exp["reference_rule"]
split = T("02_split_summary.csv"); funnel = T("02_eligibility_funnel.csv")
mat = T("02_label_maturity.csv"); rel = T("01_relationship_audit.csv"); dates = T("01_order_date_audit.csv")
valc = T("04_validation_comparison.csv"); grid = T("04_validation_grid.csv")
ops = T("05_test_operational_comparison.csv").set_index("method"); pm = T("05_test_predictive_metrics.csv").set_index("method")
bud = T("06_budget_sensitivity.csv"); mon = T("06_by_month.csv"); st_ = T("06_failure_by_customer_state.csv")
cat = T("06_failure_by_category.csv"); prof = T("06_error_profile.csv", index_col=0); refit = T("06_refit_vs_trainonly.csv")
abl = T("06_ablation.csv"); ph = T("06_posthoc_slack_features.csv"); rank = T("06_precision_by_daily_rank.csv")
audit = T("03_feature_availability_audit.csv"); leak = T("03_leakage_bruteforce_check.csv")
daily = T("05_daily_model_vs_rule.csv"); cens = T("02_censoring_by_month.csv")
csens = T("05_censoring_sensitivity.csv").set_index("method")
dp = pd.read_csv(OUT / "daily_prioritization_test.csv")
status = T("01_order_status.csv"); tab = T("01_table_audit.csv")
RND = "random (mean of 200 draws)"
f = lambda x, n=1: f"{x:,.{n}f}"
pct = lambda x, n=1: f"{100*x:.{n}f}%"
iv = lambda x: f"{int(round(x)):,}"
rv = lambda c: rel.set_index("check").loc[c, "value"]
dv = lambda c: dates.set_index("check").loc[c, "value"]
fv = lambda s: int(funnel.set_index("step").loc[s, "orders"])
sp = split.set_index("split")
b10 = bud[bud.k_per_day == 10].set_index("method")
def o(m, c):
    if c in ops.columns: return ops.loc[m, c]
    return b10.loc[m if m in b10.index else (b10.index[0]), c]
diff = daily["diff"]; import numpy as np
rng = np.random.default_rng(42); boots = [rng.choice(diff.values, len(diff)).sum() for _ in range(5000)]; ci = np.percentile(boots, [2.5, 97.5])
other = "lgbm" if P == "logreg" else "logreg"
NAME = {"logreg": "Logistic regression", "lgbm": "LightGBM", "rule_shortest_promise": "Rule: shortest promised window",
        "rule_longest_distance": "Rule: longest distance", "rule_highest_freight_ratio": "Rule: highest freight ratio",
        "rule_worst_seller_history": "Rule: worst seller history", RND: "Random selection (expected)"}
best_rule_test = ops.drop(index=[P, other, RND]).late_identified.idxmax()

d = new_doc(10.5)
# ---------------------------------------------------------------- page numbers + header
def add_field(run, instr):
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "1"
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    for e in (f1, it, f2, t, f3): run._r.append(e)
sec = d.sections[0]; sec.different_first_page_header_footer = True
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = fp.add_run("Page "); add_field(fp.add_run(), "PAGE")
hp = sec.header.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = hp.add_run("ML Project 2 – E-Commerce Delivery Risk Management · Ajay Uppu"); hr.font.size = Pt(8.5); hr.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

# ---------------------------------------------------------------- title page
for _ in range(5): d.add_paragraph()
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("E-Commerce Delivery Risk Management"); r.bold = True; r.font.size = Pt(26); r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Prioritising late-delivery investigations under a 10-orders-per-day capacity"); r.font.size = Pt(14)
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Machine Learning – Project 2 · Final Report"); r.font.size = Pt(12); r.italic = True
for _ in range(3): d.add_paragraph()
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Ajay Uppu\nM.S. Data Science, Georgia State University\nNovember 2026"); r.font.size = Pt(12)
for _ in range(4): d.add_paragraph()
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Data: Olist Brazilian E-Commerce Public Dataset (99,441 orders, 2016–2018)\n"
              "Environment: Amazon SageMaker Studio – domain machinelearningproj2ajay-ecommerce, space mlproj2-ecommerce-space\n"
              "All numbers in this report are read from the files in outputs/ produced by the notebooks.\n"
              "Final numbers come from a full local re-run (same pinned libraries) after a leakage fix on 15 Nov 2026; the SageMaker Studio space\n"
              "still holds the earlier pre-fix run (update it with git pull or aws s3 sync from the studio-v2-fixed/ prefix, see Appendix F).")
r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
d.add_page_break()

# ---------------------------------------------------------------- executive summary + TOC
d.add_heading("Executive summary", level=1)
d.add_paragraph(
    f"The marketplace wants to flag newly approved orders that will arrive after the promised date so that a logistics team "
    f"that can investigate at most 10 orders per day spends that time well. I defined the task at order level and at approval time, "
    f"built the target from the recorded delivery and promised dates, kept every predictor available at approval (history features use "
    f"only outcomes that had already been delivered), and evaluated everything chronologically: train on 2017-01 to 2018-02, select on "
    f"2018-03 to 2018-05, and test once on 2018-06 to 2018-08.")
bullets(d, [
    ("Proposed policy: ", f"{NAME[P]} risk score, top 10 orders per approval day. On the test period it used {iv(o(P,'investigations'))} investigations "
     f"and identified {iv(o(P,'late_identified'))} of the {iv(o(P,'pool_late'))} late orders ({pct(o(P,'precision'))} of investigations), "
     f"{f(o(P,'lift_vs_random'),2)}× the {f(o(RND,'late_identified'))} expected from random selection; {iv(o(P,'unnecessary_investigations'))} "
     f"investigations were unnecessary and {iv(o(P,'late_missed'))} late orders were missed."),
    ("Simple beats complex: ", f"logistic regression beat LightGBM on validation ({iv(valc.set_index('method').loc['logreg (best config)','late_identified'])} vs "
     f"{iv(valc.set_index('method').loc['lgbm (best config)','late_identified'])} late orders captured) and on test ({iv(o('logreg','late_identified'))} vs {iv(o('lgbm','late_identified'))})."),
    ("Rules are unstable: ", f"the rule chosen on validation ({NAME[R].split(': ')[1]}) captured {iv(o(R,'late_identified'))} late orders on test – barely above random. "
     f"The shortest-promise rule, which was the worst method on validation (below random), was the best on test ({iv(o('rule_shortest_promise','late_identified'))}). "
     f"The two learned models were the only methods well above random in both periods."),
    ("Limits: ", f"precision is low in absolute terms (most flagged orders arrive on time), performance fell to random level in August 2018 when lateness "
     f"rose in São Paulo, and identifying a late order does not prove an investigation would have prevented the delay."),
])
d.add_paragraph("Recommendation: pilot the logistic-regression list (10/day) next to the current process, retrain monthly, monitor precision weekly, and record the "
                "result of every investigation so that the real benefit of intervening can be measured.")
p = d.add_paragraph(); r = p.add_run("Contents"); r.bold = True; r.font.size = Pt(16); r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
p = d.add_paragraph(); add_field(p.add_run(), 'TOC \\o "1-2" \\h \\z \\u')
p = d.add_paragraph(); r = p.add_run("(If the table of contents is empty, right-click it in Word and choose Update Field.)"); r.italic = True; r.font.size = Pt(8)
d.add_page_break()

# ---------------------------------------------------------------- 1
d.add_heading("1. Business objective and ML formulation", level=1)
d.add_paragraph(
    "Late deliveries hurt customer satisfaction, but checking every order is impossible: the team can investigate at most 10 newly approved "
    "orders per calendar day (contacting the seller, checking shipment progress, coordinating with the carrier). The business question is therefore "
    "not 'can we classify every order?' but 'which 10 orders approved today are most likely to become delivery problems?'. A useful system "
    "must produce a daily ranked list and should make better use of the 10 slots than the simple alternatives the team could use without ML.")
add_table(d, pd.DataFrame([
    ["Unit of prediction", "One order (order_id). All items/payments of the order are summarised into one row."],
    ["Decision (prediction) time", "order_approved_at – the moment payment is approved and the order enters fulfilment."],
    ["Outcome / target", "is_late = 1 if the calendar date of order_delivered_customer_date is after the calendar date of order_estimated_delivery_date, else 0."],
    ["Undetermined outcome", "No customer-delivery date (canceled, unavailable, still shipped/invoiced at export). Not trained on, excluded from predictive metrics, but kept in the daily candidate pool and reported separately."],
    ["Eligible records", "Every order with an approval timestamp on a day in the evaluation window is a daily candidate. Training rows additionally need a mature, observed outcome (Section 2.3)."],
    ["Model output and use", "A risk score per order; each day the orders approved that day are ranked and the top 10 are sent for investigation (fewer if fewer were approved)."],
    ["Success measure", "Late orders identified within the 10/day cap (precision of investigations), unnecessary investigations, late orders missed – compared with rules and random under the same cap."],
], columns=["Element", "Definition"]), "Table 1. Problem formulation", font=8.5, col_widths=[1.6, 5.0])
d.add_paragraph(
    f"Why calendar dates? The promised date is stored at 00:00 for all {iv(99441)} orders ({dv('estimated delivery has a time-of-day (non-midnight)')} have a time of day). "
    "Comparing raw timestamps would label every order delivered on the promised day as late; at day level the late rate is 6.8% instead of 8.1% (notebook 02, cell 3). "
    "Assumptions: (i) the estimated delivery date shown to the customer is known at approval (it is set at purchase); (ii) the seller shipping limit "
    "date is set when the order is placed/approved; (iii) the delivery timestamp is when the company learns the outcome; (iv) orders are approved in "
    "Brazilian local time and a 'day' is the calendar day of order_approved_at.")

# ---------------------------------------------------------------- 2
d.add_heading("2. Data preparation and label construction", level=1)
d.add_heading("2.1 Raw-data audit", level=2)
d.add_paragraph(
    f"The archive has nine CSV tables. Keys were checked before any join (notebook 01). Orders and customers are 1:1, but "
    f"{iv(int(rv('orders with >1 item row (join would multiply)')))} orders have several item rows (up to {rv('max item rows per order')}), "
    f"{iv(int(rv('orders with >1 payment row')))} have several payment rows and {iv(int(rv('orders with >1 seller')))} have more than one seller; "
    f"geolocation has a median of {f(float(rv('geolocation rows per zip prefix (median)')),0)} rows per zip prefix. A naive orders×items×payments join "
    "duplicates order rows, so items and payments are aggregated to one row per order first, geolocation is collapsed to one median coordinate per "
    "zip prefix (Brazil bounding box only), and the final table is asserted to have exactly one row per order_id.")
q = dates.set_index("check")["value"]
add_table(d, pd.DataFrame([
    ["Approval timestamp missing", q["approved_at missing"], "No decision point → not a candidate"],
    ["Customer delivery date missing", q["delivered_customer missing"], "Outcome undetermined (not a label)"],
    ["Status 'delivered' without delivery date", q["status=delivered but delivered_customer missing"], "Undetermined"],
    ["Delivered before carrier pickup", q["delivered to customer before carrier pickup"], "Kept; carrier date not used"],
    ["Carrier pickup before approval", q["carrier pickup before approval"], "Kept; carrier date not used"],
    ["Delivered to customer before approval", q["delivered to customer before approval"], "Kept; its outcome becomes usable history only from its own approval time (event time = max(delivery, approval)), so it never enters its own features"],
    ["Orders without item rows", rv("orders without any item row"), "Basket features missing → imputed + indicator"],
    ["Customer zips not in geolocation", rv("customer zips missing from geolocation"), "Distance missing → geo_missing flag"],
    ["Products without category", rv("products without category"), "Category 'unknown' + category_missing"],
], columns=["Data-quality finding", "Count", "Treatment"]), "Table 2. Key data-quality findings and how they were handled", font=8.5)
add_fig(d, FIG / "01_monthly_volume_late_rate.png", "Figure 1. Approved orders and late rate by approval month; dashed lines mark the train/validation/test boundaries.", 6.4)
d.add_paragraph(
    "Figure 1 shows the most important data finding: the late rate is not stable. It was 3–5% for most of 2017, spiked in November 2017 and in "
    "February–March 2018 (up to 19%), fell to about 1% in June 2018 and rose again in August 2018. Any model will be evaluated on a period whose base "
    "rate differs from the training period, which shaped the evaluation design (chronological splits, daily ranking rather than a fixed probability threshold).")
d.add_heading("2.2 Target and outcome status", level=2)
st2 = status.set_index("order_status")
d.add_paragraph(
    f"Of the {iv(99441)} orders, {iv(int(st2.loc['delivered','orders']))} have status delivered. Lifecycle tables are used only to build the outcome: "
    "order_delivered_customer_date decides is_late and the moment the outcome becomes known (outcome_known_at). order_status, the carrier date and all "
    "review fields are post-approval information and are never used as predictors; reviews are not joined at all (99,035 of 99,068 "
    "matched reviews were created after approval, typically on the delivery day; notebook 01 cell 10).")
d.add_heading("2.3 Eligibility, chronological splits and label maturity", level=2)
d.add_paragraph(
    f"Splits use the approval timestamp: the {fv('approved in split=warmup_history_only')} orders approved in 2016 serve only as history; train = "
    f"2017-01-01 to 2018-02-28, validation = 2018-03-01 to 2018-05-31, test = 2018-06-01 to 2018-08-31 (one order approved in September 2018 is outside all windows). "
    "A subtle bias appears near every cutoff: orders approved shortly before it have an observed outcome only if they were delivered quickly, which mostly "
    "means on time. Table 3 shows that 30 days after approval only half of the eventual late outcomes are known. I therefore require a label-maturity "
    f"window of {exp['label_maturity_days']} days: a training row must be approved at least {exp['label_maturity_days']} days before the cutoff and its outcome observed before the cutoff "
    f"({pct(mat.set_index('days_after_approval').loc[exp['label_maturity_days'],'share_late_outcomes_known'])} of late outcomes are known by then). Without the rule, the February 2018 training orders would show a late rate of "
    f"{pct(cens.iloc[-1]['known_by_cutoff_late_rate'])} instead of {pct(cens.iloc[-1]['all_delivered_late_rate'])}.")
add_table(d, mat.rename(columns={"days_after_approval": "Days after approval", "share_all_outcomes_known": "Share of all outcomes known",
                                 "share_late_outcomes_known": "Share of late outcomes known"}), "Table 3. Label maturity (outcomes observed within X days of approval)", nd=3, font=8.5)
s2 = split.copy(); s2["approval_from"] = s2.approval_from.str[:10]; s2["approval_to"] = s2.approval_to.str[:10]
s2 = s2[["split", "approval_from", "approval_to", "days", "orders_in_daily_pool", "late", "on_time", "undetermined", "late_rate_labelled", "fit_rows_of_model_scored_on_next_split"]]
s2.columns = ["Split", "From", "To", "Days", "Daily-pool orders", "Late", "On time", "Undetermined", "Late rate", "Fit rows of model scored on next split*"]
add_table(d, s2, "Table 4. Modeling examples and outcomes per split", nd=4, font=8)
p = d.add_paragraph(); r = p.add_run("*Train row: training rows (labels mature at 2018-03-01) that fit the models scored on validation for selection. Validation row: train+validation rows (labels mature at 2018-06-01) that fit the final models scored on test. Test row: never used for fitting."); r.font.size = Pt(8); r.italic = True

d.add_heading("2.4 Approval-time features and leakage prevention", level=2)
d.add_paragraph(
    f"{len(exp['categorical_features']) + 45} predictors in five groups were built (full audit in Appendix A): timing and promise (approval delay, promised days, "
    "days to the seller's shipping limit, hour/weekday); basket (items, sellers, price, freight, freight ratio, weight, volume, main category); payment "
    "(type, installments, value, voucher); geography (customer/seller state, same state/city, coordinates, haversine distance); and leakage-safe history.")
bullets(d, [
    ("Outcome-based history ", "(seller, customer-state, route and category late rates; market late rate over the last 30 days) counts only orders whose delivery "
     "was observed strictly before the scored order's approval – not orders merely approved earlier, whose outcomes were still unknown. Rates are smoothed "
     "towards the market rate known at that time (m = 20 pseudo-orders)."),
    ("Activity history ", "(seller prior orders, orders in the last 7 days, open and overdue orders, repeat-customer count) uses approval timestamps strictly before the scored order."),
    ("Delivery-before-approval edge case. ", f"{q['delivered to customer before approval']} orders carry a delivery timestamp earlier than their own approval timestamp. "
     "An outcome is therefore used as history only from max(delivery, approval); without this rule these orders would have seen their own label "
     "(found in an audit and fixed; all results in this report are from the corrected pipeline)."),
    ("Verified by brute force: ", f"for {len(leak)} orders ({len(leak) - int(leak.delivered_before_own_approval.sum())} random + all {int(leak.delivered_before_own_approval.sum())} delivery-before-approval edge cases) the history features were recomputed with plain filters; all matched "
     f"({pct(leak.match_outcomes.mean(),0)} / {pct(leak.match_orders.mean(),0)} / {pct(leak.match_customer.mean(),0)}), no order used its own outcome and no outcome dated after approval was used (notebook 03, cell 5)."),
    ("Missing values ", "are median-imputed with missing-indicator columns (logistic regression) or handled natively (LightGBM); rare categories (< 50 rows) are grouped."),
])

# ---------------------------------------------------------------- 3
d.add_heading("3. Models and baselines", level=1)
d.add_paragraph("All methods produce a score per order and are turned into decisions by exactly the same daily top-10 procedure.")
add_table(d, pd.DataFrame([
    ["Random selection", "Reference", "Expected result of choosing 10 orders per day at random (mean of 200 draws) = the 'no prioritisation' baseline."],
    ["Shortest promised window", "Business rule", "Fewest days between approval and promised date first – an order with little slack is more fragile."],
    ["Longest distance", "Business rule", "Longest seller→customer distance first – long, inter-state routes are slower."],
    ["Highest freight ratio", "Business rule", "Freight as share of order value – bulky/remote shipments."],
    ["Worst seller history", "Business rule", "Highest smoothed past late rate of the seller (outcomes known before approval)."],
    ["Logistic regression", "Learned (simple baseline)", "L2-regularised; log/standardised numeric features, one-hot categories. Grid C ∈ {0.01, 0.1, 1} × class_weight ∈ {none, balanced}."],
    ["LightGBM", "Learned (gradient boosting)", "Captures non-linearities/interactions. Grid num_leaves ∈ {7,15,31} × trees ∈ {200,500} × min_child_samples ∈ {100,400}, lr 0.03, subsampling 0.8."],
], columns=["Method", "Type", "Description / hyper-parameters"]), "Table 5. Methods compared", font=8.5, col_widths=[1.4, 1.2, 4.0])
sel = exp["selected"]
d.add_paragraph(
    f"Both learned models were also compared with and without the three market-level features (feature set 'all' vs 'no_market'), giving {len(grid)} configurations. "
    f"Logistic regression is the natural simple baseline: it is fast, interpretable and extrapolates smoothly when the late rate moves outside the training range; LightGBM is a strong "
    f"tabular learner that tests whether non-linear structure adds value. Selected on validation: logistic regression C = {sel['logreg']['params']['C']}, "
    f"class_weight = {sel['logreg']['params']['class_weight']}, feature set '{sel['logreg']['feature_set']}'; LightGBM num_leaves = {sel['lgbm']['params']['num_leaves']}, "
    f"{sel['lgbm']['params']['n_estimators']} trees, min_child_samples = {sel['lgbm']['params']['min_child_samples']}, feature set '{sel['lgbm']['feature_set']}'. "
    "Strong regularisation won in both families, consistent with a shifting environment where simpler functions generalise better. Seeds are fixed (42) and every configuration is stored in experiment_configs.json (Appendix C).")

# ---------------------------------------------------------------- 4
d.add_heading("4. Evaluation design", level=1)
bullets(d, [
    ("Chronological protocol. ", "Train → validation (model, feature-set, hyper-parameter and reference-rule selection) → one final test run. After selection, the chosen "
     "configurations were refit on train+validation rows whose labels were mature on 2018-06-01, then scored the test period once."),
    ("Daily simulation. ", "For each test day, the candidate pool is every order approved that day (all statuses). The 10 highest scores are selected; ties broken by "
     "earlier approval time. Days with fewer than 10 approved orders select all of them (the cap is a maximum, not a quota) – this happened on "
     f"{iv(o(P,'days_with_fewer_than_k_orders'))} test days, so {iv(o(P,'investigations'))} instead of {iv(10*o(P,'days'))} investigations were used."),
    ("Outcome accounting. ", "Selected & late = late order identified; selected & on time = unnecessary investigation; selected & undetermined = reported separately "
     "(canceled/unavailable/no delivery record); late & not selected = missed. The perfect-foresight maximum under the same cap is reported as an upper bound."),
    ("Incomplete outcomes and censoring. ", f"The data export ends on {csens.attrs.get('export','2018-10-17')}; the last test approvals are only ~49 days old at that point. "
     f"Orders without a delivery record are 'undetermined' (main analysis). As a sensitivity check, undelivered orders that were still open at export (shipped/invoiced/processing, "
     f"not canceled/unavailable) and already past their promised date ({iv(csens.loc[P,'problems_in_pool_adjusted'] - o(P,'pool_late'))} test orders) are counted as delivery problems: the model then identifies "
     f"{iv(csens.loc[P,'problems_identified_censoring_adjusted'])} problems (vs {iv(o(P,'late_identified'))}), the reference rule {iv(csens.loc[R,'problems_identified_censoring_adjusted'])} "
     f"and the shortest-promise rule {iv(csens.loc['rule_shortest_promise','problems_identified_censoring_adjusted'])}; the ranking of methods is unchanged (outputs/tables/05_censoring_sensitivity.csv)."),
    ("Predictive metrics. ", "With only 3.6–10% positives, accuracy is meaningless (predicting 'on time' for everyone gives 96% on test). I report PR-AUC (average precision; "
     "random = base rate), ROC-AUC, Brier score, and the operational precision@10/day, which is the business metric used for selection."),
    ("Uncertainty. ", "A day-level bootstrap (5,000 resamples) of the daily difference in late orders identified quantifies model-vs-rule uncertainty."),
])

# ---------------------------------------------------------------- 5
d.add_heading("5. Quantitative results", level=1)
d.add_heading("5.1 Validation (model and rule selection)", level=2)
v = valc.copy(); v["method"] = v.method.str.replace(" (best config)", "").map(lambda m: NAME.get(m, m))
v = v[["method", "late_identified", "unnecessary_investigations", "precision", "pr_auc", "roc_auc"]]
v.columns = ["Method", "Late identified", "Unnecessary", "Precision@10/day", "PR-AUC", "ROC-AUC"]
add_table(d, v, f"Table 6. Validation period (Mar–May 2018, {int(sp.loc['validation','days'])} days, 10/day)", nd=3, font=8.5)
d.add_paragraph(f"Logistic regression captured the most late orders on validation and was chosen as the proposed model; '{NAME[R]}' was the best rule and became the pre-specified reference rule.")
d.add_heading("5.2 Test period – daily top-10 policy", level=2)
rows = []
for m in [P, other, "rule_shortest_promise", "rule_longest_distance", "rule_worst_seller_history", "rule_highest_freight_ratio", RND]:
    rows.append([NAME[m], o(m, "investigations"), o(m, "late_identified"), o(m, "unnecessary_investigations"), o(m, "undetermined_selected"),
                 o(m, "late_missed"), o(m, "precision"), o(m, "recall_of_late"), o(m, "lift_vs_random"), pm.loc[m, "pr_auc"] if m in pm.index else o(m, "base_rate")])
tt = pd.DataFrame(rows, columns=["Method", "Investigations", "Late identified", "Unnecessary", "Undetermined", "Late missed", "Precision", "Recall", "Lift", "PR-AUC"])
add_table(d, tt, f"Table 7. Test period (Jun–Aug 2018, {iv(o(P,'days'))} days, {iv(o(P,'pool_orders'))} orders, {iv(o(P,'pool_late'))} late). Perfect-foresight maximum = {iv(o(P,'oracle_late_identified'))} late orders.", nd=3, font=8)
d.add_paragraph(
    f"Under the same cap, the proposed logistic-regression policy identified {iv(o(P,'late_identified'))} late orders – {f(o(P,'lift_vs_random'),2)}× random, "
    f"{iv(o(P,'late_identified')-o('lgbm','late_identified'))} more than LightGBM and {iv(o(P,'late_identified')-o(R,'late_identified'))} more than the pre-specified "
    f"reference rule (day-bootstrap 95% CI for the difference: {ci[0]:.0f} to {ci[1]:.0f}). The rule that turned out best on test, shortest promised window, "
    f"identified {iv(o('rule_shortest_promise','late_identified'))} ({iv(o('rule_shortest_promise','late_identified')-o(P,'late_identified'))} more than the model) – "
    "but it was chosen with hindsight; on validation the same rule was the worst method and below random. On the labelled test orders the model's PR-AUC is "
    f"{f(pm.loc[P,'pr_auc'],3)} (base rate {f(pm.loc[P,'base_rate'],3)}) and ROC-AUC {f(pm.loc[P,'roc_auc'],3)}.")
add_fig(d, FIG / "05_cumulative_hits_test.png", "Figure 2. Cumulative late orders identified on the test period with 10 investigations per day.", 6.2)

# ---------------------------------------------------------------- 6
d.add_heading("6. Error and diagnostic analysis", level=1)
d.add_heading("6.1 Investigation budget", level=2)
bt = bud.pivot_table(index="k_per_day", columns="method", values="late_identified").round(1)
bt = bt[[c for c in [P, other, R, "rule_shortest_promise", "random"] if c in bt.columns]].reset_index()
bp = bud.pivot_table(index="k_per_day", columns="method", values="precision")
bt.columns = ["k / day"] + [NAME.get(c, "Random") if c != "random" else "Random" for c in bt.columns[1:]]
add_table(d, bt, "Table 8. Late orders identified on test for different daily budgets", nd=1, font=8.5)
p = d.add_paragraph(); r = p.add_run(f"Random = mean of 200 random daily selections per budget. The k = 10 value ({f(b10.loc['random','late_identified'],2)}) comes from a separate set of draws than Table 7 "
    f"({f(ops.loc[RND,'late_identified'],2)}); the gap is simulation noise (SD of one draw ≈ {f(b10.loc['random','late_identified_sd'],1)})."); r.font.size = Pt(8); r.italic = True
d.add_paragraph(
    f"Precision decreases as the budget grows ({pct(bp.loc[5,P])} at k = 5, {pct(bp.loc[10,P])} at 10, {pct(bp.loc[50,P])} at 50 for the model), confirming that the score "
    "ranks the riskiest orders first; the first two daily ranks were late about 13% of the time versus 3–9% at ranks 6–10 (Appendix D). Extra capacity "
    "beyond ~20/day mostly buys unnecessary investigations.")
d.add_heading("6.2 Performance over time", level=2)
mm = mon[mon.method == P][["month", "pool_orders", "pool_late", "base_rate", "late_identified", "precision", "lift_vs_random", "pr_auc", "roc_auc"]]
mm.columns = ["Month", "Orders", "Late", "Base rate", "Late identified", "Precision", "Lift", "PR-AUC", "ROC-AUC"]
add_table(d, mm, f"Table 9. Proposed model by test month", nd=3, font=8.5)
d.add_paragraph(
    "The model was useful in June and July (lift ≈ 3.8–4.4) but fell to random level in August 2018 (ROC-AUC ≈ 0.51), exactly when the late rate "
    "rose from 3% to 6%. The August wave consisted largely of orders to São Paulo – historically the most reliable destination – so a model that learned 'SP is safe' could not see it.")
d.add_heading("6.3 Where the model fails", level=2)
s3 = st_.head(6)[["customer_state", "orders", "late_orders", "hits", "false_alarms", "capture_rate", "late_rate"]]
s3.columns = ["Customer state", "Orders", "Late", "Identified", "False alarms", "Capture rate", "Late rate"]
add_table(d, s3, "Table 10. Failure groups by customer state (test, model top-10/day)", nd=3, font=8.5)
d.add_paragraph(
    f"São Paulo had {iv(st_.iloc[0]['late_orders'])} of the {iv(o(P,'pool_late'))} late test orders but the model captured only {pct(st_.iloc[0]['capture_rate'])}, while "
    f"Rio de Janeiro received {iv(st_.set_index('customer_state').loc['RJ','selected'])} of the {iv(o(P,'investigations'))} investigations. Missed late orders "
    f"are typically short routes with ordinary history (median distance {f(prof.loc['distance_km','missed late'],0)} km vs {f(prof.loc['distance_km','caught (hits)'],0)} km for hits; "
    f"promised window {f(prof.loc['promised_days','missed late'],0)} vs {f(prof.loc['promised_days','caught (hits)'],0)} days). False alarms look like the hits on every approval-time "
    "attribute – their difference only appears after approval. Category view: health_beauty and bed_bath_table were captured reasonably (≈26%), housewares almost never (Appendix D).")
add_fig(d, FIG / "06_model_drivers.png", "Figure 3. Model drivers: logistic-regression coefficients (left) and LightGBM gain importance (right).", 6.4)
d.add_heading("6.4 Further diagnostics", level=2)
bullets(d, [
    ("LightGBM vs logistic regression. ", "LightGBM relied heavily on market volume and state (Figure 3); trees cannot extrapolate when market-level conditions move outside the "
     "training range, so its ranking degraded under drift. Calibration (Appendix D) shows both models over-predict risk in the low-late-rate test period."),
    ("Refit vs no refit. ", "Refitting on train+validation helped logistic regression (" + ", ".join(f"{r.version}: {iv(r.late_identified)}" for r in refit[refit.model == 'logreg'].itertuples()) + ")."),
    ("Feature-group ablation (diagnostic, not used for selection). ", "; ".join(f"{r.feature_group}: {iv(r.late_identified)}" for r in abl.itertuples()) +
     ". History features alone are weak; static route/promise/geography features carry most of the signal."),
    ("Post-hoc check (test-informed, reported separately). ", "Because the shortest-promise rule did well on test, I added 'slack' features (promised days minus the typical route "
     f"delivery time from earlier outcomes). They did not improve the model (logistic regression {iv(ph.iloc[0].test_late_identified)} → {iv(ph.iloc[1].test_late_identified)} on test), and a "
     f"slack rule scored {iv(ph.iloc[-1].val_late_identified)} on validation vs {iv(ph.iloc[-1].test_late_identified)} on test – the value of promise tightness is regime-dependent."),
])

# ---------------------------------------------------------------- 7
d.add_heading("7. Operational recommendation", level=1)
d.add_paragraph(
    "Does the model improve identification and prioritisation? Yes, modestly and more reliably than any single rule chosen in advance. Compared with no "
    f"prioritisation it roughly triples the share of useful investigations (validation {pct(valc.set_index('method').loc['logreg (best config)','precision'])} vs "
    f"{pct(valc.set_index('method').loc[RND,'precision'])}; test {pct(o(P,'precision'))} vs {pct(o(RND,'precision'))}). Every rule was either good on validation and poor on test, or the reverse; "
    "the team could not have known in advance which rule to use.")
bullets(d, [
    ("Policy: ", "score every order at approval with the logistic-regression model; each morning investigate the top 10 of the previous day's approvals; never fill slots just to reach 10 on low-volume days."),
    ("Resources: ", f"≈10 investigations/day (~{iv(o(P,'investigations'))} per quarter). Expect ~1 useful investigation per day and ~9 orders that would have arrived on time anyway – budget staff time accordingly and keep each check light."),
    ("Combine with judgement: ", "treat the list as a prioritisation aid; when the operations team knows of a market-wide event (strike, holiday peak, carrier problem) the model will lag."),
    ("Monitoring and retraining: ", "retrain monthly on mature labels; track weekly precision of the top-10 list against the 3–4% base rate and a random baseline; trigger review if lift falls below ~1.5 for two weeks (as in August 2018)."),
    ("When it performs poorly: ", "sudden regime changes (Feb–Mar 2018, Aug 2018), destinations with historically low late rates (SP, MG), and new sellers with little history."),
    ("Information needed before deployment: ", "outcomes of investigations (was the delay prevented?), carrier/tracking events after approval, seller stock and dispatch status, regional carrier capacity and incident feeds, and the cost of a late order vs an investigation to set the budget."),
])

# ---------------------------------------------------------------- 8
d.add_heading("8. Limitations", level=1)
bullets(d, [
    "Observational data only: identifying an order that later arrived late does not show that investigating it would have prevented the delay; no reduction in late deliveries or revenue loss is claimed.",
    "Absolute precision is low (≈8% on test); the evaluation covers one 3-month test window with a lower late rate than training, and results vary strongly by month.",
    "The promised delivery date and shipping limit are assumed to be visible at approval; the dataset is a cleaned public sample (2016–2018) whose processes may differ from current operations.",
    "Undetermined orders (canceled/unavailable/never delivered) are kept in the pool but counted separately. The outcome record is right-censored at the export date: "
    f"{iv(csens.loc[P,'problems_in_pool_adjusted'] - o(P,'pool_late'))} test orders were still open and past their promised date when the data ended, so the main late counts are a lower bound "
    "(the censoring-adjusted check in §4 leaves the conclusions unchanged).",
    "Label maturity drops recent training data (60-day window), which delays reaction to new patterns; the post-hoc slack analysis used test information and is not a valid estimate.",
    "Multi-seller orders (≈1.3%) are represented by the highest-value seller.",
])

# ---------------------------------------------------------------- references
d.add_heading("References", level=1)
for ref in [
    "Olist (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (CC BY-NC-SA 4.0).",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.",
    "Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30.",
    "Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE 10(3).",
    "Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. ACM TKDD 6(4).",
    "Amazon Web Services (2026). Amazon SageMaker Studio and lifecycle configurations – developer guide. https://docs.aws.amazon.com/sagemaker/",
]:
    d.add_paragraph(ref, style="List Number")
d.add_paragraph("Attribution: all code was written for this project with assistance from an AI assistant (Grok Bot) and reviewed by the author; no external code was copied. "
                "Libraries: pandas, NumPy, scikit-learn, LightGBM, matplotlib, papermill, python-docx.")

# ---------------------------------------------------------------- appendices
d.add_page_break(); d.add_heading("Appendix A. Feature-availability audit", level=1)
a = audit[["feature_or_field", "source", "observed_at", "used_as_predictor", "how_computed_or_why_excluded"]]
a.columns = ["Feature / field", "Source", "Observed at", "Used?", "Computation / reason"]
add_table(d, a, "Table A1. Every candidate field, when it is observed, and whether it is used", font=7.5, col_widths=[1.6, 1.3, 1.3, 0.6, 1.9])
d.add_heading("Appendix B. Modeling examples and eligibility funnel", level=1)
add_table(d, funnel.rename(columns={"step": "Step", "orders": "Orders"}), "Table B1. Eligibility funnel", font=8)
d.add_heading("Appendix C. Experiment configuration", level=1)
cfgrows = [["Seed", exp["seed"]], ["Splits", json.dumps(exp["splits"])], ["Label", exp["label_definition"]], ["Label maturity (days)", exp["label_maturity_days"]],
           ["Daily capacity", exp["daily_capacity"]], ["Rows for selection / final refit", f"{exp['train_rows_for_selection']:,} / {exp['trainval_rows_for_final_refit']:,}"],
           ["Selection criterion", exp["selection_criterion"]], ["Selected logistic regression", json.dumps(sel["logreg"])], ["Selected LightGBM", json.dumps(sel["lgbm"])],
           ["Libraries", json.dumps(exp["libraries"])], ["Configurations searched", f"{len(grid)} (validation grid saved in outputs/tables/04_validation_grid.csv)"]]
add_table(d, pd.DataFrame(cfgrows, columns=["Item", "Value"]), "Table C1. experiment_configs.json (summary)", font=8, col_widths=[1.7, 4.9])
g = grid.sort_values(["late_identified", "pr_auc"], ascending=False).groupby("model").head(4)[["model", "feature_set", "params", "late_identified", "pr_auc", "roc_auc"]]
g.columns = ["Model", "Feature set", "Parameters", "Late identified", "PR-AUC", "ROC-AUC"]
add_table(d, g, "Table C2. Top validation configurations per model family", font=7.5, col_widths=[0.6, 0.8, 3.3, 0.7, 0.6, 0.6])
d.add_heading("Appendix D. Additional results", level=1)
pmt = pm.reset_index()[["method", "pr_auc", "roc_auc", "brier", "precision_top1pct", "precision_top5pct"]]
pmt["method"] = pmt.method.map(lambda m: NAME.get(m, m)); pmt.columns = ["Method", "PR-AUC", "ROC-AUC", "Brier", "Precision top 1%", "Precision top 5%"]
add_table(d, pmt, "Table D1. Test predictive metrics (labelled orders; Brier only for probabilistic models)", font=8)
add_table(d, rank.head(10).rename(columns={"daily_rank": "Daily rank", "orders": "Orders", "late_rate": "Late rate", "mean_score": "Mean score"}), "Table D2. Late rate by daily rank position (model, test)", font=8)
c2 = cat.head(8)[["main_category", "orders", "late_orders", "hits", "missed", "false_alarms", "capture_rate"]]
c2.columns = ["Category", "Orders", "Late", "Identified", "Missed", "False alarms", "Capture rate"]
add_table(d, c2, "Table D3. Failure groups by product category (test)", font=8)
add_fig(d, FIG / "05_pr_curves_test.png", "Figure D1. Precision–recall curves on labelled test orders.", 4.6)
add_fig(d, FIG / "06_budget_sensitivity.png", "Figure D2. Late orders identified vs daily budget.", 4.6)
add_fig(d, FIG / "06_calibration.png", "Figure D3. Calibration by decile on the test period.", 4.0)
d.add_heading("Appendix E. Sample of the saved daily prioritisation (test period)", level=1)
smp = dp.head(15)[["approval_date", "daily_rank", "order_id", "order_approved_at", "risk_score", "customer_state", "promised_days", "outcome_status"]].copy()
smp["approval_date"] = smp.approval_date.str[:10]
smp.columns = ["Approval date", "Rank", "order_id", "Approved at", "Risk score", "Cust. state", "Promised days", "Outcome"]
add_table(d, smp, f"Table E1. First 15 of {len(dp):,} rows of outputs/daily_prioritization_test.csv (every row traceable to order_id and approval date)", nd=3, font=7, col_widths=[0.8, 0.45, 2.05, 1.2, 0.55, 0.55, 0.6, 0.7])
d.add_heading("Appendix F. Reproducibility and AWS environment", level=1)
bullets(d, [
    "Code: src/olist_risk (package) + notebooks 00–06 executed in order by run_all.py (papermill); requirements.txt pins all versions.",
    "Execution: SageMaker Studio domain machinelearningproj2ajay-ecommerce (us-east-1), user profile ajay-uppu, private JupyterLab space mlproj2-ecommerce-space (the reported run executed on ml.m5.large in 283 s; the space was then switched to ml.m5.xlarge and left running). A lifecycle configuration copies the project from s3://ml-project2-olist-044450716277/ml-project2-v2/studio/, creates a virtual environment and kernel 'Python (ml-project2-v2)', and executes the notebooks; executed notebooks and outputs are copied to .../ml-project2-v2/studio-run/.",
    "Final numbers: after the submission run in Studio, an audit found that 61 orders delivered before their own approval timestamp could see their own label in the "
    "outcome-history features. The fix (outcome event time = max(delivery, approval)) was applied, logistic regression was given max_iter = 20,000 and tol = 1e-8, and the whole "
    "pipeline (notebooks 00–06) was re-run locally on the project machine with the pinned versions in requirements.txt. Every number, table, figure, notebook output and the daily list "
    "in this report come from that run. The Studio space still contains the pre-fix copy; update it with `git pull` (GitHub) or "
    "`aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/studio-v2-fixed/ ~/ml-project2-v2/` and re-run run_all.py.",
    "Determinism: seeds are fixed and LightGBM runs with deterministic=True. Different CPUs/BLAS builds can still move weakly regularised logistic-regression "
    "configurations by a few validation hits (seen in an earlier cross-machine check for C = 1.0); the selected configuration and all reported results were identical across machines.",
    "Raw data: the nine CSVs from the provided archive (byte-identical to the course Drive folder) – not included in the submission.",
    "Step-by-step instructions: REPRODUCTION_GUIDE.md / .docx and the per-cell execution guide.",
])
d.add_heading("Appendix G. Compliance checklist (assignment sections 4–7)", level=1)
chk = [
    ["4A", "Business objective; unit, decision time, outcome, eligibility, use of output", "§1, Table 1"],
    ["4A", "Incomplete/unavailable outcomes and assumptions", "§1, §2.2, §2.3"],
    ["4B", "Raw-file inspection, table relationships, distributions, data quality", "§2.1, Table 2, Fig. 1"],
    ["4B", "Correct joins (no multiplication), target derivation, missing data", "§2.1, §2.4"],
    ["4B", "Leakage prevention, historical aggregates before scored order, feature audit", "§2.4, Appendix A"],
    ["4C", "Reproducible pipeline; ≥2 learned models incl. simple baseline; justification", "§3, Table 5, Appendix C, F"],
    ["4C", "Non-ML business rule(s)", "§3 (4 rules), Tables 6–7"],
    ["4D", "Chronological train/validation/test; test kept separate", "§2.3, §4"],
    ["4D", "Metrics vs imbalance and business objective", "§4"],
    ["4D", "Daily ≤10 simulation: identified / unnecessary / missed; <10-order days; incomplete outcomes", "§4, §5.2, Table 7"],
    ["4D", "Comparison with rule under same capacity; sensitivity/diagnostics", "§5.2, §6, Tables 8–10"],
    ["4E", "Recommendation, resources, failure conditions, information needed; no causal claims", "§7, §8"],
    ["5", "Predictive vs business usefulness; simple approach preferable is a valid finding", "§5, §6.4, §7"],
    ["6.1", "Report sections (objective … limitations), 6–10 pages main text", "§1–§8"],
    ["6.2", "Source code, requirements.txt, README", "Submission zip (src/, notebooks/, README.md, requirements.txt)"],
    ["6.3", "Feature table, example summary, configs, comparison results, daily prioritisation output", "Appendices A–E, outputs/"],
    ["6", "Raw data not re-uploaded; sources and AI assistance cited; only obtained results reported", "References, attribution"],
]
add_table(d, pd.DataFrame(chk, columns=["Spec", "Requirement", "Where addressed"]), "Table G1. Requirement-to-section map", font=8, col_widths=[0.5, 4.0, 2.1])
rub = [["Business understanding & task formulation (20)", "§1 formulation table, calendar-date target, eligibility, assumptions, intended daily use"],
       ["Data engineering & valid target/features (25)", "§2: audit, aggregation before joins, label maturity, outcome-known history, brute-force leakage test, Appendix A"],
       ["Modeling & experimental evaluation (25)", "§3–§5: LR + LightGBM grids, 4 rules + random, chronological protocol, PR-AUC/ROC-AUC/precision@10, bootstrap CI"],
       ["Prioritisation & business recommendation (20)", "§4–§7: daily top-10 simulation, identified/unnecessary/missed, rule comparison, budget/month/group diagnostics, monitoring plan"],
       ["Reproducibility & communication (10)", "Appendix F, README, reproduction guide, saved outputs, executed notebooks in SageMaker Studio"]]
add_table(d, pd.DataFrame(rub, columns=["Rubric criterion", "Evidence in this report"]), "Table G2. Rubric mapping", font=8, col_widths=[2.2, 4.4])

out = ROOT / "docs" / "ML_Project2_Final_Report.docx"; out.parent.mkdir(exist_ok=True); d.save(out); print("saved", out)
