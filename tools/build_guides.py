"""Builds docs/REPRODUCTION_GUIDE.md (+.docx) from cell_guide.py and the executed notebooks."""
import sys, json, nbformat
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cell_guide import CELLS
from report_common import md_to_docx
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)

def first_line(src):
    return src.splitlines()[0].lstrip("# ").strip()

L = []
A = L.append
A("# ML Project 2 – Reproduction Guide (Amazon SageMaker Studio)")
A("")
A("**Project:** E-Commerce Delivery Risk Management (Olist) · **Author:** Ajay Uppu (GSU, MS Data Science)  ")
A("**AWS:** account 044450716277, region us-east-1 · domain `machinelearningproj2ajay-ecommerce` (d-hfxnlufab81e) · "
  "user profile `ajay-uppu` · JupyterLab space `mlproj2-ecommerce-space` (ml.m5.xlarge, 20 GB EBS) · kernel **Python (ml-project2-v2)**")
A("")
A("This guide tells you exactly which notebook and which cell to run, in which order, what each cell does, and the output "
  "you should see. All expected values below are copied from the real execution inside the SageMaker Studio space "
  "(2026-11-15, 21:48–21:54 ET). If your numbers match, your run is a faithful reproduction.")
A("")
A("## 1. What is already set up for you")
A("")
A("| Resource | Value |")
A("|---|---|")
for a, b in [
    ("Studio domain", "machinelearningproj2ajay-ecommerce (IAM auth, default VPC, public internet access)"),
    ("Domain URL", "https://d-hfxnlufab81e.studio.us-east-1.sagemaker.aws"),
    ("User profile", "ajay-uppu"),
    ("JupyterLab space", "mlproj2-ecommerce-space – private, ml.m5.xlarge, 20 GB storage, no idle shutdown configured"),
    ("Execution role", "ml-project2-olist-sagemaker-exec (SageMaker full access + read/write on the project bucket)"),
    ("Lifecycle config", "mlproj2-ecommerce-sync-and-run-v2 – copies the project from S3 on first start; runs the pipeline only if logs/RUN_DONE is missing"),
    ("Project folder in Studio", "/home/sagemaker-user/ml-project2-v2 (shown as ml-project2-v2 in the file browser)"),
    ("Data on S3", "s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/ (9 Olist CSVs)"),
    ("Run results on S3", "s3://ml-project2-olist-044450716277/ml-project2-v2/studio-run/"),
    ("Deliverables on S3", "s3://ml-project2-olist-044450716277/ml-project2-v2/deliverables/"),
]:
    A(f"| {a} | {b} |")
A("")
A("> The space is left running on purpose so you can browse the code. It costs about USD 0.23 per hour while running "
  "(ml.m5.xlarge, us-east-1). Stop it from the Studio JupyterLab page (Stop space) when you no longer need it; "
  "your files on the 20 GB volume are kept.")
A("")
A("## 2. Open the project in Studio (5 clicks)")
A("")
for s in [
    "Sign in to the AWS console (account 044450716277) and switch the region (top right) to **US East (N. Virginia) us-east-1**.",
    "Search for **Amazon SageMaker AI** → left menu **Admin configurations → Domains** → click `machinelearningproj2ajay-ecommerce`.",
    "Tab **User profiles** → row `ajay-uppu` → **Launch** ▸ **Studio**. Studio opens in a new browser tab.",
    "In Studio's left bar click **JupyterLab** → you will see the space `mlproj2-ecommerce-space`. If its status is *Stopped*, click **Run space** and wait about 1–2 minutes; then click **Open**.",
    "In JupyterLab's file browser (left) double-click `ml-project2-v2` → `notebooks`. The seven notebooks already contain the outputs of the SageMaker run.",
]:
    A(f"1. {s}")
A("")
A("## 3. Prepare the environment (only needed for a fresh space)")
A("")
A("The kernel **Python (ml-project2-v2)** and its virtual environment were created by the bootstrap script and live on the "
  "space's persistent volume, so normally you can go straight to section 4. On a new space or after deleting `.venv`, open "
  "**File → New → Terminal** and run:")
A("")
A("```bash\ncd ~/ml-project2-v2\nbash aws/studio_bootstrap.sh --no-run   # venv + pinned packages + kernel + raw data from S3 (~2 min)\n```")
A("")
A("Expected last lines: `... environment ready` followed by `environment only` (the kernel is registered and the 9 CSV files are in `data/raw`). Then reload the browser tab so JupyterLab shows the new kernel.")
A("")
A("If the project folder itself is missing (new space), copy it from S3 first:")
A("")
A("```bash\naws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/studio/ ~/ml-project2-v2/\n```")
A("")
A("## 4. Run order")
A("")
A("Run the notebooks strictly in this order; each one saves files used by the next:")
A("")
A("| # | Notebook | Purpose | Reads | Writes | Runtime* |")
A("|---|---|---|---|---|---|")
rt = {"00": "<10 s", "01": "~25 s", "02": "~15 s", "03": "~60 s", "04": "~2–3 min", "05": "~20 s", "06": "~60 s"}
io = {"00": ("data/raw (S3 if missing)", "–"), "01": ("data/raw", "tables/01_*, figures/01_*"),
      "02": ("data/raw", "data/processed/labels.pkl, tables/02_*"), "03": ("raw + labels.pkl", "data/processed/features.pkl, tables/03_*"),
      "04": ("features.pkl", "models/*.joblib, experiment_configs.json, tables/04_*"),
      "05": ("features.pkl, models", "daily_prioritization_test.csv, tables/05_*"), "06": ("features.pkl, models", "tables/06_*, figures/06_*")}
for nb, (title, cells) in CELLS.items():
    k = nb[:2]
    A(f"| {k} | `{nb}` | {title} | {io[k][0]} | {io[k][1]} | {rt[k]} |")
A("")
A("\\*Measured on ml.m5.large in the original run (whole pipeline 283 s); ml.m5.xlarge is a bit faster.")
A("")
A("For every notebook: open it → check the kernel name at the top right says **Python (ml-project2-v2)** (click it to change) → "
  "**Kernel → Restart Kernel and Clear Outputs of All Cells** (optional, for a clean run) → run the cells one by one with "
  "**Shift + Enter** and compare with the expected output below. **Run → Run All Cells** also works.")
A("")
A("Shortcut – run all seven notebooks in one go from a terminal (the notebooks are overwritten with fresh outputs):")
A("")
A("```bash\ncd ~/ml-project2-v2 && .venv/bin/python run_all.py\n```")
A("")
A("## 5. Cell-by-cell guide")
A("")
for nb, (title, cells) in CELLS.items():
    nbobj = nbformat.read(ROOT / "notebooks" / nb, 4)
    code = [c for c in nbobj.cells if c.cell_type == "code"]
    assert len(code) == len(cells), (nb, len(code), len(cells))
    A(f"### Notebook {nb[:2]} – {title} (`{nb}`)")
    A("")
    for i, (c, (what, why, exp)) in enumerate(zip(code, cells), 1):
        A(f"**Cell {i} – {first_line(c.source).split(' - ',1)[-1]}**")
        A("")
        A(f"- *What it does:* {what}")
        A(f"- *Why:* {why}")
        A(f"- *Expected output:* {exp}")
        A(f"- *Screenshot:* `screenshots/{nb[:-6]}__cell{i:02d}.png`")
        A("")
A("## 6. Checklist of key numbers")
A("")
A("| Check | Expected |")
A("|---|---|")
for a, b in [
    ("Orders / approved / delivered status", "99,441 / 99,281 / 96,478"),
    ("Late rate among delivered (calendar dates)", "6.77% (8.11% with raw timestamps)"),
    ("Train / validation / test candidate orders", "58,866 / 21,132 / 18,960"),
    ("Rows used for fitting (train) / final refit", "43,276 / 64,109"),
    ("Best rule on validation", "longest distance – 202 late orders identified"),
    ("Selected model", "logistic regression, C = 0.01, no class weights, feature set no_market (256 on validation)"),
    ("Test – logistic regression (10/day)", "70 late identified, 780 unnecessary, 19 undetermined, 602 missed (869 investigations)"),
    ("Test – LightGBM / longest-distance rule / random", "58 / 34 / 26.45"),
    ("Test – shortest-promise rule (worst on validation)", "80"),
    ("Perfect-foresight maximum", "437"),
    ("Rows in daily_prioritization_test.csv", "869"),
]:
    A(f"| {a} | {b} |")
A("")
A("## 7. Troubleshooting")
A("")
A("| Symptom | Cause | Fix |")
A("|---|---|---|")
for a, b, c in [
    ("`ModuleNotFoundError: No module named 'olist_risk'`", "Notebook not opened from ml-project2-v2/notebooks, or wrong kernel", "Open the notebook from that folder; select kernel Python (ml-project2-v2); re-run Cell 1."),
    ("Kernel 'Python (ml-project2-v2)' not in the list", "Bootstrap not run on this space", "Run `bash ~/ml-project2-v2/aws/studio_bootstrap.sh --no-run`, then reload the browser tab."),
    ("Cell 2 of notebook 00 shows ok = False", "Notebook uses the default Python 3 kernel with other versions", "Switch kernel; or run Cell 3, restart kernel, run again."),
    ("`FileNotFoundError: data/raw/olist_orders_dataset.csv`", "Raw data not downloaded", "Run notebook 00 Cell 4, or `aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/ ~/ml-project2-v2/data/raw/`."),
    ("`AccessDenied` from S3", "Space not using the execution role ml-project2-olist-sagemaker-exec", "Check the domain/user execution role in the console; the role needs s3:GetObject/PutObject/ListBucket on the bucket."),
    ("`FileNotFoundError: data/processed/features.pkl` in notebook 04–06", "Notebooks run out of order", "Run notebooks 02 and 03 first."),
    ("Kernel dies / MemoryError in notebook 03", "Very small instance (e.g. ml.t3.medium, 4 GB)", "Use ml.m5.large or larger (the space uses ml.m5.xlarge, 16 GB)."),
    ("Notebook 04 seems stuck", "36-configuration grid search runs ~2–3 minutes", "Wait; the cell prints when finished."),
    ("Slightly different 4th-decimal values", "Floating-point differences between CPUs/BLAS builds", "Counts (late identified etc.) should still match exactly; small metric differences are expected."),
    ("Space will not start: ResourceLimitExceeded", "Instance quota for Studio JupyterLab apps", "Choose ml.t3.xlarge in the space settings, or request a quota increase in Service Quotas."),
    ("Lifecycle script re-runs the pipeline on start", "logs/RUN_DONE was deleted", "Harmless (results are rewritten); recreate the file to prevent it: `date > ~/ml-project2-v2/logs/RUN_DONE`."),
]:
    A(f"| {a} | {b} | {c} |")
A("")
A("## 8. Studio screenshots you can add yourself")
A("")
A("The per-cell screenshots in `screenshots/` were rendered from the notebooks executed in the space. Screens of the AWS "
  "console/Studio UI need your console sign-in: (1) Domains list showing machinelearningproj2ajay-ecommerce; (2) the domain's "
  "User profiles tab with ajay-uppu and Launch ▸ Studio; (3) Studio's JupyterLab page with mlproj2-ecommerce-space (Running, ml.m5.xlarge); "
  "(4) the JupyterLab file browser showing ml-project2-v2/notebooks; (5) a notebook open with the kernel Python (ml-project2-v2); "
  "(6) a Terminal running the bootstrap.")
(DOCS / "REPRODUCTION_GUIDE.md").write_text("\n".join(L), encoding="utf-8")
md_to_docx(DOCS / "REPRODUCTION_GUIDE.md", DOCS / "REPRODUCTION_GUIDE.docx")
print("ok", len(L))
