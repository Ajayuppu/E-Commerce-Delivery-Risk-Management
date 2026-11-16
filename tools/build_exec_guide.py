"""Step-by-step execution guide (docx, converted to a Google Doc on upload) with one real screenshot per cell."""
import sys, nbformat
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cell_guide import CELLS
from report_common import new_doc, add_table, bullets
from docx.shared import Inches, Pt, RGBColor
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
SHOT = ROOT / "screenshots"; TMP = Path("/tmp/shots_small"); TMP.mkdir(exist_ok=True)

TROUBLE = {
 "00": ["Kernel list lacks 'Python (ml-project2-v2)': run `bash ~/ml-project2-v2/aws/studio_bootstrap.sh --no-run` in a Terminal and reload the page.",
        "Cell 4 AccessDenied: the space must use the execution role ml-project2-olist-sagemaker-exec (S3 read on the project bucket).",
        "Cell 5 assertion error on row counts: a CSV is incomplete – delete it from data/raw and re-run Cell 4."],
 "01": ["MemoryError while loading geolocation (1M rows): use an instance with ≥8 GB RAM (ml.m5.large or larger).",
        "Figures not shown: make sure Cell 1 ran (it sets the matplotlib backend and output folders)."],
 "02": ["Different late rate than 0.0677: check that you did not edit labels.py (calendar-date comparison).",
        "PermissionError writing data/processed: the folder must belong to sagemaker-user (`sudo chown -R sagemaker-user ~/ml-project2-v2`)."],
 "03": ["FileNotFoundError labels.pkl: run notebook 02 first.",
        "Cell 3 takes ~20 s; if the kernel restarts, the instance is too small.",
        "Leakage check below 1.0: features.py was modified – revert from S3 (studio/src)."],
 "04": ["FileNotFoundError features.pkl: run notebook 03 first.",
        "Cell 4 runs 36 fits (2–3 minutes); do not interrupt.",
        "LightGBM warnings 'No further splits with positive gain' are harmless."],
 "05": ["FileNotFoundError for models/*.joblib or experiment_configs.json: run notebook 04 first.",
        "Numbers must match exactly (71 / 60 / 34 / 80 / 26.45 for the final run; the pre-fix Studio copy shows 70 / 58); tiny 4th-decimal metric differences are fine."],
 "06": ["Runs ~1 minute (refits for the ablation and post-hoc analysis).",
        "The post-hoc cell (13) is test-informed – it is a diagnostic, not the reported result."],
}

def small(p):
    out = TMP / (p.stem + ".png")
    im = Image.open(p); w, h = im.size
    if w > 1400: im = im.resize((1400, int(h * 1400 / w)))
    # very tall outputs: crop to keep the doc readable
    if im.size[1] > 1500: im = im.crop((0, 0, im.size[0], 1500))
    im.save(out, optimize=True); return out

d = new_doc(10.5)
d.add_heading("ML Project 2 – Step-by-Step Execution Guide (SageMaker Studio)", level=0)
p = d.add_paragraph(); r = p.add_run("E-Commerce Delivery Risk Management (Olist) · Ajay Uppu, GSU MS Data Science · SageMaker Studio domain machinelearningproj2ajay-ecommerce (us-east-1) · final version 15 Nov 2026"); r.italic = True
d.add_paragraph("This guide walks through every notebook cell in the order you should run it. For each cell you get: what it does, "
                "why it is needed, a screenshot of the cell's code and its REAL output, the key values you should see, and what to do if something goes wrong.")
d.add_paragraph("Which run the screenshots show: the final run – a full local re-run of notebooks 00–06 with the pinned libraries, made after an audit "
                "fixed a leakage edge case (61 orders delivered before their own approval timestamp could see their own label in the history features; "
                "outcome event time is now max(delivery, approval)). The Studio space still holds the earlier pre-fix run (logistic regression 70 instead of 71 "
                "late orders on test). To update the space: open a Terminal in JupyterLab and run `cd ~/ml-project2-v2 && git pull` (GitHub clone) or "
                "`aws s3 sync s3://ml-project2-olist-044450716277/ml-project2-v2/studio-v2-fixed/ ~/ml-project2-v2/`, then `.venv/bin/python run_all.py`.")
d.add_heading("Before you start", level=1)
add_table(d, pd.DataFrame([
    ["Domain", "machinelearningproj2ajay-ecommerce (d-hfxnlufab81e), us-east-1, IAM auth"],
    ["User profile / space", "ajay-uppu / mlproj2-ecommerce-space (private JupyterLab, ml.m5.xlarge, 20 GB)"],
    ["Kernel", "Python (ml-project2-v2) – virtual env ~/ml-project2-v2/.venv with requirements.txt"],
    ["Project folder", "~/ml-project2-v2 (notebooks/, src/olist_risk/, outputs/, data/raw/)"],
    ["Data", "s3://ml-project2-olist-044450716277/ml-project2-v2/data/raw/ (copied automatically)"],
], columns=["Item", "Value"]), font=9, col_widths=[1.6, 5.0])
d.add_heading("Open the space", level=2)
for s in ["AWS console (us-east-1) → Amazon SageMaker AI → Domains → machinelearningproj2ajay-ecommerce.",
          "User profiles → ajay-uppu → Launch → Studio.",
          "JupyterLab → mlproj2-ecommerce-space → Run space (if stopped) → Open.",
          "File browser → ml-project2-v2 → notebooks. Open notebooks 00 → 06 in order.",
          "In every notebook confirm the kernel (top right) is 'Python (ml-project2-v2)', then run cells with Shift+Enter."]:
    d.add_paragraph(s, style="List Number")
d.add_paragraph("If the kernel is missing (fresh space): File → New → Terminal → `bash ~/ml-project2-v2/aws/studio_bootstrap.sh --no-run`, then reload the browser tab. "
                "To run everything at once: `cd ~/ml-project2-v2 && .venv/bin/python run_all.py` (≈5 minutes).")
d.add_heading("What you will see in the Studio UI", level=2)
for s_ in ["Domains list (SageMaker AI → Admin configurations → Domains): the row machinelearningproj2ajay-ecommerce with status InService.",
           "Domain details → User profiles tab: the row ajay-uppu; its Launch button opens a menu – choose Studio.",
           "Studio home → JupyterLab (left bar): mlproj2-ecommerce-space, instance ml.m5.xlarge, status Running (or Stopped with a Run space button). Click Open.",
           "JupyterLab file browser: folder ml-project2-v2 with notebooks/, src/, outputs/, docs/, data/, aws/, run_all.py and requirements.txt.",
           "An open notebook: kernel name 'Python (ml-project2-v2)' at the top right; a filled circle next to it means a cell is running, an empty circle means idle.",
           "Terminal (File → New → Terminal): a bash prompt; aws/studio_bootstrap.sh --no-run prints its install steps and ends with 'environment ready'."]:
    d.add_paragraph(s_, style="List Number")

for nb, (title, cells) in CELLS.items():
    k = nb[:2]
    nbobj = nbformat.read(ROOT / "notebooks" / nb, 4)
    code = [c for c in nbobj.cells if c.cell_type == "code"]
    d.add_page_break()
    d.add_heading(f"Notebook {k} – {title}", level=1)
    p = d.add_paragraph(); r = p.add_run(f"File: notebooks/{nb} · {len(code)} code cells"); r.italic = True
    for i, (c, (what, why, exp)) in enumerate(zip(code, cells), 1):
        name = c.source.splitlines()[0].lstrip("# ").split(" - ", 1)[-1]
        d.add_heading(f"Cell {i} – {name}", level=2)
        p = d.add_paragraph(); r = p.add_run("What it does: "); r.bold = True; p.add_run(what)
        p = d.add_paragraph(); r = p.add_run("Why: "); r.bold = True; p.add_run(why)
        shot = SHOT / f"{nb[:-6]}__cell{i:02d}.png"
        if shot.exists():
            d.add_picture(str(small(shot)), width=Inches(6.5))
            cp = d.add_paragraph(); rr = cp.add_run(f"Screenshot: notebook {k}, cell {i} – code and real output (final run)."); rr.italic = True; rr.font.size = Pt(8.5)
        p = d.add_paragraph(); r = p.add_run("Expected output: "); r.bold = True; p.add_run(exp)
    d.add_heading(f"Troubleshooting – notebook {k}", level=2)
    bullets(d, TROUBLE[k])

d.add_page_break()
d.add_heading("Final check – key numbers", level=1)
add_table(d, pd.DataFrame([
    ["Late rate among delivered orders", "6.77%"], ["Train / validation / test candidates", "58,866 / 21,132 / 18,960"],
    ["Selected model (validation)", "Logistic regression C=0.01, 256 late orders in the daily top-10"],
    ["Test, 10/day: logistic regression", "70 late identified, 780 unnecessary, 19 undetermined, 602 missed"],
    ["Test: LightGBM / distance rule / shortest-promise rule / random", "58 / 34 / 80 / 26.45"],
    ["daily_prioritization_test.csv", "869 rows"],
], columns=["Check", "Expected"]), font=9, col_widths=[2.6, 4.0])
out = ROOT / "docs" / "ML_Project2_Step_by_Step_Execution_Guide.docx"
d.save(out); print(out, out.stat().st_size // 1024, "KB")
