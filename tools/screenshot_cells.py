"""Render executed notebooks with the JupyterLab HTML template and save one PNG per code cell
(code + real output).  Usage: python tools/screenshot_cells.py <notebooks_dir> <out_dir>"""
import sys, json
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

nb_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
exp = HTMLExporter(template_name="lab")
index = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1.3)
    for nbp in sorted(nb_dir.glob("0*.ipynb")):
        nb = nbformat.read(nbp, as_version=4)
        html, _ = exp.from_notebook_node(nb)
        html = html.replace("</head>", "<style>body{background:#fff} .jp-Cell{padding:6px 10px} .jp-Notebook{padding:10px} .jp-OutputArea-output{overflow:visible !important} .jp-RenderedHTMLCommon table{font-size:11.5px} .jp-RenderedHTMLCommon td,.jp-RenderedHTMLCommon th{padding:2px 6px} pre{white-space:pre-wrap !important; word-break:break-word}</style></head>")
        hp = out_dir / (nbp.stem + ".html"); hp.write_text(html, encoding="utf-8")
        page.goto(hp.resolve().as_uri()); page.wait_for_timeout(800)
        cells = page.query_selector_all(".jp-Cell.jp-CodeCell")
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        for i, (el, c) in enumerate(zip(cells, code_cells), start=1):
            fn = f"{nbp.stem}__cell{i:02d}.png"
            el.scroll_into_view_if_needed(); el.screenshot(path=str(out_dir / fn))
            index.append({"notebook": nbp.name, "code_cell": i, "file": fn, "first_line": c.source.splitlines()[0] if c.source else ""})
        hp.unlink()
        print(nbp.name, len(cells), "cells")
    browser.close()
json.dump(index, open(out_dir / "index.json", "w"), indent=1)
print(len(index), "screenshots ->", out_dir)
