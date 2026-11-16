"""Shared helpers for building .docx documents from the outputs folder."""
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; TAB = OUT / "tables"; FIG = OUT / "figures"


def T(name, **kw):
    return pd.read_csv(TAB / name, **kw)


def new_doc(base_size=11):
    d = Document()
    st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(base_size)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(0.9); s.top_margin = s.bottom_margin = Inches(0.8)
    return d


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_); tcPr.append(sh)


def fmt(v, nd=3):
    if isinstance(v, float):
        if pd.isna(v): return ""
        if abs(v - round(v)) < 1e-9 and abs(v) >= 1: return f"{int(round(v)):,}"
        if abs(v) >= 10: return f"{v:,.1f}"
        return f"{v:.{nd}f}"
    if isinstance(v, int): return f"{v:,}"
    return str(v)


def add_table(d, df, caption=None, nd=3, font=8.5, col_widths=None, header_fill="1F4E79"):
    if caption:
        p = d.add_paragraph(); r = p.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; run = cell.paragraphs[0].add_run(str(c)); run.bold = True
        run.font.size = Pt(font); run.font.color.rgb = RGBColor(255, 255, 255); shade(cell, header_fill)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; run = cells[i].paragraphs[0].add_run(fmt(v, nd)); run.font.size = Pt(font)
    total = 6.7
    if not col_widths:
        n = len(df.columns)
        first = min(1.8, total / n * 1.8) if n > 2 else total / n
        col_widths = [first] + [(total - first) / (n - 1)] * (n - 1) if n > 1 else [total]
    t.autofit = False
    for i, w in enumerate(col_widths):
        t.columns[i].width = Inches(w)
    for row in t.rows:
        for i, w in enumerate(col_widths): row.cells[i].width = Inches(w)
    d.add_paragraph()
    return t


def add_fig(d, path, caption, width=6.3):
    d.add_picture(str(path), width=Inches(width)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = d.add_paragraph(); r = p.add_run(caption); r.italic = True; r.font.size = Pt(9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def bullets(d, items, style="List Bullet"):
    for it in items:
        p = d.add_paragraph(style=style)
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True; p.add_run(it[1])
        else:
            p.add_run(it)


def code(d, text, size=8.5):
    p = d.add_paragraph(); r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(size)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    pPr = p._p.get_or_add_pPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F2F2F2"); pPr.append(sh)
    return p


def md_to_docx(md_path, docx_path, title=None, image_root=None):
    """Very small Markdown -> docx converter (headings, bullets, numbered lists, code fences, tables, images, bold)."""
    import re
    d = new_doc(10.5)
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    i = 0; image_root = Path(image_root or Path(md_path).parent)
    def inline(p, text):
        parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"): r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("`") and part.endswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
            else: p.add_run(part)
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            buf = []; i += 1
            while i < len(lines) and not lines[i].startswith("```"): buf.append(lines[i]); i += 1
            code(d, "\n".join(buf)); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1)); d.add_heading(m.group(2).replace("**", ""), level=0 if (lvl == 1 and i < 3) else min(lvl, 3)); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            p = image_root / m.group(2)
            if p.exists():
                add_fig(d, p, m.group(1), width=6.2)
            else:
                para = d.add_paragraph(); r = para.add_run(f"[Figure placeholder: {m.group(1)}]"); r.italic = True
            i += 1; continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]; rows = []; i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip().replace("**", "").replace("`", "") for c in lines[i].strip().strip("|").split("|")]); i += 1
            add_table(d, pd.DataFrame(rows, columns=[h.replace("**", "") for h in hdr]), font=8.5); continue
        if re.match(r"^\s*[-*]\s+", ln):
            p = d.add_paragraph(style="List Bullet"); inline(p, re.sub(r"^\s*[-*]\s+", "", ln)); i += 1; continue
        if re.match(r"^\s*\d+\.\s+", ln):
            p = d.add_paragraph(style="List Number"); inline(p, re.sub(r"^\s*\d+\.\s+", "", ln)); i += 1; continue
        if ln.startswith(">"):
            p = d.add_paragraph(); inline(p, ln.lstrip("> ")); p.paragraph_format.left_indent = Inches(0.3)
            for r in p.runs: r.italic = True
            i += 1; continue
        if ln.strip() in ("---", "***"): i += 1; continue
        if ln.strip():
            p = d.add_paragraph(); inline(p, ln)
        i += 1
    d.save(docx_path)
