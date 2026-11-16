"""Build docs/medium_post.html (for medium.com/p/import) and docs/medium_post.docx (Google Doc) from docs/medium_post.md.
Images: HTML uses the public-repo raw URLs; the DOCX embeds the local copies from docs/medium_images/."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
md = (DOCS / "medium_post.md").read_text()
RAW = "https://raw.githubusercontent.com/Ajayuppu/E-Commerce-Delivery-Risk-Management/main/docs/medium_images/"

def blocks(text):
    lines = text.split("\n"); i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            lang = l[3:].strip(); j = i + 1; buf = []
            while not lines[j].startswith("```"): buf.append(lines[j]); j += 1
            yield ("code", lang, "\n".join(buf)); i = j + 1; continue
        if l.startswith("|"):
            buf = []
            while i < len(lines) and lines[i].startswith("|"): buf.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in buf if not re.match(r"^\|[-| ]+\|$", r)]
            yield ("table", rows); continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", l)
        if m:
            cap = ""
            if i + 1 < len(lines) and re.match(r"^\*[^*].*\*$", lines[i + 1]): cap = lines[i + 1][1:-1]; i += 1
            yield ("img", m.group(2), cap or m.group(1)); i += 1; continue
        m = re.match(r"^(#{1,3}) (.*)$", l)
        if m: yield ("h", len(m.group(1)), m.group(2)); i += 1; continue
        if re.match(r"^(\d+\.|-) ", l):
            buf = []; ordered = l[0].isdigit()
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]): buf.append(re.sub(r"^(\d+\.|-) ", "", lines[i])); i += 1
            yield ("list", ordered, buf); continue
        if l.strip() == "---": yield ("hr",); i += 1; continue
        if not l.strip(): i += 1; continue
        buf = [l]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|!\[|```|\||(\d+\.|-) |---)", lines[i]): buf.append(lines[i]); i += 1
        # a "Symptom/Likely cause/Fix" group keeps its line breaks
        yield ("p", buf)

def inline_html(s):
    import html
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r'(?<!href=")(https?://[^\s<]+[^\s<.,)])', lambda m: m.group(1) if 'href' in s[max(0, m.start()-8):m.start()] else f'<a href="{m.group(1)}">{m.group(1)}</a>', s)
    return s

def build_html():
    import html
    out = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>{}</title></head><body><article>']
    title = None
    for b in blocks(md):
        k = b[0]
        if k == "h":
            if b[1] == 1: title = b[2]; out.append(f"<h1>{inline_html(b[2])}</h1>")
            else: out.append(f"<h{b[1]}>{inline_html(b[2])}</h{b[1]}>")
        elif k == "p":
            if len(b[1]) == 1 and re.match(r"^\*[^*].*\*$", b[1][0]) and title and not any("subtitle" in o for o in out):
                out.append(f'<h2 class="subtitle"><em>{inline_html(b[1][0][1:-1])}</em></h2>'.replace('<h2 class="subtitle">', '<p class="subtitle">').replace("</h2>", "</p>"))
            else: out.append("<p>" + "<br>".join(inline_html(x) for x in b[1]) + "</p>")
        elif k == "img":
            out.append(f'<figure><img src="{b[1]}" alt="{html.escape(b[2])}"><figcaption>{inline_html(b[2])}</figcaption></figure>')
        elif k == "code":
            if b[1] == "mermaid": continue   # Medium can't render mermaid; the architecture PNG at the top shows the same flow
            out.append(f"<pre><code>{html.escape(b[2])}</code></pre>")
        elif k == "table":
            rows = b[1]; w = [max(len(re.sub(r'\*\*', '', r[c])) for r in rows) for c in range(len(rows[0]))]
            txt = "\n".join("  ".join(re.sub(r'\*\*', '', r[c]).ljust(w[c]) for c in range(len(r))).rstrip() for r in rows)
            out.append(f"<pre>{html.escape(txt)}</pre>")   # Medium has no tables; a monospace block keeps the columns aligned
        elif k == "list":
            tag = "ol" if b[1] else "ul"; out.append(f"<{tag}>" + "".join(f"<li>{inline_html(x)}</li>" for x in b[2]) + f"</{tag}>")
        elif k == "hr": out.append("<hr>")
    out.append("</article></body></html>")
    s = "\n".join(out).replace("{}", html.escape(title or "Medium post"), 1)
    (DOCS / "medium_post.html").write_text(s)

def build_docx():
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Georgia"; st.font.size = Pt(11)
    def add_runs(p, s):
        pos = 0
        for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|(?<![\w*])\*([^*]+?)\*(?![\w*])", s):
            if m.start() > pos: p.add_run(s[pos:m.start()])
            if m.group(1) is not None:
                inner = m.group(1); r = p.add_run(re.sub(r"`", "", inner)); r.bold = True
            elif m.group(2) is not None: r = p.add_run(m.group(2)); r.font.name = "Courier New"; r.font.size = Pt(10)
            elif m.group(3) is not None: r = p.add_run(f"{m.group(3)} ({m.group(4)})"); r.font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)
            else: r = p.add_run(m.group(5)); r.italic = True
            pos = m.end()
        if pos < len(s): p.add_run(s[pos:])
    first_italic_done = False
    for b in blocks(md):
        k = b[0]
        if k == "h":
            if b[1] == 1: h = doc.add_heading(b[2], 0)
            else: h = doc.add_heading(b[2], b[1] - 1)
        elif k == "p":
            if len(b[1]) == 1 and re.match(r"^\*[^*].*\*$", b[1][0]) and not first_italic_done:
                p = doc.add_paragraph(); r = p.add_run(b[1][0][1:-1]); r.italic = True; r.font.size = Pt(13); first_italic_done = True; continue
            p = doc.add_paragraph()
            for j, line in enumerate(b[1]):
                if j: p.add_run().add_break()
                add_runs(p, line)
        elif k == "img":
            f = DOCS / "medium_images" / b[1].split("/")[-1]
            doc.add_picture(str(f), width=Inches(6.3)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); r = p.add_run(b[2]); r.italic = True; r.font.size = Pt(9.5); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif k == "code":
            p = doc.add_paragraph()
            if b[1] == "mermaid":
                r = p.add_run("Mermaid source of the flow diagram (the architecture image at the top shows the same flow):"); r.italic = True
                p = doc.add_paragraph()
            r = p.add_run(b[2]); r.font.name = "Courier New"; r.font.size = Pt(9)
        elif k == "table":
            rows = b[1]; t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
            for i, row in enumerate(rows):
                for j, c in enumerate(row):
                    cell = t.cell(i, j); cell.text = ""; add_runs(cell.paragraphs[0], c)
                    for r in cell.paragraphs[0].runs: r.font.size = Pt(9.5); r.bold = r.bold or i == 0
            doc.add_paragraph()
        elif k == "list":
            for x in b[2]: add_runs(doc.add_paragraph(style="List Number" if b[1] else "List Bullet"), x)
        elif k == "hr": doc.add_paragraph("―" * 20).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(DOCS / "medium_post.docx")

build_html(); build_docx(); print("built medium_post.html and medium_post.docx")
