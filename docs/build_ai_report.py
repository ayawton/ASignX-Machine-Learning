"""Build the reviewed report from its editable Markdown source.

Requires reportlab, pypdf and pymupdf (the latter two are used for QA).
"""
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp" / "pdf_dependencies"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted,
)

OUT = ROOT / "output" / "pdf" / "AICurrentStateAndClarrificationsAndSuggestions.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
FONT_DIR = Path("C:/Windows/Fonts")
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
for name, filename in [("Body", "arial.ttf"), ("BodyBold", "arialbd.ttf"),
                       ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")

NAVY = colors.HexColor("#15304b")
TEAL = colors.HexColor("#007e87")
GRAY = colors.HexColor("#58697b")
WIDTH = A4[0] - 104
styles = {
    "title": ParagraphStyle("title", fontName="BodyBold", fontSize=27, leading=33,
                            textColor=NAVY, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="BodyBold", fontSize=20, leading=25,
                         textColor=NAVY, spaceAfter=17),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.2, leading=14.7,
                           textColor=NAVY, spaceAfter=10, splitLongWords=True),
    "sub": ParagraphStyle("sub", fontName="BodyBold", fontSize=10.6, leading=15,
                          textColor=TEAL, spaceBefore=4, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=11.2,
                           textColor=NAVY),
    "th": ParagraphStyle("th", fontName="BodyBold", fontSize=8.4, leading=11.2,
                         textColor=colors.white),
    "source": ParagraphStyle("source", fontName="Body", fontSize=8.6, leading=12.0,
                             textColor=GRAY, spaceAfter=7, splitLongWords=True),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.6, leading=12,
                           textColor=NAVY, leftIndent=12, spaceAfter=14),
}


def markup(text):
    text = escape(text)
    text = re.sub(r"https://[^\s]+", lambda m: '<link href="' + m.group(0) +
                  '" color="#007e87">' + m.group(0) + '</link>', text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#d6e2e9"))
    canvas.line(52, height - 39, width - 52, height - 39)
    canvas.setFont("BodyBold", 8)
    canvas.setFillColor(TEAL)
    canvas.drawString(52, height - 29, "ASIGNX  /  ENGINEERING REVIEW")
    canvas.setFont("Body", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(52, 30, "17 October 2026  |  Repository evidence and deployment recommendations")
    canvas.drawRightString(width - 52, 30, str(doc.page))
    canvas.restoreState()


def table(rows):
    size = len(rows[0])
    widths = {3: [WIDTH * .21, WIDTH * .40, WIDTH * .39],
              4: [WIDTH * .21, WIDTH * .28, WIDTH * .29, WIDTH * .22],
              5: [WIDTH * .16, WIDTH * .21, WIDTH * .21, WIDTH * .21, WIDTH * .21]}
    if size == 3 and rows[0][0] == "Measure":
        widths[3] = [WIDTH * .46, WIDTH * .27, WIDTH * .27]
    cells = [[Paragraph(markup(v), styles["th" if i == 0 else "cell"])
              for v in row] for i, row in enumerate(rows)]
    t = Table(cells, colWidths=widths.get(size), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#edf5f7"), colors.HexColor("#f7fafc")]),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#d6e2e9")),
    ]))
    return t


source = (ROOT / "docs" / "ai-state-review.md").read_text(encoding="utf-8")
story = []
for page_index, section in enumerate(source.split("---PAGE---")):
    if page_index:
        story.append(PageBreak())
    lines = section.strip().splitlines()
    heading = lines.pop(0).removeprefix("# ")
    story.append(Paragraph(markup(heading), styles["title" if page_index == 0 else "h1"]))
    # The document outline is linked to the section's page by the custom template.
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([v.strip() for v in lines[i].strip("|").split("|")])
                i += 1
            story.extend([table(rows), Spacer(1, 13)])
            continue
        if line.startswith("    "):
            code = []
            while i < len(lines) and lines[i].startswith("    "):
                code.append(lines[i][4:])
                i += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
            continue
        if (len(line) < 74 and not line.endswith((".", ":"))
                and not re.match(r"[RW]\d+\.", line)):
            story.append(Paragraph(markup(line), styles["sub"]))
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "    ")):
            para.append(lines[i])
            i += 1
        txt = " ".join(para)
        style = "body"
        if (len(para) == 1 and len(txt) < 74 and not txt.endswith((".", ":"))
                and not txt.startswith(("R", "W"))):
            style = "sub"
        if txt.startswith("Repository evidence:") or page_index == 22:
            style = "source"
        story.append(Paragraph(markup(txt), styles[style]))


class ReviewDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("title", "h1"):
            name = "section-" + str(self.page)
            self.canv.bookmarkPage(name)
            self.canv.addOutlineEntry(flowable.getPlainText(), name, 0)


doc = ReviewDoc(str(OUT), pagesize=A4, rightMargin=52, leftMargin=52,
                topMargin=59, bottomMargin=52,
                title="ASignX - AI Current State, Clarifications and Suggestions",
                author="ASignX project review", subject="Model audit, extraction repairs and browser deployment")
doc.build(story, onFirstPage=footer, onLaterPages=footer)

# Extract and render every page for review. Render contact sheets, too.
import pymupdf
from PIL import Image, ImageOps, ImageDraw
qa = ROOT / "tmp" / "pdfs"
qa.mkdir(parents=True, exist_ok=True)
pdf = pymupdf.open(OUT)
text_parts = []
thumbs = []
for index, page in enumerate(pdf):
    text_parts.append(page.get_text())
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35), alpha=False)
    path = qa / f"review-{index+1:02d}.png"
    pix.save(path)
    im = Image.open(path).convert("RGB")
    im.thumbnail((298, 421))
    thumb = Image.new("RGB", (318, 450), "#d9e1e7")
    thumb.paste(im, ((318-im.width)//2, 5))
    ImageDraw.Draw(thumb).text((12, 430), f"Page {index+1}", fill="black")
    thumbs.append(thumb)
for start in range(0, len(thumbs), 6):
    sheet = Image.new("RGB", (954, 900), "white")
    for pos, im in enumerate(thumbs[start:start+6]):
        sheet.paste(im, ((pos % 3)*318, (pos//3)*450))
    sheet.save(qa / f"contact-{start//6+1}.png")
(qa / "extracted.txt").write_text("\n\f\n".join(text_parts), encoding="utf-8")
print(f"Created {OUT}; {len(pdf)} pages")
for i, text in enumerate(text_parts):
    print(i + 1, len(text.split()), text.splitlines()[0][:80])
