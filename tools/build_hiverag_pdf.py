from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "HIVERAG_RESEARCH_PAPER_SHIVANSH_MUKHIA.md"
OUTPUT = ROOT / "exports" / "papers" / "HiveRAG_Research_Paper_Shivansh_Mukhia.pdf"


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "HiveTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "author": ParagraphStyle(
            "HiveAuthor",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "HiveHeading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#172554"),
        ),
        "h2": ParagraphStyle(
            "HiveHeading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#1e3a8a"),
        ),
        "body": ParagraphStyle(
            "HiveBody",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.4,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "HiveBullet",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.8,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "HiveCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#111827"),
        ),
        "table": ParagraphStyle(
            "HiveTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.5,
            alignment=TA_LEFT,
        ),
    }


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.68 * inch,
        title="HiveRAG Research Paper",
        author="Shivansh Mukhia",
    )
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(OUTPUT)


def parse_markdown(markdown: str) -> list[Flowable]:
    st = styles()
    story: list[Flowable] = []
    lines = markdown.splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(code_block("\n".join(code_lines), st))
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(markdown_table(table_lines, st))
            story.append(Spacer(1, 7))
            continue
        if line.startswith("# "):
            title = clean_inline(line[2:].strip())
            story.append(Paragraph(title, st["title"]))
            index += 1
            continue
        if line.strip() == "Shivansh Mukhia":
            story.append(Paragraph("Shivansh Mukhia", st["author"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:].strip()), st["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:].strip()), st["h2"]))
            index += 1
            continue
        if line.startswith("- "):
            bullets = []
            while index < len(lines) and lines[index].startswith("- "):
                bullets.append(ListItem(Paragraph(clean_inline(lines[index][2:].strip()), st["bullet"])))
                index += 1
            story.append(ListFlowable(bullets, bulletType="bullet", leftIndent=12))
            continue
        if re.match(r"^\d+\. ", line):
            while index < len(lines) and re.match(r"^\d+\. ", lines[index]):
                match = re.match(r"^(\d+)\. (.*)$", lines[index])
                number = match.group(1) if match else "1"
                item_text = match.group(2).strip() if match else lines[index].strip()
                story.append(Paragraph(f"{number}. {clean_inline(item_text)}", st["bullet"]))
                index += 1
            continue
        story.append(Paragraph(clean_inline(line), st["body"]))
        index += 1
    return story


def markdown_table(lines: list[str], st: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for raw in lines:
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append([Paragraph(clean_inline(cell), st["table"]) for cell in cells])
    col_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < col_count:
            row.append(Paragraph("", st["table"]))
    available = 7.1 * inch
    widths = [available / col_count] * col_count
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def code_block(text: str, st: dict[str, ParagraphStyle]) -> Flowable:
    block = Preformatted(text, st["code"], maxLineLength=90)
    return Table(
        [[block]],
        colWidths=[7.1 * inch],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        ),
    )


def clean_inline(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(doc.leftMargin, 0.38 * inch, "HiveRAG - Shivansh Mukhia")
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    build()
