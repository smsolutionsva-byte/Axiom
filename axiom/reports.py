from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .analytics import build_analytics
from .investigation import investigate_subject


@dataclass(frozen=True)
class ReportResult:
    subject: str
    output_dir: str
    markdown_path: str
    html_path: str
    json_path: str
    confidence: float
    evidence_count: int
    timeline_count: int


def generate_case_report(
    conn,
    subject: str,
    *,
    roots: list[str] | None = None,
    output_dir: str | Path = "exports/reports",
    top_k: int = 8,
) -> ReportResult:
    investigation = investigate_subject(conn, subject, roots=roots or [], top_k=top_k)
    analytics = build_analytics(conn, query=subject, limit=max(top_k * 4, 30))
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle = {"created_at": created_at, "investigation": investigation, "analytics": analytics}

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{slugify(subject)}"
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    html_path = root / f"{stem}.html"

    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(bundle), encoding="utf-8")
    html_path.write_text(render_html(bundle), encoding="utf-8")

    return ReportResult(
        subject=subject,
        output_dir=str(root),
        markdown_path=str(markdown_path),
        html_path=str(html_path),
        json_path=str(json_path),
        confidence=float(investigation["confidence"]),
        evidence_count=len(investigation["evidence"]),
        timeline_count=len(investigation["timeline"]),
    )


def render_markdown(bundle: dict[str, object]) -> str:
    investigation = bundle["investigation"]
    analytics = bundle["analytics"]
    prediction = analytics["prediction"]
    lines = [
        f"# Axiom Case Report: {investigation['subject']}",
        "",
        f"Generated: {bundle['created_at']}",
        "",
        "## Executive Brief",
        "",
        f"- Confidence: {int(float(investigation['confidence']) * 100)}%",
        f"- Summary: {investigation['summary']}",
        f"- Forecast: {prediction['forecast']}",
        f"- Hallucination guard: {investigation['hallucination_guard']['status']}",
        "",
        "## Evidence",
        "",
    ]
    if investigation["evidence"]:
        for item in investigation["evidence"]:
            lines.extend(
                [
                    f"### {item['citation']} {item['file_name']}",
                    "",
                    f"- Location: {item['location']}",
                    f"- Modality: {item['modality']}",
                    f"- Path: `{item['file_path']}`",
                    "",
                    item["snippet"],
                    "",
                ]
            )
    else:
        lines.extend(["No cited evidence found.", ""])

    lines.extend(["## Timeline", ""])
    for item in investigation["timeline"]:
        lines.append(f"- {item['when']} | {item['source']} | {item['citation']} | {item['summary']}")
    if not investigation["timeline"]:
        lines.append("- No timeline items found.")

    lines.extend(["", "## Leads", ""])
    for item in investigation["file_leads"]:
        lines.append(f"- `{item['path']}` | {item['reason']} | score {item['score']}")
    if not investigation["file_leads"]:
        lines.append("- No filesystem leads found.")

    lines.extend(["", "## Risk Flags", ""])
    for flag in investigation["risk_flags"] or ["none"]:
        lines.append(f"- {flag}")

    lines.extend(["", "## Evidence Gaps", ""])
    for gap in prediction["gaps"] or ["No major gaps detected."]:
        lines.append(f"- {gap}")

    lines.extend(["", "## Next Actions", ""])
    for action in investigation["next_actions"]:
        lines.append(f"- {action}")

    lines.extend(["", "## Guardrail Rules", ""])
    for rule in investigation["hallucination_guard"]["rules"]:
        lines.append(f"- {rule}")

    return "\n".join(lines) + "\n"


def render_html(bundle: dict[str, object]) -> str:
    investigation = bundle["investigation"]
    analytics = bundle["analytics"]
    prediction = analytics["prediction"]
    evidence_rows = "".join(
        f"""
        <tr>
          <td>{esc(item['citation'])}</td>
          <td>{esc(item['file_name'])}</td>
          <td>{esc(item['location'])}</td>
          <td>{esc(item['snippet'])}</td>
        </tr>
        """
        for item in investigation["evidence"]
    ) or '<tr><td colspan="4">No cited evidence found.</td></tr>'
    timeline_rows = "".join(
        f"<li><strong>{esc(item['when'])}</strong> {esc(item['source'])} {esc(item['citation'])}<br>{esc(item['summary'])}</li>"
        for item in investigation["timeline"]
    ) or "<li>No timeline items found.</li>"
    leads = "".join(
        f"<li><code>{esc(item['path'])}</code> · {esc(item['reason'])} · score {esc(item['score'])}</li>"
        for item in investigation["file_leads"]
    ) or "<li>No filesystem leads found.</li>"
    actions = "".join(f"<li>{esc(action)}</li>" for action in investigation["next_actions"])
    gaps = "".join(f"<li>{esc(gap)}</li>" for gap in prediction["gaps"] or ["No major gaps detected."])
    rules = "".join(f"<li>{esc(rule)}</li>" for rule in investigation["hallucination_guard"]["rules"])
    risk = ", ".join(investigation["risk_flags"] or ["none"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Axiom Case Report - {esc(investigation['subject'])}</title>
  <style>
    body {{ margin: 0; padding: 32px; font: 14px/1.55 Segoe UI, Arial, sans-serif; color: #17201b; background: #f4f6f3; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    section {{ background: #fff; border: 1px solid #cfd8d0; border-radius: 8px; padding: 18px; margin: 14px 0; }}
    h1, h2 {{ margin: 0 0 10px; }}
    .metric {{ display: inline-block; margin-right: 16px; padding: 8px 10px; background: #e7f3ec; border-radius: 8px; }}
    .score {{ font-size: 32px; font-weight: 800; color: #0f766e; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #cfd8d0; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; background: #eef2ee; }}
    code {{ overflow-wrap: anywhere; }}
    li {{ margin: 6px 0; }}
    @media print {{ body {{ background: #fff; padding: 0; }} section {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
<main>
  <h1>Axiom Case Report: {esc(investigation['subject'])}</h1>
  <p>Generated: {esc(bundle['created_at'])}</p>
  <section>
    <h2>Executive Brief</h2>
    <div class="metric"><div class="score">{int(float(investigation['confidence']) * 100)}%</div>Investigation confidence</div>
    <div class="metric">{esc(investigation['hallucination_guard']['status'])}<br>Guard status</div>
    <p>{esc(investigation['summary'])}</p>
    <p><strong>Forecast:</strong> {esc(prediction['forecast'])}</p>
    <p><strong>Risk flags:</strong> {esc(risk)}</p>
  </section>
  <section>
    <h2>Evidence</h2>
    <table>
      <thead><tr><th>Citation</th><th>Source</th><th>Location</th><th>Snippet</th></tr></thead>
      <tbody>{evidence_rows}</tbody>
    </table>
  </section>
  <section><h2>Timeline</h2><ul>{timeline_rows}</ul></section>
  <section><h2>Filesystem Leads</h2><ul>{leads}</ul></section>
  <section><h2>Evidence Gaps</h2><ul>{gaps}</ul></section>
  <section><h2>Next Actions</h2><ul>{actions}</ul></section>
  <section><h2>Guardrail Rules</h2><ul>{rules}</ul></section>
</main>
</body>
</html>
"""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or "case"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)
