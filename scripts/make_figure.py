"""Render the security–utility frontier from experiment results — stdlib only.

    python scripts/make_figure.py [--results results/results.jsonl]
                                  [--out results/security_utility.svg]

Aggregates per (model, defence) exactly like the report table and draws an
SVG scatter: x = legitimate-answer accuracy, y = canary disclosure rate.
Strong systems sit in the bottom-right corner (low canary rate, high accuracy).
No matplotlib or any other third-party dependency is required, so the figure
is reproducible on any machine that could run the experiment itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import CaseRecord, record_from_row, summarize_cell  # noqa: E402

WIDTH, HEIGHT, MARGIN = 720, 540, 64
PALETTE = ["#2f5fd0", "#c0392b", "#0c7a43", "#8a5a00", "#6b3fa0"]
SHORT = {"none": "none", "delimiter": "delim", "filter": "filter", "both": "A+B"}


def load_records(path: Path) -> list[CaseRecord]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(record_from_row(json.loads(line)))
    return records


def svg_point(x: float, y: float) -> tuple[float, float]:
    return (MARGIN + x * (WIDTH - 2 * MARGIN),
            HEIGHT - MARGIN - y * (HEIGHT - 2 * MARGIN))


def build_svg(summaries: list, model_order: list[str]) -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Verdana, sans-serif">')
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>')
    parts.append('<text x="%d" y="30" font-size="16" fill="#1c2333">'
                 'Security&#8211;utility frontier (frozen test split)</text>' % MARGIN)
    parts.append('<text x="%d" y="50" font-size="11" fill="#5b6474">'
                 'Better systems sit toward the bottom-right: low canary '
                 'disclosure rate, high legitimate-answer accuracy.</text>' % MARGIN)

    # gridlines and tick labels
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx, gy = svg_point(t, t)
        parts.append(f'<line x1="{gx:.1f}" y1="{HEIGHT - MARGIN}" x2="{gx:.1f}" '
                     f'y2="{MARGIN}" stroke="#e3e4df"/>')
        parts.append(f'<line x1="{MARGIN}" y1="{gy:.1f}" x2="{WIDTH - MARGIN}" '
                     f'y2="{gy:.1f}" stroke="#e3e4df"/>')
        parts.append(f'<text x="{gx:.1f}" y="{HEIGHT - MARGIN + 16}" font-size="10" '
                     f'fill="#5b6474" text-anchor="middle">{t:.2f}</text>')
        parts.append(f'<text x="{MARGIN - 10:.1f}" y="{gy + 3:.1f}" font-size="10" '
                     f'fill="#5b6474" text-anchor="end">{t:.2f}</text>')

    # axes and axis labels
    parts.append(f'<line x1="{MARGIN}" y1="{HEIGHT - MARGIN}" x2="{WIDTH - MARGIN}" '
                 f'y2="{HEIGHT - MARGIN}" stroke="#1c2333"/>')
    parts.append(f'<line x1="{MARGIN}" y1="{MARGIN}" x2="{MARGIN}" '
                 f'y2="{HEIGHT - MARGIN}" stroke="#1c2333"/>')
    cx = (MARGIN + WIDTH - MARGIN) / 2
    cy = (MARGIN + HEIGHT - MARGIN) / 2
    parts.append(f'<text x="{cx:.0f}" y="{HEIGHT - 18}" font-size="12" fill="#1c2333" '
                 'text-anchor="middle">legitimate-answer accuracy</text>')
    parts.append(f'<text x="16" y="{cy:.0f}" font-size="12" fill="#1c2333" '
                 f'text-anchor="middle" transform="rotate(-90 16 {cy:.0f})">'
                 'canary disclosure rate</text>')

    # one labelled point per (model, defence) cell
    for s in summaries:
        color = PALETTE[model_order.index(s.model) % len(PALETTE)]
        px, py = svg_point(s.accuracy, s.canary_rate)
        label = SHORT.get(s.defense, s.defense)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{color}" '
                     'fill-opacity="0.85" stroke="#ffffff"/>')
        parts.append(f'<text x="{px + 10:.1f}" y="{py + 4:.1f}" font-size="11" '
                     f'fill="{color}">{s.model} / {label}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    ap.add_argument("--out", default="results/security_utility.svg")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"no results file at {path} — run the experiment first "
              f"(scripts/run_experiment.py)")
        raise SystemExit(1)
    records = load_records(path)
    if not records:
        print("results file is empty")
        raise SystemExit(1)

    cells: dict[tuple[str, str], list[CaseRecord]] = {}
    for r in records:
        cells.setdefault((r.model, r.defense), []).append(r)
    summaries = [summarize_cell(recs) for _key, recs in cells.items()]
    model_order = sorted({s.model for s in summaries})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(summaries, model_order), encoding="utf-8")
    print(f"wrote {out} ({len(summaries)} cells, models: {', '.join(model_order)})")
    for s in summaries:
        print(f"  {s.model:14s} {s.defense:10s} canary_rate={s.canary_rate:.2f} acc={s.accuracy:.2f}")


if __name__ == "__main__":
    main()
