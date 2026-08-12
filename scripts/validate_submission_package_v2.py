"""Validate the technical Results in Engineering submission package.

The validator is intentionally deterministic. It checks claim artifacts, LaTeX
cross-references, references, figures, abstract/highlight constraints, and
submission-file presence. Author identity fields are reported as known
submitter-owned placeholders rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/title_page.md",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_1_evaluation_protocol.pdf",
    "paper/figures/figure_1_evaluation_protocol.png",
    "paper/figures/figure_2_controlled_effects.pdf",
    "paper/figures/figure_2_controlled_effects.png",
    "paper/figures/figure_3_source_split_sensitivity.pdf",
    "paper/figures/figure_3_source_split_sensitivity.png",
    "paper/figures/figure_metadata_v2.json",
    "reports/generated/final_statistical_summary_v2.json",
    "reports/generated/final_claim_evidence_matrix_v2.csv",
    "reports/generated/evidence_input_answer_isolation_audit_v1.json",
    "reports/generated/pid2graph_recheck_v1.json",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text))


def abstract(text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S)
    if not match:
        raise ValueError("Missing abstract environment")
    return re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", match.group(1))


def latex_ids(text: str, command: str) -> set[str]:
    values: set[str] = set()
    for match in re.finditer(rf"\\{command}\{{([^}}]+)\}}", text):
        values.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/submission_package_validation_v2.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing_files = [path for path in REQUIRED_FILES if not (root / path).exists()]
    manuscript_path = root / "paper/manuscript.tex"
    supplementary_path = root / "paper/supplementary.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    supplementary = supplementary_path.read_text(encoding="utf-8")
    abstract_words = word_count(abstract(manuscript))
    cited = latex_ids(manuscript, "cite")
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    missing_citations = sorted(cited - bib)
    uncited_bibliography = sorted(bib - cited)
    labels = latex_ids(manuscript, "label")
    refs = latex_ids(manuscript, "ref")
    missing_refs = sorted(refs - labels)
    figure_refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript)
    figures_missing = []
    for figure in figure_refs:
        candidate = root / "paper" / "figures" / figure
        if not candidate.exists():
            figures_missing.append(figure)
    highlight_lines = [
        line[2:].strip()
        for line in (root / "paper/highlights.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
    highlight_lengths = [len(line) for line in highlight_lines]
    summary = read_json(root / "reports/generated/final_statistical_summary_v2.json")
    isolation = read_json(root / "reports/generated/evidence_input_answer_isolation_audit_v1.json")
    f4 = read_json(root / "reports/generated/pid2graph_recheck_v1.json")
    required_tokens = [
        "0.213969",
        "0.3375",
        "0.2875",
        "+0.5250",
        "-0.5487",
        "+0.3100",
        "+0.2400",
        "-0.0250",
        "8,182,946,588",
    ]
    missing_numeric_tokens = [token for token in required_tokens if token not in manuscript]
    report = {
        "status": "pass",
        "required_files": {"missing": missing_files, "count": len(REQUIRED_FILES)},
        "abstract": {"word_count": abstract_words, "max_allowed": 250},
        "manuscript": {
            "word_count": word_count(manuscript),
            "supplement_word_count": word_count(supplementary),
            "citation_count": len(cited),
            "bibliography_count": len(bib),
            "missing_citations": missing_citations,
            "uncited_bibliography": uncited_bibliography,
            "missing_refs": missing_refs,
            "figures_missing": figures_missing,
            "missing_required_numeric_tokens": missing_numeric_tokens,
        },
        "highlights": {
            "count": len(highlight_lines),
            "lengths": highlight_lengths,
            "max_allowed_per_line": 85,
        },
        "evidence": {
            "final_summary_status": summary.get("status"),
            "claim_count": len(summary.get("claims", [])),
            "input_isolation_status": isolation.get("status"),
            "pid2graph_status": f4.get("status"),
            "external_score_reported": f4.get("external_score_reported"),
        },
        "submitter_owned_placeholders": [
            "author names and affiliations",
            "corresponding author contact details",
            "final author confirmation of declarations",
            "permanent public archive URL",
        ],
    }
    failures = (
        missing_files
        or abstract_words > 250
        or missing_citations
        or uncited_bibliography
        or missing_refs
        or figures_missing
        or missing_numeric_tokens
        or not 3 <= len(highlight_lines) <= 5
        or any(length > 85 for length in highlight_lengths)
        or summary.get("status") != "pass"
        or isolation.get("status") != "pass"
        or f4.get("external_score_reported") is not False
    )
    report["status"] = "fail" if failures else "pass"
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
