"""Structured VLM pilot variant that requests a best visual answer by default."""

from __future__ import annotations

import run_vlm_pilot_robust


def prompt_for(question: str, mode: str) -> str:
    if mode != "structured":
        return run_vlm_pilot_robust.run_vlm_pilot.prompt_for(question, mode)
    return (
        "You analyze a piping and instrumentation diagram (P&ID). "
        "Make your best determination from the visible diagram. Return exactly "
        "one JSON object with no Markdown using this schema: "
        '{"action":"ANSWER|ABSTAIN|INSUFFICIENT","answer":value,'
        '"entities":[],"edges":[],"evidence":[],"confidence":0.0}. '
        "Use ANSWER whenever the diagram supports a best answer; use ABSTAIN "
        "only when the image itself is not interpretable. For a list question, "
        "put a JSON array of tags in answer.\n\n"
        f"Question: {question}"
    )


run_vlm_pilot_robust.run_vlm_pilot.prompt_for = prompt_for

if __name__ == "__main__":
    raise SystemExit(run_vlm_pilot_robust.run_vlm_pilot.main())
