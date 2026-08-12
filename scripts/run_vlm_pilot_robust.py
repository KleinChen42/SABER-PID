"""VLM pilot entry point with tolerant extraction of a requested JSON object."""

from __future__ import annotations

import json

import run_vlm_pilot


def extract_prediction(raw: str, mode: str):
    if mode != "structured":
        return "ANSWER", raw.strip()
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    parsed = None
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        return "INVALID", None
    action = str(parsed.get("action", "ANSWER")).upper()
    if action not in {"ANSWER", "ABSTAIN", "INSUFFICIENT"}:
        action = "INVALID"
    return action, parsed.get("answer")


run_vlm_pilot.extract_prediction = extract_prediction

if __name__ == "__main__":
    raise SystemExit(run_vlm_pilot.main())
