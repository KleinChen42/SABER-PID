from pathlib import Path

from pidbench.pidqa import load_pidqa, summarize_pidqa


def test_load_pidqa_preserves_source_identity() -> None:
    # A checked-in fixture avoids platform-specific temporary-directory ACLs
    # while exercising the same four released PIDQA CSV schemas.
    fixture = Path(__file__).parent / "fixtures" / "pidqa_minimal"
    records = load_pidqa(fixture)
    assert {record["source_id"] for record in records} == {"pidqa-sheet-007"}
    assert summarize_pidqa(records)["record_count"] == 4
