"""Render a DOCX through an isolated Microsoft Word COM instance."""

from __future__ import annotations

import argparse
from pathlib import Path

import win32com.client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        document = word.Documents.Open(
            str(source), ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False
        )
        try:
            document.ExportAsFixedFormat(str(output), 17)
        finally:
            document.Close(False)
    finally:
        word.Quit()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
