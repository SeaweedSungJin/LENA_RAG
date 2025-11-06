"""Utility helpers for parsing EchoSight section payloads (LENA)."""
from __future__ import annotations

from typing import Tuple


def parse_section_payload(text: str) -> Tuple[str, str, str]:
    doc_title = "unknown"
    section_title = "unknown"
    body_lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("# Wiki Article:"):
            doc_title = stripped.split(":", 1)[-1].strip()
        elif stripped.startswith("## Section Title:"):
            section_title = stripped.split(":", 1)[-1].strip()
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return doc_title, section_title, body

