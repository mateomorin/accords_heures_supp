"""Row assembly for the RAG v2 preprocessing output."""

from __future__ import annotations

import json
from typing import Any

from repair import process_document


def format_number(number: tuple | None) -> str | None:
    """Format a canonical number tuple: (3, 1, 'b') -> '3.1b'."""
    if not number:
        return None
    out: list[str] = []
    for part in number:
        if isinstance(part, str):
            out.append(part)
        else:
            if out:
                out.append(".")
            out.append(str(part))
    return "".join(out)


def attach_tables(parts: list[dict[str, Any]], doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach table markdown payloads to parts."""
    tables_by_id = {
        table.get("id"): table for table in (doc.get("tables") or []) if table.get("id")
    }
    for part in parts:
        tables = []
        for table_id in part.get("table_ids") or []:
            table = tables_by_id.get(table_id)
            if table:
                tables.append({"id": table_id, "md": table.get("md") or ""})
        part["tables"] = tables
    return parts


def part_to_dict(part: dict[str, Any]) -> dict[str, Any]:
    """Public JSON representation of one part."""
    body_md = part.get("body_md") or ""
    tables = part.get("tables") or []
    return {
        "section_id": part["section_id"],
        "original_section_id": part["original_section_id"],
        "section_order": part["section_order"],
        "level": part["level"],
        "number": format_number(part["number"]),
        "title": part["title"],
        "parent_id": part["parent"]["section_id"] if part.get("parent") else None,
        "path": part["path"],
        "body_md": body_md,
        "n_chars": len(body_md),
        "table_ids": part.get("table_ids") or [],
        "tables": tables,
        "n_tables": len(tables),
        "origin": part["origin"],
        "merged_ids": part.get("merged_ids") or [],
    }


def doc_to_row(doc: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert one document into one output row (parts packed in a JSON column)."""
    parts = attach_tables(process_document(doc), doc)

    date_signature = metadata.get("dateSignature")
    try:
        year = int(str(date_signature)[:4]) if date_signature else None
    except ValueError:
        year = None

    parts_json = [part_to_dict(part) for part in parts]
    return {
        "cid": metadata.get("cid"),
        "reference": metadata.get("reference"),
        "dateSignature": date_signature,
        "anneeSignature": year,
        "doc_type": metadata.get("doc_type"),
        "format": metadata.get("format"),
        "doc_n_sections": len(doc.get("sections") or []),
        "doc_n_tables": len(doc.get("tables") or []),
        "n_parts": len(parts_json),
        "parts_json": json.dumps(parts_json, ensure_ascii=False),
    }
