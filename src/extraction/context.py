"""
Structural context building for extraction.

Pure functions over the v2 parts of one document: anchors, outline, sibling completion,
budget and rendering. No I/O, no type-specific logic: everything that depends on what
is being extracted comes from the retrieval profile and from the prompts.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from math import log
from typing import Any

from pydantic import BaseModel, Field


class SelectedSections(BaseModel):
    """Sections retenues par le sélecteur d'arborescence."""

    section_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiants des sections à lire, recopiés depuis les crochets de "
            "l'arborescence (ex: ['S12', 'S5_split_1']). Aucun identifiant inventé."
        ),
    )


def fold_title(text: str | None) -> str:
    """Case, accent, dash, apostrophe and whitespace insensitive form of a title."""
    text = text or ""
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015"):
        text = text.replace(dash, "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def find_anchor_ids(parts: list[dict[str, Any]], anchor_titles: list[str]) -> list[str]:
    """
    Every part whose title literally contains one of the anchor titles.

    A document may carry several application-scope sections (one per block, per
    population, ...): all of them are returned, in document order.
    """
    anchors = [fold_title(title) for title in anchor_titles if title and title.strip()]
    return [
        part["section_id"]
        for part in parts
        if any(anchor in fold_title(part.get("title")) for anchor in anchors)
    ]


def preamble_ids(parts: list[dict[str, Any]]) -> list[str]:
    """Fallback anchors when the document has no application-scope section."""
    return [part["section_id"] for part in parts if part.get("origin") == "preamble"]


def expand_subtrees(parts: list[dict[str, Any]], root_ids: list[str]) -> list[str]:
    """Root ids plus every descendant, in document order."""
    children: dict[str | None, list[str]] = defaultdict(list)
    order = {part["section_id"]: part["section_order"] for part in parts}
    for part in parts:
        children[part.get("parent_id")].append(part["section_id"])

    selected: set[str] = set()
    stack = list(root_ids)
    while stack:
        section_id = stack.pop()
        if section_id in selected:
            continue
        selected.add(section_id)
        stack.extend(children.get(section_id, []))

    return sorted((section_id for section_id in selected if section_id in order), key=order.get)


def _outline_lines(
    parts: list[dict[str, Any]], child_counts: Counter
) -> list[str]:
    lines = []
    for part in parts:
        indent = "  " * max(part.get("level", 1) - 1, 0)
        number = part.get("number") or "-"
        title = part.get("title") or ""
        n_children = child_counts.get(part["section_id"], 0)
        suffix = f" (+{n_children} sous-sections)" if n_children else ""
        n_chars = part.get("n_chars", len(part.get("body_md") or ""))
        lines.append(f"{indent}[{part['section_id']}] {number} | {title}{suffix} | {n_chars} c")
    return lines


def build_outline(parts: list[dict[str, Any]], max_chars: int) -> str:
    """Compact, deterministic view of the document tree shown to the selector."""
    ordered = sorted(parts, key=lambda part: part["section_order"])
    child_counts = Counter(part["parent_id"] for part in parts if part.get("parent_id"))

    text = "\n".join(_outline_lines(ordered, child_counts))
    if len(text) <= max_chars:
        return text

    shallow = [part for part in ordered if part.get("level", 1) <= 2]
    text = "\n".join(_outline_lines(shallow, child_counts))
    marker = "\n[... arborescence tronquée ...]"
    if len(text) <= max_chars:
        return text + marker
    return text[:max_chars] + marker


def part_ancestors(part: dict[str, Any]) -> list[str]:
    """Ancestors of the part, without the part itself (the path ends with its title)."""
    title = fold_title(part.get("title"))
    ancestors: list[str] = []
    for crumb in (part.get("path") or "").split(">"):
        crumb = crumb.strip()
        if not crumb or (ancestors and fold_title(crumb) == fold_title(ancestors[-1])):
            continue
        ancestors.append(crumb)
    while ancestors and title and fold_title(ancestors[-1]) == title:
        ancestors.pop()
    return ancestors


def strip_heading(text: str, title: str) -> str:
    """Body without the leading heading line that repeats the part title."""
    heading = fold_title(title)
    lines = text.split("\n")
    index = 0
    removed = False
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if heading and fold_title(lines[index]) == heading:
            index += 1
            removed = True
            continue
        break
    return "\n".join(lines[index:]) if removed else text


def render_part(part: dict[str, Any], anchor_ids: set[str], anchor_tag: str) -> str:
    """One block: stable id header, breadcrumb, body, tables."""
    section_id = part["section_id"]
    title = part.get("title") or ""
    tag = f"   ⇐ {anchor_tag}" if section_id in anchor_ids else ""
    lines = [f"### [{section_id}] {title}{tag}"]

    ancestors = part_ancestors(part)
    if ancestors:
        lines.append(f"chemin : {' > '.join(ancestors)}")

    body = strip_heading(part.get("body_md") or "", title).strip()
    if body:
        lines.append(body)
    for table in part.get("tables") or []:
        markdown = (table.get("md") or "").strip()
        if markdown:
            lines.append(markdown)
    return "\n".join(lines)


def render_document(parts: list[dict[str, Any]], anchor_ids: set[str], anchor_tag: str) -> str:
    """Whole document, in document order (full-reading mode)."""
    ordered = sorted(parts, key=lambda part: part["section_order"])
    return "\n\n".join(render_part(part, anchor_ids, anchor_tag) for part in ordered)


def selector_section_budget(
    n_parts: int,
    *,
    min_parts: int,
    max_parts: int,
    small_doc_parts: int,
    large_doc_parts: int,
) -> int:
    """
    How many sections the selector may pick in a document of `n_parts` parts.

    Log interpolation between two anchors (e.g. 2 picks for a 5-part document, 10 for a
    30-part one): the number of relevant sections grows much slower than the document, so
    a linear rule over-selects on the big agreements. Clamped to `[min_parts, max_parts]`.
    """
    if large_doc_parts <= small_doc_parts:
        return max_parts
    if n_parts <= small_doc_parts:
        return min_parts
    if n_parts >= large_doc_parts:
        return max_parts
    ratio = log(n_parts / small_doc_parts) / log(large_doc_parts / small_doc_parts)
    return round(min_parts + (max_parts - min_parts) * ratio)


def build_context(
    parts: list[dict[str, Any]],
    anchor_ids: list[str],
    selected_ids: list[str],
    similarity_ids: list[str],
    sibling_scores: dict[str, float],
    *,
    max_siblings: int,
    max_sibling_chars: int,
    max_family_size: int,
    max_picked_chars: int,
    anchor_tag: str,
) -> tuple[str, dict[str, int]]:
    """
    Assemble the hybrid context, layer by layer.

    Layers: anchors (never dropped), selector picks, siblings of selector picks,
    similarity hits. The final text is always rendered in document order.

    There is no character budget here: the caller compares the rendered length with its
    safety ceiling and falls back to chunk retrieval when the selection is still too big.

    Sibling completion is deliberately narrow. At most `max_siblings` neighbours of a
    picked section, ranked by their similarity to the majoration query
    (`sibling_scores`), and only when the sibling is at most `max_sibling_chars` long.
    Big families (more than `max_family_size` children) and big picked sections (more
    than `max_picked_chars`) get no sibling at all: they are containers, and completing
    them would just resend the document.
    """
    ordered = sorted(parts, key=lambda part: part["section_order"])
    by_id = {part["section_id"]: part for part in ordered}
    order = {part["section_id"]: part["section_order"] for part in ordered}
    blocks = {
        part["section_id"]: render_part(part, set(anchor_ids), anchor_tag) for part in ordered
    }
    sizes = {section_id: len(block) + 2 for section_id, block in blocks.items()}

    anchors = sorted({sid for sid in anchor_ids if sid in by_id}, key=order.get)
    selected = sorted(
        {sid for sid in selected_ids if sid in by_id and sid not in anchors}, key=order.get
    )
    similar = sorted(
        {
            sid
            for sid in similarity_ids
            if sid in by_id and sid not in anchors and sid not in selected
        },
        key=order.get,
    )

    chosen: set[str] = set(anchors)
    after_anchors = set(chosen)
    chosen.update(selected)
    after_selected = set(chosen)

    by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for part in ordered:
        by_parent[part.get("parent_id")].append(part)

    for section_id in selected:
        family = by_parent.get(by_id[section_id].get("parent_id"), [])
        if not 1 < len(family) <= max_family_size:
            continue
        if sizes[section_id] > max_picked_chars:
            continue
        siblings = [
            part["section_id"]
            for part in family
            if part["section_id"] != section_id
            and sizes[part["section_id"]] <= max_sibling_chars
        ]
        siblings.sort(key=lambda sid: (-sibling_scores.get(sid, -1.0), order[sid]))
        chosen.update(siblings[:max_siblings])
    after_siblings = set(chosen)

    chosen.update(similar)
    after_similarity = set(chosen)

    rendered = "\n\n".join(blocks[section_id] for section_id in sorted(chosen, key=order.get))
    info = {
        "n_anchor_parts": len(after_anchors),
        "n_selected_parts": len(after_selected) - len(after_anchors),
        "n_sibling_parts": len(after_siblings) - len(after_selected),
        "n_similarity_parts": len(after_similarity) - len(after_siblings),
        "n_context_parts": len(after_similarity),
        "n_anchor_chars": sum(sizes[sid] for sid in after_anchors),
        "n_selected_chars": sum(sizes[sid] for sid in after_selected - after_anchors),
        "n_sibling_chars": sum(sizes[sid] for sid in after_siblings - after_selected),
        "n_similarity_chars": sum(sizes[sid] for sid in after_similarity - after_siblings),
        "context_chars": len(rendered),
    }
    return rendered, info
