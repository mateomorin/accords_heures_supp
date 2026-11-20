"""
Deterministic repair passes that rebuild document structure from final shards.

No LLM is involved. The passes, in order:

1. Canonical number per section: the title is authoritative, the ``number``
   field is the fallback, and on a prefix conflict the deeper number wins.
2. Body splits: a heading embedded in a section body starts a new part when
   its number is (a) a container heading, (b) a parent number that exists
   nowhere else, (c) the prefix of the next section's number, (d) a missing
   number between two consecutive siblings (OCR dropped the digits), or
   (e) the next top-level number after the previous one.
3. Gap fill: when the missing sibling number is absent everywhere, an
   unnumbered heading-like line (number OCR-dropped) in the previous section's
   body is split as the missing part.
4. P.1 renumber: an unnumbered short title sitting right before ``P.k``
   (k >= 2) where ``P.1`` is absent is renumbered ``P.1``.
5. Leading merge: unnumbered parts before the first numbered part collapse
   into a single ``PREAMBULE`` part (``Champ d'application`` stays separate).
6. Tree build: a single stack over the document order; containers and
   single-component numbers reset the scope, duplicates stay siblings.
"""

from __future__ import annotations

import re

from number_parser import (
    CHAMP_RE,
    LEGAL_RE,
    PAGE_RE,
    ROMAN,
    ROMAN_RE,
    canon_of,
    fold,
    is_container_title,
    is_prefix,
    line_container,
    line_number,
    line_number_parts,
    numbered_line_ok,
)


def _new_part(
    section_id: str,
    title: str,
    number: tuple | None,
    body_md: str,
    table_ids: list[str],
    origin: str,
    merged_ids: list[str] | None = None,
) -> dict:
    return {
        "section_id": section_id,
        "original_section_id": section_id,
        "title": title,
        "number": number,
        "body_md": body_md,
        "table_ids": table_ids,
        "origin": origin,
        "is_container": is_container_title(title),
        "merged_ids": merged_ids or [],
    }


def _split_points(
    lines: list[str],
    title: str,
    idx: int,
    n: int,
    canons: list,
    missing_parents: set,
    gaps: dict,
    prev_top: list,
    num_set: set,
) -> list[tuple[int, dict]]:
    """Numbered/container/champ split points inside one section body."""
    points: list[tuple[int, dict]] = []
    next_c = canons[idx + 1] if idx + 1 < n else None
    t_fold = fold(title)
    for j in range(1, len(lines)):
        ln = lines[j].strip()
        if not ln or lines[j - 1].strip() != "":
            continue
        if fold(ln) == t_fold:
            continue
        lc = line_container(ln)
        if lc and len(ln) <= 120:
            points.append((j, {"number": lc[1], "title": ln}))
            continue
        nl = line_number_parts(ln)
        if nl:
            lv, numstr = nl
            if numbered_line_ok(ln, fold(ln).strip(), numstr):
                if lv in missing_parents and len(ln) <= 120:
                    points.append((j, {"number": lv, "title": ln}))
                elif len(ln) >= 8 and next_c and len(next_c) >= 2 and lv == next_c[:-1]:
                    points.append((j, {"number": lv, "title": ln}))
                elif len(ln) >= 8 and lv in gaps.get(idx, set()):
                    points.append((j, {"number": lv, "title": ln}))
                elif (
                    len(ln) >= 8
                    and len(lv) == 1
                    and isinstance(lv[0], int)
                    and prev_top[idx] is not None
                    and lv[0] == prev_top[idx] + 1
                    and lv not in num_set
                    and len(ln) <= 80
                ):
                    points.append((j, {"number": lv, "title": ln}))
            continue
        if CHAMP_RE.match(fold(ln)) and len(ln) <= 80:
            points.append((j, {"number": None, "title": "Champ d'application"}))
    return points


def _gap_fill_points(
    lines: list[str],
    points: list[tuple[int, dict]],
    idx: int,
    gaps: dict,
) -> None:
    """
    Split an unnumbered heading-like line for each still-unresolved gap.

    The OCR often drops the digits of a heading, leaving a bare line
    ("Contingent annuel") in the previous section's body.
    """
    resolved = {p["number"] for _, p in points if p["number"]}
    for g in sorted(gaps.get(idx, set()) - resolved):
        tail_lines = lines[points[-1][0] :] if points else lines
        cands = [k for k in range(1, len(tail_lines)) if _heading_like(tail_lines, k)]
        if not cands:
            continue
        k = cands[-1]
        abs_j = (points[-1][0] if points else 0) + k
        ln = lines[abs_j].strip()
        points.append((abs_j, {"number": g, "title": ln}))


def _heading_like(lines: list[str], i: int) -> bool:
    """Return whether the line looks like a bare (number-dropped) heading."""
    s = lines[i].strip()
    t = fold(s)
    if not (3 <= len(s) <= 80):
        return False
    if len(s.split()) > 15:
        return False
    if t[-1] in ".!?;,":
        return False
    first = s[0]  # original (pre-fold) char: lowercase => mid-sentence, reject
    if not (first.isupper() or first.isdigit() or first in "-–•*()"):
        return False
    if first == "•":  # markdown bullet list item
        return False
    if first in "-–*":  # dash bullet: the title word must be capitalized
        nxt = s[1:].lstrip()
        if not nxt or not (nxt[0].isupper() or nxt[0].isdigit()):
            return False
    if re.search(r"[:]\s*\d", s):  # list value: "Pour les ETAM : 130 heures"
        return False
    # long determiner-led lines are usually sentences, not headings
    if re.match(r"^(Le |La |Les |L')", s) and len(s.split()) >= 8:
        return False
    if i > 1 and lines[i - 1].strip() != "":
        return False
    if LEGAL_RE.match(t):
        return False
    if line_number(s) is not None:
        return False
    if s[0].isdigit() and line_number(s) is None:  # '151,67h': a value, not a heading
        return False
    rm = ROMAN_RE.match(t)  # roman heading (its number is not the gap's)
    if rm and rm.group(1) in ROMAN:
        return False
    # heading paragraphs are delimited: blank line before (checked above) and after
    if i + 1 >= len(lines) or lines[i + 1].strip() != "":
        return False
    if not any(line.strip() for line in lines[i + 2 :]):
        return False
    return True


def _renumber_ok(title: str, part: dict) -> bool:
    """
    Return whether the unnumbered part right before P.k (k>=2, P.1 absent) is the dropped P.1.

    Guards against party names ("L'ORGANISATION SYNDICALE CGT"), sentences
    ("Les horaires... inchangés."), list headers ("PREAMBULE :") and
    cross-references ("L'article 2.1 ...").
    """
    t = (title or "").strip()
    ft = fold(t.replace("\u2019", "'").replace("\u2018", "'"))
    if not (3 <= len(t) <= 50):
        return False
    if not t[0].isupper():
        return False
    if len(t.split()) > 5:
        return False
    if t[-1] in ".!?;:," or ft[-1] in ".!?;,":
        return False
    if ft.startswith(("CHAMP D", "PREAMBULE", "LE ", "LA ", "LES ", "L'", "DE ", "DES ", "D' ")):
        return False
    if re.search(r"\d", t):
        return False
    if part.get("table_ids"):
        return False
    return True


def process_document(doc: dict) -> list[dict]:
    """Rebuild the structure of one document into an ordered list of parts."""
    sections = [s for s in doc.get("sections", []) if not s.get("is_toc")]
    n = len(sections)
    canons = [canon_of(s) for s in sections]
    num_set = set(c for c in canons if c)

    # missing parents: parent prefixes of existing children that exist nowhere
    missing_parents = set()
    for c in num_set:
        if len(c) >= 2:
            missing_parents.add(c[:-1])
    missing_parents -= num_set

    # gaps: consecutive siblings a, b (all-int, same depth, same prefix, a+1 < b)
    gaps: dict[int, set] = {}
    last_idx = None
    for i, c in enumerate(canons):
        if not c or not all(isinstance(x, int) for x in c):
            last_idx = None
            continue
        if last_idx is not None:
            pc = canons[last_idx]
            if (
                len(pc) == len(c)
                and all(isinstance(x, int) for x in pc)
                and pc[:-1] == c[:-1]
                and c[-1] > pc[-1] + 1
            ):
                # the swallowed heading sits in the PREVIOUS section's body
                g = gaps.setdefault(last_idx, set())
                for k in range(pc[-1] + 1, c[-1]):
                    g.add(pc[:-1] + (k,))
        last_idx = i

    # previous top-level number per section (depth-1 continuation rule)
    prev_top: list = [None] * n
    last_t = None
    for i, c in enumerate(canons):
        prev_top[i] = last_t
        if c and len(c) == 1 and isinstance(c[0], int):
            last_t = c[0]

    parts: list[dict] = []
    for i, s in enumerate(sections):
        sid = s.get("id") or f"S{i + 1}"
        title = (s.get("title") or "").strip()
        body = (s.get("body_md") or title).strip()
        lines = body.split("\n")
        tables = list(s.get("table_ids") or [])

        points = _split_points(lines, title, i, n, canons, missing_parents, gaps, prev_top, num_set)
        _gap_fill_points(lines, points, i, gaps)

        points.sort(key=lambda p: p[0])
        dedup: list[tuple[int, dict]] = []
        last_j = None
        for j, meta in points:
            if j == last_j:
                continue
            dedup.append((j, meta))
            last_j = j
        points = dedup

        if not points:
            parts.append(_new_part(sid, title, canons[i], body, tables, "original"))
            continue

        bounds = [0] + [j for j, _ in points] + [len(lines)]
        parts.append(
            _new_part(
                sid,
                title,
                canons[i],
                "\n".join(lines[bounds[0] : bounds[1]]).strip(),
                tables,
                "split",
            )
        )
        for k, (_, meta) in enumerate(points, start=1):
            nb = "\n".join(lines[bounds[k] : bounds[k + 1]]).strip()
            sp = _new_part(f"{sid}_split_{k}", meta["title"], meta["number"], nb, [], "split")
            sp["original_section_id"] = sid
            parts.append(sp)

    # drop trivial page fragments (unnumbered, page title, tiny body)
    parts = [
        p
        for p in parts
        if p["number"] or not (PAGE_RE.match(fold(p["title"])) and len(p["body_md"]) <= 40)
    ]

    # P.1 renumber: unnumbered part right before the first numbered child P.k
    first_num = next((i for i, p in enumerate(parts) if p["number"]), None)
    if first_num:
        q = parts[first_num]["number"]
        if (
            len(q) >= 2
            and all(isinstance(x, int) for x in q)
            and q[-1] >= 2
            and q[:-1] not in {p["number"] for p in parts if p["number"]}
            and first_num > 1  # not the document's first part (intro blocks)
        ):
            cand = parts[first_num - 1]
            if (
                not cand["number"]
                and not cand["is_container"]
                and _renumber_ok(cand["title"], cand)
            ):
                cand["number"] = q[:-1] + (1,)
                cand["origin"] = "renumbered"

    # leading merge: unnumbered, non-champ parts before the first numbered part
    first_num = next((i for i, p in enumerate(parts) if p["number"]), None)
    if first_num:
        new_parts: list[dict] = []
        lead: dict | None = None
        lead_n = 0
        for k in range(first_num):
            p = parts[k]
            if fold(p["title"]).startswith("CHAMP D"):
                new_parts.append(p)  # champ d'application stays its own part
                lead = None
            elif lead is None:
                lead_n += 1
                lead = _new_part(
                    f"LEAD_{lead_n}" if lead_n > 1 else "LEAD",
                    "PREAMBULE",
                    None,
                    p["body_md"],
                    [],
                    "merged",
                    merged_ids=[p["section_id"]],
                )
                new_parts.append(lead)
            else:
                lead["merged_ids"].append(p["section_id"])
                lead["body_md"] = (lead["body_md"] + "\n\n" + p["body_md"]).strip()
        parts = new_parts + parts[first_num:]

    # doc-level preamble / signature
    preamble_md = (doc.get("preamble_md") or "").strip()
    if preamble_md:
        parts.insert(
            0,
            _new_part(
                "PREAMBULE",
                "PREAMBULE",
                None,
                preamble_md,
                list(doc.get("preamble_table_ids") or []),
                "preamble",
            ),
        )
    signature_md = (doc.get("signature_md") or "").strip()
    if signature_md:
        signature_table_ids = [
            t.get("id") for t in (doc.get("tables") or []) if t.get("section") == "signature"
        ]
        parts.append(
            _new_part(
                "SIGNATURE",
                "SIGNATURE",
                None,
                signature_md,
                signature_table_ids,
                "signature",
            )
        )

    # tree build: single stack over the document order
    stack: list[dict] = []
    anchor: dict | None = None
    for p in parts:
        if p["origin"] in ("preamble", "signature"):
            p["parent"] = None
            p["level"] = 1
            continue
        num = p["number"]
        if num:
            if p["is_container"] or len(num) == 1:
                p["parent"] = None
                p["level"] = 1
                anchor = p
                stack = [p]
            else:
                while stack and not is_prefix(stack[-1]["number"], num):
                    stack.pop()
                p["parent"] = stack[-1] if stack else anchor
                p["level"] = len(num)
                stack.append(p)
        else:
            p["parent"] = anchor
            p["level"] = 1 if anchor is None else anchor["level"] + 1

    for order, p in enumerate(parts):
        p["section_order"] = order
        anc: list[str] = []
        cur = p["parent"]
        while cur is not None and len(anc) < 2:
            anc.append(cur["title"])
            cur = cur["parent"]
        p["path"] = " > ".join(list(reversed(anc)) + [p["title"]])

    return parts
