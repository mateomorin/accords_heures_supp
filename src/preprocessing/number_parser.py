"""
Deterministic parsing of French collective-agreement section numbers.

The parser works on raw title/line text and never calls an LLM. It is
conservative by design: when a token is ambiguous (anonymized party initials,
French thousands separators, ordinal markers), it returns ``None`` rather
than guess a number.
"""

from __future__ import annotations

import re
import unicodedata

#: A part of a section number: an integer or a single lowercase letter.
Part = int | str

#: Strict whitelist of roman numerals accepted as section numbers (I..XX).
#: Wider roman tokens are rejected: "IL" comes from "Il est convenu...",
#: and XX+/XXX/XXXX are often anonymized party names.
ROMAN: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
}

#: Structural keywords that introduce a top-level container heading.
CONT_RE = re.compile(
    r"^(TITRE|CHAPITRE|PARTIE|SECTION|CLASSE|ARTICLE|ART)\s*[.\-–]?\s+"
    r"([IVXLC]{1,4}|\d{1,3}(?:[.\-–]\d{1,3})*)\b"
)
#: Leading arabic number in a title, with optional single-letter sub-part
#: ("1.b Horaires..."). The number must be delimited, not glued to letters.
ARABIC_RE = re.compile(
    r"^(\d{1,3}(?:[.\-–]\d{1,3})*)(?:[.\-–] ?([A-Z])(?![A-Z0-9]))?"
    r"(?=[.\-–:\s]|[^0-9A-Z]|$)"
)
#: Leading roman numeral in a title; the next char must not be a letter so
#: that "Vu l'accord" / "Vingt-cinq" are rejected.
ROMAN_RE = re.compile(r"^([IVXLC]{1,4})(?=[.\-–:\s])")
#: Glued arabic sub-parts after a roman number ("III.1 Durée", "II. 2-1").
SUB_RE = re.compile(r"^[.\-–] ?(\d{1,3}(?:[.\-–]\d{1,3})*)")
#: Leading number in a body line; requires title text after the number.
LINE_NUM_RE = re.compile(
    r"^(\d{1,3}(?:[.\-–]\d{1,3})*)(?:[.\-–] ?([A-Z])(?![A-Z0-9]))?"
    r"(?=[.\-–:\s]|[^0-9A-Z])\s*\S"
)

#: Page-header fragments ("P. 2/5", "PAGE 11").
PAGE_RE = re.compile(r"^(P\.? ?\d+ ?/ ?\d+|PAGE ?\d+)$")
#: "Champ d'application" heading (accent/apostrophe-insensitive).
CHAMP_RE = re.compile(r"^CHAMP D. ?APPLICATION")
#: Lines that reference legal articles, never headings.
LEGAL_RE = re.compile(r"^(ARTICLE|ART\.|CODE DU TRAVAIL|ANNEXE|L \d|R \d|ARTICLES)")


def fold(text: str) -> str:
    """Uppercase and strip accents for pattern matching."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", (text or "").upper())
        if unicodedata.category(c) != "Mn"
    )


def _subs(raw: str) -> tuple[int, ...] | None:
    """Parse glued sub-parts, rejecting thousands-separator components."""
    parts = tuple(int(x) for x in re.split(r"[.\-–]", raw))
    if any(p >= 100 for p in parts):
        return None
    return parts


def _arabic_parts(raw: str) -> tuple[Part, ...] | None:
    parts = tuple(int(x) for x in re.split(r"[.\-–]", raw))
    # "1.607 heures" / "7.146 988 €": a French thousands separator, not a hierarchy.
    if len(parts) >= 2 and any(p >= 100 for p in parts):
        return None
    # "607 heures" / "227 j / 5": a quantity, not a section number.
    if len(parts) == 1 and parts[0] >= 100:
        return None
    return parts


def container_of(title: str) -> tuple[str, tuple[Part, ...]] | None:
    """Return ``(kind, number)`` if the title is a TITRE/CHAPITRE/.../ARTICLE heading."""
    t = fold(title).strip()
    m = CONT_RE.match(t)
    if not m:
        return None
    kind, raw = m.group(1), m.group(2)
    if raw[0].isdigit():
        # Article-style numbers may be large ("Article 142-4"); only the
        # sub-components are guarded against thousands separators.
        parts = tuple(int(x) for x in re.split(r"[.\-–]", raw))
        if any(p >= 100 for p in parts[1:]):
            return None
        return kind, parts
    if raw not in ROMAN:
        return None
    parts: list[Part] = [ROMAN[raw]]
    sm = SUB_RE.match(t[m.end():])
    if sm:
        subs = _subs(sm.group(1))
        if subs is None:
            return None
        parts.extend(subs)
    return kind, tuple(parts)


def title_number(title: str) -> tuple[Part, ...] | None:
    """Extract the leading section number from a title."""
    t = fold(title).strip()
    if not t:
        return None
    c = container_of(t)
    if c:
        return c[1]
    if re.match(r"^\d[^0-9]*°", t):  # ordinal marker (1°, 2°), not a section number
        return None
    m = ARABIC_RE.match(t)
    if m:
        parts = _arabic_parts(m.group(1))
        if parts is None:
            return None
        if m.group(2):
            parts = parts + (m.group(2).lower(),)
        return parts
    m = ROMAN_RE.match(t)
    if m and m.group(1) in ROMAN:
        parts = [ROMAN[m.group(1)]]
        sm = SUB_RE.match(t[m.end() :])
        if sm:
            subs = _subs(sm.group(1))
            if subs is None:
                return None
            parts.extend(subs)
        return tuple(parts)
    return None


def parse_field(raw: str | None) -> tuple[Part, ...] | None:
    """Parse the ``number`` field of a section (fallback source)."""
    if not raw:
        return None
    return title_number(str(raw))


def canon_of(section: dict) -> tuple[Part, ...] | None:
    """
    Return the canonical number of a section.

    The title is the authoritative source; the ``number`` field is the
    fallback. On a prefix conflict the deeper number wins.
    """
    pt = title_number(section.get("title") or "")
    pf = parse_field(section.get("number"))
    if pt and pf and pt != pf:
        if pt == pf[: len(pt)]:
            return pf
        return pt
    return pt or pf


def line_number_parts(line: str) -> tuple[tuple[Part, ...], str] | None:
    """
    Return the number of a body line, as ``(parts, matched number string)``.

    The number must be followed by title text. Roman numerals only count
    when glued to arabic sub-parts ("III.1 Durée"), never bare.
    """
    t = fold(line).strip()
    m = LINE_NUM_RE.match(t)
    if m:
        if re.match(r"^\d[^0-9]*°", t):  # ordinal marker (1°, 2°), not a number
            return None
        numstr = m.group(1)
        parts = _arabic_parts(numstr)
        if parts is None:
            return None
        if m.group(2):
            parts = parts + (m.group(2).lower(),)
        return parts, numstr
    m = ROMAN_RE.match(t)
    if m and m.group(1) in ROMAN:
        sm = SUB_RE.match(t[m.end() :])
        if sm:
            subs = _subs(sm.group(1))
            if subs is not None and re.match(r"\s*\S", t[m.end() + sm.end() :]):
                numstr = m.group(1) + t[m.end() : m.end() + sm.end()]
                return (ROMAN[m.group(1)],) + subs, numstr
    return None


def line_number(line: str) -> tuple[Part, ...] | None:
    """Return the number of a body line, or ``None``."""
    res = line_number_parts(line)
    return res[0] if res else None


def line_container(line: str) -> tuple[str, tuple[Part, ...]] | None:
    """Container heading of a body line, or ``None``."""
    return container_of(line)


def numbered_line_ok(orig: str, folded: str, numstr: str) -> bool:
    """
    Return whether a numbered body line is a real heading.

    Guards against lines that are not headings:

    - truncated deeper number / cross-reference: "3.7Travail", "3- 5- 2- 1",
      "15 .1" (space-injected) — the line's real number is deeper than ``numstr``
    - fraction: "4/5e", "4 / 6"
    - quantity sentence: "6 jours travaillés", "1 jour à partir de..." —
      the char after the number must be a separator or an uppercase letter
      in the ORIGINAL case (``fold`` uppercases everything)
    - digit after the separator: "3 : 80% des objectifs", "20- 3 Mutuelle"
    - formula / address: "=" in the line, 5-digit postal code
    """
    if re.match(re.escape(numstr) + r"\s*[.\-–]\s*\d", folded):
        return False
    if "=" in orig or re.search(r"\b\d{5}\b", orig):
        return False
    after = orig[len(numstr) :]
    if re.match(r"\s*/\s*\d", after):
        return False
    rest = after.lstrip()
    if not rest:
        return False
    c = rest[0]
    if c in "./-–:/":
        nxt = rest[1:].lstrip()
        if nxt and nxt[0].isdigit():
            return False
        return True
    return c.isupper()


def is_container_title(title: str) -> bool:
    """Return whether the title is a top-level anchor (container + single-part number)."""
    c = container_of(title)
    return c is not None and len(c[1]) == 1


def is_prefix(a: tuple[Part, ...], b: tuple[Part, ...]) -> bool:
    """Return whether ``a`` is a strict numeric prefix of ``b``."""
    return len(a) < len(b) and b[: len(a)] == a
