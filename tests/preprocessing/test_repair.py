"""Golden-document tests for the deterministic structure repair."""

import json

from repair import process_document
from structure import doc_to_row


def _section(sid, title, body=None, number=None, **kw):
    section = {
        "id": sid,
        "level": kw.pop("level", 1),
        "number": number,
        "title": title,
        "body_md": body if body is not None else title,
        "table_ids": kw.pop("table_ids", []),
        "parent": kw.pop("parent", None),
        "is_toc": False,
    }
    section.update(kw)
    return section


def _doc(sections, **kw):
    doc = {
        "preamble_md": kw.pop("preamble_md", ""),
        "signature_md": kw.pop("signature_md", ""),
        "tables": kw.pop("tables", []),
        "preamble_table_ids": kw.pop("preamble_table_ids", []),
        "sections": sections,
    }
    doc.update(kw)
    return doc


def test_container_alias_becomes_parent() -> None:
    # "Article V." (roman) is the logical parent of the 5.x sections
    doc = _doc(
        [
            _section("S1", "ARTICLE V. HEURES SUPPLEMENTAIRES"),
            _section(
                "S2", "5.1 Contingent annuel", "5.1 Contingent annuel\n\nBody 5.1", "5.1", level=2
            ),
            _section("S3", "5.2 Majoration", "5.2 Majoration\n\nBody 5.2", "5.2", level=2),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert by_id["S1"]["number"] == (5,)
    assert by_id["S1"]["level"] == 1
    assert by_id["S2"]["parent"] is by_id["S1"]
    assert by_id["S3"]["parent"] is by_id["S1"]
    assert by_id["S2"]["level"] == 2


def test_embedded_container_split() -> None:
    doc = _doc(
        [
            _section(
                "S1",
                "Article 2 : Données",
                "Article 2 : Données\n\nIntro body\n\n"
                "Article 3 : Durée du travail\n\n3.1 - Définition\n\ndef body",
                "2",
            ),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert set(by_id) == {"S1", "S1_split_1"}
    assert by_id["S1"]["origin"] == "split"
    assert "Intro body" in by_id["S1"]["body_md"]
    assert "Article 3" not in by_id["S1"]["body_md"]
    assert by_id["S1_split_1"]["number"] == (3,)
    assert by_id["S1_split_1"]["original_section_id"] == "S1"
    assert by_id["S1_split_1"]["parent"] is None  # a new top-level container


def test_gap_fill_recovers_ocr_dropped_number() -> None:
    # 5.2 is missing: its heading ("– Contingent annuel") sits in 5.1's body
    doc = _doc(
        [
            _section("S1", "Article 5 : Heures supplémentaires", number="5"),
            _section(
                "S2",
                "5.1 Définition",
                "5.1 Définition\n\nText 5.1\n\n– Contingent annuel\n\nText 5.2",
                "5.1",
                level=2,
            ),
            _section("S3", "5.3 Majoration", "5.3 Majoration\n\nText", "5.3", level=2),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert by_id["S2"]["number"] == (5, 1)
    assert "Contingent" not in by_id["S2"]["body_md"]
    assert by_id["S2_split_1"]["number"] == (5, 2)
    assert by_id["S2_split_1"]["title"] == "– Contingent annuel"
    assert by_id["S2_split_1"]["parent"] is by_id["S1"]
    assert by_id["S3"]["parent"] is by_id["S1"]


def test_champ_d_application_split() -> None:
    doc = _doc(
        [
            _section(
                "S1",
                "Article 1 : Objet",
                "Article 1 : Objet\n\nBody\n\nCHAMP D'APPLICATION\n\nTous les salariés.",
                "1",
            ),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert by_id["S1_split_1"]["title"] == "Champ d'application"
    assert by_id["S1_split_1"]["number"] is None
    assert "Tous les salariés." in by_id["S1_split_1"]["body_md"]
    assert by_id["S1_split_1"]["parent"] is by_id["S1"]


def test_p1_renumber() -> None:
    # unnumbered short title right before 2.2 (2.1 absent) becomes 2.1
    doc = _doc(
        [
            _section("S1", "Champ d'application", "Champ d'application\n\nChamp body"),
            _section("S2", "Définition", "Définition\n\nDéfinition body"),
            _section("S3", "2.2 Répartition", "2.2 Répartition\n\nBody", "2.2", level=2),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert by_id["S2"]["number"] == (2, 1)
    assert by_id["S2"]["origin"] == "renumbered"
    assert by_id["S1"]["origin"] == "original"  # champ stays separate
    assert [p["section_id"] for p in parts] == ["S1", "S2", "S3"]


def test_renumber_rejects_determiner_led_title() -> None:
    doc = _doc(
        [
            _section("S1", "Préambule", "Préambule\n\nParties..."),
            _section("S2", "L'organisation du travail", "L'organisation du travail\n\nBody"),
            _section("S3", "2.2 Répartition", "2.2 Répartition\n\nBody", "2.2", level=2),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}
    # the determiner-led title is not renumbered: it stays unnumbered and is
    # folded into the leading preamble along with the other intro parts
    assert "S2" not in by_id
    assert "S2" in by_id["LEAD"]["merged_ids"]
    assert all(p["number"] != (2, 1) for p in parts)


def test_duplicates_are_kept_as_siblings() -> None:
    doc = _doc(
        [
            _section("S1", "Article 3 : Durée", number="3"),
            _section("S2", "3.1 A", "3.1 A\n\nBody A", "3.1", level=2),
            _section("S3", "3.1 A", "3.1 A\n\nBody B", "3.1", level=2),
            _section("S4", "3.2 B", "3.2 B\n\nBody", "3.2", level=2),
        ]
    )

    parts = process_document(doc)
    by_id = {p["section_id"]: p for p in parts}

    assert len(parts) == 4
    assert by_id["S2"]["parent"] is by_id["S1"]
    assert by_id["S3"]["parent"] is by_id["S1"]  # both duplicates under Article 3
    assert by_id["S4"]["parent"] is by_id["S1"]


def test_leading_merge_and_doc_level_parts() -> None:
    doc = _doc(
        [
            _section("S1", "Préambule", "Préambule\n\nParties..."),
            _section("S2", "Objet de l'accord", "Objet de l'accord\n\nBody"),
            _section("S3", "Champ d'application", "Champ d'application\n\n..."),
            _section("S4", "Article 1 : Durée", "Article 1 : Durée\n\nBody", "1"),
        ],
        preamble_md="Préambule doc-level",
        signature_md="Fait à X, le 1er janvier 2020",
        tables=[{"id": "T1", "section": "signature", "md": "| s |"}],
    )

    parts = process_document(doc)
    assert [p["section_id"] for p in parts] == [
        "PREAMBULE",
        "LEAD",
        "S3",
        "S4",
        "SIGNATURE",
    ]
    lead = parts[1]
    assert lead["origin"] == "merged"
    assert lead["merged_ids"] == ["S1", "S2"]
    assert "Parties..." in lead["body_md"]
    assert parts[0]["origin"] == "preamble"
    assert parts[4]["origin"] == "signature"
    assert parts[4]["table_ids"] == ["T1"]
    assert parts[4]["parent"] is None


def test_page_fragment_dropped() -> None:
    doc = _doc(
        [
            _section("S1", "P. 3/5"),
            _section("S2", "Article 1 : Durée", "Article 1 : Durée\n\nBody", "1"),
        ]
    )

    parts = process_document(doc)
    assert [p["section_id"] for p in parts] == ["S2"]


def test_doc_to_row_schema() -> None:
    doc = _doc(
        [
            _section("S1", "ARTICLE V. HEURES SUPPLEMENTAIRES"),
            _section(
                "S2", "5.1 Contingent annuel", "5.1 Contingent annuel\n\nBody 5.1", "5.1", level=2
            ),
        ],
        signature_md="Fait à X",
    )
    metadata = {
        "cid": "C1",
        "reference": "R1",
        "dateSignature": "2017-09-01 00:00:00+00:00",
        "doc_type": "word",
        "format": "docx",
    }

    row = doc_to_row(doc, metadata=metadata)
    parts = json.loads(row["parts_json"])

    assert [p["section_id"] for p in parts] == ["S1", "S2", "SIGNATURE"]
    assert [p["section_order"] for p in parts] == [0, 1, 2]
    assert [p["number"] for p in parts] == ["5", "5.1", None]
    assert parts[1]["parent_id"] == "S1"
    assert parts[1]["path"] == "ARTICLE V. HEURES SUPPLEMENTAIRES > 5.1 Contingent annuel"
    assert parts[1]["origin"] == "original"
    assert parts[2]["origin"] == "signature"
    assert row["anneeSignature"] == 2017
    assert row["doc_n_sections"] == 2
    assert row["n_parts"] == 3
    for part in parts:
        assert part["merged_ids"] == []
        assert part["n_chars"] == len(part["body_md"])
