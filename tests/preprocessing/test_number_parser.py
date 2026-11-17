"""Tests for the deterministic section-number parser."""

from number_parser import (
    canon_of,
    fold,
    line_container,
    line_number,
    numbered_line_ok,
    title_number,
)


def test_title_number_arabic_dotted() -> None:
    assert title_number("3.1 Définition") == (3, 1)
    assert title_number("12.3.4 Suivi") == (12, 3, 4)


def test_title_number_letter_subpart() -> None:
    assert title_number("1.b Horaires") == (1, "b")
    assert title_number("2-3 Modalités") == (2, 3)


def test_title_number_container_keywords() -> None:
    assert title_number("Article 6 : Durée du travail") == (6,)
    assert title_number("ARTICLE V. HEURES SUPPLEMENTAIRES") == (5,)
    assert title_number("Chapitre III – Organisation") == (3,)
    assert title_number("Article 142-4 Dérogations") == (142, 4)


def test_title_number_glued_roman_subparts() -> None:
    assert title_number("III.1 Durée") == (3, 1)
    assert title_number("II. 2-1 Modalités") == (2, 2, 1)


def test_title_number_rejects_roman_false_positives() -> None:
    # "IL" comes from "Il est convenu..."; XX+ are often anonymized names
    assert title_number("Il est convenu ce qui suit") is None
    assert title_number("Vingt-cinq salariés") is None


def test_title_number_rejects_thousands_separators() -> None:
    assert title_number("1.607 heures annuelles") is None
    assert title_number("7.146 988 € de masse salariale") is None


def test_title_number_rejects_quantities_and_pages() -> None:
    assert title_number("227 j / 5 = 45,4 semaines") is None
    assert title_number("P. 2/5") is None
    assert title_number("PAGE 11") is None
    assert title_number("2°)Monsieur X, délégué") is None


def test_canon_of_title_wins_on_conflict() -> None:
    # title is authoritative: deeper number wins over a shallower field
    assert canon_of({"title": "3.1 Définition", "number": "3"}) == (3, 1)
    # the field completes the title: shallower title, deeper field
    assert canon_of({"title": "3 Durée", "number": "3.1"}) == (3, 1)
    assert canon_of({"title": "Définition", "number": "5.2"}) == (5, 2)
    assert canon_of({"title": "Définition", "number": None}) is None


def test_line_number_and_container() -> None:
    assert line_number("3.2 - Durée annuelle") == (3, 2)
    assert line_number("6 jours travaillés") == (6,)
    assert line_container("Article VI. HEURES SUPPLEMENTAIRES") == ("ARTICLE", (6,))
    assert line_number("ARTICLE VI. HEURES SUPPLEMENTAIRES") is None


def test_numbered_line_ok_guards() -> None:
    # quantity sentence: char after the number must be separator or uppercase
    assert numbered_line_ok("3.2 - Durée annuelle", fold("3.2 - Durée annuelle"), "3.2")
    assert not numbered_line_ok("6 jours travaillés", fold("6 jours travaillés"), "6")
    # truncated deeper number / cross-reference
    assert not numbered_line_ok("3.7Travail en équipes", fold("3.7Travail en équipes"), "3")
    # fraction
    assert not numbered_line_ok("4/5e du contingent", fold("4/5e du contingent"), "4")
    # digit after the separator
    assert not numbered_line_ok("3 : 80% des objectifs", fold("3 : 80% des objectifs"), "3")
    # formula / postal code
    assert not numbered_line_ok("2/ + 60 € = 80 €", fold("2/ + 60 € = 80 €"), "2")
    assert not numbered_line_ok("15 Place de la République CS 70527", fold("15 Place"), "15")
