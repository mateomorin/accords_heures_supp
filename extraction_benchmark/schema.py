"""
Structured output used by the benchmark only.

Same shape as `structured_outputs.majoration` minus `section_ids`: the flat arms get
chunks without `[section_id]` headers, so asking for provenance would make the task
impossible for them and the instruction meaningless for the comparison. Everything
else (nature, populations, thresholds, paliers, verbatim citation) is shared with
production.
"""

from pydantic import BaseModel, Field
from structured_outputs.majoration import NatureMention, PalierMajoration


class MentionMajorationBench(BaseModel):
    """One overtime majoration mention, without a section reference."""

    nature: NatureMention = Field(
        description="Nature de la mention : règle fixée par l'accord, simple rappel d'un "
        "autre texte, ou nature indécidable."
    )
    population_concernee: str | None = Field(
        default=None,
        description="Population ciblée (ex: 'Tous les salariés', 'Cadres en heures', "
        "'Ouvriers', 'Personnel roulant', ...).",
    )
    seuil_declenchement: int | float | None = Field(
        default=None,
        description="Durée de travail au-delà de laquelle les heures accomplies sont "
        "considérées comme des heures supplémentaires, en heures (par exemple, 35). "
        "Correspond généralement à la durée de travail propre à l'entreprise.",
    )
    paliers: list[PalierMajoration] = Field(
        default_factory=list,
        description="Liste ordonnée des paliers de majoration applicables pour cette "
        "population/cette mention. Peut être vide si la mention est un rappel sans aucun "
        "taux chiffré.",
    )
    citation_exacte: str = Field(
        description="Extrait exact du texte source (copié mot à mot) qui justifie la mention."
    )
    commentaire: str | None = Field(
        default=None,
        description="Précision utile : ambiguïté du texte, renvoi à un autre texte, réserve "
        "d'interprétation. Laisser null si la mention est claire.",
    )


class ExtractionMajorationsBench(BaseModel):
    """All overtime majoration mentions found in the provided context."""

    mentions: list[MentionMajorationBench] = Field(
        default_factory=list,
        description="Toutes les mentions de majoration des heures supplémentaires trouvées "
        "dans le texte, y compris les simples rappels sans taux chiffré. Liste vide si le "
        "texte n'en contient aucune.",
    )
