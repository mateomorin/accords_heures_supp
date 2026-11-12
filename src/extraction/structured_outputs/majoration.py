from enum import Enum

from pydantic import BaseModel, Field


class TypeMajoration(str, Enum):
    POURCENTAGE = "pourcentage"  # Ex: 25 for 25%
    COEFFICIENT = "coefficient"  # Ex: 1.25 for 125%
    FORFAIT_EURO = "forfait_euro"  # Ex: 250€


class PeriodeDecompte(str, Enum):
    HEBDOMADAIRE = "hebdomadaire"
    MENSUEL = "mensuel"
    TRIMESTRIEL = "trimestriel"
    ANNUEL = "annuel"


class PalierMajoration(BaseModel):
    periode_decompte: PeriodeDecompte | None = Field(
        default=None,
        description="Période sur laquelle s'effectue le décompte des heures supplémentaires (hebdomadaire, mensuel, ...)"
    )
    rang_debut: int | float | None = Field(
        default=None,
        description="Heure de début du palier (ex: 36 pour la 36e heure, soit la 1re heure sup). Null si non précisé.",
    )
    rang_fin: int | float | None = Field(
        default=None,
        description="Heure de fin du palier inclus (ex: 43 pour la 43e heure). Null si le palier s'applique sans limite supérieure.",
    )
    volume_heures_debut: int | float | None = Field(
        default=None,
        description="Alternative au rang si le décompte se fait en nombre d'heures supplémentaires (ex: 0 pour 'les 8 premières heures sup').",
    )
    volume_heures_fin: int | float | None = Field(
        default=None,
        description="Alternative au rang si le décompte se fait en nombre d'heures supplémentaires (ex: 8 pour 'les 8 premières heures sup').",
    )
    taux_valeur: float = Field(
        description="Valeur numérique de la majoration. Ex: 25.0 pour 25%, 1.25 pour un coefficient 1,25, ou 10.0 pour 10€/h.",
    )
    type_majoration: TypeMajoration = Field(
        default=TypeMajoration.POURCENTAGE,
        description="Unité de la valeur numérique renseignée.",
    )
    application_globale: bool = Field(
        default=False,
        description="True si la majoration s'applique à l'ensemble des heures supplémentaires sans distinction de palier."
    )
    borne_incomplete: bool = Field(
        default=False,
        description="True si le texte évoque un palier dont les limites exactes sont incomplètes, floues ou ambigües."
    )
    condition_particuliere: str | None = Field(
        default=None,
        description="Condition spécifique au palier (ex: 'Uniquement pour le travail du dimanche', 'De nuit', 'Si dépassement du contingent').",
    )


class RegleMajoration(BaseModel):
    population_concernee: str | None = Field(
        default=None,
        description="Population ciblée (ex: 'Tous les salariés', 'Cadres en heures', 'Ouvriers', 'Personnel roulant', ...).",
    )
    seuil_declenchement: int | float | None = Field(
        default=None,
        description="Durée de travail au-delà de laquelle les heures accomplies sont comptées considérées comme des heures supplémentaires, en heures (par exemple, 35). "
                    "Correspond généralement à la durée de travail propre à l'entreprise."
    )
    paliers: list[PalierMajoration] = Field(
        default_factory=list,
        description="Liste ordonnée des paliers de majoration applicables pour cette population/règle.",
    )


class ExtractionMajorations(BaseModel):
    presence_regle_conventionnelle: bool = Field(
        description=(
            "True si le document définit explicitement des taux ou règles de majoration d'heures supplémentaires. "
            "False si le document est muet ou renvoie purement au Code du travail sans précision."
        )
    )
    regles_majorations: list[RegleMajoration] = Field(
        default_factory=list,
        description="Ensemble des règles de majoration identifiées par population ou cas de figure.",
    )
    citations_exactes: list[str] = Field(
        default_factory=list,
        description="Extrait(s) exact(s) du texte source mentionnant les taux de majoration.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques explicatives (ex: renvoi à la convention collective nationale, dispositions temporaires).",
    )
    rappel_majoration: bool = Field(
        default=False,
        description="True si la majoration présentée est un rappel d'un autre texte légal (Code du Travail, accords d'entreprise, conventions collectives, ...) "
                    "ou si un extrait mentionne explicitement un rappel pour la majoration."
    )
