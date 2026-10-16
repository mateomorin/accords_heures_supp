from enum import Enum

from pydantic import BaseModel, Field


class TypeMajoration(str, Enum):
    POURCENTAGE = "pourcentage"  # Ex: 25 for 25%
    COEFFICIENT = "coefficient"  # Ex: 1.25 for 125%
    FORFAIT_EURO = "forfait_euro"  # Ex: 250€


class PalierMajoration(BaseModel):
    rang_debut: int | None = Field(
        default=None,
        description="Heure de début du palier (ex: 36 pour la 36e heure, soit la 1re heure sup). Null si non précisé.",
    )
    rang_fin: int | None = Field(
        default=None,
        description="Heure de fin du palier inclus (ex: 43 pour la 43e heure). Null si le palier s'applique sans limite supérieure.",
    )
    volume_heures: int | None = Field(
        default=None,
        description="Alternative au rang : nombre d'heures concernées par ce palier (ex: 8 pour 'les 8 premières heures sup').",
    )
    taux_valeur: float = Field(
        description="Valeur numérique de la majoration. Ex: 25.0 pour 25%, 1.25 pour un coefficient 1,25, ou 10.0 pour 10€/h.",
    )
    type_majoration: TypeMajoration = Field(
        default=TypeMajoration.POURCENTAGE,
        description="Unité de la valeur numérique renseignée.",
    )
    condition_particuliere: str | None = Field(
        default=None,
        description="Condition spécifique au palier (ex: 'Uniquement pour le travail du dimanche', 'De nuit', 'Si dépassement du contingent').",
    )


class RegleMajoration(BaseModel):
    categorie_ou_population: str | None = Field(
        default=None,
        description="Population ciblée (ex: 'Tous les salariés', 'Cadres en heures', 'Ouvriers', 'Temps partiel').",
    )
    duree_hebdomadaire_de_travail: int | None = Field(
        default=None,
        description="Durée hebdomadaire de travail fixée par l'accord, en heures (par exemple, 35)"
    )
    paliers: list[PalierMajoration] = Field(
        default_factory=list,
        description="Liste ordonnée des paliers de majoration applicables pour cette population/règle.",
    )
    remplacement_repos: bool | None = Field(
        default=None,
        description="True si la majoration peut être remplacée par un repos compensateur équivalent (total ou partiel).",
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
