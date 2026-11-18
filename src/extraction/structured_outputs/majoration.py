from enum import Enum

from pydantic import BaseModel, Field


class NatureMention(str, Enum):
    REGLE_CONVENTIONNELLE = "regle_conventionnelle"  # The agreement sets the rule itself
    RAPPEL = "rappel"  # The text only recalls another text (law, CCN, prior agreement)
    INDETERMINEE = "indeterminee"  # The text does not allow deciding


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
    taux_valeur: float | None = Field(
        default=None,
        description="Valeur numérique de la majoration. Ex: 25.0 pour 25%, 1.25 pour un coefficient 1,25, ou 10.0 pour 10€/h. "
                    "Null si le palier est évoqué sans qu'aucun taux ne soit chiffré dans le texte.",
    )
    type_majoration: TypeMajoration | None = Field(
        default=None,
        description="Unité de la valeur numérique renseignée. Null si aucun taux n'est chiffré.",
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


class MentionMajoration(BaseModel):
    nature: NatureMention = Field(
        description="Nature de la mention : règle fixée par l'accord, simple rappel d'un autre texte, "
                    "ou nature indécidable."
    )
    population_concernee: str | None = Field(
        default=None,
        description="Population ciblée (ex: 'Tous les salariés', 'Cadres en heures', 'Ouvriers', 'Personnel roulant', ...).",
    )
    seuil_declenchement: int | float | None = Field(
        default=None,
        description="Durée de travail au-delà de laquelle les heures accomplies sont considérées comme des heures supplémentaires, en heures (par exemple, 35). "
                    "Correspond généralement à la durée de travail propre à l'entreprise."
    )
    paliers: list[PalierMajoration] = Field(
        default_factory=list,
        description="Liste ordonnée des paliers de majoration applicables pour cette population/cette mention. "
                    "Peut être vide si la mention est un rappel sans aucun taux chiffré.",
    )
    section_ids: list[str] = Field(
        default_factory=list,
        description="Identifiants des sections du contexte d'où provient la mention, recopiés depuis "
                    "les crochets des en-têtes (ex: ['S12']). Ne jamais inventer d'identifiant.",
    )
    citation_exacte: str = Field(
        description="Extrait exact du texte source (copié mot à mot) qui justifie la mention."
    )
    commentaire: str | None = Field(
        default=None,
        description="Précision utile : ambiguïté du texte, renvoi à un autre texte, réserve "
                    "d'interprétation. Laisser null si la mention est claire.",
    )


class ExtractionMajorations(BaseModel):
    mentions: list[MentionMajoration] = Field(
        default_factory=list,
        description="Toutes les mentions de majoration des heures supplémentaires trouvées dans le "
                    "texte, y compris les simples rappels sans taux chiffré. Liste vide si le texte "
                    "n'en contient aucune.",
    )
