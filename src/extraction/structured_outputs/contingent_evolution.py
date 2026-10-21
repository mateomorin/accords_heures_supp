from enum import Enum
from pydantic import BaseModel, Field


class TypeSubstitution(str, Enum):
    REMPLACEMENT_CONVENTION_COLLECTIVE = "remplacement_convention_collective"
    REMPLACEMENT_ACCORD_ANTERIEUR = "remplacement_accord_anterieur"
    REMPLACEMENT_CONTINGENT_LEGAL = "remplacement_contingent_legal"
    FIXATION_DIRECTE = "fixation_directe"


class DetailContingent(BaseModel):
    categorie_juridique: str | None = Field(
        default="FIXATION DIRECTE",
        description="Manière dont le contingent est annoncé : en remplacement d'un précédent, en fixation directe, en simple rappel de textes existants.",
    )
    nombre_heures_fixe: int = Field(
        description="Nombre d'heures du nouveau contingent fixé par l'accord (ex: 180, 220)."
    )
    type_substitution: TypeSubstitution = Field(
        default=TypeSubstitution.FIXATION_DIRECTE,
        description=(
            "Type de lien avec le contingent précédent identifié dans le texte :\n"
            "- 'remplacement_convention_collective' si l'accord remplace expressément la convention collective ou de branche.\n"
            "- 'remplacement_contingent_legal' si l'accord remplace le contingent légal.\n"
            "- 'remplacement_accord_anterieur' si l'accord remplace un précédent accord d'entreprise.\n"
            "- 'fixation_directe' si le texte se contente de fixer un contingent à XXX heures sans mentionner de texte remplacé."
        ),
    )
    population_concernee: str | None = Field(
        default="Tous les salariés",
        description="Population concernée (ex: 'Non-cadres', 'Salariés en modulation', 'Tous les salariés').",
    )
    nombre_heures_avant_accord: int | None = Field(
        default=None,
        description=(
            "Nombre d'heures du contingent précédent s'il est EXPLICITEMENT chiffré dans le texte "
            "(ex: 'remplace l'ancien contingent de 220h'). Laisser None si le texte ne donne pas le chiffre précédent."
        ),
    )
    periode: str = Field(
        default="annuel",
        description="Période de référence du contingent (ex: 'annuel', 'mensuel', 'par exercice').",
    )


class ExtractionContingentsEvolutions(BaseModel):
    presence_contingent_conventionnel: bool = Field(
        description=(
            "True uniquement si le document fixe ou modifie un contingent d'heures supplémentaires "
            "propre à l'entreprise. False si le document est muet ou se contente de rappeler la loi sans fixer de règles."
        )
    )
    contingents: list[DetailContingent] = Field(
        default_factory=list,
        description="Liste des contingents définis ou modifiés par l'accord (généralement 1, ou plusieurs si distinctions par catégorie).",
    )
    citation_exacte: str | None = Field(
        default=None,
        description="Extrait exact du texte traitant du contingent d'heures supplémentaires et de sa substitution. Obligatoire si un contingent est trouvé.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques éventuelles ou subtilités textuelles (ex: 'L'accord cite la branche mais ne donne pas son ancien chiffre').",
    )
