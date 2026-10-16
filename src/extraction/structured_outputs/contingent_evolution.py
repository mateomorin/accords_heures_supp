from enum import Enum
from pydantic import BaseModel, Field


class TypeSubstitution(str, Enum):
    REMPLACEMENT_BRANCHE = "remplacement_branche"
    REMPLACEMENT_ACCORD_ANTERIEUR = "remplacement_accord_anterieur"
    REMPLACEMENT_CONTINGENT_LEGAL = "remplacement_contingent_legal"
    AUCUN = "aucun"


class DetailContingent(BaseModel):
    categorie_ou_population: str | None = Field(
        default=None,
        description="Population concernée si le contingent est spécifique (ex: 'Non-cadres', 'Salariés en modulation', 'Tous les salariés').",
    )
    type_substitution: TypeSubstitution = Field(
        default=TypeSubstitution.AUCUN,
        description="Type de substitution identifié dans l'accord.",
    )
    nombre_heures_avant_accord: int | None = Field(
        default=None,
        description=(
            "Nombre d'heures du contingent précédent s'il est EXPLICITEMENT chiffré dans le texte "
            "(ex: 220). Mettre None si le texte dit juste qu'il remplace la branche/loi sans donner le chiffre."
        ),
    )
    nombre_heures_apres_accord: int | None = Field(
        default=None,
        description="Nombre d'heures du nouveau contingent fixé par l'accord (ex: 180). Mettre None si non chiffré.",
    )
    periode: str | None = Field(
        default="annuel",
        description="Période de référence du contingent (ex: 'annuel', 'mensuel', 'par exercice').",
    )


class ExtractionContingentsEvolutions(BaseModel):
    presence_contingent_conventionnel: bool = Field(
        description=(
            "True uniquement si le document fixe un contingent d'heures supplémentaires "
            "propre à l'entreprise. False si le document est muet ou ne fait que citer le contingent légal de référence."
        )
    )
    presence_substitution: bool = Field(
        default=False,
        description=(
            "True uniquement si le document mentionne un contingent précédent "
            "(issu d'un accord d'entreprise antérieur, d'une convention de branche "
            "ou du contingent légal) ET indique explicitement que ce contingent "
            "est remplacé, révisé, annulé, ou jugé inadapté par le présent accord "
            "(ex: 'annule et remplace', 'se substitue à', 'ce contingent se révèle inadapté', "
            "'en dérogation au contingent légal de 220h'). "
            "False si aucun contingent antérieur/légal n'est mentionné comme étant remplacé."
        ),
    )
    contingents: list[DetailContingent] = Field(
        default_factory=list,
        description="Liste des contingents d'heures supplémentaires définis ou modifiés par l'accord.",
    )
    citation_exacte: str | None = Field(
        default=None,
        description="Extrait exact du texte traitant du contingent d'heures supplémentaires et de sa substitution. Obligatoire si un contingent est trouvé.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques éventuelles (ex: 'L'accord cite la branche mais ne donne pas son ancien chiffre').",
    )
