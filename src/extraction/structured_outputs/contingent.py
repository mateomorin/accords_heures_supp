from pydantic import BaseModel, Field


class DetailContingent(BaseModel):
    categorie_ou_population: str | None = Field(
        default=None,
        description="Population concernée si le contingent est spécifique (ex: 'Non-cadres', 'Salariés en modulation', 'Tous les salariés').",
    )
    nombre_heures: int = Field(
        description="Nombre d'heures du contingent d'heures supplémentaires fixé par l'accord."
    )
    periode: str | None = Field(
        default="annuel",
        description="Période de référence du contingent (ex: 'annuel', 'par exercice').",
    )


class ExtractionContingents(BaseModel):
    presence_contingent_conventionnel: bool = Field(
        description=(
            "True uniquement si le document fixe un contingent d'heures supplémentaires "
            "propre à l'entreprise. False si le document est muet ou ne fait que citer le contingent légal de référence."
        )
    )
    contingents: list[DetailContingent] = Field(
        default_factory=list,
        description="Liste des contingents d'heures supplémentaires définis spécifiquement par l'accord.",
    )
    citation_exacte: str | None = Field(
        default=None,
        description="Extrait exact du texte traitant du contingent d'heures supplémentaires. Obligatoire si un contingent est trouvé.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques éventuelles (ex: rappel du droit commun, contingent variable, proratisation).",
    )
