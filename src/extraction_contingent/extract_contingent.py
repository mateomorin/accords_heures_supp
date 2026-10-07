import argparse
import logging
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from qdrant_client.http import models
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).parent


class DetailContingent(BaseModel):
    """
    Ask detailed information about the contigent.
    """
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


class ExtractionContingentHeuresSup(BaseModel):
    """
    Ask context information to avoid hallucinations
    """
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


def create_prompt():
    with open(SCRIPT_DIR / "system_prompt.txt", "r") as f:
        system_prompt = f.read()

    with open(SCRIPT_DIR / "user_prompt.txt", "r") as f:
        user_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    return prompt


def create_query():
    with open(SCRIPT_DIR / "query.txt", "r") as f:
        query = f.read()

    return query


def default_references():
    with open(SCRIPT_DIR / "annotation_references.txt", "r") as f:
        references = f.read().split("\n")
    return references


def filter_documents(filter_key: str, filter_value: str):
    return models.Filter(
        must=[
            models.FieldCondition(
                key=filter_key,
                match=models.MatchValue(value=filter_value)
            )
        ]
    )


def filter_by_reference(reference: str):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.reference",
                match=models.MatchValue(value=reference)
            )
        ]
    )


def extract_data_from_filter(structured_llm, prompt, query, fil):
    retriever = config.vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": fil
        }
    )

    docs = retriever.invoke(query)

    context_text = "\n\n".join([d.page_content for d in docs])

    chain = prompt | structured_llm
    result = chain.invoke({"context": context_text})

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant information from ACCO documents."
    )
    parser.add_argument(
        "--filter_key", type=str, default="metadata.reference",
        help="Name of a column to filter by it during RAG (e.g. cid, reference)"
    )
    parser.add_argument(
        "--references", nargs="+", type=str, default=[],
        help="List of references to filter by. If not provided, all references in the vector store will be used (warning: might be slow)."
    )

    args = parser.parse_args()

    structured_llm = config.llm.with_structured_output(ExtractionContingentHeuresSup)
    prompt = create_prompt()
    query = create_query()

    if not args.references:
        logger.warning("No references provided. Defaulting to annotation_references.txt ...")
        args.references = default_references()

    results = []
    for ref in tqdm(args.references):
        fil = filter_by_reference(ref)
        try:
            res = extract_data_from_filter(structured_llm, prompt, query, fil)
            # Flatten the result for parquet
            row = {
                "reference": ref,
                "presence_contingent_conventionnel": res.presence_contingent_conventionnel,
                "citation_exacte": res.citation_exacte,
                "justification_ou_reserve": res.justification_ou_reserve,
            }
            # Add contingents as a list of dicts (parquet supports this)
            row["contingents"] = [c.model_dump() for c in res.contingents]
            results.append(row)
        except Exception as e:
            logger.info(f"Error processing {ref}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_file = f"{config.S3_OUTPUT_PATH}/extraction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        logger.info(f"Saving results to {output_file}")
        df.to_parquet(output_file, storage_options=config.fs.storage_options)
    else:
        logger.info("No results to save.")


if __name__ == "__main__":
    main()
