# Accords Heures Supp'

This project extracts structured information about **overtime (heures supplémentaires)** from French company agreements (accords) published on [LegiFrance](https://www.legifrance.gouv.fr/) — e.g. the overtime majoration rate, the contingent, its evolution over time — using a retrieval-augmented (RAG) pipeline backed by an LLM.

The pipeline runs end to end over the S3 corpus:

```
final documents (S3)  →  preprocessing (parts)  →  embedding / indexing (Qdrant)
                        →  extraction (RAG + LLM)  →  structured results (S3)
```

## Repository structure

```
.
├── src/
│   ├── preprocessing/            # Deterministic preprocessing: raw docs → ordered parts
│   │   └── README.md             #   (structure, tables, numbering)
│   ├── rag/                      # Embedding & chunking of the parts into Qdrant
│   │   └── EMBED_CHUNKS.md
│   └── extraction/               # RAG-based structured extraction of overtime data
│       └── EXTRACTION.md
├── extraction_benchmark/         # Benchmark of context-construction strategies
│   └── README.md                 #   (protocol, results, findings)
├── notebooks/                    # Exploration, annotation and visualisation notebooks
├── tests/                        # Unit tests
├── .env.example                  # Template of the environment variables
├── AGENTS.md                     # Working constraints for AI agents
└── pyproject.toml                # Project metadata, dependencies and tooling config
```

## Requirements

- **Python 3.13**, managed with **uv** (not `pip`).
- External services, all configured through environment variables:
  - **S3 / MinIO** — document and shard storage.
  - **Qdrant** — vector database for the parts and chunk embeddings.
  - An **OpenAI-compatible** endpoint for the LLM and a separate one for embeddings.
  - **MLflow** — tracing (or **Langfuse** if prefered).
- The Python package manager, linter and test runner are:

```bash
uv sync                 # install dependencies
uv run ruff check       # lint
uv run pytest           # tests
```

### Configuration

Copy `.env.example` to `.env` and fill in the values. The variables group around four services: Qdrant, the LLM, the embedding model, and the tracing backends(Langfuse, MLflow).

## Main parts

Each stage is self-contained under `src/` and documents how to run it. The extraction stage reads the output of the two stages before it, so run them in order.

| Stage | Purpose | Documentation |
| ----- | ------- | ------------- |
| **Preprocessing** | Converts the final document shards into structured, ordered parts (sections, tables, numbering). Fully deterministic — no LLM. | [src/preprocessing/README.md](src/preprocessing/README.md) |
| **RAG indexing** | Embeds the parts (and their chunks) into the Qdrant collections used for retrieval. | [src/rag/EMBED_CHUNKS.md](src/rag/EMBED_CHUNKS.md) |
| **Extraction** | Retrieves the relevant context for an agreement and asks the LLM to extract the overtime fields into a structured schema. | [src/extraction/EXTRACTION.md](src/extraction/EXTRACTION.md) |
| **Benchmark** | Compares the context-construction strategies (full reading, structured, flat) on sampled documents, measuring coverage, token length and cost. | [extraction_benchmark/README.md](extraction_benchmark/README.md) |

The `notebooks/` folder holds exploratory and evaluation notebooks against sample data.

## Remark on Vibe Coding

This repository was originaly created and maintained by hand, then I switched to vibe coding (deepseek-v4.1-flash) for RAG refactoring. Benchmark is a full becnhmark made by AI agents (mistral-medium), not so good to be honest, looking at the report is the best way to take a glance at a previous step of the refactoring.