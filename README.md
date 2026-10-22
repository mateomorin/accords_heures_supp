# Accords Heures Supp'

This project aims to extract useful information about overtime at work in French company agreements.

# Structure of the repository

```{bash}
.
├── notebooks/                                  # Notebooks for exploration, visualization, and evaluation
│   ├── sources/                                # Different data sources output by notebooks(samples of documents, images, ...)
│   ├── annotations.ipynb                       # Evaluation with annotations
│   ├── false_negative_themes.ipynb             # Assessing theme correctness from LegiFrance
│   ├── results_rag.ipynb                       # RAG
│   ├── visualisation_evo_contingent.ipynb      # Visualization for an extraction
│   └── visualisation_heures_supp.ipynb         # Visualization for an extraction
├── src/
│   ├── extraction/                             # Extraction from RAG script
│   │   ├── configs/
│   │   ├── prompts/
│   │   ├── structured_outputs/
│   │   ├── argo_extract.yaml
│   │   ├── EXTRACTION.md                       # How to run the script
│   │   ├── main.py                             # Main scrip to run
│   │   ├── registry.py
│   │   └── services.py
│   └── rag                                     # Creating RAG
│       ├── argo_chunk.yaml
│       ├── config.py
│       ├── EMBED_CHUNKS.md                     # How to run the script
│       └── embed_chunks.py                     # Main script to run
├── pyproject.toml
└── uv.lock
```

# Run scripts

See `EXTRACTION.md` and `EMBED_CHUNKS.md` for all the information you need to run scripts.
