from structured_outputs.contingent import ExtractionContingents
from structured_outputs.contingent_evolution import ExtractionContingentsEvolutions
from structured_outputs.majoration import ExtractionMajorations

STRUCTURED_OUTPUT_REGISTRY = {
    "contingent": ExtractionContingents,
    "contingent_evolution": ExtractionContingentsEvolutions,
    "majoration": ExtractionMajorations
}


def get_structured_output(structure_output_name: str):
    if structure_output_name not in STRUCTURED_OUTPUT_REGISTRY:
        raise ValueError(f"Structured output '{structure_output_name}' is not recognized. "
                         f"Choices: {list(STRUCTURED_OUTPUT_REGISTRY.keys())}")
    return STRUCTURED_OUTPUT_REGISTRY[structure_output_name]
