from __future__ import annotations

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState


def uncertainty_router(state: MASState, config: MASConfig) -> dict[str, str]:
    """Determine the extraction path based on retrieval uncertainty.
    
    ABCI insight: focus compute budget on edges with highest posterior entropy.
    VS cosine similarity is the proxy for extraction certainty.
    """
    examples = state.get("few_shot_examples", [])

    # Cold start or insufficient examples
    if not examples or len(examples) < config.router.min_vs_results:
        return {"extraction_path": "careful"}

    top_similarity = examples[0].get("similarity_score", 0.0)

    if top_similarity >= config.router.high_similarity_threshold:
        return {"extraction_path": "fast"}
    elif top_similarity < config.router.low_similarity_threshold:
        return {"extraction_path": "careful"}

    return {"extraction_path": "standard"}
