"""ChromaDB-backed semantic ride search."""

import json
import os

import chromadb

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
_COLLECTION_NAME = "disney_rides_info"

try:
    _client = chromadb.PersistentClient(path=os.path.abspath(_CHROMA_PATH))
    _collection = _client.get_collection(_COLLECTION_NAME)
except Exception as _e:
    _client = None
    _collection = None
    print(f"[rag] WARNING: ChromaDB collection could not be loaded: {_e}")


def search_rides_semantic(query: str, top_k: int = 5) -> str:
    """Search 400+ Disney ride descriptions by meaning using vector similarity.

    Use this to find rides matching preferences like:
      - 'family-friendly rides for a 5-year-old'
      - 'thrilling roller coasters for adults'
      - 'water ride', 'dark ride', 'no height requirement'
      - 'soaring or flying experience'

    Do NOT use for live wait times — use get_current_wait_natural_language for that.

    Args:
        query: Natural language description of the kind of ride the user wants.
        top_k: Max results to return (default 5, max 10).

    Returns:
        JSON with matching rides: name, park, intensity, min height, and a description snippet.
    """
    if _collection is None:
        return json.dumps({
            "status": "error",
            "message": "Ride search database unavailable. ChromaDB collection failed to load.",
            "results": [],
        })

    try:
        results = _collection.query(query_texts=[query], n_results=min(top_k, 10))
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {e}",
            "results": [],
        })

    rides = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        rides.append({
            "name": meta.get("name", "unknown"),
            "park": meta.get("park", "unknown"),
            "intensity": meta.get("intensity", "unknown"),
            "min_height_inches": meta.get("min_height", 0),
            "relevance": round(1 - dist, 3),
            "description_snippet": doc[:500] + "..." if len(doc) > 500 else doc,
        })

    return json.dumps({
        "status": "ok",
        "query": query,
        "results": rides,
    })
