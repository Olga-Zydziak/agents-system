"""
Helper functions for reconstructing mission memory entries from a Vertex AI Agent
Engine Memory Bank.

When using the `persist_missions_to_vertex_memory` helper to store mission
records, long entries (e.g. final plans, aggregator reasoning or
transcripts) are split into multiple memory records, each with a
``content_json_chunk`` field, a ``chunk_index`` and a ``chunk_total``.  This
module provides a utility to fetch and reassemble these parts into a single
Python object for downstream processing.

Example usage::

    from vertexai import Client
    from tools.retrieve_mission_memory import retrieve_mission_memory

    client = Client(project=PROJECT_ID, location=LOCATION)
    mission_data = retrieve_mission_memory(
        engine_name=AGENT_ENGINE_NAME,
        mission_id="abcdefg-1234",
        client=client
    )
    # mission_data is a dict keyed by kind (e.g. 'final_plan', 'mission_overview')
    full_plan = mission_data["final_plan"]["content"]
    # ...

"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Iterable

try:
    from vertexai import Client  # type: ignore
except ImportError:
    Client = Any  # pragma: no cover

__all__ = ["retrieve_mission_memory"]


def _assemble_chunks(items: Iterable[Dict[str, Any]]) -> Any:
    """
    Given an iterable of memory entries (dictionaries) that represent parts
    of a single logical fact, concatenate the ``content_json_chunk`` fields in
    order of ``chunk_index``, parse the resulting JSON and return the
    reconstructed Python object.  If the concatenated string cannot be
    deserialised, return the raw string.
    """
    items_sorted = sorted(items, key=lambda d: d.get("chunk_index", 0))
    combined_str = "".join(d.get("content_json_chunk", "") for d in items_sorted)
    try:
        return json.loads(combined_str)
    except json.JSONDecodeError:
        return combined_str


def retrieve_mission_memory(
    *,
    engine_name: str,
    mission_id: str,
    client: Client,
    view: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve and reassemble all memory records for a given mission.

    Parameters
    ----------
    engine_name: str
        Fully qualified name of the Agent Engine (e.g. ``projects/.../locations/.../reasoningEngines/...``).

    mission_id: str
        The identifier of the mission whose memories should be retrieved.

    client: vertexai.Client
        An initialised Vertex AI client.  Used to call ``retrieve_memories``.

    view: str, optional
        If provided, restricts retrieval to memories with this ``view`` scope
        (e.g. "overview", "plan", "aggregator", "critic", "transcript").

    Returns
    -------
    Dict[str, Any]
        A dictionary keyed by ``kind`` where each value is a reconstructed
        memory object.  For kinds that were split into multiple parts, the
        returned value is a dict with ``content`` holding the full Python
        object and any replicated metadata (e.g. ``final_score``, ``verdict``).
        For kinds that were stored in a single record, the original fact
        dictionary is returned.

    Notes
    -----
    This helper assumes that long facts are stored using the convention
    ``{kind}_part`` with ``chunk_index`` and ``chunk_total`` keys, and that
    the payload is serialised as JSON in the ``content_json_chunk`` field.
    """
    if not engine_name:
        raise ValueError("engine_name must be provided")
    if not mission_id:
        raise ValueError("mission_id must be provided")
    if client is None:
        raise ValueError("client must be provided")

    scope: Dict[str, Any] = {"mission_id": mission_id}
    if view:
        scope["view"] = view

    # Retrieve all memories matching the mission and optional view
    memories_iter = client.agent_engines.retrieve_memories(
        name=engine_name,
        scope=scope,
    )
    memories: List[Any] = list(memories_iter)

    # Group by base kind (strip off '_part' suffix if present)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in memories:
        try:
            fact_dict = json.loads(m.memory.fact)
        except Exception:
            # fall back to raw string in case of unexpected format
            fact_dict = {"kind": "unknown", "content": m.memory.fact}
        kind = str(fact_dict.get("kind", ""))
        base_kind = kind[:-5] if kind.endswith("_part") else kind
        grouped.setdefault(base_kind, []).append(fact_dict)

    # Reconstruct each kind
    result: Dict[str, Any] = {}
    for kind_key, items in grouped.items():
        # Determine if this kind was split
        if any("content_json_chunk" in it for it in items):
            # Assemble the content
            full_content = _assemble_chunks(items)
            # Collect replicated metadata (first occurrence wins)
            metadata: Dict[str, Any] = {
                key: it[key]
                for it in items
                for key in ("final_score", "verdict", "score", "weaknesses", "tags", "timestamp", "mission_type")
                if key in it and key not in result.get(kind_key, {})
            }
            assembled = {"kind": kind_key, "mission_id": mission_id, "content": full_content}
            assembled.update(metadata)
            result[kind_key] = assembled
        else:
            # Single record: return as-is
            # Use the first entry (in case multiple matches exist)
            result[kind_key] = items[0]
    return result