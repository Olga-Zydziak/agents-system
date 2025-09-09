"""
Utilities for persisting mission records and their corresponding debate artefacts
into a Vertex AI Agent Engine memory bank.

This module contains a single helper function, ``persist_missions_to_vertex_memory``,
which expects a path to a JSON file containing missions (in the same shape
as produced by your multi‑agent workflow) and writes several structured
memories for each mission into a specified Agent Engine.  Unlike earlier
implementations that attempted to write directly through the high‑level
``vertexai.agent_engines`` module, this version uses a ``vertexai.Client``
instance to perform the memory writes.  The Vertex SDK requires that
``create_memory`` be invoked via ``client.agent_engines.create_memory(...)``
when interacting with an existing engine by name.

Example usage::

    from vertexai import agent_engines
    from tools.persist_missions_memory import persist_missions_to_vertex_memory

    # create or fetch your Agent Engine up front
    engine = agent_engines.create(display_name="my-engine")

    # initialise a Vertex client (project and location can be omitted if
    # configured via environment variables)
    client = vertexai.Client(project="my-project", location="us-central1")

    # persist the missions into memory
    persist_missions_to_vertex_memory(
        json_path="/path/to/learned_strategies.json",
        engine_name=engine.resource_name,
        client=client
    )

Each mission record will produce several memory entries: an overview,
the final plan (graph), a summary of aggregator contributions, a critic
report and optionally the full debate transcript split into chunks.  The
``scope`` of each memory contains identifiers like ``mission_id`` and
``mission_type`` so that downstream components can query the memory bank
precisely.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Callable

try:
    import vertexai  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "vertexai package is required for persist_missions_to_vertex_memory"
    ) from e

__all__ = ["persist_missions_to_vertex_memory"]


def _md5(text: str) -> str:
    """Return an MD5 hex digest for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _now() -> str:
    """Return current UTC time in ISO8601 format with a 'Z' suffix."""
    return datetime.utcnow().isoformat() + "Z"


def _merge_dicts(
    base: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Helper to merge two dictionaries.  ``extra`` entries override those in
    ``base``.  None values are skipped.
    """
    result = dict(base)
    if extra:
        for k, v in extra.items():
            if v is not None:
                result[k] = v
    return result


def persist_missions_to_vertex_memory(
    json_path: str,
    *,
    engine_name: str,
    client: "vertexai.Client",
    include_transcript: bool = True,
    max_transcript_chunk_chars: int = 15000,
    make_scope: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    ] = None,
) -> None:
    """
    Persist missions and their associated debate artefacts into a Vertex AI Agent
    Engine memory bank.

    Parameters
    ----------
    json_path: str
        Path to the JSON file containing mission records.  Each record should
        resemble the entries under ``full_mission_records`` in your data.

    engine_name: str
        The fully qualified resource name of the Agent Engine (e.g.
        ``projects/123/locations/us-central1/reasoningEngines/456``).  The
        engine must already exist; this function does not create or look up
        the engine.

    client: vertexai.Client
        An initialised Vertex AI client.  Creation of memories must be
        performed via ``client.agent_engines.create_memory(...)``.

    include_transcript: bool, default True
        Whether to write the full debate transcript (``full_transcript``) into
        memory.  When ``True``, the transcript will be stored in one or
        multiple memory records.  To satisfy the Vertex AI limitation that
        ``fact`` strings must be shorter than ~2k characters, transcripts are
        automatically split into chunks of approximately 2040 characters.

    max_transcript_chunk_chars: int, default 15000
        **Deprecated.** Previously controlled the size of transcript chunks,
        but is now ignored.  All memory entries are automatically split to
        satisfy the Vertex AI fact size limit of ~2k characters.

    make_scope: Callable, optional
        Optional callback to build a scope dictionary given the base mission
        scope and a view-specific override.  If not provided, a simple
        merge of dictionaries is used.  This can be used to inject custom
        metadata or enforce specific scoping rules.

    Notes
    -----
    Each mission produces the following memory entries:

    1. ``mission_overview``: contains the prompt, mission type, tags and timestamps.
    2. ``final_plan``: stores the ``final_plan`` object with nodes and edges and
       the mission's final score.
    3. ``aggregator_summary``: summarises the aggregator's reasoning and
       proposer contributions, if present.
    4. ``critic_report``: stores the critic's verdict, score and any
       weaknesses.
    5. ``debate_transcript``: optional, contains the full list of messages
       exchanged during the debate.  For long transcripts a ``content_json_chunk``
       field is used to store raw JSON segments.

    All entries are written using the Vertex SDK's ``create_memory`` method on
    ``client.agent_engines``, which attaches the memory to the specified engine
    and associates a scope.  The scope includes identifiers like ``mission_id``
    and ``mission_type`` so that consumers can later retrieve only the
    relevant memories.
    """
    # Validate inputs early
    if not engine_name:
        raise ValueError("engine_name must be provided and non-empty")
    if client is None:
        raise ValueError(
            "A vertexai.Client instance must be provided via the 'client' parameter"
        )

    # Limit for the fact field.  The Vertex AI Agent Engine requires the fact
    # string to be under 2KiB (roughly 2048 characters).  We leave
    # a comfortable margin so that after adding JSON overhead (field names,
    # mission_id, imported_at etc.) the final string stays within the limit.
    MAX_FACT_LENGTH = 1500

    # Load the JSON file and compute a corpus signature to group related entries
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON at {json_path}: {exc}") from exc

    corpus_sig = _md5(json.dumps(data, ensure_ascii=False, sort_keys=True))

    # Use provided or default scope builder
    build_scope = (
        make_scope
        if make_scope is not None
        else lambda base, extra: _merge_dicts(base, extra)
    )

    def _create_memory_fact(
        *,
        kind: str,
        fact_dict: Dict[str, Any],
        view: str,
        replic_fields: Optional[tuple[str, ...]] = None,
    ) -> None:
        """
        Construct a JSON fact from ``fact_dict`` (ensuring ``mission_id``,
        ``kind`` and ``imported_at`` are present) and persist it to Vertex AI
        memory.  If the resulting JSON string is small, it is stored directly.
        Otherwise the payload is split into chunks small enough that each
        serialized memory entry, including metadata, does not exceed 2048
        characters in UTF‑8 encoding.  Each chunk entry has a
        ``content_json_chunk`` field along with ``chunk_index`` and ``chunk_total``.
        Selected fields listed in ``replic_fields`` are copied onto every
        chunk to make retrieval easier.
        """
        # Compose the complete payload once (includes the entire fact_dict)
        payload = dict(fact_dict)
        payload["kind"] = kind
        payload["mission_id"] = mission_id
        payload["imported_at"] = _now()
        payload_str = json.dumps(payload, ensure_ascii=False)
        # Quick path: small payload
        if len(payload_str.encode("utf-8")) < 2000:
            # Under limit: write directly
            client.agent_engines.create_memory(
                name=engine_name,
                fact=payload_str,
                scope=build_scope(base_scope, {"view": view}),
            )
            return
        # Otherwise, split the payload into parts that will fit into the 2 KiB limit
        # Precompute replic field values (small data)
        replic_values: Dict[str, Any] = {}
        if replic_fields:
            for field in replic_fields:
                if field in fact_dict:
                    replic_values[field] = fact_dict[field]
        # Build the list of chunk strings (content_json_chunk) with dynamic sizing
        parts: list[str] = []
        # We will reduce the candidate content size until the entire serialized
        # memory entry fits within the 2 KiB limit.  Initial candidate is
        # MAX_FACT_LENGTH, which is conservatively small.
        start_idx = 0
        total_len = len(payload_str)
        while start_idx < total_len:
            # Start with the maximum allowed raw content size
            candidate_len = min(MAX_FACT_LENGTH, total_len - start_idx)
            # Dynamically shrink the candidate until the resulting JSON entry is within limit
            while candidate_len > 0:
                part_content = payload_str[start_idx : start_idx + candidate_len]
                test_fact: Dict[str, Any] = {
                    "kind": f"{kind}_part",
                    "mission_id": mission_id,
                    "chunk_index": 0,  # placeholder; true index set later
                    "chunk_total": 0,  # placeholder; true total set later
                    "imported_at": _now(),
                    "content_json_chunk": part_content,
                }
                # copy replic field values to test_fact to measure overhead
                test_fact.update(replic_values)
                test_json = json.dumps(test_fact, ensure_ascii=False).encode("utf-8")
                if len(test_json) < 2040:
                    # Accept this candidate size
                    parts.append(part_content)
                    start_idx += candidate_len
                    break
                # Too large: shrink the candidate and try again
                candidate_len -= 50
            else:
                # If no candidate size worked, raise an error
                raise ValueError(
                    f"Unable to fit payload part into memory fact for mission {mission_id}; content may be too large."
                )
        # Now persist each part with accurate chunk_index and chunk_total
        total_parts = len(parts)
        for idx, part in enumerate(parts):
            part_fact: Dict[str, Any] = {
                "kind": f"{kind}_part",
                "mission_id": mission_id,
                "chunk_index": idx,
                "chunk_total": total_parts,
                "imported_at": _now(),
                "content_json_chunk": part,
            }
            part_fact.update(replic_values)
            client.agent_engines.create_memory(
                name=engine_name,
                fact=json.dumps(part_fact, ensure_ascii=False),
                scope=build_scope(base_scope, {"view": view}),
            )

    missions = data.get("full_mission_records", [])
    for rec in missions:
        # Derive mission identifiers; fall back to a digest of the record if missing
        mission_id = (
            rec.get("memory_id")
            or rec.get("mission_id")
            or _md5(json.dumps(rec, ensure_ascii=False, sort_keys=True))
        )
        mission_prompt = rec.get("mission_prompt")
        mission_type = rec.get("mission_type")
        final_plan = rec.get("final_plan")
        final_score = rec.get("final_score")
        tags = rec.get("tags", [])
        timestamp = rec.get("timestamp")
        aggregator_reasoning = rec.get("aggregator_reasoning")
        proposer_contributions = rec.get("proposer_contributions")
        critic = rec.get("critic", {}) or {}
        verdict = critic.get("verdict")
        critic_score = critic.get("score")
        weaknesses = critic.get("weaknesses", [])
        transcript = rec.get("full_transcript", [])

        # Base scope for all memories of this mission
        base_scope = {
            "corpus_signature": corpus_sig,
            "mission_id": mission_id,
            "mission_type": mission_type,
            "source": "learned_strategies_json",
        }

        # 1. Mission overview
        overview_fact = {
            "mission_prompt": mission_prompt,
            "mission_type": mission_type,
            "tags": tags,
            "timestamp": timestamp,
        }
        _create_memory_fact(
            kind="mission_overview", fact_dict=overview_fact, view="overview"
        )

        # 2. Final plan (if present)
        if final_plan:
            plan_fact = {
                "plan": final_plan,
                "final_score": final_score,
            }
            _create_memory_fact(
                kind="final_plan",
                fact_dict=plan_fact,
                view="plan",
                replic_fields=("final_score",),
            )

        # 3. Aggregator summary (if present)
        if aggregator_reasoning or proposer_contributions:
            agg_fact = {
                "aggregator_reasoning": aggregator_reasoning,
                "proposer_contributions": proposer_contributions,
            }
            _create_memory_fact(
                kind="aggregator_summary",
                fact_dict=agg_fact,
                view="aggregator",
            )

        # 4. Critic report (if any critic info exists)
        if verdict or critic_score is not None or weaknesses:
            critic_fact = {
                "verdict": verdict,
                "score": critic_score,
                "weaknesses": weaknesses,
            }
            _create_memory_fact(
                kind="critic_report",
                fact_dict=critic_fact,
                view="critic",
                replic_fields=("verdict", "score", "weaknesses"),
            )

        # 5. Debate transcript (optional)
        if include_transcript and transcript:
            trans_fact = {
                "content": transcript,
            }
            _create_memory_fact(
                kind="debate_transcript",
                fact_dict=trans_fact,
                view="transcript",
            )

    # End of for loop
    # Explicitly return None for clarity
    return None
