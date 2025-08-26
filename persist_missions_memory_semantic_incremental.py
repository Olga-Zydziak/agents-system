from __future__ import annotations
import json, hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Callable
from google.cloud import storage  # pip install google-cloud-storage

def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _record_uid(rec: Dict[str, Any]) -> str:
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    return _md5(raw)

def _plan_outline(plan: Dict[str, Any], limit: int = 12) -> list[str]:
    names = [str(n.get("name") or n.get("implementation") or "").strip()
             for n in (plan or {}).get("nodes") or []]
    return [x for x in names if x][:limit]

def _counts(plan: Dict[str, Any]) -> Dict[str, int]:
    return {"node_count": len((plan or {}).get("nodes") or []),
            "edge_count": len((plan or {}).get("edges") or [])}

def _summary(text: str, n: int) -> str:
    t = (text or "").strip()
    return t[:n] + ("…" if len(t) > n else "")

def _collect_keywords(rec: Dict[str, Any], outline: list[str]) -> str:
    kws = set()
    mt = rec.get("mission_type")
    if mt: kws.add(str(mt))
    for t in rec.get("tags", []): kws.add(str(t))
    for n in outline:
        if n: kws.add(n.lower())
    return ", ".join(sorted(kws))

def _upload_json(gcs: storage.Client, bucket: str, obj_path: str, data: Any) -> Dict[str, Any]:
    b = gcs.bucket(bucket)
    blob = b.blob(obj_path)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob.upload_from_string(payload, content_type="application/json; charset=utf-8")
    import hashlib as _h
    return {"uri": f"gs://{bucket}/{obj_path}",
            "sha256": _h.sha256(payload).hexdigest(), "bytes": len(payload)}

def _exists_in_memory(*, client_vertex, engine_name: str, mission_id: str, record_uid: str) -> bool:
    """Czy istnieje już overview dla (mission_id, record_uid) — jeśli tak, pomijamy."""
    it = client_vertex.agent_engines.retrieve_memories(
        name=engine_name,
        scope={"mission_id": mission_id, "record_uid": record_uid, "view": "overview"},
    )
    for _ in it:
        return True
    return False

def _write_fact_short(*, client_vertex, engine_name: str, fact: Dict[str, Any], scope: Dict[str, Any]) -> None:
    """Pilnuje limitu ~2k znaków przez skracanie długich pól."""
    def shrink(d: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("summary", "title", "preview", "keywords"):
            if k in d and isinstance(d[k], str):
                d[k] = _summary(d[k], 600 if k == "summary" else 140)
        return d
    fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
    if len(fact_json) > 1900:
        fact = shrink(fact)
        fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
    client_vertex.agent_engines.create_memory(name=engine_name, fact=fact_json, scope=scope)

def persist_missions_to_vertex_memory_semantic_incremental(
    *,
    json_path: str,
    engine_name: str,               # resource_name istniejącego Agent Engine
    client_vertex,                  # vertexai.Client
    gcs_bucket: str,
    gcs_prefix: str = "agent-memory",
    # FILTRY:
    mission_id_filter: Optional[Iterable[str] | str | Callable[[Dict[str, Any]], bool]] = None,
    record_uid_filter: Optional[Iterable[str] | str] = None,
    # OPCJE:
    write_plan_nodes_topk: int = 6,  # ile atomów węzłowych indeksować (0 = wyłącz)
) -> Dict[str, int]:
    """
    Przyrostowy import z learned_strategies.json:
     - jeśli podasz mission_id_filter/record_uid_filter → zapisze tylko wskazane,
     - jeśli dany (mission_id, record_uid) już istnieje w Memory → pominie (idempotencja),
     - do Vertex Memory zapisuje KRÓTKIE indeksy (+ pointery), pełne treści trafiają do GCS.

    Zwraca: {'processed': N, 'skipped_existing': X, 'published_new': Y}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get("full_mission_records", [])
    gcs = storage.Client()

    # Ujednolicenie filtrów
    def _as_set(x):
        if x is None: return None
        if isinstance(x, str): return {x}
        return set(x)

    record_uid_set = _as_set(record_uid_filter)

    def _pass_mission(rec: Dict[str, Any]) -> bool:
        if mission_id_filter is None:
            return True
        mid = rec.get("memory_id") or rec.get("mission_id") or _record_uid(rec)
        if callable(mission_id_filter):
            return bool(mission_id_filter(rec))
        if isinstance(mission_id_filter, str):
            return mission_id_filter == mid
        return mid in set(mission_id_filter)

    stats = {"processed": 0, "skipped_existing": 0, "published_new": 0}

    for rec in recs:
        if not _pass_mission(rec):
            continue

        mission_id = rec.get("memory_id") or rec.get("mission_id") or _record_uid(rec)
        ruid = _record_uid(rec)

        if record_uid_set is not None and ruid not in record_uid_set:
            continue

        stats["processed"] += 1

        # Idempotencja
        if _exists_in_memory(client_vertex=client_vertex, engine_name=engine_name,
                             mission_id=mission_id, record_uid=ruid):
            stats["skipped_existing"] += 1
            continue

        prompt = (rec.get("mission_prompt") or "").strip()
        mtype = rec.get("mission_type")
        tags = rec.get("tags", [])
        plan = rec.get("final_plan") or {}
        critic = rec.get("critic") or rec.get("critic_report") or {}
        transcript = rec.get("full_transcript") or rec.get("debate_transcript") or []
        agg_reason = rec.get("aggregator_reasoning")
        contribs = rec.get("proposer_contributions")
        final_score = rec.get("final_score")

        # Upload do GCS
        base = f"{gcs_prefix}/{mission_id}/{ruid}"
        pointers: Dict[str, Dict[str, Any]] = {}
        if plan:
            pointers["final_plan"] = _upload_json(gcs, gcs_bucket, f"{base}/final_plan.json", plan)
            pointers["final_plan"].update(_counts(plan))
        if critic:
            pointers["critic"] = _upload_json(gcs, gcs_bucket, f"{base}/critic.json", critic)
        if agg_reason or contribs:
            pointers["aggregator"] = _upload_json(
                gcs, gcs_bucket, f"{base}/aggregator.json",
                {"aggregator_reasoning": agg_reason, "proposer_contributions": contribs},
            )
        if transcript:
            pointers["transcript"] = _upload_json(gcs, gcs_bucket, f"{base}/transcript.json", transcript)

        # Manifest
        manifest = {
            "mission_id": mission_id,
            "record_uid": ruid,
            "created_at": _now(),
            "artifacts": pointers,
            "metrics": {"final_score": final_score, **(_counts(plan) if plan else {})},
            "tags": tags,
            "plan_outline": _plan_outline(plan),
        }
        manifest_ptr = _upload_json(gcs, gcs_bucket, f"{base}/manifest.json", manifest)
        pointers["manifest"] = manifest_ptr

        # Scope
        base_scope = {"mission_id": mission_id, "record_uid": ruid, "source": "learned_strategies_json"}

        # Indeksy (krótkie)
        outline = manifest["plan_outline"]
        keywords = _collect_keywords(rec, outline)

        # overview
        _write_fact_short(
            client_vertex=client_vertex, engine_name=engine_name,
            fact={
                "kind": "mission_overview",
                "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                "title": _summary(prompt.split("\n", 1)[0], 140) or f"Mission {mission_id}",
                "summary": _summary(prompt, 700),
                "keywords": _summary(keywords, 300),
                "metrics": manifest["metrics"],
                "preview": f"type={mtype}, tags={tags}",
                "pointers": pointers,
            },
            scope={**base_scope, "view": "overview"},
        )

        # plan index
        if plan:
            _write_fact_short(
                client_vertex=client_vertex, engine_name=engine_name,
                fact={
                    "kind": "final_plan_index",
                    "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                    "title": "Plan index",
                    "summary": f"Nodes/Edges: {manifest['metrics'].get('node_count',0)}/{manifest['metrics'].get('edge_count',0)}",
                    "plan_outline": outline[:12],
                    "keywords": _summary(", ".join(outline), 300),
                    "metrics": manifest["metrics"],
                    "pointers": pointers,
                },
                scope={**base_scope, "view": "plan"},
            )
            if write_plan_nodes_topk > 0:
                for n in (plan.get("nodes") or [])[:write_plan_nodes_topk]:
                    _write_fact_short(
                        client_vertex=client_vertex, engine_name=engine_name,
                        fact={
                            "kind": "plan_node_index",
                            "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                            "node_name": n.get("name") or n.get("implementation"),
                            "node_role": n.get("role"),
                            "summary": _summary(json.dumps(n, ensure_ascii=False), 600),
                            "keywords": _summary((n.get("name") or "") + ", " + (n.get("implementation") or ""), 200),
                            "metrics": {"importance": n.get("importance", 0)},
                            "pointers": pointers,
                        },
                        scope={**base_scope, "view": "plan-node"},
                    )

        # critic
        if critic:
            _write_fact_short(
                client_vertex=client_vertex, engine_name=engine_name,
                fact={
                    "kind": "critic_report_index",
                    "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                    "verdict": critic.get("verdict") or critic.get("decision"),
                    "score": critic.get("score"),
                    "weaknesses_topk": (critic.get("weaknesses") or critic.get("identified_weaknesses") or [])[:5],
                    "summary": _summary(str(critic), 700),
                    "pointers": pointers,
                },
                scope={**base_scope, "view": "critic"},
            )

        # aggregator
        if agg_reason or contribs:
            _write_fact_short(
                client_vertex=client_vertex, engine_name=engine_name,
                fact={
                    "kind": "aggregator_summary_index",
                    "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                    "key_reasons": (_summary(str(agg_reason), 500) if agg_reason else ""),
                    "contributors_topk": list((contribs or {}).keys())[:5] if isinstance(contribs, dict) else [],
                    "summary": "Aggregator reasoning + proposer contributions",
                    "pointers": pointers,
                },
                scope={**base_scope, "view": "aggregator"},
            )

        # transcript (tylko indeks)
        if transcript:
            _write_fact_short(
                client_vertex=client_vertex, engine_name=engine_name,
                fact={
                    "kind": "debate_transcript_index",
                    "mission_id": mission_id, "record_uid": ruid, "imported_at": _now(),
                    "summary": f"Debate transcript (~{len(transcript)} msgs)",
                    "pointers": pointers,
                },
                scope={**base_scope, "view": "transcript"},
            )

        stats["published_new"] += 1

    return stats