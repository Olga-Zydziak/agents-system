from __future__ import annotations
import json, hashlib
from datetime import datetime
from typing import Any, Dict, Optional
from google.cloud import storage  # pip install google-cloud-storage


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    import hashlib as _h

    return _h.sha256(b).hexdigest()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _record_uid(rec: Dict[str, Any]) -> str:
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    return _md5(raw)


def _summary(text: str, n: int) -> str:
    t = (text or "").strip()
    return t[:n] + ("…" if len(t) > n else "")


def _plan_outline(plan: Dict[str, Any], limit: int = 8) -> list[str]:
    nodes = (plan or {}).get("nodes") or []
    names = [str(n.get("name") or n.get("implementation") or "").strip() for n in nodes]
    return [x for x in names if x][:limit]


def _counts(plan: Dict[str, Any]) -> Dict[str, int]:
    return {
        "node_count": len((plan or {}).get("nodes") or []),
        "edge_count": len((plan or {}).get("edges") or []),
    }


def _collect_keywords(rec: Dict[str, Any], plan_outline: list[str]) -> str:
    kws = set()
    mt = rec.get("mission_type")
    if mt:
        kws.add(str(mt))
    for t in rec.get("tags", []):
        kws.add(str(t))
    for n in plan_outline:
        if n:
            kws.add(n.lower())
    return ", ".join(sorted(kws))


def _upload_json(
    client_gcs: storage.Client, bucket: str, obj_path: str, data: Any
) -> Dict[str, Any]:
    b = client_gcs.bucket(bucket)
    blob = b.blob(obj_path)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    blob.upload_from_string(payload, content_type="application/json; charset=utf-8")
    return {
        "uri": f"gs://{bucket}/{obj_path}",
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def persist_missions_to_vertex_memory_semantic(
    *,
    json_path: str,
    engine_name: str,  # już istniejący engine (resource_name)
    client_vertex,  # vertexai.Client
    gcs_bucket: str,
    gcs_prefix: str = "agent-memory",
) -> None:
    """
    Zapisuje KRÓTKIE indeksy (≤~2 KB) do Vertex Memory + pełne treści do GCS.
    Scope per misja + record_uid + view.
    """
    # 1) Wczytaj źródło
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("full_mission_records", [])

    # 2) Klient GCS
    gcs = storage.Client()

    # 3) Iteruj po rekordach misji
    for rec in records:
        mission_id = rec.get("memory_id") or rec.get("mission_id") or _record_uid(rec)
        ruid = _record_uid(rec)
        prompt = (rec.get("mission_prompt") or "").strip()
        mtype = rec.get("mission_type")
        tags = rec.get("tags", [])
        plan = rec.get("final_plan") or {}
        critic = rec.get("critic") or rec.get("critic_report") or {}
        transcript = rec.get("full_transcript") or rec.get("debate_transcript") or []
        agg_reason = rec.get("aggregator_reasoning")
        contribs = rec.get("proposer_contributions")
        final_score = rec.get("final_score")

        base = f"{gcs_prefix}/{mission_id}/{ruid}"

        # 3a) Upload pełnych artefaktów + manifest
        pointers: Dict[str, Dict[str, Any]] = {}
        if plan:
            pointers["final_plan"] = _upload_json(
                gcs, gcs_bucket, f"{base}/final_plan.json", plan
            )
            pointers["final_plan"].update(_counts(plan))
        if critic:
            pointers["critic"] = _upload_json(
                gcs, gcs_bucket, f"{base}/critic.json", critic
            )
        if agg_reason or contribs:
            pointers["aggregator"] = _upload_json(
                gcs,
                gcs_bucket,
                f"{base}/aggregator.json",
                {
                    "aggregator_reasoning": agg_reason,
                    "proposer_contributions": contribs,
                },
            )
        if transcript:
            pointers["transcript"] = _upload_json(
                gcs, gcs_bucket, f"{base}/transcript.json", transcript
            )

        manifest = {
            "mission_id": mission_id,
            "record_uid": ruid,
            "created_at": _now(),
            "artifacts": pointers,
            "metrics": {"final_score": final_score, **(_counts(plan) if plan else {})},
            "tags": tags,
            "plan_outline": _plan_outline(plan, 12),
        }
        manifest_ptr = _upload_json(gcs, gcs_bucket, f"{base}/manifest.json", manifest)
        pointers["manifest"] = manifest_ptr  # dodaj do pointerów

        # 3b) Zbuduj indeksy semantyczne (≤2k znaków) i zapisz do Vertex Memory
        def _write(kind: str, payload: Dict[str, Any], view: str):
            fact = {
                "kind": kind,
                "mission_id": mission_id,
                "record_uid": ruid,
                "imported_at": _now(),
                **payload,
                "pointers": pointers,  # zawiera manifest i artefakty
            }
            fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
            if len(fact_json) > 1900:
                # minimalny „cięcie” – skróć najdłuższe pola
                for key in ("summary", "title", "preview", "keywords"):
                    if key in fact:
                        fact[key] = _summary(
                            str(fact[key]), 600 if key == "summary" else 140
                        )
                fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))

            client_vertex.agent_engines.create_memory(
                name=engine_name,
                fact=fact_json,
                scope={
                    "mission_id": mission_id,
                    "record_uid": ruid,
                    "view": view,
                    "source": "learned_strategies_json",
                },
            )

        outline = manifest["plan_outline"]
        keywords = _collect_keywords(rec, outline)

        # overview
        _write(
            "mission_overview",
            {
                "title": _summary(prompt.split("\n", 1)[0], 140)
                or f"Mission {mission_id}",
                "summary": _summary(prompt, 700),
                "keywords": _summary(keywords, 300),
                "metrics": manifest["metrics"],
                "preview": f"type={mtype}, tags={tags}",
            },
            view="overview",
        )
        # plan index
        if plan:
            _write(
                "final_plan_index",
                {
                    "title": "Plan index",
                    "summary": f"Nodes/Edges: {manifest['metrics'].get('node_count',0)}/{manifest['metrics'].get('edge_count',0)}",
                    "plan_outline": outline[:12],
                    "keywords": _summary(", ".join(outline), 300),
                    "metrics": manifest["metrics"],
                },
                view="plan",
            )
            # (opcjonalnie) atomy węzłów – TOP-6
            nodes = (plan.get("nodes") or [])[:6]
            for n in nodes:
                _write(
                    "plan_node_index",
                    {
                        "node_name": n.get("name") or n.get("implementation"),
                        "node_role": n.get("role"),
                        "summary": _summary(json.dumps(n, ensure_ascii=False), 600),
                        "keywords": _summary(
                            (n.get("name") or "")
                            + ", "
                            + (n.get("implementation") or ""),
                            200,
                        ),
                        "metrics": {"importance": n.get("importance", 0)},
                    },
                    view="plan-node",
                )

        if critic:
            _write(
                "critic_report_index",
                {
                    "verdict": critic.get("verdict") or critic.get("decision"),
                    "score": critic.get("score"),
                    "weaknesses_topk": (
                        critic.get("weaknesses")
                        or critic.get("identified_weaknesses")
                        or []
                    )[:5],
                    "summary": _summary(str(critic), 700),
                },
                view="critic",
            )

        if agg_reason or contribs:
            _write(
                "aggregator_summary_index",
                {
                    "key_reasons": (
                        _summary(str(agg_reason), 500) if agg_reason else ""
                    ),
                    "contributors_topk": (
                        list((contribs or {}).keys())[:5]
                        if isinstance(contribs, dict)
                        else []
                    ),
                    "summary": "Aggregator reasoning + proposer contributions",
                },
                view="aggregator",
            )

        if transcript:
            _write(
                "debate_transcript_index",
                {
                    "summary": f"Debate transcript (~{len(transcript)} msgs)",
                    "preview": "",
                },
                view="transcript",
            )

    print("✅ Semantic indices saved to Vertex Memory; full artifacts saved to GCS.")
