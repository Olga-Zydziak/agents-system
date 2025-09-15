# memory_helpers.py

from datetime import datetime
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage


# ---------- Generatory i serializacja ----------

def gen_mission_id() -> str:
    """Generuje unikalny identyfikator misji."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"mission_{timestamp}_{short}"


def safe_json_dumps(obj: Any) -> str:
    """Bezpieczne serializowanie do JSON; w razie błędów serializuje repr."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)


def count_nodes_edges(plan: Dict) -> Tuple[int, int]:
    """Zlicza liczbę węzłów i krawędzi w finalnym planie."""
    nodes = plan.get("nodes", []) if isinstance(plan, dict) else []
    edges = plan.get("edges", []) if isinstance(plan, dict) else []
    return (len(nodes), len(edges))


def infer_flags_and_tags(plan: Dict, transcript: List[str]) -> Tuple[Dict[str, bool], List[str]]:
    """Wykrywa flagi (retry, rollback, optimize) i generuje listę tagów."""
    joined = " ".join(str(x).lower() for x in transcript)[:50000]
    flags = {
        "has_retry": any(k in joined for k in ["retry", "ponów", "ponowienie", "backoff"]),
        "has_rollback": "rollback" in joined,
        "has_optimization": any(k in joined for k in ["optimiz", "optymaliz"]),
    }
    tags: List[str] = []
    if flags["has_retry"]:
        tags.append("retry")
    if flags["has_rollback"]:
        tags.append("rollback")
    if flags["has_optimization"]:
        tags.append("optimize")
    return flags, tags


# ---------- Główna funkcja zapisu ----------

def save_mission_to_gcs(
    bucket_name: str,
    base_prefix: str,
    mission: str,
    final_plan: Dict,
    all_messages: List[Dict],
    orchestrator_state: Dict,
    approved: bool = True,
    final_score: Optional[float] = None
) -> str:
    
    
    print(">>> save_mission_to_gcs CALLED")
    print(">>> module:", __name__)
    print(">>> file:  ", __file__)
    logging.warning("save_mission_to_gcs CALLED from %s", __file__)
    """
    Zapisuje plan, transkrypt i metadane do Google Cloud Storage w strukturze:
      gs://{bucket}/{base_prefix}/{mission_id}/plan.json
      gs://{bucket}/{base_prefix}/{mission_id}/transcript.jsonl
      gs://{bucket}/{base_prefix}/{mission_id}/metadata.json

    Zwraca mission_id.
    """
    # 1. Id i ścieżki
    mission_id = gen_mission_id()
    base_path = f"{base_prefix}/{mission_id}"
    plan_path = f"{base_path}/plan.json"
    transcript_path = f"{base_path}/transcript.jsonl"
    meta_path = f"{base_path}/metadata.json"

    plan_uri = f"gs://{bucket_name}/{plan_path}"
    transcript_uri = f"gs://{bucket_name}/{transcript_path}"
    meta_uri = f"gs://{bucket_name}/{meta_path}"

    # 2. Przygotuj dane
    nodes_count, edges_count = count_nodes_edges(final_plan)

    # Zamień transkrypt na JSONL (po jednej linii na wiadomość)
    transcript_lines: List[str] = []
    transcript_texts: List[str] = []
    for m in all_messages:
        mm = dict(m)
        c = mm.get("content")
        transcript_texts.append(c if isinstance(c, str) else safe_json_dumps(c))
        # Serializacja do JSONL – konwertuj content na string jeśli to np. dict
        if not isinstance(c, (str, dict)):
            mm["content"] = str(c)
        transcript_lines.append(safe_json_dumps(mm))

    flags, tags = infer_flags_and_tags(final_plan, transcript_texts)

    metadata = {
        "mission_id": mission_id,
        "mission_prompt": mission,
        "approved": approved,
        "final_score": float(final_score) if final_score is not None else None,
        "nodes_count": nodes_count,
        "edges_count": edges_count,
        "has_optimization": flags["has_optimization"],
        "has_rollback": flags["has_rollback"],
        "has_retry": flags["has_retry"],
        "tags": tags,
        "orchestrator_state": orchestrator_state or {},
        "timestamp": datetime.now().isoformat(),
        "links": {
            "plan_uri": plan_uri,
            "transcript_uri": transcript_uri,
            "metadata_uri": meta_uri,
        },
        "preview": {
            "entry_point": (final_plan or {}).get("entry_point"),
        },
    }

    # 3. Zapis do GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    bucket.blob(plan_path).upload_from_string(
        safe_json_dumps(final_plan), content_type="application/json; charset=utf-8"
    )
    bucket.blob(transcript_path).upload_from_string(
        "\n".join(transcript_lines), content_type="application/x-ndjson; charset=utf-8"
    )
    bucket.blob(meta_path).upload_from_string(
        safe_json_dumps(metadata), content_type="application/json; charset=utf-8"
    )
    
    
    
    #zapisywanie indeksow
    ts_dt = datetime.now(timezone.utc)
    ts_file = ts_dt.strftime("%Y%m%d_%H%M%S")
    ts_iso  = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # preview.txt (content.uri)
    preview_txt = (
        f"mission_id: {mission_id}\n"
        f"timestamp:  {ts_iso}\n"
        f"approved:   {bool(approved)}\n"
        f"final_score:{final_score if final_score is not None else 'null'}\n"
        f"tags:       {', '.join(tags) if tags else ''}\n"
    )
    preview_path = f"{base_path}/preview.txt"
    bucket.blob(preview_path).upload_from_string(
        preview_txt, content_type="text/plain; charset=utf-8"
    )
    preview_uri = f"gs://{bucket_name}/{preview_path}"

    # dokument NDJSON (1 linia)
    ndjson_doc = {
        "id": f"{ts_file}_{(mission_id[-8:] if len(mission_id) >= 8 else mission_id)}",
        "structData": {
            "mission_id": mission_id,
            "timestamp": ts_iso,
            "mission_type": "general",
            "tags": tags,
            "outcome": "Success" if approved else "Partial" if final_score else "Failure",
            "final_score": float(final_score) if final_score is not None else None,
            "approved": bool(approved),
            "nodes_count": int(nodes_count),
            "edges_count": int(edges_count),
            "has_retry": bool(flags.get("has_retry")) if isinstance(flags, dict) else False,
            "has_rollback": bool(flags.get("has_rollback")) if isinstance(flags, dict) else False,
            "has_optimization": bool(flags.get("has_optimization")) if isinstance(flags, dict) else False,
            "lang": "pl",
            "display_id": f"{ts_file}-{mission_id}",
            "links": {
                "txt_uri": preview_uri,
                "plan_uri": plan_uri,
                "transcript_uri": transcript_uri,
                "metrics_uri": meta_uri,
                "metadata_uri": meta_uri,
            },
        },
        "content": {
            "mimeType": "text/plain",
            "uri": preview_uri,
        },
    }

    # zapis NDJSON pod {base_prefix}/index/
    index_dir  = f"{base_prefix}/index"
    index_path = f"{index_dir}/metadata_{ts_file}.ndjson"
    bucket.blob(index_path).upload_from_string(
        json.dumps(ndjson_doc, ensure_ascii=False) + "\n",
        content_type="application/x-ndjson; charset=utf-8",
    )

    # twardy log z pełnym URI
    ndjson_uri = f"gs://{bucket_name}/{index_path}"
    print("NDJSON ->", ndjson_uri)
    logging.warning("NDJSON wrote to %s", ndjson_uri)

    # mały kanarek, żeby łatwo złapać prefiks (ten sam katalog co NDJSON)
    canary_path = f"{index_dir}/_canary_{ts_file}.txt"
    bucket.blob(canary_path).upload_from_string(
        f"ok {ts_iso} mission_id={mission_id}",
        content_type="text/plain; charset=utf-8",
    )
    print("CANARY ->", f"gs://{bucket_name}/{canary_path}")
    
    #koniec zapisu indeksow

    logging.getLogger(__name__).info(f"[MEMORY:GCS] Saved mission {mission_id} at {plan_uri}")
    return mission_id
