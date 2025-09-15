# exporter_missions_gcs.py
from __future__ import annotations
import json, os, glob
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from google.cloud import storage
from datetime import datetime, timezone
# Re-use logiki z Twojej biblioteki (identyczne wyliczenia i pola!)
from exporter_missions_lib import (
    _to_str_content,
    _extract_plan,
    _build_txt,
    _build_transcript,
    _build_metrics,
    _ndjson_line,
)

def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"output_root_gcs must start with gs://, got: {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")

def _mission_id_from_snapshot(snap: dict, fallback_path: Path) -> str:
    mid = snap.get("memory_id") or snap.get("mission_id")
    if isinstance(mid, str) and mid:
        return mid
    base = fallback_path.stem
    if base.lower().startswith("mission_"):
        return base
    # ostateczny fallback
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"mission_{stamp}"

def export_local_by_filename_date(
    input_dir: str,
    output_root_gcs: str,
    pattern: str = "*.json",
    skip_existing: bool = True,
) -> List[str]:
    """
    Eksportuje lokalne misje do GCS w formacie zgodnym z exporter_missions_lib:
      missions/<mission_id>/<mission_id>.txt
      missions/<mission_id>/<mission_id>.plan.json
      missions/<mission_id>/<mission_id>.transcript.json
      missions/<mission_id>/<mission_id>.metrics.json
      missions/<mission_id>/<mission_id>.ndjson   (1 linia na misję)

    Zwraca listę GCS URI do *.ndjson (po jednym na misję).
    """
    input_dir_p = Path(input_dir).resolve()
    files = sorted(
        glob.glob(os.path.join(str(input_dir_p), pattern)),
        key=lambda p: os.path.basename(p).split("_")[1:3]  # sort: YYYYMMDD, HHMMSS
    )

    bucket_name, root_prefix = _parse_gs_uri(output_root_gcs)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    ndjson_uris: List[str] = []

    for p in files:
        src = Path(p)
        try:
            snap = json.loads(src.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[EXPORT][SKIP] {src.name}: cannot load JSON ({e})")
            continue

        mission_id = _mission_id_from_snapshot(snap, src)
        plan = _extract_plan(snap)
        txt = _build_txt(snap, plan)
        transcript = _build_transcript(snap)
        metrics = _build_metrics(snap, plan)

        # Ścieżki w GCS – jak w Twojej bibliotece (katalog per misja)
        base_prefix = f"{root_prefix}/{mission_id}".strip("/")
        txt_path     = f"{base_prefix}/{mission_id}.txt"
        plan_path    = f"{base_prefix}/{mission_id}.plan.json"
        trans_path   = f"{base_prefix}/{mission_id}.transcript.json"
        metrics_path = f"{base_prefix}/{mission_id}.metrics.json"
        ndjson_path  = f"{base_prefix}/{mission_id}.ndjson"

        txt_uri     = f"gs://{bucket_name}/{txt_path}"
        plan_uri    = f"gs://{bucket_name}/{plan_path}"
        trans_uri   = f"gs://{bucket_name}/{trans_path}"
        metrics_uri = f"gs://{bucket_name}/{metrics_path}"
        ndjson_uri  = f"gs://{bucket_name}/{ndjson_path}"

        
        if skip_existing:
            
            metrics_blob = bucket.blob(metrics_path)
            ndjson_blob  = bucket.blob(ndjson_path)
            if metrics_blob.exists(client) or ndjson_blob.exists(client):
                print(f"[EXPORT][SKIP] {mission_id} już istnieje w GCS (metrics/ndjson).")
                continue
        
        
        # 1) Upload artefaktów
        bucket.blob(txt_path).upload_from_string(txt, content_type="text/plain; charset=utf-8")
        bucket.blob(plan_path).upload_from_string(
            json.dumps(plan, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )
        bucket.blob(trans_path).upload_from_string(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )
        bucket.blob(metrics_path).upload_from_string(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )

        # 2) Zbuduj linię NDJSON (identyczna z Twoją funkcją _ndjson_line)
        ndjson_line = _ndjson_line(
            mission_id=mission_id,
            txt_uri=txt_uri,
            plan_uri=plan_uri,
            transcript_uri=trans_uri,
            metrics_uri=metrics_uri,
            metrics=metrics,
        )

        # 3) Upload pojedynczego pliku NDJSON dla misji
        bucket.blob(ndjson_path).upload_from_string(
            ndjson_line + "\n",
            content_type="application/x-ndjson; charset=utf-8",
        )
        
        
        #zapisywanie indeksow
        
        
         # helper do slugów (zachowuje PL znaki, ogranicza długość)
        def _slug_u(text: str) -> str:
            t = (text or "").strip().lower()
            t = re.sub(r"\s+", "-", t)                         # spacje -> '-'
            t = re.sub(r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-]+", "-", t)  # tylko sensowne znaki
            t = re.sub(r"-{2,}", "-", t).strip("-")
            return t[:120]

        # 1) Timestamp i identyfikatory
        ts_dt   = datetime.now(timezone.utc)                   # jeśli wolisz bez timezone: datetime.utcnow()
        ts_file = ts_dt.strftime("%Y%m%d_%H%M%S")              # do nazwy pliku
        ts_iso  = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")         # do pola timestamp (ISO8601Z)

        # mission_id pełny vs skrócony (jak w Twoim załączniku)
        mid_full  = mission_id                                 # np. "mission_20250829_212413_92ed8ebc"
        mid_short = mission_id.replace("mission_", "")         # np. "20250829_212413_92ed8ebc"
        tail8     = mid_short[-8:] if len(mid_short) >= 8 else mid_short

        
        
        # --- SAFE BINDINGS: używaj *_val zamiast gołych nazw ---
        _locals = locals()
        _md = _locals.get("metadata") if isinstance(_locals.get("metadata"), dict) else {}

        approved_val     = _locals.get("approved", _md.get("approved", True))
        final_score_val  = _locals.get("final_score", _md.get("final_score"))
        nodes_count_val  = _locals.get("nodes_count", _md.get("nodes_count"))
        edges_count_val  = _locals.get("edges_count", _md.get("edges_count"))
        mission_type_val = _locals.get("mission_type", _md.get("mission_type", "general"))
        lang_val         = _locals.get("lang", _md.get("lang", "pl"))

        # tags jako lista
        _tags = _locals.get("tags", _md.get("tags", []))
        tags_list = list(_tags) if isinstance(_tags, (list, tuple)) else ([] if _tags is None else [str(_tags)])

        # flags jako dict + fallback z metadanych (has_* bywa trzymane płasko)
        flags_val = _locals.get("flags", {})
        if not isinstance(flags_val, dict):
            flags_val = {}
        flags_val = {
            "has_retry":        bool(flags_val.get("has_retry",        _md.get("has_retry", False))),
            "has_rollback":     bool(flags_val.get("has_rollback",     _md.get("has_rollback", False))),
            "has_optimization": bool(flags_val.get("has_optimization", _md.get("has_optimization", False))),
        }
        
        
        
        # 2) Źródło tytułu do display_id z metadanych (BEZ użycia 'mission')
        _display_src = ""
        try:
            if isinstance(metadata, dict):
                _display_src = (
                    metadata.get("mission_prompt")
                    or metadata.get("mission")
                    or metadata.get("title")
                    or ""
                )
        except NameError:
            _display_src = ""

        display_base = _slug_u(_display_src) if isinstance(_display_src, str) and _display_src.strip() else ""
        display_id   = f"{ts_file}-{display_base}-{tail8}" if display_base else f"{ts_file}-{tail8}"

        # 3) Lekki .txt jako content.uri (tak jak w załączniku)
        txt_name = f"{display_id}.txt"
        # UWAGA: 'root_prefix' to katalog dnia (ten sam, w którym lądują artefakty tej misji)
        txt_path = f"{root_prefix}/{txt_name}"
        txt_body = (
            f"mission_id: {mid_full}\n"
            f"timestamp:  {ts_iso}\n"
            f"approved:   {bool(approved)}\n"
            f"final_score:{final_score if final_score is not None else 'null'}\n"
            f"tags:       {', '.join(tags) if isinstance(tags, (list, tuple)) else ''}\n"
        )
        bucket.blob(txt_path).upload_from_string(txt_body, content_type="text/plain; charset=utf-8")
        txt_uri = f"gs://{bucket_name}/{txt_path}"

        # 4) Składamy dokument NDJSON 1:1 jak w załączniku
        tags_list = list(tags) if isinstance(tags, (list, tuple)) else ([] if tags is None else [str(tags)])
        has_retry = bool(flags.get("has_retry")) if isinstance(flags, dict) else False
        has_rb    = bool(flags.get("has_rollback")) if isinstance(flags, dict) else False
        has_opt   = bool(flags.get("has_optimization")) if isinstance(flags, dict) else False

        doc = {
            "id": mid_short,  # uwaga: w załączniku 'id' NIE ma prefiksu 'mission_'
            "structData": {
                "mission_id": mid_full,
                "timestamp": ts_iso,
                "mission_type": mission_type if 'mission_type' in locals() else "general",
                "tags": tags_list,
                "outcome": "Success" if approved else ("Partial" if (final_score not in (None, 0)) else "Failure"),
                "final_score": float(final_score) if final_score is not None else None,
                "approved": bool(approved),
                "nodes_count": int(nodes_count) if 'nodes_count' in locals() and nodes_count is not None else None,
                "edges_count": int(edges_count) if 'edges_count' in locals() and edges_count is not None else None,
                "has_retry": has_retry,
                "has_rollback": has_rb,
                "has_optimization": has_opt,
                "lang": "pl",
                "display_id": display_id,
                "links": {
                    "txt_uri": txt_uri,
                    "plan_uri": plan_uri,               # wcześniej policzone
                    "transcript_uri": transcript_uri,   # jw.
                    "metrics_uri": meta_uri,            # nazwa jak w załączniku
                },
            },
            "content": {
                "mimeType": "text/plain",
                "uri": txt_uri,
            },
        }

        # 5) Jednowierszowy plik NDJSON do folderu index/
        index_dir  = f"{root_prefix}/index"    # jeśli chcesz top-level: index_dir = "index"
        index_path = f"{index_dir}/metadata_{ts_file}.ndjson"
        bucket.blob(index_path).upload_from_string(
            json.dumps(doc, ensure_ascii=False) + "\n",
            content_type="application/x-ndjson; charset=utf-8",
        )
        print(f"[INDEX] NDJSON -> gs://{bucket_name}/{index_path}")
        
        
        
        #koniec zapisu indeksu

        print(f"[EXPORT] {mission_id} -> {ndjson_uri}")
        ndjson_uris.append(ndjson_uri)

    return ndjson_uris
