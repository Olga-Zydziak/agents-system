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
    Eksportuje lokalne misje do GCS w formacie:
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
        plan       = _extract_plan(snap)
        txt        = _build_txt(snap, plan)
        transcript = _build_transcript(snap)
        metrics    = _build_metrics(snap, plan)

        # Ścieżki w GCS – katalog per misja
        base_prefix = f"{root_prefix}/{mission_id}".strip("/")
        txt_path     = f"{base_prefix}/{mission_id}.txt"
        plan_path    = f"{base_prefix}/{mission_id}.plan.json"
        trans_path   = f"{base_prefix}/{mission_id}.transcript.json"
        metrics_path = f"{base_prefix}/{mission_id}.metrics.json"
        ts_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ndjson_path = f"index/metadata_{ts_file}.ndjson"

        txt_uri     = f"gs://{bucket_name}/{txt_path}"
        plan_uri    = f"gs://{bucket_name}/{plan_path}"
        trans_uri   = f"gs://{bucket_name}/{trans_path}"
        metrics_uri = f"gs://{bucket_name}/{metrics_path}"
        ndjson_uri  = f"gs://{bucket_name}/{ndjson_path}"

        # Skip, jeśli już istnieją artefakty (lub samo NDJSON)
        if skip_existing:
            if bucket.blob(metrics_path).exists(client) or bucket.blob(ndjson_path).exists(client):
                print(f"[EXPORT][SKIP] {mission_id} już istnieje w GCS (metrics/ndjson).")
                continue

        # 1) Upload artefaktów
        bucket.blob(txt_path).upload_from_string(
            txt, content_type="text/plain; charset=utf-8"
        )
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

        # 2) NDJSON przez kanoniczny builder
        ndjson_line = _ndjson_line(
            mission_id=mission_id,
            txt_uri=txt_uri,
            plan_uri=plan_uri,
            transcript_uri=trans_uri,
            metrics_uri=metrics_uri,
            metrics=metrics,  # lub rozbij na płaskie argumenty, jeśli taka jest Twoja wersja
        )

        # 3) Upload pojedynczego pliku NDJSON dla misji
        bucket.blob(ndjson_path).upload_from_string(
            ndjson_line + "\n",
            content_type="application/x-ndjson; charset=utf-8",
        )
        print(f"[EXPORT] {mission_id} -> {ndjson_uri}")
        ndjson_uris.append(ndjson_uri)

    return ndjson_uris

