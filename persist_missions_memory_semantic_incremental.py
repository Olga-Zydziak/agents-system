from __future__ import annotations
import json
import hashlib
from datetime import datetime
from typing import Any, Dict
from google.cloud import storage
import vertexai


# --------------------


# --- Funkcje pomocnicze (bez zmian, są w porządku) ---
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _record_uid(rec: Dict[str, Any]) -> str:
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    return _md5(raw)

def _summary(text: str, n: int) -> str:
    t = (text or "").strip()
    return t[:n] + ("…" if len(t) > n else "")

# --- 2. ULEPSZONE FUNKCJE ZAPISU Z OBSŁUGĄ BŁĘDÓW ---

def _upload_json_robust(gcs_client: storage.Client, bucket: str, obj_path: str, data: Any) -> Dict[str, Any]:
    """Niezawodnie zapisuje obiekt JSON do GCS, zgłaszając błąd w razie niepowodzenia."""
    try:
        bucket_obj = gcs_client.bucket(bucket)
        blob = bucket_obj.blob(obj_path)
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        blob.upload_from_string(payload, content_type="application/json; charset=utf-8")
        
        uri = f"gs://{bucket}/{obj_path}"
        print(f"  ✅ [GCS] Zapisano obiekt: {uri}")
        return {"uri": uri, "bytes": len(payload)}
    except Exception as e:
        print(f"  ❌ [GCS BŁĄD KRYTYCZNY] Nie udało się zapisać obiektu {obj_path}. Przerywam przetwarzanie tego rekordu.")
        # Rzucamy błąd dalej, aby zatrzymać przetwarzanie tego jednego rekordu,
        # ponieważ bez zapisu w GCS, zapis w Vertex AI nie ma sensu.
        raise e

def _write_fact_robust(vertex_client: vertexai.Client, engine_name: str, fact: Dict[str, Any], scope: Dict[str, Any]) -> bool:
    """Niezawodnie zapisuje fakt w Vertex AI Memory, logując sukces lub porażkę."""
    fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
    # Ograniczenie rozmiaru faktu, aby uniknąć błędów API
    if len(fact_json) > 15000: # Bezpieczny, wysoki limit dla faktów
        fact["summary"] = _summary(fact.get("summary", ""), 1000)
        fact["plan_outline"] = fact.get("plan_outline", [])[:10]
        fact_json = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))

    try:
        vertex_client.agent_engines.create_memory(name=engine_name, fact=fact_json, scope=scope)
        print(f"    ✅ [Vertex AI] Zapisano fakt '{fact.get('kind')}' (widok: {scope.get('view')})")
        return True
    except Exception as e:
        print(f"    ❌ [Vertex AI BŁĄD] Nie udało się zapisać faktu '{fact.get('kind')}'.")
        print(f"       Szczegóły błędu API: {e}")
        return False

# --- 3. GŁÓWNA LOGIKA ORKIESTRUJĄCA ZAPIS ---

def run_persistent_memory_ingestion(
    json_path: str,
    engine_name: str,
    vertex_client: vertexai.Client,
    gcs_client: storage.Client,
    gcs_bucket: str
):
    print("--- 🏁 Rozpoczynam proces zapisu pamięci ---")
    print(f"➡️  Silnik agenta: {engine_name}")
    print(f"➡️  Bucket GCS: {gcs_bucket}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_records = data.get("full_mission_records", [])
    stats = {"processed": 0, "succeeded": 0, "failed": 0}

    for i, record in enumerate(all_records):
        mission_id = record.get("memory_id") or record.get("mission_id") or _record_uid(record)
        ruid = _record_uid(record)
        print(f"\n--- Przetwarzam rekord {i+1}/{len(all_records)} | Misja: {mission_id} ---")
        stats["processed"] += 1

        try:
            # Krok 1: Zapisz pełne dane w GCS
            pointers = {}
            gcs_prefix = "agent-memory-artifacts"
            base_path = f"{gcs_prefix}/{mission_id}/{ruid}"
            
            if record.get("final_plan"):
                pointers["final_plan"] = _upload_json_robust(gcs_client, gcs_bucket, f"{base_path}/final_plan.json", record["final_plan"])
            if record.get("critic") or record.get("critic_report"):
                critic_data = record.get("critic") or record.get("critic_report")
                pointers["critic"] = _upload_json_robust(gcs_client, gcs_bucket, f"{base_path}/critic.json", critic_data)
            
            # Krok 2: Przygotuj i zapisz skrócony fakt (indeks) w Vertex AI
            base_scope = {"mission_id": mission_id, "record_uid": ruid, "source": "learned_strategies_json"}
            
            overview_fact = {
                "kind": "mission_overview",
                "mission_id": mission_id,
                "record_uid": ruid,
                "imported_at": _now(),
                "title": _summary(record.get("mission_prompt", ""), 140),
                "summary": _summary(record.get("mission_prompt", ""), 700),
                "tags": record.get("tags", []),
                "final_score": record.get("final_score"),
                "pointers": pointers, # Linki do pełnych danych w GCS
            }

            # Zapisujemy JEDEN, główny fakt dla tego rekordu
            success = _write_fact_robust(
                vertex_client=vertex_client,
                engine_name=engine_name,
                fact=overview_fact,
                scope={**base_scope, "view": "overview"}
            )
            
            if success:
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1

        except Exception as e:
            # Ten błąd złapie głównie problemy z GCS
            stats["failed"] += 1
            continue # Przejdź do następnego rekordu

    print("\n--- ✅ Proces zapisu zakończony ---")
    print(f"📊 Statystyki: Przetworzono {stats['processed']}, Sukces {stats['succeeded']}, Porażki {stats['failed']}")