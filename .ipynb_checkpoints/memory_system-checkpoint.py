"""
System pamięci kontekstowej z uczeniem się z poprzednich iteracji
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import numpy as np
from collections import deque
import os
from config_api import (
    GCS_BUCKET, GCS_MISSIONS_PREFIX, GCS_INDEX_PREFIX, GCS_INDEX_FILENAME_TEMPLATE,
    gcs_index_path, gcs_mission_base_prefix,
)
# Zewnętrzne biblioteki do obliczania podobieństwa tekstu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lokalny logger procesu
from process_logger import log as process_log


import re, unicodedata
from autogen_vertex_mcp_system_claude import SystemConfig, VertexSearchTool
from exporter_missions_lib import _ndjson_line

def _to_text_safe(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _slugify(text: str, maxlen: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:maxlen] or "mission"


def _gcs_upload_json(bucket_name: str, blob_name: str, obj: dict):
    """Wrzuca JSON do GCS pod gs://{bucket_name}/{blob_name}"""
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError(
            "Brak pakietu google-cloud-storage. Zainstaluj: pip install google-cloud-storage"
        )
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.cache_control = "no-cache"
    blob.upload_from_string(
        data=json.dumps(obj, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


class ContextMemory:
    def __init__(
        self,
        max_episodes: int = 100,
        gcs_bucket: str | None = None,
        gcs_prefix: str = "",
    ):
        # Existing
        self.episodes = deque(maxlen=max_episodes)
        self.learned_patterns = {}
        self.successful_strategies = []

        # NOWE - Pełne dane misji
        self.full_mission_records = []  # Bez limitu - wszystko zapisujemy
        self.mission_index = {}  # Szybkie wyszukiwanie po ID

        self._load_persistent_memory()
        self.gcs_bucket = gcs_bucket or GCS_BUCKET
        self.gcs_prefix = (gcs_prefix or "").strip().strip("/")
        self.use_gcs = bool(self.gcs_bucket)

        
        self.gcs_missions_prefix = GCS_MISSIONS_PREFIX
        self.gcs_index_prefix = GCS_INDEX_PREFIX
        self.gcs_index_filename_template = GCS_INDEX_FILENAME_TEMPLATE
        
        self.iteration_feedback = []
        self.last_feedback = ""
        
        
    #poprawki
    
        
    def add_iteration_feedback(self, iteration: int, feedback: str, timestamp: datetime):
        """
        Zapisuje feedback z iteracji do pamięci operacyjnej + lekki zapis lokalny.
        Orchestrator woła to po każdej odpowiedzi Critica.
        """
        try:
            entry = {
                "iteration": int(iteration),
                "feedback": str(feedback),
                "timestamp": timestamp.isoformat(),
            }
            # w RAM
            if not hasattr(self, "iteration_feedback"):
                self.iteration_feedback = []
            self.iteration_feedback.append(entry)
            # do szybkiego kontekstu
            self.last_feedback = entry["feedback"]

            # lekki zapis lokalny (bez zależności od pełnego mission_id)
            os.makedirs("memory/iterations", exist_ok=True)
            with open(f"memory/iterations/iter_{iteration:03d}.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)

            process_log(f"[MEMORY] add_iteration_feedback(iter={iteration}) ok")
        except Exception as e:
            process_log(f"[MEMORY ERROR] add_iteration_feedback failed: {e}")

    def get_relevant_context(self, mission: str) -> Dict[str, Any]:
        """
        Zwraca wstrzyknięcie kontekstu dla promptów (to woła orchestrator).
        Minimalnie: rekomendacje, pułapki, ostatni feedback.
        Wersja bez TF-IDF: heurystyki + ostatnie sukcesy.
        """
        context = {
            "recommended_strategies": [],
            "common_pitfalls": [],
            "last_feedback": getattr(self, "last_feedback", ""),
        }

        # Heurystyka po treści misji (szybkie tagi)
        m = (mission or "").lower()
        if "csv" in m or "etl" in m or "pipeline" in m:
            context["recommended_strategies"].append("Add robust error handling with retry & dead-letter queue.")
            context["common_pitfalls"].append("Schema drift unchecked; missing data validation gates.")
        if "continuous" in m or "online" in m or "adapt" in m:
            context["recommended_strategies"].append("Introduce drift detection + gated retraining with rollback.")
            context["common_pitfalls"].append("No cap on retraining loops; missing convergence/abort criteria.")
        if "causal" in m or "przyczyn" in m:
            context["recommended_strategies"].append("Add causal shift analysis before blind retraining.")

        # Na bazie ostatnich udanych planów zapamiętanych w tym procesie
        best_practices = []
        if hasattr(self, "successful_strategies") and self.successful_strategies:
            best_practices = self.successful_strategies[-5:]
        for bp in best_practices:
            tip = bp.get("tip") or bp.get("note")
            if tip:
                context["recommended_strategies"].append(tip)

        # Deduplicate i przytnij, żeby prompt był zwięzły
        context["recommended_strategies"] = list(dict.fromkeys(context["recommended_strategies"]))[:6]
        context["common_pitfalls"] = list(dict.fromkeys(context["common_pitfalls"]))[:6]

        return context

    
    
    #przeszukiwanie pamieci
    
    def get_vertex_context(self, mission: str, min_score: float = 80.0, top_k: int = 5) -> dict:
        ctx = {"recommended_strategies": [], "common_pitfalls": [], "examples": []}
        if VertexSearchTool is None or SystemConfig is None:
            return ctx
        try:
            cfg = SystemConfig()
            vst = VertexSearchTool(cfg)
            raw = vst.search_mission_memory(query=mission, top_k=top_k)
            results = (json.loads(raw) or {}).get("results", [])
            for r in results:
                try:
                    score = float(r.get("score") or 0)
                except Exception:
                    score = 0.0
                if score < min_score:
                    continue
                tags = r.get("tags") or []
                links = r.get("links") or {}
                if "retry" in tags:
                    ctx["recommended_strategies"].append("Use retry with backoff + DLQ.")
                if "rollback" in tags:
                    ctx["recommended_strategies"].append("Add rollback path for irreversible ops.")
                if "optimize" in tags:
                    ctx["recommended_strategies"].append("Add optimize_performance guarded loop.")
                if links.get("plan_uri"):
                    ctx["examples"].append({
                    "mission_id": r.get("mission_id"),
                    "plan_uri": links["plan_uri"],
                    })
            # dedup + limit
            dedup = []
            seen = set()
            for t in ctx["recommended_strategies"]:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            ctx["recommended_strategies"] = dedup[:6]
            return ctx
        except Exception as e:
            process_log(f"[MEMORY] Vertex ctx skipped: {e}")
            return {"recommended_strategies": [], "common_pitfalls": [], "examples": []}
    
    #koniec poprawki
    
    
    
    def add_successful_plan(self, plan: Dict[str, Any], mission: str, metadata: Dict[str, Any]):
        """
        Zapisuje „udany plan” lokalnie (folder per misja) oraz aktualizuje proste „best practices”.
        To woła orchestrator po wyciągnięciu finalnego planu.
        """
        try:
            os.makedirs("memory/success", exist_ok=True)

            # id misji z misji (spójne z save_complete_mission)
            from datetime import datetime as _dt
            import hashlib as _h
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            h = _h.md5(mission.encode("utf-8")).hexdigest()[:8]
            mission_id = f"mission_{ts}_{h}"

            # folder per misja
            mission_dir = os.path.join("memory", "success", mission_id)
            os.makedirs(mission_dir, exist_ok=True)

            # zapis planu i meta
            with open(os.path.join(mission_dir, f"{mission_id}.plan.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            payload = {
                "mission": mission,
                "metadata": metadata or {},
                "saved_at": _dt.now().isoformat(),
            }
            with open(os.path.join(mission_dir, f"{mission_id}.meta.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            # bardzo proste „best practices” (do wykorzystania przez get_relevant_context)
            node_types = {str(n.get("implementation") or n.get("name") or "").lower()
                          for n in (plan.get("nodes") or [])}
            tiplist = []
            if "error_handler" in node_types:
                tiplist.append("Always include a dedicated error_handler with routing to notify/report.")
            if "rollback" in node_types:
                tiplist.append("Keep rollback path for any irreversible change.")
            if "validate_data" in node_types:
                tiplist.append("Gate the main path by validate_data with measurable thresholds.")
            if "validate_model" in node_types and "train_model" in node_types:
                tiplist.append("Retrain only after drift detection; compare and rollback on regression.")
            if "optimize_performance" in node_types:
                tiplist.append("Use optimize_performance with loop guards to avoid infinite cycles.")

            if tiplist:
                self.successful_strategies.append({"mission_id": mission_id, "tip": tiplist[0]})

            process_log(f"[MEMORY] add_successful_plan saved under {mission_dir}")
        except Exception as e:
            process_log(f"[MEMORY ERROR] add_successful_plan failed: {e}")

    
    
    #koniec poprawek
        
        
        
        
    def _clean_agent_content(
        self, content: str
    ) -> any:  # Zmieniamy typ zwracany na 'any'
        """
        Usuwa bloki kodu markdown i parsuje wewnętrzny JSON,
        jeśli to możliwe.
        """
        if not isinstance(content, str):
            return content

        cleaned_content = content.strip()

        # Krok 1: Usuń bloki kodu markdown (tak jak poprzednio)
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, cleaned_content, re.DOTALL)
        if match:
            cleaned_content = match.group(1).strip()

        # === NOWY, KLUCZOWY KROK ===
        # Krok 2: Spróbuj sparsować string jako JSON
        try:
            # Jeśli się uda, zwróć prawdziwy obiekt (słownik)
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            # Jeśli to nie jest JSON, zwróć po prostu oczyszczony tekst
            return cleaned_content

    def _gcs_path(self, relative: str) -> str:
        """Buduje pełną ścieżkę do pliku w GCS z uwzględnieniem prefiksu"""
        if self.gcs_prefix:
            return f"{self.gcs_prefix}/{relative}"
        return relative

    def _learn_from_success(self, mission_record: Dict):
        """Ekstraktuje i zapisuje PRAWDZIWE wzorce z udanej misji"""

        # 1. Zapisz wzorzec sukcesu dla tego typu misji
        pattern_key = f"success_pattern_{mission_record['mission_type']}"

        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "occurrences": 0,
                "examples": [],
                "common_elements": {},
                "avg_score": 0,
                "best_practices": [],
            }

        # 2. Aktualizuj statystyki
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        current_score = mission_record.get("final_score", 0)
        pattern["avg_score"] = (
            pattern["avg_score"] * (pattern["occurrences"] - 1) + current_score
        ) / pattern["occurrences"]

        # 3. Znajdź kluczowe elementy sukcesu
        success_elements = []

        # Sprawdź co było w tym planie
        plan = mission_record.get("final_plan", {})
        nodes = plan.get("nodes", [])

        # Zapisz które węzły były użyte
        node_types = [n.get("implementation") for n in nodes]

        if "error_handler" in node_types:
            success_elements.append("comprehensive_error_handling")
        if "rollback" in node_types:
            success_elements.append("rollback_mechanism")
        if "validate_data" in node_types:
            success_elements.append("data_validation")
        if "optimize_performance" in node_types:
            success_elements.append("performance_optimization")

        # 4. Znajdź unikalne innowacje z tej misji
        if "Adaptive_Router" in str(nodes):
            success_elements.append("adaptive_routing")

        # 5. Zapisz jako best practice jeśli score > 90
        if current_score > 90:
            best_practice = {
                "mission_id": mission_record["memory_id"],
                "score": current_score,
                "key_success_factors": success_elements,
                "node_count": len(nodes),
                "complexity": mission_record["performance_metrics"].get(
                    "convergence_rate", 0
                ),
            }
            pattern["best_practices"].append(best_practice)

        # 6. Zaktualizuj common_elements (co występuje najczęściej)
        for element in success_elements:
            if element not in pattern["common_elements"]:
                pattern["common_elements"][element] = 0
            pattern["common_elements"][element] += 1

        # 7. Dodaj przykład
        pattern["examples"].append(
            {
                "mission_prompt": mission_record["mission_prompt"],
                "success_factors": success_elements,
                "score": current_score,
            }
        )

        process_log(
            f"[MEMORY] Learned from success: {pattern_key}, "
            f"occurrences={pattern['occurrences']}, "
            f"avg_score={pattern['avg_score']:.2f}"
        )

    def export_temporal_report(self, filepath: str = "memory/temporal_patterns.json"):
        """Eksportuje raport wzorców czasowych"""
        patterns = self.analyze_temporal_patterns()

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_missions": len(self.full_mission_records),
            "patterns": patterns,
            "insights": [],
        }

        # Znajdź najlepszy/najgorszy czas
        best_day = max(
            patterns["by_weekday"].items(), key=lambda x: x[1].get("avg_score", 0)
        )
        worst_day = min(
            patterns["by_weekday"].items(), key=lambda x: x[1].get("avg_score", 100)
        )

        report["insights"].append(
            f"Best day: {best_day[0]} (avg: {best_day[1]['avg_score']:.1f})"
        )
        report["insights"].append(
            f"Worst day: {worst_day[0]} (avg: {worst_day[1]['avg_score']:.1f})"
        )

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        return report

    # ------
    
    def save_complete_mission(
    self,
    mission: str,
    final_plan: Dict,
    all_messages: List[Dict],
    orchestrator_state: Dict,
) -> str:
        """
        Zapisuje KOMPLETNY rekord misji.
        - GCS: pełny rekord JSON + lekkie artefakty + indeks JSON + indeks NDJSON
        - Lokalnie: pełny rekord JSON + lokalny indeks
        """
        from datetime import datetime as _dt
        import hashlib as _h
        import json as _json
        import os as _os

        # === 1) ID i wstępne czyszczenie ===
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        mission_hash = _h.md5(mission.encode("utf-8")).hexdigest()[:8]
        mission_id = f"mission_{timestamp}_{mission_hash}"

        cleaned_messages = []
        for msg in all_messages:
            new_msg = msg.copy()
            if "content" in new_msg:
                new_msg["content"] = self._clean_agent_content(new_msg["content"])
            cleaned_messages.append(new_msg)

        # Wyciągi i metadane
        iterations_data = self._extract_iterations_from_transcript(cleaned_messages)
        mission_type = self._classify_mission(mission)
        tags = self._extract_tags(mission, final_plan)
        critical_moments = self._identify_critical_moments(all_messages)

        mission_record = {
            "memory_id": mission_id,
            "timestamp": _dt.now().isoformat(),
            "mission_prompt": mission,
            "mission_type": mission_type,
            "tags": tags,
            "outcome": "Success" if final_plan else "Failed",
            "total_iterations": orchestrator_state.get("iteration_count", 0),
            "total_messages": len(all_messages),
            "time_taken_seconds": orchestrator_state.get("execution_time", 0),
            "final_plan": final_plan,
            "final_score": self._extract_final_score(all_messages),
            "iterations": iterations_data,
            "critique_evolution": self._track_critique_evolution(iterations_data),
            "aggregator_reasoning": self._extract_aggregator_reasoning(all_messages),
            "proposer_contributions": self._analyze_proposer_contributions(all_messages),
            "llm_generated_summary": self._generate_mission_summary(all_messages, final_plan),
            "identified_patterns": self._extract_patterns_from_debate(all_messages),
            "success_factors": self._identify_success_factors(final_plan, iterations_data),
            "failure_points": self._identify_failure_points(iterations_data),
            "critical_moments": critical_moments,
            "turning_points": self._identify_turning_points(iterations_data),
            "full_transcript": cleaned_messages,
            "performance_metrics": {
                "token_usage": orchestrator_state.get("total_tokens", 0),
                "api_calls": orchestrator_state.get("api_calls", 0),
                "convergence_rate": self._calculate_convergence_rate(iterations_data),
            },
        }

        # === 2) Zapis: GCS lub lokalnie ===
        if getattr(self, "use_gcs", False) and getattr(self, "gcs_bucket", None):
            # Czytelna ścieżka: missions/YYYY/MM/DD/<plik>.json
            ts_date, ts_time = timestamp.split("_")  # np. 20250917, 213045
            base_prefix = gcs_mission_base_prefix(ts_date) 
            y, m, d = ts_date[:4], ts_date[4:6], ts_date[6:8]
            slug = _slugify(mission)
            full_name = f"{ts_date}_{ts_time}-{slug}-{mission_hash}.json"
            mission_blob_rel = f"{base_prefix}/{full_name}"
            mission_blob = self._gcs_path(mission_blob_rel)

            # 2a) Pełny rekord JSON
            try:
                _gcs_upload_json(self.gcs_bucket, mission_blob, mission_record)
                process_log(f"[MEMORY] Saved mission to GCS: gs://{self.gcs_bucket}/{mission_blob}")
            except Exception as e:
                process_log(f"[MEMORY ERROR] Failed to save mission to GCS: {e}")

            # 2b) Artefakty lekkie do linkowania w indeksie
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(self.gcs_bucket)

                
                ts_date, ts_time = timestamp.split("_")
                base_prefix = gcs_mission_base_prefix(ts_date) 
                
                base_name = mission_id                      # gwarantuje unikalność per run

                # preview.txt
                preview_txt = (
                    f"mission_id: {mission_id}\n"
                    f"timestamp:  {mission_record.get('timestamp')}\n"
                    f"outcome:    {mission_record.get('outcome')}\n"
                    f"final_score:{mission_record.get('final_score')}\n"
                    f"tags:       {', '.join(mission_record.get('tags', []))}\n"
                )
                preview_path = f"{base_prefix}/{base_name}.preview.txt"
                bucket.blob(preview_path).upload_from_string(
                    preview_txt, content_type="text/plain; charset=utf-8"
                )
                preview_uri = f"gs://{self.gcs_bucket}/{preview_path}"

                # plan.json (sam plan)
                plan_path = f"{base_prefix}/{base_name}.plan.json"
                bucket.blob(plan_path).upload_from_string(
                    _json.dumps(final_plan, ensure_ascii=False, indent=2),
                    content_type="application/json; charset=utf-8",
                )
                plan_uri = f"gs://{self.gcs_bucket}/{plan_path}"

                # transcript.ndjson
                transcript_path = f"{base_prefix}/{base_name}.transcript.ndjson"
                transcript_lines = []
                for m in cleaned_messages:
                    transcript_lines.append(_json.dumps({
                        "name": m.get("name"),
                        "role": m.get("role"),
                        "content": self._clean_agent_content(m.get("content", "")),
                        "ts": m.get("timestamp") or None,
                    }, ensure_ascii=False))
                bucket.blob(transcript_path).upload_from_string(
                    "\n".join(transcript_lines),
                    content_type="application/x-ndjson; charset=utf-8",
                )
                transcript_uri = f"gs://{self.gcs_bucket}/{transcript_path}"

                # metadata.json (kompakt)
                meta_compact = {
                    "mission_id": mission_id,
                    "timestamp": mission_record.get("timestamp"),
                    "mission_type": mission_record.get("mission_type"),
                    "tags": mission_record.get("tags", []),
                    "outcome": mission_record.get("outcome"),
                    "final_score": mission_record.get("final_score"),
                    "nodes_count": len((final_plan or {}).get("nodes", [])),
                    "edges_count": len((final_plan or {}).get("edges", [])),
                    "performance_metrics": mission_record.get("performance_metrics", {}),
                }
                meta_path = f"{base_prefix}/{base_name}.metadata.json"
                bucket.blob(meta_path).upload_from_string(
                    _json.dumps(meta_compact, ensure_ascii=False, indent=2),
                    content_type="application/json; charset=utf-8",
                )
                meta_uri = f"gs://{self.gcs_bucket}/{meta_path}"
            except Exception as e:
                process_log(f"[MEMORY ERROR] Failed to save lightweight artifacts: {e}")
                # Na wszelki wypadek puste linki (żeby _ndjson_line się nie wysypało)
                preview_uri = plan_uri = transcript_uri = meta_uri = f"gs://{self.gcs_bucket}/{mission_blob}"

            # 2c) Lekki indeks JSON (zostaje jak był)
            index_entry = {
                "mission_id": mission_id,
                "gcs_path": f"gs://{self.gcs_bucket}/{mission_blob}",
                "timestamp": mission_record.get("timestamp"),
                "mission_prompt": mission_record.get("mission_prompt", "")[:100],
                "mission_type": mission_record.get("mission_type"),
                "final_score": mission_record.get("final_score"),
                "tags": mission_record.get("tags", []),
                "outcome": mission_record.get("outcome"),
            }
            index_blob = self._gcs_path(f"index/{mission_id}.json")
            try:
                _gcs_upload_json(self.gcs_bucket, index_blob, index_entry)
            except Exception as e:
                process_log(f"[MEMORY ERROR] Failed to save index to GCS: {e}")

            # 2d) Indeks NDJSON (kanoniczny, 1 linia)
            try:
                ndjson_line = _ndjson_line(
                    mission_id=mission_id,
                    txt_uri=preview_uri,
                    plan_uri=plan_uri,
                    transcript_uri=transcript_uri,
                    metrics_uri=meta_uri,
                    metrics={
                        "mission_id": mission_id,
                        "timestamp": mission_record.get("timestamp"),
                        "mission_type": mission_record.get("mission_type"),
                        "tags": mission_record.get("tags", []),
                        "outcome": mission_record.get("outcome"),
                        "final_score": mission_record.get("final_score"),
                        "approved": mission_record.get("outcome") == "Success",
                        "nodes_count": len((final_plan or {}).get("nodes", [])),
                        "edges_count": len((final_plan or {}).get("edges", [])),
                        "has_retry": "retry" in mission_record.get("tags", []),
                        "has_rollback": "rollback" in mission_record.get("tags", []),
                        "has_optimization": "optimization" in mission_record.get("tags", []),
                        "lang": "pl",
                    },
                )
                
                
                
                ts_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                ndjson_rel = gcs_index_path(ts_file)          # np. "index/metadata_20250918_090001.ndjson"
                ndjson_blob_rel = self._gcs_path(ndjson_rel)
                bucket.blob(ndjson_blob_rel).upload_from_string(
                    ndjson_line + "\n",
                    content_type="application/x-ndjson; charset=utf-8",
                )
                
                
                
                # Upload NDJSON
                from google.cloud import storage as _storage2
                _storage_client = _storage2.Client()
                _bucket2 = _storage_client.bucket(self.gcs_bucket)
                _bucket2.blob(ndjson_blob_rel).upload_from_string(
                    ndjson_line + "\n",
                    content_type="application/x-ndjson; charset=utf-8",
                )
                process_log(f"[MEMORY] Saved NDJSON index: gs://{self.gcs_bucket}/{ndjson_blob_rel}")
            except Exception as e:
                process_log(f"[MEMORY ERROR] Failed to save NDJSON index: {e}")

        else:
            # === Zapis lokalny ===
            mission_dir = "memory/missions"
            _os.makedirs(mission_dir, exist_ok=True)
            mission_file = _os.path.join(mission_dir, f"{mission_id}.json")

            try:
                with open(mission_file, "w", encoding="utf-8") as f:
                    _json.dump(mission_record, f, ensure_ascii=False, indent=2)
                process_log(f"[MEMORY] Saved mission to file: {mission_file}")
            except Exception as e:
                process_log(f"[MEMORY ERROR] Failed to save mission file: {e}")

            self._update_mission_index(mission_id, mission_file, mission_record)

        # === 3) Runtime + patterns ===
        self.full_mission_records.append(mission_record)
        self.mission_index[mission_id] = len(self.full_mission_records) - 1

        if final_plan:
            self._learn_from_success(mission_record)

        if len(self.full_mission_records) % 5 == 0:
            patterns = self.analyze_temporal_patterns()
            process_log(f"[MEMORY] Temporal patterns update: {len(patterns['by_weekday'])} weekdays analyzed")

        self._persist_lightweight_memory()
        process_log(f"[MEMORY] Saved complete mission: {mission_id}")
        return mission_id
    
    

    #koniec save completed mission
    def _update_mission_index(self, mission_id: str, file_path: str, record: Dict):
        """Aktualizuje lekki indeks wszystkich misji"""
        index_file = "memory/mission_index.json"

        try:
            if os.path.exists(index_file):
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {"missions": [], "metadata": {}}
        except:
            index = {"missions": [], "metadata": {}}

        # Dodaj wpis do indeksu
        index_entry = {
            "mission_id": mission_id,
            "file_path": file_path,
            "timestamp": record.get("timestamp"),
            "mission_prompt": record.get("mission_prompt", "")[:100],
            "mission_type": record.get("mission_type"),
            "final_score": record.get("final_score"),
            "tags": record.get("tags", []),
            "outcome": record.get("outcome"),
        }

        index["missions"].append(index_entry)
        index["metadata"]["last_updated"] = datetime.now().isoformat()
        index["metadata"]["total_missions"] = len(index["missions"])

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _persist_lightweight_memory(self):
        """Zapisuje tylko patterns i strategies (bez pełnych rekordów misji)"""
        data = {
            "patterns": self.learned_patterns,
            "strategies": self.successful_strategies,
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "mission_count": len(self.full_mission_records),
            },
        }

        if self.use_gcs:
            try:
                blob = self._gcs_path("learned_strategies.json")
                _gcs_upload_json(self.gcs_bucket, blob, data)
                process_log(
                    f"[MEMORY] Saved lightweight memory to GCS: gs://{self.gcs_bucket}/{blob}"
                )
                return
            except Exception as e:
                process_log(
                    f"[MEMORY ERROR] Failed to save lightweight memory to GCS: {e}"
                )

        # fallback/local
        os.makedirs("memory", exist_ok=True)
        memory_file = "memory/learned_strategies.json"
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Nie udało się zapisać pamięci: {e}")

    def load_specific_mission(self, mission_id: str) -> Optional[Dict]:
        """Ładuje konkretną misję z pliku"""
        mission_file = f"memory/missions/{mission_id}.json"

        if os.path.exists(mission_file):
            try:
                with open(mission_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                process_log(f"[MEMORY ERROR] Cannot load mission {mission_id}: {e}")

        return None

    def search_missions(self, query: str, limit: int = 10) -> List[Dict]:
        """Przeszukuje indeks misji bez ładowania wszystkich plików"""
        index_file = "memory/mission_index.json"

        if not os.path.exists(index_file):
            return []

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        results = []
        query_lower = query.lower()

        for entry in index["missions"]:
            # Proste wyszukiwanie tekstowe
            if (
                query_lower in entry.get("mission_prompt", "").lower()
                or query_lower in entry.get("mission_type", "").lower()
                or any(query_lower in tag.lower() for tag in entry.get("tags", []))
            ):

                results.append(entry)
                if len(results) >= limit:
                    break

        return results



    def _extract_iterations_from_transcript(self, messages: List[Dict]) -> List[Dict]:
        """Ekstraktuje dane każdej iteracji z transkryptu"""
        iterations = []
        current_iteration = {"proposers": [], "aggregator": None, "critic": None}

        for msg in messages:
            role = msg.get("name", "").lower()

            if "proposer" in role or "analyst" in role or "planner" in role:
                current_iteration["proposers"].append(
                    {
                        "agent": msg.get("name"),
                        "content": msg.get("content"),
                        "key_ideas": self._extract_key_ideas(msg.get("content", "")),
                    }
                )

            elif "aggregator" in role:
                current_iteration["aggregator"] = {
                    "content": msg.get("content"),
                    "synthesis": self._extract_synthesis(msg.get("content", "")),
                }

            elif "critic" in role:
                current_iteration["critic"] = {
                    "content": msg.get("content"),
                    "verdict": self._extract_verdict(msg.get("content", "")),
                    "score": self._extract_score(msg.get("content", "")),
                    "weaknesses": self._extract_weaknesses(msg.get("content", "")),
                }

                # Koniec iteracji - zapisz i zacznij nową
                if current_iteration["proposers"]:
                    iterations.append(current_iteration)
                    current_iteration = {
                        "proposers": [],
                        "aggregator": None,
                        "critic": None,
                    }

        return iterations

    def _generate_mission_summary(self, messages: List[Dict], final_plan: Dict) -> str:
        """Generuje BOGATE podsumowanie misji"""
        summary_parts = []

        # 1. Liczba iteracji i czas
        iteration_count = sum(
            1 for m in messages if "critic" in m.get("name", "").lower()
        )
        summary_parts.append(f"Misja zakończona w {iteration_count} iteracji.")

        # 2. Kluczowe innowacje (szukaj w transkrypcie)
        innovations = set()
        for msg in messages:

            content_str = str(msg.get("content", ""))
            content = content_str.lower()

            if "adaptive" in content or "adaptacyjny" in content:
                innovations.add("adaptive routing")
            if "rollback" in content:
                innovations.add("rollback mechanism")
            if "optimiz" in content or "optymali" in content:
                innovations.add("optimization")

        if innovations:
            summary_parts.append(f"Zastosowano: {', '.join(innovations)}.")

        # 3. Analiza struktury planu
        if final_plan:
            nodes = final_plan.get("nodes", [])
            edges = final_plan.get("edges", [])

            # Policz typy ścieżek
            success_paths = len(
                [e for e in edges if e.get("condition") == "on_success"]
            )
            failure_paths = len(
                [e for e in edges if e.get("condition") == "on_failure"]
            )

            summary_parts.append(
                f"Struktura: {len(nodes)} węzłów, "
                f"{success_paths} ścieżek sukcesu, "
                f"{failure_paths} ścieżek obsługi błędów."
            )

            # Znajdź kluczowe węzły
            key_nodes = []
            for node in nodes:
                impl = node.get("implementation", "")
                if impl in [
                    "error_handler",
                    "rollback",
                    "validate_data",
                    "optimize_performance",
                ]:
                    key_nodes.append(impl)

            if key_nodes:
                summary_parts.append(
                    f"Kluczowe komponenty: {', '.join(set(key_nodes))}."
                )

        # 4. Końcowy verdykt
        for msg in reversed(messages):
            if "critic" in msg.get("name", "").lower() and "ZATWIERDZONY" in msg.get(
                "content", ""
            ):
                summary_parts.append("Plan zatwierdzony przez krytyka bez zastrzeżeń.")
                break

        return " ".join(summary_parts)

    def _extract_tags(self, mission: str, final_plan: Dict) -> List[str]:
        """Automatycznie taguje misję"""
        tags = []
        mission_lower = mission.lower()

        # Mission-based tags
        tag_keywords = {
            "error_handling": ["error", "błęd", "obsługa", "handler"],
            "optimization": ["optym", "performance", "wydajność"],
            "causality": ["causal", "przyczyn"],
            "validation": ["valid", "walidac"],
            "retry": ["retry", "ponow"],
            "rollback": ["rollback", "cofn"],
            "ml": ["model", "train", "uczenie"],
            "data": ["data", "dane", "csv", "pipeline"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in mission_lower for kw in keywords):
                tags.append(tag)

        # Plan-based tags
        if final_plan:
            nodes_str = str(final_plan.get("nodes", []))
            if "error_handler" in nodes_str:
                tags.append("robust")
            if "optimize" in nodes_str:
                tags.append("optimized")

        return list(set(tags))  # Unique tags

    # ------------
    def _load_persistent_memory(self):
        """Ładuje pamięć - patterns z głównego pliku, misje z indeksu"""
        json_file = "memory/learned_strategies.json"

        # Załaduj patterns i strategies
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.learned_patterns = data.get("patterns", {})
                self.successful_strategies = data.get("strategies", [])
            except Exception as e:
                print(f"⚠ Nie udało się załadować pamięci: {e}")

        # Załaduj listę misji z indeksu
        index_file = "memory/mission_index.json"
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)

                # Załaduj ostatnie 10 misji do pamięci runtime
                recent_missions = index["missions"][-10:] if "missions" in index else []
                for entry in recent_missions:
                    if "file_path" in entry and os.path.exists(entry["file_path"]):
                        with open(entry["file_path"], "r", encoding="utf-8") as f:
                            record = json.load(f)
                            self.full_mission_records.append(record)
                            self.mission_index[entry["mission_id"]] = (
                                len(self.full_mission_records) - 1
                            )

                print(f"✔ Załadowano {len(self.full_mission_records)} ostatnich misji")
            except Exception as e:
                print(f"⚠ Problem z indeksem misji: {e}")
        else:
            print("🔍 Tworzę nową pamięć (brak istniejącego indeksu)")

        os.makedirs("memory/missions", exist_ok=True)


    def _persist_full_memory(self):
        """Alias dla _persist_memory"""
        # self._persist_memory()
        self._persist_lightweight_memory()

    # def _extract_key_ideas(self, content: str) -> List[str]:
    #     """Ekstraktuje kluczowe pomysły z contentu"""
    #     # Prosta heurystyka - możesz ulepszyć
    #     ideas = []
    #     if "error_handler" in content.lower():
    #         ideas.append("error_handling")
    #     if "rollback" in content.lower():
    #         ideas.append("rollback_mechanism")
    #     if "optimiz" in content.lower():
    #         ideas.append("optimization")
    #     return ideas

    
    
    def _extract_key_ideas(self, content):
        """
        Zwraca listę 'idei' (słów-kluczy) wykrytych w treści.
        Treść może być stringiem albo dict-em (np. plan/odpowiedź krytyka).
        """
        ideas = set()

        # Gdy dostajemy JSON-owy plan/krytykę (dict) — wyciągnijmy sygnały bez tekstu
        if isinstance(content, dict):
            # 1) plan: nodes/edges → dorzuć implementacje/nazwy
            nodes = content.get("nodes") \
                or content.get("final_synthesized_plan", {}).get("nodes") \
                or content.get("final_plan", {}).get("nodes") \
                or []
            if isinstance(nodes, list):
                for n in nodes:
                    if isinstance(n, dict):
                        impl = n.get("implementation") or n.get("name")
                        if impl:
                            ideas.add(str(impl).lower())

            # 2) krytyka: weaknesses → dorzuć nazwy słabości
            cs = content.get("critique_summary") or {}
            for w in (cs.get("identified_weaknesses") or []):
                if isinstance(w, dict):
                    wk = w.get("weakness")
                    if wk:
                        ideas.add(str(wk).lower())

            # 3) przejdź do heurystyk tekstowych na zserializowanym stringu
            text = _to_text_safe(content)
        else:
            text = _to_text_safe(content)

        t = text.lower()

        # Heurystyki — jak wcześniej, ale działają na stringu
        if "error_handler" in t:
            ideas.add("error_handler")
        if "rollback" in t:
            ideas.add("rollback")
        if "retry" in t or "ponów" in t or "ponowienie" in t or "backoff" in t:
            ideas.add("retry")
        if "optimiz" in t or "optymaliz" in t:
            ideas.add("optimize")
        if "validate" in t or "walidac" in t:
            ideas.add("validate_data")
        if "clean" in t or "czyszczen" in t:
            ideas.add("clean_data")
        if "load" in t or "wczyt" in t:
            ideas.add("load_data")

        return sorted(i for i in ideas if i)

    
    def _extract_synthesis(self, content: str) -> str:
        """Ekstraktuje syntezę z odpowiedzi aggregatora"""
        # Szukaj "synthesis_reasoning" w JSON
        try:
            data = json.loads(content) if isinstance(content, str) else content
            return data.get("synthesis_reasoning", "")
        except:
            return ""

    def _extract_verdict(self, content: str) -> str:
        """Ekstraktuje werdykt z odpowiedzi krytyka"""
        content_str = str(content)

        if "ZATWIERDZONY" in content_str:
            return "ZATWIERDZONY"
        return "ODRZUCONY"

    def _extract_score(self, content: str) -> float:
        """Ekstraktuje score z odpowiedzi krytyka"""
        content_str = str(content)
        try:
            import re

            score_match = re.search(r'"Overall_Quality_Q":\s*([\d.]+)', content_str)
            if score_match:
                return float(score_match.group(1))
        except:
            pass
        return 0.0

    def _extract_weaknesses(self, content: str) -> List[str]:
        """Ekstraktuje weaknesses z odpowiedzi krytyka"""
        weaknesses = []
        try:
            if isinstance(content, dict):
                data = content
            else:  # Jeśli nie, spróbuj sparsować go jako JSON
                data = json.loads(str(content))

            weak_list = data.get("critique_summary", {}).get(
                "identified_weaknesses", []
            )
            for w in weak_list:
                if isinstance(w, dict):
                    weaknesses.append(w.get("weakness", ""))
                else:
                    weaknesses.append(str(w))
        except:
            pass
        return weaknesses

    def add_successful_plan(self, plan: Dict[str, Any], mission: str, metadata: Dict):
        """Zapisuje udany plan do pamięci proceduralnej"""
        strategy = {
            "mission_type": self._classify_mission(mission),
            "plan_structure": self._extract_plan_structure(plan),
            "success_factors": metadata.get("success_factors", []),
            "performance_metrics": metadata.get("metrics", {}),
            "timestamp": datetime.now().isoformat(),
        }

        self.successful_strategies.append(strategy)
        self._persist_lightweight_memory()  # Zapisz od razu

        # Loguj dodanie udanego planu
        process_log(
            f"[MEMORY] Added successful plan for mission_type={strategy['mission_type']}, "
            f"nodes={strategy['plan_structure']['num_nodes']}"
        )

    def _classify_mission(self, mission: str) -> str:
        """Klasyfikuje typ misji"""
        mission_lower = mission.lower()

        if "przyczynow" in mission_lower or "causal" in mission_lower:
            return "causal_analysis"
        elif (
            "dane" in mission_lower or "data" in mission_lower or "csv" in mission_lower
        ):
            return "data_processing"
        elif "model" in mission_lower:
            return "model_validation"
        elif "optymali" in mission_lower:
            return "optimization"
        else:
            return "general"

    def _extract_plan_structure(self, plan: Dict) -> Dict:
        """Ekstraktuje strukturalne cechy planu"""
        return {
            "num_nodes": len(plan.get("nodes", [])),
            "num_edges": len(plan.get("edges", [])),
            "has_error_handling": any(
                "error" in str(node).lower() for node in plan.get("nodes", [])
            ),
            "has_validation": any(
                "valid" in str(node).lower() for node in plan.get("nodes", [])
            ),
            "graph_complexity": self._calculate_complexity(plan),
        }

    def _calculate_complexity(self, plan: Dict) -> float:
        """Oblicza złożoność grafu"""
        nodes = len(plan.get("nodes", []))
        edges = len(plan.get("edges", []))

        if nodes == 0:
            return 0.0

        # Złożoność cyklomatyczna aproksymowana
        return (edges - nodes + 2) / nodes

    def _identify_critical_moments(self, messages: List[Dict]) -> List[Dict]:
        """Identyfikuje krytyczne momenty w debacie"""
        critical = []
        for i, msg in enumerate(messages):
            content_str = str(msg.get("content", ""))  # Konwertujemy na string
            content = content_str.lower()

            # Moment krytyczny = duża zmiana w score lub verdict
            if "zatwierdzony" in content or "odrzucony" in content:
                critical.append(
                    {
                        "index": i,
                        "type": "verdict",
                        "agent": msg.get("name"),
                        "summary": "Decyzja krytyka",
                    }
                )
        return critical

    def _extract_final_score(self, messages: List[Dict]) -> float:
        """Znajduje finalny score z ostatniej odpowiedzi krytyka"""
        for msg in reversed(messages):
            if "critic" in msg.get("name", "").lower():
                score = self._extract_score(msg.get("content", ""))
                if score > 0:
                    return score
        return 0.0

    def _track_critique_evolution(self, iterations: List[Dict]) -> List[Dict]:
        """Śledzi jak zmieniała się krytyka między iteracjami"""
        evolution = []
        for i, iteration in enumerate(iterations):
            if iteration.get("critic"):
                evolution.append(
                    {
                        "iteration": i,
                        "score": iteration["critic"].get("score", 0),
                        "verdict": iteration["critic"].get("verdict", ""),
                        "main_issues": iteration["critic"].get("weaknesses", [])[:2],
                    }
                )
        return evolution

    def _extract_aggregator_reasoning(self, messages: List[Dict]) -> str:
        """Wyciąga reasoning agregatora"""
        for msg in reversed(messages):
            if "aggregator" in msg.get("name", "").lower():
                return self._extract_synthesis(msg.get("content", ""))
        return ""

    def _analyze_proposer_contributions(
        self, messages: List[Dict]
    ) -> Dict[str, List[str]]:
        """Analizuje wkład każdego proposera"""
        contributions = {}
        for msg in messages:
            name = msg.get("name", "")
            if any(role in name.lower() for role in ["analyst", "planner", "proposer"]):
                if name not in contributions:
                    contributions[name] = []
                ideas = self._extract_key_ideas(msg.get("content", ""))
                contributions[name].extend(ideas)
        return contributions

    def _extract_patterns_from_debate(self, messages: List[Dict]) -> List[str]:
        """Ekstraktuje wzorce z całej debaty"""
        patterns = []

        all_text = " ".join(str(m.get("content", "")) for m in messages).lower()

        # Szukaj powtarzających się konceptów
        all_text = " ".join(m.get("content", "") for m in messages).lower()

        if all_text.count("error_handler") > 3:
            patterns.append("Częste odniesienia do obsługi błędów")
        if all_text.count("rollback") > 2:
            patterns.append("Rollback jako kluczowy element")
        if all_text.count("optimiz") > 2:
            patterns.append("Focus na optymalizację")

        return patterns

    def _identify_success_factors(
        self, final_plan: Dict, iterations: List[Dict]
    ) -> List[str]:
        """Identyfikuje co przyczyniło się do sukcesu"""
        factors = []

        if final_plan:
            # Analiza struktury planu
            if any("error" in str(n).lower() for n in final_plan.get("nodes", [])):
                factors.append("Comprehensive error handling")
            if any("valid" in str(n).lower() for n in final_plan.get("nodes", [])):
                factors.append("Data validation steps")

            # Analiza iteracji
            if len(iterations) > 1:
                factors.append(f"Iterative improvement ({len(iterations)} rounds)")

        return factors

    def _identify_failure_points(self, iterations: List[Dict]) -> List[Dict]:
        """Identyfikuje gdzie były problemy"""
        failures = []
        for i, iteration in enumerate(iterations):
            if iteration.get("critic", {}).get("verdict") == "ODRZUCONY":
                failures.append(
                    {
                        "iteration": i,
                        "issues": iteration["critic"].get("weaknesses", []),
                        "score": iteration["critic"].get("score", 0),
                    }
                )
        return failures

    def _identify_turning_points(self, iterations: List[Dict]) -> List[Dict]:
        """Znajduje punkty zwrotne w debacie"""
        turning_points = []
        prev_score = 0

        for i, iteration in enumerate(iterations):
            curr_score = iteration.get("critic", {}).get("score", 0)
            if curr_score - prev_score > 20:  # Duży skok w score
                turning_points.append(
                    {
                        "iteration": i,
                        "score_jump": curr_score - prev_score,
                        "reason": "Significant improvement",
                    }
                )
            prev_score = curr_score

        return turning_points

    def _calculate_convergence_rate(self, iterations: List[Dict]) -> float:
        """Oblicza jak szybko system doszedł do rozwiązania"""
        if not iterations:
            return 0.0

        scores = [it.get("critic", {}).get("score", 0) for it in iterations]
        if len(scores) < 2:
            return 1.0

        # Średni przyrost score na iterację
        improvements = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        # Normalizuj do 0-1 (im wyższy przyrost, tym lepsza convergence)
        return min(
            avg_improvement / 20, 1.0
        )  # 20 punktów na iterację = max convergence

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analizuje wzorce czasowe w performance systemu"""
        from datetime import datetime

        patterns = {"by_weekday": {}, "by_hour": {}, "by_day_hour": {}}

        if not self.full_mission_records:
            return patterns

        # Analiza per dzień tygodnia
        for record in self.full_mission_records:
            timestamp = datetime.fromisoformat(record["timestamp"])
            weekday = timestamp.strftime("%A")
            hour = timestamp.hour
            day_hour = f"{weekday}_{hour:02d}h"

            # Per weekday
            if weekday not in patterns["by_weekday"]:
                patterns["by_weekday"][weekday] = {
                    "missions": [],
                    "avg_score": 0,
                    "avg_iterations": 0,
                    "common_issues": [],
                }

            patterns["by_weekday"][weekday]["missions"].append(record["memory_id"])

            # Per hour
            if hour not in patterns["by_hour"]:
                patterns["by_hour"][hour] = {
                    "missions": [],
                    "avg_score": 0,
                    "avg_iterations": 0,
                }

            patterns["by_hour"][hour]["missions"].append(record["memory_id"])

            # Per day+hour combo
            if day_hour not in patterns["by_day_hour"]:
                patterns["by_day_hour"][day_hour] = {"missions": [], "scores": []}

            patterns["by_day_hour"][day_hour]["missions"].append(record["memory_id"])
            patterns["by_day_hour"][day_hour]["scores"].append(
                record.get("final_score", 0)
            )

        # Oblicz średnie
        for weekday, data in patterns["by_weekday"].items():
            if data["missions"]:
                scores = [
                    r["final_score"]
                    for r in self.full_mission_records
                    if r["memory_id"] in data["missions"]
                ]
                data["avg_score"] = sum(scores) / len(scores) if scores else 0

        return patterns

    def get_current_context_hints(self) -> str:
        """Zwraca wskazówki kontekstowe na podstawie aktualnego czasu"""
        from datetime import datetime

        now = datetime.now()
        patterns = self.analyze_temporal_patterns()

        hints = []

        # Sprawdź wzorce dla aktualnego dnia
        weekday = now.strftime("%A")
        if weekday in patterns["by_weekday"]:
            weekday_data = patterns["by_weekday"][weekday]
            if weekday_data["avg_score"] < 90:
                hints.append(
                    f"Uwaga: {weekday} historycznie mają niższe score ({weekday_data['avg_score']:.1f})"
                )

        # Sprawdź wzorce dla aktualnej godziny
        hour = now.hour
        if hour in patterns["by_hour"]:
            hour_data = patterns["by_hour"][hour]
            if len(hour_data["missions"]) > 2:  # Jeśli mamy wystarczająco danych
                hints.append(
                    f"O godzinie {hour}:00 zazwyczaj wykonywane są misje tego typu"
                )

        return " | ".join(hints) if hints else ""
