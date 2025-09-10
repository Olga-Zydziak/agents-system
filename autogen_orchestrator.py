"""
Pełny orchestrator MOA używający AutoGen do zarządzania debatą agentów
"""
import json
from datetime import datetime
import config_api
from autogen import UserProxyAgent, ConversableAgent, GroupChat, GroupChatManager
from typing import Dict, List, Any, Optional
from datetime import datetime
import autogen
from google.cloud import secretmanager
from models_config import AgentRole
from moa_prompts import MOAPrompts
from memory_system import ContextMemory
# Używamy structured parsera zamiast heurystycznego response_parser
from structured_response_parser import StructuredResponseParser
from process_logger import log as process_log
import os, json, time, traceback
from typing import Any, Dict, Optional
from process_logger import log as process_log
import vertexai
from process_logger import log_exception as log_exc
from config_api import basic_config_agent
from exporter_missions_gcs import export_local_by_filename_date
from explainability_layer import ExplainabilityHooks
from memory_helpers import save_mission_to_gcs
EXPLAINABILITY = ExplainabilityHooks()

class AutoGenMOAOrchestrator:
    """
    Orchestrator systemu MOA używający AutoGen do wieloturowej debaty
    """
    
    def __init__(self, mission: str, node_library: Dict[str, Any], config_file: str = "agents_config.json"):
        self.mission = mission
        self.node_library = node_library
        self.memory = ContextMemory(
        max_episodes=50,
        gcs_bucket=None,   # ten sam bucket co w Memory_bank.ipynb
        gcs_prefix=""                   # opcjonalnie; usuń lub zostaw ""
        )
        # Parser oparty na Pydantic – oczekuje czystego JSON zgodnego ze schematem
        self.parser = StructuredResponseParser()
        
        # Wczytaj konfigurację
        self._load_config(config_file)
        
        # Stan debaty
        self.iteration_count = 0
        self.max_iterations = 5
        self.current_context = {}
        self.final_plan = None
        self._forced_speaker: Optional[str] = None
        # Inicjalizuj agentów AutoGen
        self.enable_sanity_ping = False
        process_log(f"[CFG] enable_sanity_ping={self.enable_sanity_ping}")
        self._initialize_autogen_agents()
        self._secret_cache = {}
        
        process_log(f"=== AutoGen MOA Orchestrator initialized for mission: {mission[:100]}... ===")
    
    
    #nowe helpery
    
    @staticmethod
    def _valid_memory_json(js: dict) -> bool:
        if not isinstance(js, dict):
            return False
        required = ("recommended_strategies", "common_pitfalls", "examples", "notes")
        if any(k not in js for k in required):
            return False
        return (
            isinstance(js["recommended_strategies"], list) and
            isinstance(js["common_pitfalls"], list) and
            isinstance(js["examples"], list) and
            isinstance(js["notes"], str)
        )

    
    def _extract_name_and_len(self, msg):
        """
        Zwraca (name, content_len, type_name) z wiadomości w wielu możliwych formatach:
        - dict z kluczami name/content/sender/role
        - list/tuple w stylu [role, name, content] lub [name, content]
        - obiekt z atrybutami .name/.content
        - plain string
        """
        try:
            tname = type(msg).__name__
            # 1) dict
            if isinstance(msg, dict):
                name = msg.get("name") or msg.get("sender") or msg.get("role") or None
                content = msg.get("content") or msg.get("text") or ""
                return name, len(content or ""), tname

            # 2) list/tuple – najczęstsze warianty
            if isinstance(msg, (list, tuple)):
                if len(msg) >= 3:
                    # [role, name, content]
                    name = str(msg[1])
                    content = str(msg[2])
                elif len(msg) == 2:
                    # [name, content] lub [role, content]
                    name = str(msg[0])
                    content = str(msg[1])
                elif len(msg) == 1:
                    name = None
                    content = str(msg[0])
                else:
                    name, content = None, ""
                return name, len(content or ""), tname

            # 3) obiekt z atrybutami
            if hasattr(msg, "name") or hasattr(msg, "content"):
                name = getattr(msg, "name", None)
                content = getattr(msg, "content", "") or ""
                return name, len(content or ""), tname

            # 4) fallback: traktuj jako string
            s = str(msg)
            return None, len(s), tname

        except Exception:
            # ostateczny fallback
            return None, 0, "unknown"

        
        
        
    #liczenie uzycia pamieci
    
    def _log_memory_alignment(self, plan_text: str, phase: str):
        """
        Miękka telemetria: sprawdza, na ile 'plan_text' (JSON/tekst Aggregatora)
        pokrywa się z seedowaną pamięcią (self._last_seeded_memory).
        Nie modyfikuje promptów ani toku debaty – tylko loguje.
        """
        import json, re

        mem = getattr(self, "_last_seeded_memory", None)
        if not mem or not plan_text:
            process_log(f"[MEMORY ALIGNMENT] skipped (mem_or_plan_missing) phase={phase}")
            return

        # --- 1) wyciągnij czysty tekst planu (jeśli to JSON w stringu)
        txt = plan_text
        # spróbuj znaleźć największy blok JSON (ostrożny regex)
        try:
            match = re.search(r"\{[\s\S]*\}", plan_text)
            if match:
                txt = match.group(0)
        except Exception:
            pass

        # --- 2) zbuduj 'work_text' do prostych dopasowań (lowercase)
        work_text = txt.lower()

        # --- 3) źródło pamięci
        strategies = [s.strip().lower() for s in (mem.get("recommended_strategies") or []) if s and isinstance(s, str)]
        pitfalls   = [p.strip().lower() for p in (mem.get("common_pitfalls")       or []) if p and isinstance(p, str)]
        examples   = mem.get("examples") or []

        # --- 4) proste dopasowania substring (bez magii – chodzi o telemetrię, nie scoring naukowy)
        def _hitcount(items, text):
            hits = []
            for it in items:
                if len(it) >= 5 and it in text:  # mini-próg, żeby „csv” itp. nie łapać
                    hits.append(it)
            return hits

        used_strats   = _hitcount(strategies, work_text)
        addressed_pts = _hitcount(pitfalls,   work_text)

        # --- 5) examples: sprawdzimy, czy wystąpiły ID lub URI w planie
        ex_total = len(examples)
        ex_hits = 0
        for ex in examples:
            mid = str(ex.get("mission_id","")).strip().lower()
            uri = str(ex.get("plan_uri","")).strip().lower()
            if (mid and mid in work_text) or (uri and uri in work_text):
                ex_hits += 1

        # --- 6) prosty score 0..1 (wagi możesz zmienić)
        import math
        def frac(num, den): 
            return (num/den) if den else 0.0

        s_frac = frac(len(used_strats),   len(strategies))
        p_frac = frac(len(addressed_pts), len(pitfalls))
        e_frac = frac(ex_hits,            ex_total)

        score = round(0.5 * s_frac + 0.3 * p_frac + 0.2 * e_frac, 3)

        # --- 7) log + opcjonalna integracja z EXPLAINABILITY (jeśli jest)
        process_log(
            "[MEMORY ALIGNMENT] phase=%s score=%.3f "
            "strategies_used=%d/%d pitfalls_addressed=%d/%d examples_ref=%d/%d" % (
                phase, score,
                len(used_strats), len(strategies),
                len(addressed_pts), len(pitfalls),
                ex_hits, ex_total
            )
        )
        try:
            if "EXPLAINABILITY" in globals() and hasattr(EXPLAINABILITY, "on_memory_alignment"):
                EXPLAINABILITY.on_memory_alignment({
                    "phase": phase, "score": score,
                    "strategies_used": used_strats,
                    "pitfalls_addressed": addressed_pts,
                    "examples_hits": ex_hits,
                    "examples_total": ex_total
                })
        except Exception as e:
            process_log(f"[MEMORY ALIGNMENT][WARN] explainability hook failed: {type(e).__name__}: {e}")

        # --- 8) (opcjonalnie) krótki, niewpływający podgląd w transkrypcie
        try:
            if True:
            # if getattr(self.config, "show_memory_alignment_preview", False):
                preview = {
                    "phase": phase, "score": score,
                    "strategies_used": len(used_strats), "strategies_total": len(strategies),
                    "pitfalls_addressed": len(addressed_pts), "pitfalls_total": len(pitfalls),
                    "examples_ref": ex_hits, "examples_total": ex_total
                }
                # zapisz do historii – jako wiadomość Orchestratora (nie zmienia mówcy)
                self.groupchat.messages.append({
                    "role": "assistant", "name": "Orchestrator",
                    "content": f"[MEMORY ALIGNMENT PREVIEW] {preview}"
                })
        except Exception as e:
            process_log(f"[MEMORY ALIGNMENT][DEBUG] preview append failed: {type(e).__name__}: {e}")
    
    
    #koniec liczenia
    
    
    def _make_memory_message_once(self):
        """LIGHT -> walidacja -> MID fallback -> lokalny fallback (bez LLM). Zwraca pojedynczą wiadomość do historii czatu."""
        user_msg = {"role":"user","content": f"mission={self.mission}"}

        # 1) LIGHT (PRIMARY)
        if self.memory_analyst_agent:
            try:
                out = self.memory_analyst_agent.generate_reply(messages=[user_msg])
                try: js = json.loads(out)
                except Exception: js = None
                if self._valid_memory_json(js) and (len(js["recommended_strategies"])>=2 or len(js["examples"])>=1):
                    self._last_seeded_memory = js
                    process_log("[MEMORY][LIGHT] Valid memory JSON produced")
                    return {"name":"Memory_Analyst","role":"assistant","content": out}
            except Exception as e:
                log_exc(f"[MEMORY] LIGHT failed:", e)

            # 2) MID (FALLBACK)
            if self.memory_analyst_fallback_model:
                try:
                    mid_llm = self._build_llm_config(self.memory_analyst_fallback_model)
                    tmp = autogen.ConversableAgent(
                        name="Memory_Analyst_MID",
                        llm_config=mid_llm,
                        system_message=MOAPrompts.get_memory_analyst_prompt(),
                        human_input_mode="NEVER"
                    )
                    out = tmp.generate_reply(messages=[user_msg])
                    try: js = json.loads(out)
                    except Exception: js = None
                    if self._valid_memory_json(js):
                        self._last_seeded_memory = js
                        process_log("[MEMORY][MID] Valid memory JSON produced")
                        return {"name":"Memory_Analyst","role":"assistant","content": out}
                except Exception as e:
                    log_exc(f"[MEMORY] MID failed:", e)

        # 3) Fallback bez LLM — lokalny + (opcjonalnie) Vertex
        try:
            ctx_local = self.memory.get_relevant_context(self.mission) if self.memory else {}
            get_v = getattr(self.memory, "get_vertex_context", None)
            ctx_vertex = get_v(self.mission) if callable(get_v) else {}
            tips = list(dict.fromkeys(
                (ctx_local.get("recommended_strategies") or []) +
                (ctx_vertex.get("recommended_strategies") or [])
            ))[:6]
            examples = (ctx_vertex.get("examples") or [])[:3]
            minimal = {
                "recommended_strategies": tips,
                "common_pitfalls": ctx_local.get("common_pitfalls") or [],
                "examples": [{"mission_id": e.get("mission_id",""), "plan_uri": e.get("plan_uri","")} for e in examples],
                "notes": ""
            }
            self._last_seeded_memory = minimal
            return {"name":"Memory_Analyst","role":"assistant","content": json.dumps(minimal, ensure_ascii=False)}
        except Exception as e:
            log_exc(f"[MEMORY] local fallback failed:",e)
            return {"name":"Memory_Analyst","role":"assistant","content": '{"recommended_strategies":[],"common_pitfalls":[],"examples":[],"notes":""}'}
    
    #koniec nowych helperow
    
    
    
    def reset(self):
        # ... resetuje liczniki ...
        # Używa wbudowanej metody .reset() do wyczyszczenia historii każdego agenta
        all_agents = [self.user_proxy, *self.proposer_agents, self.aggregator_agent, self.critic_agent]
        for agent in all_agents:
            if agent:
                agent.reset()
    
    def _get_api_key_from_gcp_secret_manager(self, model_cfg: Dict) -> str | None:
        """
        Czyta klucz z GCP Secret Manager.
        Oczekuje: model_cfg["secret_manager"] = {"project_id": "...", "secret_id": "...", "version": "latest"|"1"|...}
        Zwraca: string lub None (gdy brak/nieudane).
        """
        sm = model_cfg.get("secret_manager") or {}
        project_id = (sm.get("project_id") or "").strip()
        secret_id  = (sm.get("secret_id") or "").strip()
        version    = (sm.get("version") or "latest").strip()

        if not project_id or not secret_id:
            return None

        cache_key = (project_id, secret_id, version)
        if cache_key in self._secret_cache:
            return self._secret_cache[cache_key]

        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
            resp = client.access_secret_version(name=name)
            value = resp.payload.data.decode("utf-8")
            # cache in-memory (nie logujemy!)
            self._secret_cache[cache_key] = value
            return value
        except Exception as e:
            # Nie loguj wartości sekretu. Możesz zalogować TYLKO metadane.
            from process_logger import log as process_log
            process_log(f"[SECRETS] Failed to read {secret_id}@{project_id}/{version}: {type(e).__name__}: {e}")
            return None
    
    
    #pamiec:
    
    def _build_memory_analyst_message(self) -> str:
        ctx_local = self.memory.get_relevant_context(self.mission) if self.memory else {}
        get_v = getattr(self.memory, "get_vertex_context", None)
        ctx_vertex = get_v(self.mission) if callable(get_v) else {}


        tips = []
        tips += ctx_local.get("recommended_strategies", []) or []
        tips += ctx_vertex.get("recommended_strategies", []) or []


        seen, dedup = set(), []
        for t in tips:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        tips = dedup[:8]


        examples = (ctx_vertex.get("examples") or [])[:3]
        lines = ["# MemoryAnalyst Summary"]
        if tips:
            lines.append("## Recommended strategies:")
            lines += [f"- {t}" for t in tips]
        if examples:
            lines.append("\n## Example plans:")
            for e in examples:
                mid = e.get("mission_id") or "unknown"
                puri = e.get("plan_uri") or ""
                lines.append(f"- {mid}: {puri}")


        return "\n".join(lines) if len(lines) > 1 else "# MemoryAnalyst Summary\n(no relevant memory found)"
    
    
    #Raport:
    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _now_stamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def _extract_llm_hint(self, text: str) -> Optional[str]:
        """Prosta heurystyka do rozpoznawania typowych problemów LLM-a."""
        if not text:
            return None
        
        #poprawka:
        if not isinstance(text, str):
            try:
                import json as _json
                text = _json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)
        
        
        #koniec_poprawki
        
        
        
        t = text.lower()
        hints = {
            "quota/rate_limit": ["rate limit", "too many requests", "quota", "insufficient_quota"],
            "context_length": ["maximum context length", "token limit", "context window", "too many tokens"],
            "safety": ["safety", "blocked", "content filter"],
            "auth/api": ["invalid api key", "unauthorized", "forbidden", "permission"],
            "timeout": ["timeout", "timed out", "deadline exceeded"]
        }
        for label, kws in hints.items():
            if any(k in t for k in kws):
                return label
        return None

    def _write_failure_report(
        self,
        reason: str,
        stage: str,
        aggregator_raw: Optional[str],
        critic_raw: Optional[str],
        exception: Optional[BaseException] = None,
        parsed_aggregator: Optional[Dict[str, Any]] = None
    ) -> str:
        
        
        #poprawka:
        def _safe_str(x):
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            try:
                import json as _json
                return _json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)
        
        
        
        #koniec poprawki:
        
        """Zapisuje raport awaryjny JSON + MD i zwraca ścieżkę do pliku JSON."""
        self._ensure_dir("reports")
        ts = self._now_stamp()
        jpath = f"reports/failure_report_{ts}.json"
        mpath = f"reports/failure_report_{ts}.md"

        agg_hint = self._extract_llm_hint(aggregator_raw or "")
        crit_hint = self._extract_llm_hint(critic_raw or "")

        report = {
            "timestamp": ts,
            "mission": self.mission,
            "stage": stage,  # np. "aggregator", "groupchat", "critic"
            "reason": reason,  # np. "AGGREGATOR_NO_VALID_JSON", "EXCEPTION_DURING_DEBATE"
            "aggregator_model": getattr(self, "aggregator_config", {}).get("model", {}),
            "critic_model": getattr(self, "critic_config", {}).get("model", {}),
            "aggregator_output_excerpt": _safe_str(aggregator_raw or "")[:4000],
            "critic_output_excerpt": _safe_str(critic_raw or "")[:4000],
            "aggregator_llm_hint": agg_hint,
            "critic_llm_hint": crit_hint,
            "parsed_aggregator": parsed_aggregator,
            "exception": None if not exception else {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        }

        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # krótkie MD dla ludzi
        with open(mpath, "w", encoding="utf-8") as f:
            f.write(f"# Failure Report ({ts})\n\n")
            f.write(f"**Mission:** {self.mission}\n\n")
            f.write(f"**Stage:** {stage}\n\n")
            f.write(f"**Reason:** {reason}\n\n")
            if agg_hint:
                f.write(f"**Aggregator LLM hint:** `{agg_hint}`\n\n")
            if crit_hint:
                f.write(f"**Critic LLM hint:** `{crit_hint}`\n\n")
            if exception:
                f.write(f"**Exception:** `{type(exception).__name__}: {exception}`\n\n")
            f.write("## Last Aggregator Output (excerpt)\n\n")
            f.write("```\n" +  _safe_str(aggregator_raw or "")[:4000] + "\n```\n\n")
            f.write("## Last Critic Output (excerpt)\n\n")
            f.write("```\n" +  _safe_str(critic_raw or "")[:4000] + "\n```\n")

        
        process_log(f"[FAILSAFE] Saved failure report: {jpath}")

        return jpath

    def _get_last_message_from(self, groupchat, agent_name: str) -> Optional[str]:
        """Zwraca tekst ostatniej wiadomości danego agenta z obiektu GroupChat."""
        try:
            msgs = getattr(groupchat, "messages", [])
            for m in reversed(msgs):
                if (m.get("name") or m.get("role")) == agent_name:
                    return m.get("content") or ""
        except Exception:
            pass
        return None
    
    
    # ========== UNIVERSAL JSON REPAIR ==========

    MAX_REPAIR_ATTEMPTS = 2
    REPAIR_JSON_SUFFIX = "\n\nZWRÓĆ TYLKO I WYŁĄCZNIE JSON, bez komentarzy, bez dodatkowego tekstu."

    def _schema_example_for(self, role: str) -> str:
        if role == "proposer":
            return (
                '{\n'
                '  "thought_process": ["Krok 1...", "Krok 2..."],\n'
                '  "plan": {\n'
                '    "entry_point": "Start_Node",\n'
                '    "nodes": [ {"name":"Start_Node","implementation":"load_data"} ],\n'
                '    "edges": [ {"from":"Start_Node","to":"Next_Node","condition":"on_success"} ]\n'
                '  },\n'
                '  "confidence": 0.80\n'
                '}'
            )
        if role == "aggregator":
            return (
                '{\n'
                '  "thought_process": ["Agreguję elementy X i Y..."],\n'
                '  "final_plan": {\n'
                '    "entry_point": "Start_Node",\n'
                '    "nodes": [ {"name":"Start_Node","implementation":"load_data"} ],\n'
                '    "edges": [ {"from":"Start_Node","to":"Next_Node","condition":"on_success"} ]\n'
                '  },\n'
                '  "confidence_score": 0.90\n'
                '}'
            )
        if role == "critic":
            return (
                '{\n'
                '  "critique_summary": {\n'
                '    "verdict": "ZATWIERDZONY",\n'
                '    "statement": "Uzasadnienie...",\n'
                '    "key_strengths": ["..."],\n'
                '    "identified_weaknesses": [{"weakness":"...", "severity":"Low", "description":"..."}]\n'
                '  },\n'
                '  "quality_metrics": {\n'
                '    "Complexity_Score_C": 3.1,\n'
                '    "Robustness_Score_R": 50,\n'
                '    "Innovation_Score_I": 100,\n'
                '    "Completeness_Score": 100,\n'
                '    "Overall_Quality_Q": 84.07\n'
                '  },\n'
                '  "final_synthesized_plan": {\n'
                '    "entry_point": "Start_Node",\n'
                '    "nodes": [ {"name":"Start_Node","implementation":"load_data"} ],\n'
                '    "edges": [ {"from":"Start_Node","to":"Next_Node","condition":"on_success"} ]\n'
                '  }\n'
                '}'
            )
        return "{}"

    def _try_parse_by_role(self, role: str, text: str):
        try:
            if role == "proposer":
                parsed = self.parser.parse_agent_response(text)
            elif role == "aggregator":
                parsed = self.parser.parse_aggregator_response(text)
            elif role == "critic":
                parsed = self.parser.parse_critic_response(text)
            else:
                return None, f"Unknown role: {role}"
            if parsed:
                return parsed, None
            return None, "Parser returned None"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _repair_prompt_for(self, role: str, err_msg: str) -> str:
        return (
            f"Twoja poprzednia odpowiedź NIE SPEŁNIA wymaganego schematu dla roli '{role}'.\n"
            f"Błąd/diagnoza parsera: {err_msg}\n\n"
            f"Wymagana struktura JSON (minimalny przykład):\n{self._schema_example_for(role)}\n"
            f"{REPAIR_JSON_SUFFIX}"
        )

    def _force_one_turn(self, agent, manager) -> str:
        self._forced_speaker = agent.name
        try:
            manager.step()  # jeśli Twoja wersja AG2 nie wspiera .step(), użyj run(max_round=1)
        except Exception:
            pass
        return self._get_last_message_from(manager.groupchat, agent.name) or ""

    def _auto_repair_and_parse(self, role: str, agent, manager, last_text: str):
        parsed, err = self._try_parse_by_role(role, last_text or "")
        if parsed:
            return parsed
        
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            repair_msg = self._repair_prompt_for(role, err or "Invalid JSON")
            manager.groupchat.messages.append({
                "role": "user",
                "name": "Orchestrator",
                "content": repair_msg
            })
            process_log(f"[REPAIR][{role}] attempt {attempt}: requesting strictly JSON output.")
            repaired_text = self._force_one_turn(agent, manager)
            parsed, err2 = self._try_parse_by_role(role, repaired_text or "")
            if parsed:
                return parsed
            err = err2
        return None
    
    
    
    def _load_config(self, config_file: str):
        """Wczytuje konfigurację agentów"""
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    
    def _is_final_plan_message(self, m: dict) -> bool:
        """Kończymy TYLKO na odpowiedzi CRITICA, gdy kończy się markerem."""
        content = (m.get("content") or "").strip()
        name = (m.get("name") or "").lower()
        role = (m.get("role") or "").lower()
        return role == "assistant" and content.endswith("PLAN_ZATWIERDZONY") and "critic" in name
    
    

    def custom_speaker_selection_logic(self, last_speaker, groupchat):
        """
        Proposers → Aggregator → Critic. Jeśli Critic nie zatwierdzi, nowa iteracja od pierwszego Proposera.
        Porównujemy po NAZWACH z historii wiadomości (AutoGen może podawać inne instancje agentów).
        """
        msgs = groupchat.messages
        
        for msg in msgs:
            if "PLAN_ZATWIERDZONY" in msg.get("content", ""):
                raise StopIteration("Plan zatwierdzony - kończymy debatę")
        
        last_name = (msgs[-1].get("name") or "").lower() if msgs else ""
        last_content = (msgs[-1].get("content") or "")

        if last_name and last_content:
            # Znajdź prompt który wywołał tę odpowiedź
            # To będzie przedostatnia wiadomość lub początkowy bootstrap
            prompt_for_last = ""
            if len(msgs) >= 2:
                prompt_for_last = msgs[-2].get("content", "")
            elif len(msgs) == 1:
                # Pierwsza odpowiedź - prompt to bootstrap z run_full_debate_cycle
                prompt_for_last = getattr(self, '_initial_bootstrap', '')

            # Zapisz wyjaśnialność
            EXPLAINABILITY.on_response_received(
                prompt=prompt_for_last,
                response=last_content,
                agent_name=last_name
            )
        
        
        
        # --- [DODANE] Memory Analyst ma tylko jedną wypowiedź na starcie ---
        ma_name = (getattr(self, "memory_analyst_agent", None).name or "").lower() \
                  if getattr(self, "memory_analyst_agent", None) else ""
        if last_name == ma_name:
            # Po jednorazowym komunikacie pamięci przechodzimy do pierwszego Proposera
            return self.proposer_agents[0]

        # ❶ Po bootstrapie (ostatni był Orchestrator → wybieramy pierwszego proposera)
        if last_name == (self.user_proxy.name or "").lower() and last_content.strip():
            return self.proposer_agents[0]

        # ❷ Po Aggregatorze → czas na Critica
        if last_name == (self.aggregator_agent.name or "").lower():
            try:
                # zapamiętaj ostatni tekst planu do późniejszego użytku (np. post-approval)
                self._last_aggregated_plan_text = last_content
                # policz adhezję tuż po agregacji
                self._log_memory_alignment(last_content, phase="post_aggregation")
            except Exception as e:
                process_log(f"[MEMORY ALIGNMENT][WARN] post_aggregation failed: {type(e).__name__}: {e}")
            return self.critic_agent

        # ❸ Po Criticu → zatwierdzenie albo nowa iteracja
        if last_name == (self.critic_agent.name or "").lower():
            self._save_iteration_to_memory(last_content, self.iteration_count)
            if "PLAN_ZATWIERDZONY" in last_content:
                try:
                    plan_text = getattr(self, "_last_aggregated_plan_text", "") or last_content
                    self._log_memory_alignment(plan_text, phase="post_approval")
                except Exception as e:
                    process_log(f"[MEMORY ALIGNMENT][WARN] post_approval failed: {type(e).__name__}: {e}")
                return None
                
                
                
                
                # return None
            # nowa iteracja
            self.iteration_count += 1
            if self.iteration_count >= self.max_iterations:
                process_log(f"[FAILSAFE] Osiągnięto maksymalną liczbę iteracji ({self.max_iterations}). Koniec debaty.")
                return None
            process_log(f"===== ROZPOCZYNAM ITERACJĘ DEBATY NR {self.iteration_count + 1} =====")
            self._update_context_from_last_critique(last_content)
            return self.proposer_agents[0]

        # ❹ Wewnątrz puli proposerów – leć kolejno po nazwach
        proposer_names = [p.name.lower() for p in self.proposer_agents]
        if last_name in proposer_names:
            idx = proposer_names.index(last_name)
            if idx < len(self.proposer_agents) - 1:
                return self.proposer_agents[idx + 1]
            return self.aggregator_agent  # po ostatnim proposerze mówi Aggregator

        # ❺ Domyślnie – zacznij od pierwszego proposera
        return self.proposer_agents[0]
    
    
    def _save_iteration_to_memory(self, critic_response: str, iteration: int):
        """Zapisuje dane z iteracji do pamięci"""
        try:
            
            if not self.memory:
                return
            
            # Parse odpowiedzi krytyka
            parsed = self.parser.parse_critic_response(critic_response)
            if not parsed:
                process_log(f"[MEMORY] Nie mogę sparsować odpowiedzi krytyka w iteracji {iteration}")
                return

            # Wyciągnij kluczowe dane
            score = parsed.get("quality_metrics", {}).get("Overall_Quality_Q", 0)
            weaknesses = parsed.get("critique_summary", {}).get("identified_weaknesses", [])
            verdict = parsed.get("critique_summary", {}).get("verdict", "")

            # Stwórz feedback string
            feedback_data = {
                "score": score,
                "verdict": verdict,
                "weaknesses": [w.get("weakness", "") for w in weaknesses if isinstance(w, dict)],
                "iteration": iteration
            }

            # ZAPISZ DO PAMIĘCI
            self.memory.add_iteration_feedback(
                iteration=iteration,
                feedback=json.dumps(feedback_data),
                timestamp=datetime.now()
            )

            process_log(f"[MEMORY] Zapisano iterację {iteration}: score={score}, verdict={verdict}")

        except Exception as e:
            process_log(f"[MEMORY ERROR] Błąd zapisu iteracji {iteration}: {e}")

    
    def _initialize_autogen_agents(self):
        """Inicjalizuje agentów AutoGen dla debaty — minimalistycznie i niezawodnie."""
        # Atrybuty ZAWSZE istnieją
        self.proposer_agents = []
        self.aggregator = None
        self.critic = None
        self.aggregator_agent = None
        self.critic_agent = None
        self.memory_analyst_agent = None
        self.memory_analyst_fallback_model = None
        
        
        # User Proxy – nigdy nie kończy rozmowy
        self.user_proxy = autogen.ConversableAgent( # <-- POPRAWKA
        name="Orchestrator",
        human_input_mode="NEVER",
        llm_config=False,  # Ten agent nie potrzebuje LLM, tylko rozpoczyna rozmowę
        system_message="You are the orchestrator who starts the debate and then observes."
        )
        self.user_proxy.silent = False
        process_log("[INIT] UserProxy initialized")

        # Proposerzy
        for agent_config in self.config['agents']:
            rn = agent_config['role_name'].lower()
            if 'aggregator' in rn or 'critic' in rn or 'memory analyst' in rn:
                continue
            role = AgentRole(
                role_name=agent_config['role_name'],
                expertise_areas=agent_config['expertise_areas'],
                thinking_style=agent_config['thinking_style']
            )
            prompt = self._build_proposer_prompt(role)
            ag = autogen.ConversableAgent(
                name=agent_config['role_name'].replace(" ", "_"),
                llm_config=self._build_llm_config(agent_config['model']),
                system_message=prompt,
                human_input_mode="NEVER"
            )
            ag.silent = False
            self.proposer_agents.append(ag)
            process_log(f"[INIT] Proposer initialized: {ag.name}")

        #memory agent
        
        # --- Memory Analyst (lekki agent, mówi tylko raz na starcie; prompt z MOAPrompts) ---
        mem_cfg = next((a for a in self.config['agents'] if 'memory analyst' in a['role_name'].lower()), None)
        if mem_cfg:
            try:
                mem_llm = self._build_llm_config(mem_cfg['model'])
                self.memory_analyst_agent = autogen.ConversableAgent(
                    name="Memory_Analyst",
                    llm_config=mem_llm,
                    system_message=MOAPrompts.get_memory_analyst_prompt(),
                    human_input_mode="NEVER"
                )
                self.memory_analyst_agent.silent = False
                self.memory_analyst_fallback_model = mem_cfg.get("fallback")  # dict lub None
                process_log("[INIT] Memory Analyst initialized")
            except Exception as e:
                process_log(f"[INIT][WARN] Memory Analyst init failed: {e}")
        else:
            process_log("[INIT] Memory Analyst not configured")
        
        
        
        #koniec memory agent
        
            
        # Aggregator
        aggregator_config = next((a for a in self.config['agents'] if 'aggregator' in a['role_name'].lower()), None)
        if aggregator_config:
            self.aggregator = autogen.ConversableAgent(
                name="Master_Aggregator",
                llm_config=self._build_llm_config(aggregator_config['model']),
                system_message=MOAPrompts.get_aggregator_prompt(),
                human_input_mode="NEVER",
                is_termination_msg=lambda m: False,
            )
        else:
            self.aggregator = autogen.ConversableAgent(
                name="Master_Aggregator",
                llm_config={"config_list": [{"model": "dummy", "api_type": "dummy"}]},
                system_message=MOAPrompts.get_aggregator_prompt(),
                human_input_mode="NEVER",
                is_termination_msg=lambda m: False,
            )
        self.aggregator.silent = False
        self.aggregator_agent = self.aggregator
        process_log("[INIT] Aggregator initialized")

        # Critic
        critic_config = next((a for a in self.config['agents'] if 'critic' in a['role_name'].lower()), None)
        if critic_config:
            self.critic = autogen.ConversableAgent(
                name="Quality_Critic",
                llm_config=self._build_llm_config(critic_config['model']),
                system_message=self._build_critic_prompt(),
                human_input_mode="NEVER",
                is_termination_msg=self._is_final_plan_message,
            )
        else:
            self.critic = autogen.ConversableAgent(
                name="Quality_Critic",
                llm_config={"config_list": [{"model": "dummy", "api_type": "dummy"}]},
                system_message=self._build_critic_prompt(),
                human_input_mode="NEVER",
                is_termination_msg=self._is_final_plan_message,
            )
        self.critic.silent = False
        self.critic_agent = self.critic
        process_log("[INIT] Critic initialized")

        process_log(f"Initialized {len(self.proposer_agents)} proposers, 1 aggregator, 1 critic using AutoGen")
    
    
    
    def reset(self):
        """
        Minimalny reset: czyści historię agentów i liczniki sesji,
        bez dotykania cache'u LLM i bez zmian w konfiguracji.
        """
        agents = []

        # Zbierz agentów, jeśli istnieją (nie zakładamy, że wszystkie są zainicjalizowane)
        if getattr(self, "user_proxy", None):           agents.append(self.user_proxy)
        if getattr(self, "proposer_agents", None):      agents.extend(self.proposer_agents)
        if getattr(self, "aggregator_agent", None):     agents.append(self.aggregator_agent)
        if getattr(self, "critic_agent", None):         agents.append(self.critic_agent)
        if getattr(self, "memory_analyst_agent", None): agents.append(self.memory_analyst_agent)

        # Czyść historie czatu agentów (jeśli wspierają .reset())
        for ag in agents:
            try:
                if hasattr(ag, "reset") and callable(ag.reset):
                    ag.reset()
            except Exception as e:
                process_log(f"[RESET][WARN] {getattr(ag,'name','<agent>')}: {e}")

        # Zresetuj licznik iteracji i bootstrap
        self.iteration_count = 0
        self._initial_bootstrap = ""

        # (opcjonalnie) miękko wyczyść lokalny kontekst pamięci, jeśli ma takie API
        try:
            mem = getattr(self, "memory", None)
            if mem and hasattr(mem, "reset") and callable(mem.reset):
                mem.reset()
            elif mem and hasattr(mem, "clear") and callable(mem.clear):
                mem.clear()
        except Exception as e:
            process_log(f"[RESET][WARN] memory: {e}")

        process_log("[RESET] Orchestrator state cleared")
    
    
    
    
    def _runtime_env_snapshot(self) -> dict:
        # tylko presence, bez wartości
        def present(k): return bool(os.getenv(k))
        return {
            "VERTEXAI_PROJECT": present("VERTEXAI_PROJECT") or present("GOOGLE_CLOUD_PROJECT") or present("GCP_PROJECT"),
            "VERTEXAI_LOCATION": present("VERTEXAI_LOCATION") or present("GOOGLE_CLOUD_REGION"),
            "ANTHROPIC_API_KEY": present("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": present("OPENAI_API_KEY"),
        }

    def _agent_signature(self, agent) -> dict:
        llm = getattr(agent, "llm_config", {})
        # wyciągamy pierwszy wpis z config_list dla krótkiego podpisu
        vendor = None; model = None
        try:
            entry = (llm.get("config_list") or [{}])[0]
            if "google" in entry:
                vendor = "google"; model = entry["google"].get("model")
            elif "anthropic" in entry:
                vendor = "anthropic"; model = entry["anthropic"].get("model")
            elif "openai" in entry:
                vendor = "openai"; model = entry["openai"].get("model")
            else:
                vendor = (entry.get("api_type") or "unknown")
                model = entry.get("model")
        except Exception:
            pass
        return {"name": getattr(agent, "name", "?"), "vendor": vendor, "model": model}

    def _sanity_ping_agent(self, agent) -> None:
        tmp_user = UserProxyAgent(
            "sanity_user", human_input_mode="NEVER", max_consecutive_auto_reply=1,
            is_termination_msg=lambda m: True, code_execution_config=False
        )
        try:
            tmp_user.initiate_chat(agent, message="Odpowiedz dokładnie słowem: PONG")
        except Exception as e:
            sig = self._agent_signature(agent)
            snap = self._runtime_env_snapshot()
            raise RuntimeError(
                f"[SANITY PING FAILED] agent={sig} | env={snap} | err={type(e).__name__}: {e}"
            ) from e
    
    
    
    
    
    
    def _build_llm_config(self, model_config: dict) -> dict:
        """
        Buduje llm_config dla AutoGen na bazie agents_config.json,
        używając config_api.basic_config_agent (ten sam format co w solo).
        Google => Vertex/ADC (bez api_key), Anthropic/OpenAI => klucze z ENV/SM.
        """
        from config_api import basic_config_agent, PROJECT_ID as DEFAULT_PROJECT_ID, LOCATION as DEFAULT_LOCATION

        # --- helpery ---
        DEFAULT_MODEL_BY_PROVIDER = {
            "google": "gemini-2.5-pro",
            "anthropic": "claude-3-7-sonnet",
            "openai": "gpt-4o-mini",
        }

        def _map_provider_to_api_type(provider: str) -> str:
            p = (provider or "google").strip().lower()
            return {
                "google": "google", "gemini": "google", "vertex": "google",
                "anthropic": "anthropic",
                "openai": "openai", "azure_openai": "openai",
            }.get(p, p)

        def _validate_provider_model_pair(api_type: str, model: str) -> None:
            m = (model or "").lower()
            if api_type == "google" and not m.startswith("gemini"):
                raise ValueError(f"Model '{model}' nie pasuje do providera 'google' (Vertex/Gemini).")
            if api_type == "anthropic" and not m.startswith("claude"):
                raise ValueError(f"Model '{model}' nie pasuje do 'anthropic'.")
            if api_type == "openai" and not ("gpt" in m or m.startswith("o")):
                raise ValueError(f"Model '{model}' nie wygląda na model OpenAI.")

        # 1) provider -> api_type
        api_type = _map_provider_to_api_type(model_config.get("provider"))

        # 2) model + sanity
        agent_name = model_config.get("model_name") or DEFAULT_MODEL_BY_PROVIDER.get(api_type, "gemini-2.5-pro")
        _validate_provider_model_pair(api_type, agent_name)

        # 3) projekt/region tylko dla Google/Vertex
        project_id = model_config.get("project_id") or DEFAULT_PROJECT_ID
        location   = model_config.get("location")   or DEFAULT_LOCATION
        if api_type == "google" and not project_id:
            raise RuntimeError("Vertex/Gemini: brak project_id. Ustaw VERTEXAI_PROJECT/GOOGLE_CLOUD_PROJECT albo podaj 'project_id' w agents_config.json.")

        # 4) api_key tylko dla nie-Google
        api_key_arg = None if api_type == "google" else model_config.get("api_key")

        # 5) wołamy Twój builder
        flat_list = basic_config_agent(
            agent_name = agent_name,
            api_type   = api_type,
            location   = (location if api_type == "google" else None),   # <-- KLUCZOWA ZMIANA
            project_id = (project_id if api_type == "google" else None), # <-- KLUCZOWA ZMIANA
            api_key    = api_key_arg,
        )
        if not isinstance(flat_list, list) or not flat_list:
            raise ValueError("basic_config_agent powinien zwrócić niepustą listę.")

        entry = dict(flat_list[0])  # kopia, żeby móc czyścić

        # 6) sanity: dla nie-Google WYTNJIJ project/location (gdyby kiedyś znów wpadły)
        if api_type != "google":
            entry.pop("project_id", None)
            entry.pop("location", None)

        # 7) finalny llm_config
        return {
            "config_list": [entry],
            "temperature": float(model_config.get("temperature", 0.0)),
            "seed": 42,
            "cache_seed": 42,
        }
       
    
    def _build_proposer_prompt(self, role: AgentRole) -> str:
        """Buduje prompt dla proposera z kontekstem"""
        base_prompt = MOAPrompts.get_proposer_prompt(role, self.mission, self.node_library)
        
        # Dodaj dynamiczny kontekst
        if self.current_context:
            context_injection = self._build_context_injection()
            base_prompt = base_prompt + "\n\n" + context_injection
            # return base_prompt + "\n\n" + context_injection
        
        base_prompt = EXPLAINABILITY.on_prompt_build(base_prompt, role.role_name)
        return base_prompt
    
    def _build_critic_prompt(self) -> str:
        """Buduje prompt dla krytyka"""
        base_prompt = MOAPrompts.get_critic_prompt()

        # Dodaj specjalną instrukcję o frazie kończącej i nowej strukturze JSON
        additional_instruction = """

        ## CRITICAL OUTPUT STRUCTURE
        - If you REJECT the plan, provide your standard critique with weaknesses and suggestions.
        - If you APPROVE the plan, your JSON response MUST contain a top-level key named `plan_approved`. Inside this key, you MUST place the complete, final, synthesized plan object. The other keys (like critique_summary) should still be present.

        Example of an APPROVED response structure:
        ```json
        {
          "critique_summary": {
            "verdict": "ZATWIERDZONY",
            "statement": "Plan jest doskonały, spełnia wszystkie wymagania.",
            ...
          },
          "plan_approved": {
            "entry_point": "Start_Node",
            "nodes": [ ... ],
            "edges": [ ... ]
          },
          ...
        }
        ```

        ## GOLDEN TERMINATION RULE
        If you approve the plan, you MUST end your ENTIRE response with the exact phrase on a new line, after the JSON block:
        PLAN_ZATWIERDZONY
        """
        return EXPLAINABILITY.on_prompt_build(base_prompt + additional_instruction, "Quality_Critic")
        # return base_prompt + additional_instruction
    
    def _build_context_injection(self) -> str:
        """Buduje wstrzyknięcie kontekstu"""
        parts = []
        
        if self.current_context.get('recommended_strategies'):
            parts.append("## 💡 RECOMMENDED STRATEGIES (from memory):")
            for strategy in self.current_context['recommended_strategies']:
                parts.append(f"• {strategy}")
        
        if self.current_context.get('common_pitfalls'):
            parts.append("\n## ⚠️ COMMON PITFALLS TO AVOID:")
            for pitfall in self.current_context['common_pitfalls']:
                parts.append(f"• {pitfall}")
        
        if self.current_context.get('last_feedback'):
            parts.append(f"\n## 📝 LAST FEEDBACK:\n{self.current_context['last_feedback']}")
        
        return "\n".join(parts)
    

    def run_full_debate_cycle(self):
        from autogen import GroupChat, GroupChatManager
        import json, os, traceback
        from datetime import datetime
        self.reset()
        # Lazy-guard: jeśli ktoś zawoła przed init
        for must in ("user_proxy", "proposer_agents", "aggregator_agent", "critic_agent"):
            if not hasattr(self, must) or getattr(self, must) is None:
                self._initialize_autogen_agents()
                break

        # Szybkie asserty z czytelnym komunikatem
        if not self.proposer_agents:
            raise RuntimeError("Brak proposerów. Sprawdź agents_config.json (role bez 'aggregator'/'critic').")
        if not self.aggregator_agent:
            raise RuntimeError("Brak agregatora. Sprawdź agents_config.json (rola 'Aggregator').")
        if not self.critic_agent:
            raise RuntimeError("Brak krytyka. Sprawdź agents_config.json (rola 'Critic').")

        max_rounds = len(self.proposer_agents) + 2

        # Bootstrap misji – bez 'PLAN_ZATWIERDZONY' w treści, żeby manager nie kończył po 1 msg
        bootstrap = (
            f"## MISJA\n{self.mission}\n\n"
    "Zaproponuj kompletny PLAN w formacie JSON {entry_point, nodes[], edges[]}.\n"
    "Rola: Proposerzy proponują swoje wersje planu. Następnie Aggregator scala je w jedną, spójną propozycję. "
    "Na końcu, Quality_Critic oceni finalny, zagregowany plan."
        )

        self._initial_bootstrap = bootstrap
        
        
        #nowe wczytywanie pamieci
        memory_msg = None
        try:
            memory_msg = self._make_memory_message_once()
            if memory_msg and isinstance(memory_msg, dict) and memory_msg.get("content"):
                process_log("[MEMORY] Seeded memory message into history")
            else:
                process_log("[MEMORY][WARN] No usable memory message produced; seeding skipped")
                memory_msg = None
        except Exception as e:
            process_log(f"[MEMORY][ERROR] Could not build memory message: {type(e).__name__}: {e}")
            memory_msg = None
        
        #koniec 
        
        
        
        
        agents = (
        ([self.memory_analyst_agent] if self.memory_analyst_agent else []) +
        [*self.proposer_agents, self.aggregator_agent, self.critic_agent]
        )
        
        # Uczestnicy – tylko agenci
        # agents = [*self.proposer_agents, self.aggregator_agent, self.critic_agent]

        turns_per_iteration = len(self.proposer_agents) + 2 
        max_rounds = self.max_iterations * turns_per_iteration + 5 # Dodajemy bufor bezpieczeństwa

        start_messages = []
        if memory_msg:
            start_messages.append(memory_msg)

        
        
        gc = GroupChat(
            agents=agents,
            messages = start_messages,
            max_round=max_rounds, # Używamy nowej, dynamicznie obliczonej wartości
            speaker_selection_method=self.custom_speaker_selection_logic)
        
        self.groupchat = gc
        
        try:
            for i, m in enumerate(gc.messages[:3]):  # pokaż do 3 pierwszych
                n, l, tn = self._extract_name_and_len(m)
                process_log(f"[MEMORY][DEBUG] m{i}: type={tn} name={n} len={l}")
        except Exception as e:
            process_log(f"[MEMORY][DEBUG] inspect failed: {type(e).__name__}: {e}")
        
        
        
        manager = GroupChatManager(
            groupchat=gc,
            llm_config=self.aggregator_agent.llm_config,
            human_input_mode="NEVER",
            system_message=MOAPrompts.get_aggregator_prompt(),
            is_termination_msg=self._is_final_plan_message
        )

        try:
            # Start rozmowy – to uruchamia całą maszynkę
            self.user_proxy.initiate_chat(manager, message=bootstrap, max_turns=max_rounds)
        except StopIteration:
            # To jest OK - plan został zatwierdzony
            process_log("[SUCCESS] Debata zakończona przez StopIteration - plan zatwierdzony")
        try:
            # Szukamy finalnej odpowiedzi
            final_plan_message_content = None
            messages = manager.groupchat.messages
            for msg in reversed(messages):
                
                content_str = str(msg.get("content", ""))
                # Używamy Twojej nowej, precyzyjnej funkcji sprawdzającej
                if "PLAN_ZATWIERDZONY" in content_str:
                    final_plan_message_content = msg.get("content")
                    break
                    

            # Jeśli znaleziono zatwierdzoną wiadomość, sparsuj ją
            if final_plan_message_content:
                process_log("[SUCCESS] Krytyk zatwierdził plan. Rozpoczynam parsowanie...")
                try:
                    parsed_critic_response = self.parser.parse_critic_response(final_plan_message_content)

                    # TUTAJ WKLEJ NOWY KOD (zamiast linii 67-75):
                    if parsed_critic_response:
                        # Szukaj planu w różnych możliwych miejscach
                        final_plan = None

                        # Lista możliwych kluczy
                        possible_keys = [
                            "plan_approved",
                            "final_synthesized_plan", 
                            "final_plan",
                            "synthesized_plan",
                            "approved_plan",
                            "plan"
                        ]

                        for key in possible_keys:
                            if key in parsed_critic_response:
                                candidate = parsed_critic_response[key]
                                # Sprawdź czy to wygląda jak plan (ma entry_point i nodes)
                                if isinstance(candidate, dict) and "entry_point" in candidate and "nodes" in candidate:
                                    final_plan = candidate
                                    process_log(f"[SUCCESS] Znaleziono plan pod kluczem: '{key}'")
                                    break

                        if final_plan:
                            self.final_plan = final_plan
                            self._save_successful_plan()
                            
                            
                            
                            # Zbierz stan orchestratora
                            orchestrator_state = {
                                "iteration_count": self.iteration_count,
                                "execution_time": (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0,
                                "total_tokens": getattr(self, 'token_counter', 0),
                                "api_calls": getattr(self, 'api_call_counter', 0)
                            }
                            
                            # try:
                                # Jeśli masz obliczony final_score z jakości planu – możesz go tutaj podać, inaczej None
                            #     mission_id = save_mission_to_gcs(
                            #         bucket_name="memory_rag_for_agents",
                            #         base_prefix="missions",
                            #         mission=self.mission,
                            #         final_plan=self.final_plan,
                            #         all_messages=manager.groupchat.messages,
                            #         orchestrator_state=orchestrator_state,
                            #         approved=True,
                            #         final_score=None,
                            #     )
                            #     process_log(f"[ORCHESTRATOR] Mission completed and saved as: {mission_id}")
                            # except Exception as err:
                            #     # jedno wywołanie loguje zarówno nagłówek, jak i pełny traceback
                            #     log_exc("[MEMORY:GCS][ERROR] Nie udało się zapisać misji", err)
                            
                            if self.memory:
                                # Zapisz KOMPLETNĄ misję
                                mission_id = self.memory.save_complete_mission(
                                mission=self.mission,
                                final_plan=self.final_plan,
                                all_messages=manager.groupchat.messages,
                                orchestrator_state=orchestrator_state
                                )

                                process_log(f"[ORCHESTRATOR] Mission completed and saved as: {mission_id}")
    
                                try:
                                    ndjson_uris = export_local_by_filename_date(
                                        input_dir="memory/missions",                # folder z lokalnymi plikami mission_*.json
                                        output_root_gcs="gs://external_memory/missions",
                                        pattern="*.json",
                                    )
                                    process_log(f"[EXPORT] Wyeksportowano {len(ndjson_uris)} misji do GCS")
                                except Exception as e:
                                    # w razie błędu logujemy pełen traceback
                                    log_exc("[EXPORT][ERROR] Eksport misji do GCS nie powiódł się", e)
                            
                            return self.final_plan
                        else:
                            # Jeśli nie znaleziono planu w żadnym kluczu
                            raise RuntimeError(f"Nie znaleziono planu w odpowiedzi. Dostępne klucze: {list(parsed_critic_response.keys())}")
                    else:
                        raise RuntimeError("Parser zwrócił None - nie udało się sparsować JSON")
            
                except Exception as parse_error:
                    #poprawka
                    log_exc("[ERROR] Nie udało się sparsować odpowiedzi krytyka", parse_error)
                    
                    #koniec poprawki
                    
                    
                    # Sytuacja awaryjna: nie udało się sparsować odpowiedzi krytyka
                    # process_log(f"[ERROR] Nie udało się sparsować odpowiedzi krytyka: {parse_error}")
                    
                    
                    # Zapisz raport z surową odpowiedzią do analizy
                    self._write_failure_report(
                        reason="CRITIC_RESPONSE_PARSE_FAILURE",
                        stage="post-debate_parsing",
                        aggregator_raw=None, # Nieistotne na tym etapie
                        critic_raw=final_plan_message_content,
                        exception=parse_error
                    )
                    
                    return None # Zwracamy None w przypadku błędu parsowania
                
                
                explainability_report = EXPLAINABILITY.generate_final_report()
                process_log(f"[ORCHESTRATOR] Explainability report generated: {explainability_report['debate_id']}")
            else:
                explainability_report = EXPLAINABILITY.generate_final_report()
                process_log(f"[ORCHESTRATOR] Explainability report generated: {explainability_report['debate_id']}")
                # Jeśli pętla się zakończyła i nie znaleziono zatwierdzonej wiadomości
                raise RuntimeError("Debata zakończona, ale krytyk nigdy nie zwrócił wiadomości z 'PLAN_ZATWIERDZONY'.")

        except Exception as e:
            # Raport diagnostyczny
            tb = traceback.format_exc()
            os.makedirs("reports", exist_ok=True)
            path = f"reports/failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"error_type": type(e).__name__,
                           "error_message": str(e),
                           "stacktrace": tb}, f, ensure_ascii=False, indent=2)
            process_log(f"[FAILSAFE] Saved failure report: {path}")
            process_log(tb)
            
            explainability_report = EXPLAINABILITY.generate_final_report()
            process_log(f"[ORCHESTRATOR] Explainability report generated: {explainability_report['debate_id']}")
            
            return None
    
   

    
        
    
    def _update_context_from_last_critique(self, critique_message: str):
        """Aktualizuje kontekst na podstawie krytyki"""
        # Parsuj krytykę
        if not self.memory:
            return
        
        
        parsed = self.parser.parse_critic_response(critique_message)
        
        if parsed:
            feedback = f"Score: {parsed.get('score', 'N/A')}. "
            feedback += f"Weaknesses: {', '.join(parsed.get('weaknesses', []))}. "
            feedback += f"Improvements: {', '.join(parsed.get('improvements', []))}"
            
            self.current_context['last_feedback'] = feedback
            
            # Zapisz do pamięci
            self.memory.add_iteration_feedback(
                iteration=self.iteration_count,
                feedback=feedback,
                timestamp=datetime.now()
            )
        
        # Odśwież kontekst z pamięci
        self.current_context = self.memory.get_relevant_context(self.mission)
        
        process_log(f"Context updated for iteration {self.iteration_count}")
    
    def _extract_final_plan(self, messages: List[Dict]):
        """Wyodrębnia zatwierdzony plan z historii wiadomości"""
        # Szukaj od końca
        for msg in reversed(messages):
            content = msg.get("content", "")
            name = msg.get("name", "")
            
            # Jeśli krytyk zatwierdził
            if name == "Quality_Critic" and "PLAN_ZATWIERDZONY" in content:
                # Znajdź ostatni plan od agregatora
                for prev_msg in reversed(messages):
                    if prev_msg.get("name") == "Master_Aggregator":
                        parsed = self.parser.parse_agent_response(prev_msg.get("content", ""))
                        if parsed:
                            self.final_plan = parsed.get("final_plan", parsed.get("plan"))
                            break
                break
        
        process_log(f"Final plan extracted: {self.final_plan is not None}")
    
    def _save_successful_plan(self):
        """Zapisuje udany plan do pamięci i pliku"""
        if not self.final_plan:
            return
        if self.memory:
           
        # Zapisz do pamięci
            self.memory.add_successful_plan(
                plan=self.final_plan,
                mission=self.mission,
                metadata={
                    'iterations': self.iteration_count,
                    'agents_count': len(self.proposer_agents)
                }
            )
        
        # Zapisz do pliku
        output = {
            "mission": self.mission,
            "final_plan": self.final_plan,
            "metadata": {
                "iterations": self.iteration_count,
                "timestamp": datetime.now().isoformat(),
                "autogen_debate": True
            }
        }
        
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/autogen_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Plan saved to: {output_file}")
        process_log(f"Successful plan saved to {output_file}")
        
        
    def _debug_dump_transcript(self, groupchat, tail: int = 30):
        """Wypisz ostatnie ~N wiadomości debaty, żeby było je widać w notebooku."""
        from process_logger import log as process_log
        try:
            msgs = getattr(groupchat, "messages", [])[-tail:]
            process_log("----- TRANSCRIPT (tail) -----")
            for m in msgs:
                role = m.get("role") or m.get("name") or "?"
                name = m.get("name") or ""
                content = m.get("content") or ""
                head = (content[:400] + "...") if len(content) > 400 else content
                process_log(f"{role} {name}: {head}")
            process_log("----- END TRANSCRIPT -----")
        except Exception as e:
            process_log(f"[TRANSCRIPT_DUMP_FAIL] {type(e).__name__}: {e}")