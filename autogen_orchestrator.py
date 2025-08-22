"""
Pełny orchestrator MOA używający AutoGen do zarządzania debatą agentów
"""
import json
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

from config_api import basic_config_agent

class AutoGenMOAOrchestrator:
    """
    Orchestrator systemu MOA używający AutoGen do wieloturowej debaty
    """
    
    def __init__(self, mission: str, node_library: Dict[str, Any], config_file: str = "agents_config.json"):
        self.mission = mission
        self.node_library = node_library
        self.memory = ContextMemory(max_episodes=50)
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
    
    #Raport:
    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _now_stamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def _extract_llm_hint(self, text: str) -> Optional[str]:
        """Prosta heurystyka do rozpoznawania typowych problemów LLM-a."""
        if not text:
            return None
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
            "aggregator_output_excerpt": (aggregator_raw or "")[:4000],
            "critic_output_excerpt": (critic_raw or "")[:4000],
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
            f.write("```\n" + (aggregator_raw or "")[:4000] + "\n```\n\n")
            f.write("## Last Critic Output (excerpt)\n\n")
            f.write("```\n" + (critic_raw or "")[:4000] + "\n```\n")

        
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
    
    
#     def custom_speaker_selection_logic(self, last_speaker: ConversableAgent, groupchat: GroupChat):
#         """
#         Zarządza cyklem debaty: Proposerzy -> Aggregator -> Krytyk.
#         """
#         messages = groupchat.messages

#         # **POPRAWKA:** Jeśli rozmowa dopiero się zaczyna (tylko 1 wiadomość od Orchestratora),
#         # zawsze zaczynaj od pierwszego proposera.
#         if len(messages) <= 1:
#             return self.proposer_agents[0]

#         if last_speaker.name == "Master_Aggregator":
#             return self.critic_agent

#         if last_speaker.name == "Quality_Critic":
#             last_message_content = messages[-1].get("content", "").upper()
#             if "PLAN_ZATWIERDZONY" in last_message_content:
#                 return None  # Zakończ debatę

#             self.iteration_count += 1
#             if self.iteration_count >= self.max_iterations:
#                 process_log(f"Max iterations ({self.max_iterations}) reached. Ending debate.")
#                 return None
            
#             process_log(f"--- Starting iteration {self.iteration_count + 1} ---")
#             self._update_context_from_last_critique(messages[-1].get("content", ""))
#             return self.proposer_agents[0]

#         if last_speaker in self.proposer_agents:
#             try:
#                 idx = self.proposer_agents.index(last_speaker)
#                 if idx < len(self.proposer_agents) - 1:
#                     return self.proposer_agents[idx + 1]
#                 else:
#                     return self.aggregator_agent
#             except ValueError:
#                 return self.aggregator_agent
        
#         # Domyślny fallback na wszelki wypadek
#         return self.proposer_agents[0]

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

        # ❶ Po bootstrapie (ostatni był Orchestrator → wybieramy pierwszego proposera)
        if last_name == (self.user_proxy.name or "").lower() and last_content.strip():
            return self.proposer_agents[0]

        # ❷ Po Aggregatorze → czas na Critica
        if last_name == (self.aggregator_agent.name or "").lower():
            return self.critic_agent

        # ❸ Po Criticu → zatwierdzenie albo nowa iteracja
        if last_name == (self.critic_agent.name or "").lower():
            if "PLAN_ZATWIERDZONY" in last_content:
                return None
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
    
    
    
    def _initialize_autogen_agents(self):
        """Inicjalizuje agentów AutoGen dla debaty — minimalistycznie i niezawodnie."""
        # Atrybuty ZAWSZE istnieją
        self.proposer_agents = []
        self.aggregator = None
        self.critic = None
        self.aggregator_agent = None
        self.critic_agent = None

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
            if 'aggregator' in rn or 'critic' in rn:
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
            return base_prompt + "\n\n" + context_injection
        
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

        return base_prompt + additional_instruction
    
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

        # Uczestnicy – tylko agenci
        agents = [*self.proposer_agents, self.aggregator_agent, self.critic_agent]

        turns_per_iteration = len(self.proposer_agents) + 2 
        max_rounds = self.max_iterations * turns_per_iteration + 5 # Dodajemy bufor bezpieczeństwa

        gc = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_rounds, # Używamy nowej, dynamicznie obliczonej wartości
            speaker_selection_method=self.custom_speaker_selection_logic)
        
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
                # Używamy Twojej nowej, precyzyjnej funkcji sprawdzającej
                if "PLAN_ZATWIERDZONY" in msg.get("content", ""):
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
                            return self.final_plan
                        else:
                            # Jeśli nie znaleziono planu w żadnym kluczu
                            raise RuntimeError(f"Nie znaleziono planu w odpowiedzi. Dostępne klucze: {list(parsed_critic_response.keys())}")
                    else:
                        raise RuntimeError("Parser zwrócił None - nie udało się sparsować JSON")
            
                except Exception as parse_error:
                    # Sytuacja awaryjna: nie udało się sparsować odpowiedzi krytyka
                    process_log(f"[ERROR] Nie udało się sparsować odpowiedzi krytyka: {parse_error}")
                    # Zapisz raport z surową odpowiedzią do analizy
                    self._write_failure_report(
                        reason="CRITIC_RESPONSE_PARSE_FAILURE",
                        stage="post-debate_parsing",
                        aggregator_raw=None, # Nieistotne na tym etapie
                        critic_raw=final_plan_message_content,
                        exception=parse_error
                    )
                    return None # Zwracamy None w przypadku błędu parsowania
            else:
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
            return None
    
   

    
        
    
    def _update_context_from_last_critique(self, critique_message: str):
        """Aktualizuje kontekst na podstawie krytyki"""
        # Parsuj krytykę
        parsed = self.parser.parse_agent_response(critique_message)
        
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