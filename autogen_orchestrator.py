"""
Pełny orchestrator MOA używający AutoGen do zarządzania debatą agentów
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import autogen

from models_config import AgentRole
from moa_prompts import MOAPrompts
from memory_system import ContextMemory
# Używamy structured parsera zamiast heurystycznego response_parser
from structured_response_parser import StructuredResponseParser
from process_logger import log as process_log

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
        
        # Inicjalizuj agentów AutoGen
        self._initialize_autogen_agents()
        
        process_log(f"=== AutoGen MOA Orchestrator initialized for mission: {mission[:100]}... ===")
    
    
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

        # log do Twojego loggera
        from process_logger import log as process_log
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
    
    
    
    
    
    
    def _load_config(self, config_file: str):
        """Wczytuje konfigurację agentów"""
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def _initialize_autogen_agents(self):
        """Inicjalizuje agentów AutoGen dla debaty"""
        
        # User Proxy - zarządza przepływem konwersacji
        self.user_proxy = autogen.UserProxyAgent(
            name="Orchestrator",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "PLAN_ZATWIERDZONY" in x.get("content", ""),
            code_execution_config=False,
            system_message="You orchestrate the planning discussion."
        )
        
        # Proposer Agents
        self.proposer_agents = []
        for agent_config in self.config['agents']:
            if 'aggregator' in agent_config['role_name'].lower() or 'critic' in agent_config['role_name'].lower():
                continue  # Skip aggregator and critic for now
                
            role = AgentRole(
                role_name=agent_config['role_name'],
                expertise_areas=agent_config['expertise_areas'],
                thinking_style=agent_config['thinking_style']
            )
            
            # Generuj prompt z kontekstem
            prompt = self._build_proposer_prompt(role)
            
            agent = autogen.ConversableAgent(
                name=agent_config['role_name'].replace(" ", "_"),
                llm_config=self._build_llm_config(agent_config['model']),
                system_message=prompt,
                human_input_mode="NEVER"
            )
            self.proposer_agents.append(agent)
        
        # Aggregator Agent
        aggregator_config = next((a for a in self.config['agents'] 
                                 if 'aggregator' in a['role_name'].lower()), None)
        if aggregator_config:
            self.aggregator = autogen.ConversableAgent(
                name="Master_Aggregator",
                llm_config=self._build_llm_config(aggregator_config['model']),
                system_message=MOAPrompts.get_aggregator_prompt(),
                human_input_mode="NEVER"
            )
        else:
            # Default aggregator
            self.aggregator = autogen.ConversableAgent(
                name="Master_Aggregator",
                llm_config={"config_list": [{"model": "dummy", "api_type": "dummy"}]},
                system_message=MOAPrompts.get_aggregator_prompt(),
                human_input_mode="NEVER"
            )
        
        # Critic Agent
        critic_config = next((a for a in self.config['agents'] 
                            if 'critic' in a['role_name'].lower()), None)
        if critic_config:
            self.critic = autogen.ConversableAgent(
                name="Quality_Critic",
                llm_config=self._build_llm_config(critic_config['model']),
                system_message=self._build_critic_prompt(),
                human_input_mode="NEVER"
            )
        else:
            # Default critic
            self.critic = autogen.ConversableAgent(
                name="Quality_Critic",
                llm_config={"config_list": [{"model": "dummy", "api_type": "dummy"}]},
                system_message=self._build_critic_prompt(),
                human_input_mode="NEVER"
            )
        
        process_log(f"Initialized {len(self.proposer_agents)} proposers, 1 aggregator, 1 critic using AutoGen")
    
    def _build_llm_config(self, model_config: Dict) -> Dict:
        """
        Buduje llm_config zgodny z AutoGen (AG2).
        - config_list: płaska lista wpisów dla modeli (model, api_key, ewentualnie base_url, api_version, deployment_name).
        - Parametry generacji (temperature, top_p, max_tokens, ...) NA POZIOMIE GŁÓWNYM.
        """
        provider   = (model_config.get("provider") or "openai").lower()
        model_name = model_config.get("model_name", "gpt-4o-mini")

        # API key z ENV (NIE trzymaj kluczy w JSON!)
        api_key_env = model_config.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None

        # Płaski wpis config_list
        config_item = {"model": model_name}
        if api_key:
            config_item["api_key"] = api_key

        # Opcjonalne endpointy/parametry providerów:
        # base_url (alias api_base) – np. dla Azure/OpenAI proxy/self-host
        if "base_url" in model_config:
            config_item["base_url"] = model_config["base_url"]
        if "api_base" in model_config and "base_url" not in config_item:
            config_item["base_url"] = model_config["api_base"]

        # Providerowe hinty (opcjonalne — dodaj tylko, gdy potrzebujesz)
        if provider in ("anthropic",):
            config_item["api_type"] = "anthropic"
        elif provider in ("azure", "azure_openai"):
            # Azure zwykle wymaga wersji API i deploymentu
            if "api_version" in model_config:
                config_item["api_version"] = model_config["api_version"]
            if "azure_deployment" in model_config:
                config_item["deployment_name"] = model_config["azure_deployment"]
        elif provider in ("google", "gemini", "vertex"):
            config_item["api_type"] = "google"
        else:
            # OpenAI / inne kompatybilne – najczęściej nic nie trzeba dopisywać
            pass

        # Składamy llm_config: temperatura na TOP-LEVEL, nie w config_list!
        llm_config = {
            "config_list": [config_item],
            "temperature": model_config.get("temperature", 0.5),
            # opcjonalnie:
            # "top_p": model_config.get("top_p", 1.0),
            # "max_tokens": model_config.get("max_tokens", 2000),
            "seed": 42,
            "cache_seed": 42,
        }
        return llm_config
    
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
        
        # Dodaj specjalną instrukcję o frazie kończącej
        additional_instruction = """
        
        ## GOLDEN TERMINATION RULE
        If you approve the plan, you MUST end your ENTIRE response with the exact phrase:
        PLAN_ZATWIERDZONY
        
        This phrase must appear AFTER your JSON response, on a new line.
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
    
    def run_full_debate_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Uruchamia pełny cykl debaty używając AutoGen GroupChat
        """
        print("\n" + "="*70)
        print("🚀 STARTING AUTOGEN MOA DEBATE CYCLE")
        print("="*70)
        
        # Funkcja wyboru mówcy - kontroluje przepływ debaty
        def custom_speaker_selection(last_speaker, groupchat):
            """Kontroluje kolejność mówców w debacie"""
            messages = groupchat.messages
            
            # Sprawdź czy krytyk zaakceptował plan
            if last_speaker == self.critic and messages:
                last_msg = messages[-1].get("content", "")
                if "PLAN_ZATWIERDZONY" in last_msg:
                    return None  # Kończy debatę
            
            # Logika przepływu
            if last_speaker == self.user_proxy:
                # Po orchestratorze -> pierwszy proposer
                return self.proposer_agents[0] if self.proposer_agents else self.aggregator
            
            elif last_speaker in self.proposer_agents:
                # Po proposerze -> następny proposer lub aggregator
                idx = self.proposer_agents.index(last_speaker)
                if idx < len(self.proposer_agents) - 1:
                    return self.proposer_agents[idx + 1]
                else:
                    return self.aggregator
            
            elif last_speaker == self.aggregator:
                # Po aggregatorze -> krytyk
                return self.critic
            
            elif last_speaker == self.critic:
                # Po krytyku -> wróć do proposerów (nowa iteracja)
                self.iteration_count += 1
                if self.iteration_count < self.max_iterations:
                    # Aktualizuj kontekst dla nowej iteracji
                    self._update_context_from_last_critique(messages[-1].get("content", ""))
                    return self.proposer_agents[0] if self.proposer_agents else None
                else:
                    return None  # Koniec po max iteracjach
            
            return self.proposer_agents[0]  # Default
        
        # Stwórz GroupChat
        all_agents = [self.user_proxy] + self.proposer_agents + [self.aggregator, self.critic]
        
        groupchat = autogen.GroupChat(
            agents=all_agents,
            messages=[],
            max_round=50,  # Dużo rund na wszelki wypadek
            speaker_selection_method=custom_speaker_selection
        )
        
        # Manager zarządza konwersacją
        manager = autogen.GroupChatManager(
            groupchat=groupchat
        )
        
        # Aktualizuj kontekst początkowy
        self.current_context = self.memory.get_relevant_context(self.mission)
        
        # Wiadomość startowa
        initial_message = f"""
        ## MISSION: {self.mission}
        
        ## AVAILABLE TOOLS:
        {json.dumps(list(self.node_library.keys()))}
        
        ## TASK:
        Create a robust workflow plan. Each agent should propose their approach,
        then the aggregator will synthesize, and the critic will evaluate.
        
        Iteration: {self.iteration_count + 1}/{self.max_iterations}
        
        {self._build_context_injection() if self.current_context else ""}
        
        Please provide your proposal in JSON format with: plan, thought_process, confidence.
        """
        
        # Rozpocznij debatę
        print("\n📍 Starting AutoGen group discussion...")
        self.user_proxy.initiate_chat(manager, message=initial_message)
        
        # Wyodrębnij finalny plan z konwersacji
        self._extract_final_plan(groupchat.messages)
        
        # Zapisz do pamięci jeśli sukces
        if self.final_plan:
            self._save_successful_plan()
            print("\n✅ PLAN APPROVED AND SAVED!")
        else:
            print("\n⚠️ No approved plan after debate")
        
        return self.final_plan
    
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