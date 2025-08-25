"""
System pamięci kontekstowej z uczeniem się z poprzednich iteracji
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import numpy as np
from collections import deque
import os

# Zewnętrzne biblioteki do obliczania podobieństwa tekstu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lokalny logger procesu
from process_logger import log as process_log

class ContextMemory:
    def __init__(self, max_episodes: int = 100):
        # Existing
        self.episodes = deque(maxlen=max_episodes)
        self.learned_patterns = {}
        self.successful_strategies = []
        
        # NOWE - Pełne dane misji
        self.full_mission_records = []  # Bez limitu - wszystko zapisujemy
        self.mission_index = {}  # Szybkie wyszukiwanie po ID
        
        self._load_persistent_memory()
    
    
    
    
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
                "best_practices": []
            }

        # 2. Aktualizuj statystyki
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        current_score = mission_record.get("final_score", 0)
        pattern["avg_score"] = (
            (pattern["avg_score"] * (pattern["occurrences"] - 1) + current_score) 
            / pattern["occurrences"]
        )

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
                "complexity": mission_record["performance_metrics"].get("convergence_rate", 0)
            }
            pattern["best_practices"].append(best_practice)

        # 6. Zaktualizuj common_elements (co występuje najczęściej)
        for element in success_elements:
            if element not in pattern["common_elements"]:
                pattern["common_elements"][element] = 0
            pattern["common_elements"][element] += 1

        # 7. Dodaj przykład
        pattern["examples"].append({
            "mission_prompt": mission_record["mission_prompt"],
            "success_factors": success_elements,
            "score": current_score
        })

        process_log(f"[MEMORY] Learned from success: {pattern_key}, "
                    f"occurrences={pattern['occurrences']}, "
                    f"avg_score={pattern['avg_score']:.2f}")
    
    
    
    
    def save_complete_mission(self, 
                            mission: str,
                            final_plan: Dict,
                            all_messages: List[Dict],
                            orchestrator_state: Dict) -> str:
        """
        Zapisuje KOMPLETNY rekord misji z wszystkimi danymi
        """
        from datetime import datetime
        import hashlib
        
        # Generuj unikalne ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mission_hash = hashlib.md5(mission.encode()).hexdigest()[:8]
        mission_id = f"mission_{timestamp}_{mission_hash}"
        
        # Ekstraktuj kluczowe informacje z transcript
        iterations_data = self._extract_iterations_from_transcript(all_messages)
        
        # Klasyfikuj misję i tagi
        mission_type = self._classify_mission(mission)
        tags = self._extract_tags(mission, final_plan)
        
        # Znajdź krytyczne momenty w debacie
        critical_moments = self._identify_critical_moments(all_messages)
        
        # Przygotuj pełny rekord
        mission_record = {
            # === METADATA ===
            "memory_id": mission_id,
            "timestamp": datetime.now().isoformat(),
            "mission_prompt": mission,
            "mission_type": mission_type,
            "tags": tags,
            
            # === OUTCOME ===
            "outcome": "Success" if final_plan else "Failed",
            "total_iterations": orchestrator_state.get("iteration_count", 0),
            "total_messages": len(all_messages),
            "time_taken_seconds": orchestrator_state.get("execution_time", 0),
            
            # === FINAL ARTIFACTS ===
            "final_plan": final_plan,
            "final_score": self._extract_final_score(all_messages),
            
            # === ITERATION DETAILS ===
            "iterations": iterations_data,
            
            # === KEY INSIGHTS ===
            "critique_evolution": self._track_critique_evolution(iterations_data),
            "aggregator_reasoning": self._extract_aggregator_reasoning(all_messages),
            "proposer_contributions": self._analyze_proposer_contributions(all_messages),
            
            # === LEARNING DATA ===
            "llm_generated_summary": self._generate_mission_summary(all_messages, final_plan),
            "identified_patterns": self._extract_patterns_from_debate(all_messages),
            "success_factors": self._identify_success_factors(final_plan, iterations_data),
            "failure_points": self._identify_failure_points(iterations_data),
            
            # === CRITICAL MOMENTS ===
            "critical_moments": critical_moments,
            "turning_points": self._identify_turning_points(iterations_data),
            
            # === FULL TRANSCRIPT ===
            "full_transcript": all_messages,  # Kompletny zapis
            
            # === METRICS ===
            "performance_metrics": {
                "token_usage": orchestrator_state.get("total_tokens", 0),
                "api_calls": orchestrator_state.get("api_calls", 0),
                "convergence_rate": self._calculate_convergence_rate(iterations_data)
            }
        }
        
        # Zapisz do pamięci
        self.full_mission_records.append(mission_record)
        self.mission_index[mission_id] = len(self.full_mission_records) - 1
        
        if final_plan:  # Jeśli misja się udała
            self._learn_from_success(mission_record)
        
        
        # Persist immediately
        self._persist_full_memory()
        
        process_log(f"[MEMORY] Saved complete mission: {mission_id}")
        return mission_id
    
    def _extract_iterations_from_transcript(self, messages: List[Dict]) -> List[Dict]:
        """Ekstraktuje dane każdej iteracji z transkryptu"""
        iterations = []
        current_iteration = {"proposers": [], "aggregator": None, "critic": None}
        
        for msg in messages:
            role = msg.get("name", "").lower()
            
            if "proposer" in role or "analyst" in role or "planner" in role:
                current_iteration["proposers"].append({
                    "agent": msg.get("name"),
                    "content": msg.get("content"),
                    "key_ideas": self._extract_key_ideas(msg.get("content", ""))
                })
            
            elif "aggregator" in role:
                current_iteration["aggregator"] = {
                    "content": msg.get("content"),
                    "synthesis": self._extract_synthesis(msg.get("content", ""))
                }
            
            elif "critic" in role:
                current_iteration["critic"] = {
                    "content": msg.get("content"),
                    "verdict": self._extract_verdict(msg.get("content", "")),
                    "score": self._extract_score(msg.get("content", "")),
                    "weaknesses": self._extract_weaknesses(msg.get("content", ""))
                }
                
                # Koniec iteracji - zapisz i zacznij nową
                if current_iteration["proposers"]:
                    iterations.append(current_iteration)
                    current_iteration = {"proposers": [], "aggregator": None, "critic": None}
        
        return iterations
    
#     def _generate_mission_summary(self, messages: List[Dict], final_plan: Dict) -> str:
#         """Generuje podsumowanie misji (możesz tu użyć LLM)"""
#         # Prosta heurystyka - w przyszłości możesz wywołać LLM
#         summary_parts = []
        
#         # Analiza iteracji
#         iteration_count = sum(1 for m in messages if "critic" in m.get("name", "").lower())
#         summary_parts.append(f"Misja wymagała {iteration_count} iteracji.")
        
#         # Kluczowe poprawki
#         weaknesses_mentioned = set()
#         for msg in messages:
#             if "weakness" in msg.get("content", "").lower():
#                 # Ekstraktuj weakness (uproszczenie)
#                 weaknesses_mentioned.add("obsługa błędów")
        
#         if weaknesses_mentioned:
#             summary_parts.append(f"Główne wyzwania: {', '.join(weaknesses_mentioned)}.")
        
#         # Finalny sukces
#         if final_plan:
#             node_count = len(final_plan.get("nodes", []))
#             summary_parts.append(f"Finalny plan zawiera {node_count} węzłów.")
        
#         return " ".join(summary_parts)
    
    
    def _generate_mission_summary(self, messages: List[Dict], final_plan: Dict) -> str:
        """Generuje BOGATE podsumowanie misji"""
        summary_parts = []

        # 1. Liczba iteracji i czas
        iteration_count = sum(1 for m in messages if "critic" in m.get("name", "").lower())
        summary_parts.append(f"Misja zakończona w {iteration_count} iteracji.")

        # 2. Kluczowe innowacje (szukaj w transkrypcie)
        innovations = set()
        for msg in messages:
            content = msg.get("content", "").lower()
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
            success_paths = len([e for e in edges if e.get("condition") == "on_success"])
            failure_paths = len([e for e in edges if e.get("condition") == "on_failure"])

            summary_parts.append(
                f"Struktura: {len(nodes)} węzłów, "
                f"{success_paths} ścieżek sukcesu, "
                f"{failure_paths} ścieżek obsługi błędów."
            )

            # Znajdź kluczowe węzły
            key_nodes = []
            for node in nodes:
                impl = node.get("implementation", "")
                if impl in ["error_handler", "rollback", "validate_data", "optimize_performance"]:
                    key_nodes.append(impl)

            if key_nodes:
                summary_parts.append(f"Kluczowe komponenty: {', '.join(set(key_nodes))}.")

        # 4. Końcowy verdykt
        for msg in reversed(messages):
            if "critic" in msg.get("name", "").lower() and "ZATWIERDZONY" in msg.get("content", ""):
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
            "data": ["data", "dane", "csv", "pipeline"]
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
    
    
    def _load_persistent_memory(self):
        """
        Ładuje pamięć z pliku JSON
        """
        json_file = "memory/learned_strategies.json"

        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.learned_patterns = data.get("patterns", {})
                self.successful_strategies = data.get("strategies", [])

                # Załaduj też nowe full_mission_records jeśli istnieją
                if "full_mission_records" in data:
                    self.full_mission_records = data["full_mission_records"]
                    # Odbuduj index
                    for i, record in enumerate(self.full_mission_records):
                        self.mission_index[record["memory_id"]] = i

                print(f"✔ Załadowano pamięć: {len(self.successful_strategies)} strategies, {len(self.full_mission_records)} full records")
            except Exception as e:
                print(f"⚠ Nie udało się załadować pamięci: {e}")
        else:
            print("📝 Tworzę nową pamięć (brak istniejącego pliku)")
            os.makedirs("memory", exist_ok=True)

    def _persist_memory(self):
        """
        Zapisuje pamięć do pliku JSON
        """
        os.makedirs("memory", exist_ok=True)
        memory_file = "memory/learned_strategies.json"

        data = {
            "patterns": self.learned_patterns,
            "strategies": self.successful_strategies,
            "full_mission_records": self.full_mission_records  # NOWE!
        }

        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Nie udało się zapisać pamięci: {e}")

    def _persist_full_memory(self):
        """Alias dla _persist_memory"""
        self._persist_memory()


        
    def _extract_key_ideas(self, content: str) -> List[str]:
        """Ekstraktuje kluczowe pomysły z contentu"""
        # Prosta heurystyka - możesz ulepszyć
        ideas = []
        if "error_handler" in content.lower():
            ideas.append("error_handling")
        if "rollback" in content.lower():
            ideas.append("rollback_mechanism")
        if "optimiz" in content.lower():
            ideas.append("optimization")
        return ideas

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
        if "ZATWIERDZONY" in content:
            return "ZATWIERDZONY"
        return "ODRZUCONY"

    def _extract_score(self, content: str) -> float:
        """Ekstraktuje score z odpowiedzi krytyka"""
        try:
            import re
            score_match = re.search(r'"Overall_Quality_Q":\s*([\d.]+)', content)
            if score_match:
                return float(score_match.group(1))
        except:
            pass
        return 0.0

    def _extract_weaknesses(self, content: str) -> List[str]:
        """Ekstraktuje weaknesses z odpowiedzi krytyka"""
        weaknesses = []
        try:
            data = json.loads(content) if isinstance(content, str) else content
            weak_list = data.get("critique_summary", {}).get("identified_weaknesses", [])
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
            "timestamp": datetime.now().isoformat()
        }

        self.successful_strategies.append(strategy)
        self._persist_memory()  # Zapisz od razu

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
        elif "dane" in mission_lower or "data" in mission_lower or "csv" in mission_lower:
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
            "has_error_handling": any("error" in str(node).lower() 
                                     for node in plan.get("nodes", [])),
            "has_validation": any("valid" in str(node).lower() 
                                 for node in plan.get("nodes", [])),
            "graph_complexity": self._calculate_complexity(plan)
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
            content = msg.get("content", "").lower()
            # Moment krytyczny = duża zmiana w score lub verdict
            if "zatwierdzony" in content or "odrzucony" in content:
                critical.append({
                    "index": i,
                    "type": "verdict",
                    "agent": msg.get("name"),
                    "summary": "Decyzja krytyka"
                })
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
                evolution.append({
                    "iteration": i,
                    "score": iteration["critic"].get("score", 0),
                    "verdict": iteration["critic"].get("verdict", ""),
                    "main_issues": iteration["critic"].get("weaknesses", [])[:2]
                })
        return evolution

    def _extract_aggregator_reasoning(self, messages: List[Dict]) -> str:
        """Wyciąga reasoning agregatora"""
        for msg in reversed(messages):
            if "aggregator" in msg.get("name", "").lower():
                return self._extract_synthesis(msg.get("content", ""))
        return ""

    def _analyze_proposer_contributions(self, messages: List[Dict]) -> Dict[str, List[str]]:
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
        # Szukaj powtarzających się konceptów
        all_text = " ".join(m.get("content", "") for m in messages).lower()

        if all_text.count("error_handler") > 3:
            patterns.append("Częste odniesienia do obsługi błędów")
        if all_text.count("rollback") > 2:
            patterns.append("Rollback jako kluczowy element")
        if all_text.count("optimiz") > 2:
            patterns.append("Focus na optymalizację")

        return patterns

    def _identify_success_factors(self, final_plan: Dict, iterations: List[Dict]) -> List[str]:
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
                failures.append({
                    "iteration": i,
                    "issues": iteration["critic"].get("weaknesses", []),
                    "score": iteration["critic"].get("score", 0)
                })
        return failures

    def _identify_turning_points(self, iterations: List[Dict]) -> List[Dict]:
        """Znajduje punkty zwrotne w debacie"""
        turning_points = []
        prev_score = 0

        for i, iteration in enumerate(iterations):
            curr_score = iteration.get("critic", {}).get("score", 0)
            if curr_score - prev_score > 20:  # Duży skok w score
                turning_points.append({
                    "iteration": i,
                    "score_jump": curr_score - prev_score,
                    "reason": "Significant improvement"
                })
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
        improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        # Normalizuj do 0-1 (im wyższy przyrost, tym lepsza convergence)
        return min(avg_improvement / 20, 1.0)  # 20 punktów na iterację = max convergence
    
# class ContextMemory:
#     """
#     Zaawansowany system pamięci dla agentów MOA
#     Implementuje episodic memory, semantic memory i procedural memory
#     """
    
#     def __init__(self, max_episodes: int = 100):
#         # Episodic Memory - konkretne wydarzenia/iteracje
#         self.episodes: deque = deque(maxlen=max_episodes)
        
#         # Semantic Memory - wyuczone wzorce i koncepty
#         self.learned_patterns: Dict[str, Any] = {}
        
#         # Procedural Memory - sprawdzone strategie
#         self.successful_strategies: List[Dict] = []
        
#         # Working Memory - bieżący kontekst
#         self.current_context: Dict[str, Any] = {}
        
#         # Meta-Memory - informacje o skuteczności pamięci
#         self.memory_performance: Dict[str, float] = {
#             "retrieval_accuracy": 1.0,
#             "pattern_recognition_rate": 0.0
#         }
        
#         self._load_persistent_memory()
    
#     def add_iteration_feedback(self, iteration: int, feedback: str, timestamp: datetime):
#         """Dodaje feedback z iteracji do pamięci epizodycznej"""
#         episode = {
#             "iteration": iteration,
#             "feedback": feedback,
#             "timestamp": timestamp.isoformat(),
#             "extracted_issues": self._extract_issues(feedback),
#             "success": False  # Będzie zaktualizowane jeśli plan zostanie zatwierdzony
#         }
        
#         self.episodes.append(episode)
#         self._update_learned_patterns(episode)
#         # Zaloguj dodanie feedbacku wraz z potencjalnymi problemami
#         try:
#             process_log(
#                 f"Add iteration feedback (iter={iteration}) issues={episode['extracted_issues']}"
#             )
#         except Exception:
#             pass
    
#     def add_successful_plan(self, plan: Dict[str, Any], mission: str, metadata: Dict):
#         """Zapisuje udany plan do pamięci proceduralnej"""
#         strategy = {
#             "mission_type": self._classify_mission(mission),
#             "plan_structure": self._extract_plan_structure(plan),
#             "success_factors": metadata.get("success_factors", []),
#             "performance_metrics": metadata.get("metrics", {}),
#             "timestamp": datetime.now().isoformat()
#         }
        
#         self.successful_strategies.append(strategy)
#         self._persist_memory()
#         # Loguj dodanie udanego planu
#         try:
#             process_log(
#                 f"Add successful plan for mission_type={strategy['mission_type']}, structure={strategy['plan_structure']}"
#             )
#         except Exception:
#             pass
    
#     def get_relevant_context(self, mission: str) -> Dict[str, Any]:
#         """
#         Pobiera relevantny kontekst dla danej misji
#         Używa similarity search i pattern matching
#         """
#         context = {
#             "similar_missions": self._find_similar_missions(mission),
#             "relevant_patterns": self._get_relevant_patterns(mission),
#             "recommended_strategies": self._recommend_strategies(mission),
#             "common_pitfalls": self._get_common_pitfalls(),
#             "last_feedback": self._get_last_feedback()
#         }
#         # Zaloguj pobranie kontekstu
#         try:
#             process_log(
#                 f"Retrieve context for mission='{mission}', suggestions={context['recommended_strategies']}"
#             )
#         except Exception:
#             pass
#         return context
    
#     def _extract_issues(self, feedback: str) -> List[str]:
#         """Ekstraktuje konkretne problemy z feedbacku"""
#         issues: List[str] = []
#         # Rozszerzona lista wskaźników problemów (różne formy i synonimy)
#         problem_indicators = [
#             "brak", "niewystarczający", "niewystarczająca", "niepoprawny", "niepoprawna",
#             "błąd", "problem", "wadliwy", "wadliwa", "niekompletny", "niekompletna",
#             "niespójny", "niespójna", "niedostateczny", "niedostateczna",
#             "nieprawidłowy", "nieprawidłowa", "awaria", "usterka"
#         ]
#         for sentence in feedback.split("."):
#             s_low = sentence.lower()
#             if any(ind in s_low for ind in problem_indicators):
#                 stripped = sentence.strip()
#                 if stripped:
#                     issues.append(stripped)
#         return issues
    
#     def _update_learned_patterns(self, episode: Dict):
#         """Aktualizuje wyuczone wzorce na podstawie nowego epizodu"""
#         for issue in episode["extracted_issues"]:
#             # Tworzymy hash problemu dla grupowania podobnych
#             issue_hash = self._hash_issue(issue)
            
#             if issue_hash not in self.learned_patterns:
#                 self.learned_patterns[issue_hash] = {
#                     "occurrences": 0,
#                     "examples": [],
#                     "solutions": []
#                 }
            
#             self.learned_patterns[issue_hash]["occurrences"] += 1
#             self.learned_patterns[issue_hash]["examples"].append(issue)
    
#     def _classify_mission(self, mission: str) -> str:
#         """Klasyfikuje typ misji"""
#         mission_lower = mission.lower()
        
#         if "przyczynow" in mission_lower or "causal" in mission_lower:
#             return "causal_analysis"
#         elif "dane" in mission_lower or "data" in mission_lower:
#             return "data_processing"
#         elif "model" in mission_lower:
#             return "model_validation"
#         elif "optymali" in mission_lower:
#             return "optimization"
#         else:
#             return "general"
    
#     def _extract_plan_structure(self, plan: Dict) -> Dict:
#         """Ekstraktuje strukturalne cechy planu"""
#         return {
#             "num_nodes": len(plan.get("nodes", [])),
#             "num_edges": len(plan.get("edges", [])),
#             "has_error_handling": any("error" in str(node).lower() 
#                                      for node in plan.get("nodes", [])),
#             "has_validation": any("valid" in str(node).lower() 
#                                  for node in plan.get("nodes", [])),
#             "graph_complexity": self._calculate_complexity(plan)
#         }
    
#     def _calculate_complexity(self, plan: Dict) -> float:
#         """Oblicza złożoność grafu"""
#         nodes = len(plan.get("nodes", []))
#         edges = len(plan.get("edges", []))
        
#         if nodes == 0:
#             return 0.0
        
#         # Złożoność cyklomatyczna aproksymowana
#         return (edges - nodes + 2) / nodes
    
#     def _find_similar_missions(self, mission: str, top_k: int = 3) -> List[Dict]:
#         """Znajduje podobne misje z historii"""
#         similar = []
        
#         for strategy in self.successful_strategies[-20:]:  # Ostatnie 20 strategii
#             similarity = self._calculate_similarity(
#                 mission, 
#                 strategy.get("mission_type", "")
#             )
#             similar.append({
#                 "strategy": strategy,
#                 "similarity": similarity
#             })
        
#         similar.sort(key=lambda x: x["similarity"], reverse=True)
#         return similar[:top_k]
    
#     def _calculate_similarity(self, text1: str, text2: str) -> float:
#         """
#         Oblicza podobieństwo między dwoma tekstami za pomocą TF‑IDF i kosinusowej miary odległości.
#         Jeżeli którykolwiek tekst jest pusty, zwraca 0.0. Użycie TF‑IDF pozwala na lepsze
#         odzwierciedlenie znaczenia słów w różnych kontekstach.
#         """
#         if not text1 or not text2:
#             return 0.0
#         try:
#             vectorizer = TfidfVectorizer().fit([text1, text2])
#             vectors = vectorizer.transform([text1, text2])
#             sim = cosine_similarity(vectors[0], vectors[1])[0][0]
#             return float(sim)
#         except Exception:
#             # W razie błędu zwróć minimalne podobieństwo
#             return 0.0
    
#     def _get_relevant_patterns(self, mission: str) -> List[Dict]:
#         """Pobiera wzorce relevantne dla misji"""
#         relevant = []
        
#         for pattern_hash, pattern_data in self.learned_patterns.items():
#             if pattern_data["occurrences"] >= 2:  # Wzorzec musi wystąpić co najmniej 2 razy
#                 relevant.append({
#                     "pattern": pattern_data["examples"][0] if pattern_data["examples"] else "",
#                     "frequency": pattern_data["occurrences"],
#                     "solutions": pattern_data["solutions"]
#                 })
        
#         return sorted(relevant, key=lambda x: x["frequency"], reverse=True)[:5]
    
#     def _recommend_strategies(self, mission: str) -> List[str]:
#         """Rekomenduje strategie na podstawie historii"""
#         recommendations: List[str] = []
#         mission_type = self._classify_mission(mission)
#         for strat in self.successful_strategies:
#             if strat.get("mission_type") == mission_type:
#                 plan_struct = strat.get("plan_structure", {})
#                 success_factors = strat.get("success_factors", [])
#                 if plan_struct.get("has_error_handling"):
#                     recommendations.append(
#                         "Dodaj obsługę błędów – zwiększa odporność na nieprzewidziane sytuacje"
#                     )
#                 if plan_struct.get("has_validation"):
#                     recommendations.append(
#                         "Włącz kroki walidacji – pomaga wykryć odchylenia i błędne dane"
#                     )
#                 for factor in success_factors:
#                     recommendations.append(f"Zastosuj czynnik sukcesu: {factor}")
#         # Zwróć unikalne rekomendacje (maksymalnie 5)
#         return list(dict.fromkeys(recommendations))[:5]
    
#     def _get_common_pitfalls(self) -> List[str]:
#         """Zwraca najczęstsze problemy z historii"""
#         pitfalls = []
        
#         for pattern_data in self.learned_patterns.values():
#             if pattern_data["occurrences"] >= 3:
#                 pitfalls.append(f"Częsty problem ({pattern_data['occurrences']}x): {pattern_data['examples'][0]}")
        
#         return pitfalls[:5]
    
#     def _get_last_feedback(self) -> Optional[str]:
#         """Pobiera ostatni feedback jeśli istnieje"""
#         if self.episodes:
#             return self.episodes[-1]["feedback"]
#         return None
    
#     def _hash_issue(self, issue: str) -> str:
#         """Tworzy hash dla grupowania podobnych problemów"""
#         # Usuń liczby i szczegóły, zostaw istotę problemu
#         core_words = []
#         for word in issue.lower().split():
#             if len(word) > 3 and not word.isdigit():
#                 core_words.append(word)
        
#         return "_".join(sorted(core_words)[:5])
    
#     def _persist_memory(self):
#         """
#         Zapisuje pamięć do pliku JSON. Plik JSON jest bezpieczniejszy i pozwala na łatwiejszy
#         podgląd zawartości niż pickle.
#         """
#         os.makedirs("memory", exist_ok=True)
#         memory_file = "memory/learned_strategies.json"
#         data = {
#             "patterns": self.learned_patterns,
#             "strategies": self.successful_strategies
#         }
#         try:
#             with open(memory_file, "w", encoding="utf-8") as f:
#                 json.dump(data, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             print(f"⚠ Nie udało się zapisać pamięci: {e}")
    
#     def _load_persistent_memory(self):
#         """
#         Ładuje pamięć z pliku JSON, a w razie braku – z pliku pickle.
#         """
#         json_file = "memory/learned_strategies.json"
#         pickle_file = "memory/learned_strategies.pkl"
#         if os.path.exists(json_file):
#             try:
#                 with open(json_file, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 self.learned_patterns = data.get("patterns", {})
#                 self.successful_strategies = data.get("strategies", [])
#                 print("✓ Załadowano pamięć z poprzednich sesji (JSON)")
#             except Exception as e:
#                 print(f"⚠ Nie udało się załadować pamięci JSON: {e}")
#         elif os.path.exists(pickle_file):
#             try:
#                 import pickle
#                 with open(pickle_file, "rb") as f:
#                     data = pickle.load(f)
#                 self.learned_patterns = data.get("patterns", {})
#                 self.successful_strategies = data.get("strategies", [])
#                 print("✓ Załadowano pamięć z poprzednich sesji (pickle)")
#             except Exception as e:
#                 print(f"⚠ Nie udało się załadować pamięci pickle: {e}")