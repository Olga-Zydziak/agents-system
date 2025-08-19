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
    """
    Zaawansowany system pamięci dla agentów MOA
    Implementuje episodic memory, semantic memory i procedural memory
    """
    
    def __init__(self, max_episodes: int = 100):
        # Episodic Memory - konkretne wydarzenia/iteracje
        self.episodes: deque = deque(maxlen=max_episodes)
        
        # Semantic Memory - wyuczone wzorce i koncepty
        self.learned_patterns: Dict[str, Any] = {}
        
        # Procedural Memory - sprawdzone strategie
        self.successful_strategies: List[Dict] = []
        
        # Working Memory - bieżący kontekst
        self.current_context: Dict[str, Any] = {}
        
        # Meta-Memory - informacje o skuteczności pamięci
        self.memory_performance: Dict[str, float] = {
            "retrieval_accuracy": 1.0,
            "pattern_recognition_rate": 0.0
        }
        
        self._load_persistent_memory()
    
    def add_iteration_feedback(self, iteration: int, feedback: str, timestamp: datetime):
        """Dodaje feedback z iteracji do pamięci epizodycznej"""
        episode = {
            "iteration": iteration,
            "feedback": feedback,
            "timestamp": timestamp.isoformat(),
            "extracted_issues": self._extract_issues(feedback),
            "success": False  # Będzie zaktualizowane jeśli plan zostanie zatwierdzony
        }
        
        self.episodes.append(episode)
        self._update_learned_patterns(episode)
        # Zaloguj dodanie feedbacku wraz z potencjalnymi problemami
        try:
            process_log(
                f"Add iteration feedback (iter={iteration}) issues={episode['extracted_issues']}"
            )
        except Exception:
            pass
    
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
        self._persist_memory()
        # Loguj dodanie udanego planu
        try:
            process_log(
                f"Add successful plan for mission_type={strategy['mission_type']}, structure={strategy['plan_structure']}"
            )
        except Exception:
            pass
    
    def get_relevant_context(self, mission: str) -> Dict[str, Any]:
        """
        Pobiera relevantny kontekst dla danej misji
        Używa similarity search i pattern matching
        """
        context = {
            "similar_missions": self._find_similar_missions(mission),
            "relevant_patterns": self._get_relevant_patterns(mission),
            "recommended_strategies": self._recommend_strategies(mission),
            "common_pitfalls": self._get_common_pitfalls(),
            "last_feedback": self._get_last_feedback()
        }
        # Zaloguj pobranie kontekstu
        try:
            process_log(
                f"Retrieve context for mission='{mission}', suggestions={context['recommended_strategies']}"
            )
        except Exception:
            pass
        return context
    
    def _extract_issues(self, feedback: str) -> List[str]:
        """Ekstraktuje konkretne problemy z feedbacku"""
        issues: List[str] = []
        # Rozszerzona lista wskaźników problemów (różne formy i synonimy)
        problem_indicators = [
            "brak", "niewystarczający", "niewystarczająca", "niepoprawny", "niepoprawna",
            "błąd", "problem", "wadliwy", "wadliwa", "niekompletny", "niekompletna",
            "niespójny", "niespójna", "niedostateczny", "niedostateczna",
            "nieprawidłowy", "nieprawidłowa", "awaria", "usterka"
        ]
        for sentence in feedback.split("."):
            s_low = sentence.lower()
            if any(ind in s_low for ind in problem_indicators):
                stripped = sentence.strip()
                if stripped:
                    issues.append(stripped)
        return issues
    
    def _update_learned_patterns(self, episode: Dict):
        """Aktualizuje wyuczone wzorce na podstawie nowego epizodu"""
        for issue in episode["extracted_issues"]:
            # Tworzymy hash problemu dla grupowania podobnych
            issue_hash = self._hash_issue(issue)
            
            if issue_hash not in self.learned_patterns:
                self.learned_patterns[issue_hash] = {
                    "occurrences": 0,
                    "examples": [],
                    "solutions": []
                }
            
            self.learned_patterns[issue_hash]["occurrences"] += 1
            self.learned_patterns[issue_hash]["examples"].append(issue)
    
    def _classify_mission(self, mission: str) -> str:
        """Klasyfikuje typ misji"""
        mission_lower = mission.lower()
        
        if "przyczynow" in mission_lower or "causal" in mission_lower:
            return "causal_analysis"
        elif "dane" in mission_lower or "data" in mission_lower:
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
    
    def _find_similar_missions(self, mission: str, top_k: int = 3) -> List[Dict]:
        """Znajduje podobne misje z historii"""
        similar = []
        
        for strategy in self.successful_strategies[-20:]:  # Ostatnie 20 strategii
            similarity = self._calculate_similarity(
                mission, 
                strategy.get("mission_type", "")
            )
            similar.append({
                "strategy": strategy,
                "similarity": similarity
            })
        
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:top_k]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Oblicza podobieństwo między dwoma tekstami za pomocą TF‑IDF i kosinusowej miary odległości.
        Jeżeli którykolwiek tekst jest pusty, zwraca 0.0. Użycie TF‑IDF pozwala na lepsze
        odzwierciedlenie znaczenia słów w różnych kontekstach.
        """
        if not text1 or not text2:
            return 0.0
        try:
            vectorizer = TfidfVectorizer().fit([text1, text2])
            vectors = vectorizer.transform([text1, text2])
            sim = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(sim)
        except Exception:
            # W razie błędu zwróć minimalne podobieństwo
            return 0.0
    
    def _get_relevant_patterns(self, mission: str) -> List[Dict]:
        """Pobiera wzorce relevantne dla misji"""
        relevant = []
        
        for pattern_hash, pattern_data in self.learned_patterns.items():
            if pattern_data["occurrences"] >= 2:  # Wzorzec musi wystąpić co najmniej 2 razy
                relevant.append({
                    "pattern": pattern_data["examples"][0] if pattern_data["examples"] else "",
                    "frequency": pattern_data["occurrences"],
                    "solutions": pattern_data["solutions"]
                })
        
        return sorted(relevant, key=lambda x: x["frequency"], reverse=True)[:5]
    
    def _recommend_strategies(self, mission: str) -> List[str]:
        """Rekomenduje strategie na podstawie historii"""
        recommendations: List[str] = []
        mission_type = self._classify_mission(mission)
        for strat in self.successful_strategies:
            if strat.get("mission_type") == mission_type:
                plan_struct = strat.get("plan_structure", {})
                success_factors = strat.get("success_factors", [])
                if plan_struct.get("has_error_handling"):
                    recommendations.append(
                        "Dodaj obsługę błędów – zwiększa odporność na nieprzewidziane sytuacje"
                    )
                if plan_struct.get("has_validation"):
                    recommendations.append(
                        "Włącz kroki walidacji – pomaga wykryć odchylenia i błędne dane"
                    )
                for factor in success_factors:
                    recommendations.append(f"Zastosuj czynnik sukcesu: {factor}")
        # Zwróć unikalne rekomendacje (maksymalnie 5)
        return list(dict.fromkeys(recommendations))[:5]
    
    def _get_common_pitfalls(self) -> List[str]:
        """Zwraca najczęstsze problemy z historii"""
        pitfalls = []
        
        for pattern_data in self.learned_patterns.values():
            if pattern_data["occurrences"] >= 3:
                pitfalls.append(f"Częsty problem ({pattern_data['occurrences']}x): {pattern_data['examples'][0]}")
        
        return pitfalls[:5]
    
    def _get_last_feedback(self) -> Optional[str]:
        """Pobiera ostatni feedback jeśli istnieje"""
        if self.episodes:
            return self.episodes[-1]["feedback"]
        return None
    
    def _hash_issue(self, issue: str) -> str:
        """Tworzy hash dla grupowania podobnych problemów"""
        # Usuń liczby i szczegóły, zostaw istotę problemu
        core_words = []
        for word in issue.lower().split():
            if len(word) > 3 and not word.isdigit():
                core_words.append(word)
        
        return "_".join(sorted(core_words)[:5])
    
    def _persist_memory(self):
        """
        Zapisuje pamięć do pliku JSON. Plik JSON jest bezpieczniejszy i pozwala na łatwiejszy
        podgląd zawartości niż pickle.
        """
        os.makedirs("memory", exist_ok=True)
        memory_file = "memory/learned_strategies.json"
        data = {
            "patterns": self.learned_patterns,
            "strategies": self.successful_strategies
        }
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Nie udało się zapisać pamięci: {e}")
    
    def _load_persistent_memory(self):
        """
        Ładuje pamięć z pliku JSON, a w razie braku – z pliku pickle.
        """
        json_file = "memory/learned_strategies.json"
        pickle_file = "memory/learned_strategies.pkl"
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.learned_patterns = data.get("patterns", {})
                self.successful_strategies = data.get("strategies", [])
                print("✓ Załadowano pamięć z poprzednich sesji (JSON)")
            except Exception as e:
                print(f"⚠ Nie udało się załadować pamięci JSON: {e}")
        elif os.path.exists(pickle_file):
            try:
                import pickle
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                self.learned_patterns = data.get("patterns", {})
                self.successful_strategies = data.get("strategies", [])
                print("✓ Załadowano pamięć z poprzednich sesji (pickle)")
            except Exception as e:
                print(f"⚠ Nie udało się załadować pamięci pickle: {e}")