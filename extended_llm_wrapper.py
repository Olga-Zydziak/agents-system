"""
Rozszerzony wrapper LLM z bardziej realistycznymi dummy responses dla różnych ról
"""
import json
import random
from typing import Dict, Any

class ExtendedLLMWrapper:
    """
    Rozszerzona wersja wrappera z różnorodnymi odpowiedziami dla demo
    """
    
    @staticmethod
    def generate_dummy_response(model_name: str, prompt: str) -> str:
        """Generuje różne odpowiedzi w zależności od typu agenta"""
        
        # Sprawdź typ agenta na podstawie nazwy modelu lub promptu
        if "causal" in model_name.lower() or "Causal" in prompt:
            return ExtendedLLMWrapper._causal_analyst_response()
        elif "creative" in model_name.lower() or "Creative" in prompt:
            return ExtendedLLMWrapper._creative_planner_response()
        elif "risk" in model_name.lower() or "Risk" in prompt:
            return ExtendedLLMWrapper._risk_analyst_response()
        elif "aggregator" in model_name.lower() or "Aggregator" in prompt:
            return ExtendedLLMWrapper._aggregator_response(prompt)
        elif "critic" in model_name.lower() or "Critic" in prompt:
            return ExtendedLLMWrapper._critic_response(prompt)
        else:
            return ExtendedLLMWrapper._default_response()
    
    @staticmethod
    def _causal_analyst_response() -> str:
        """Odpowiedź analityka przyczynowego"""
        response = {
            "thought_process": [
                "Analizuję potencjalne relacje przyczynowe w przepływie danych",
                "Identyfikuję zmienne confounding i mediatory",
                "Projektuję DAG (Directed Acyclic Graph) dla workflow"
            ],
            "plan": {
                "entry_point": "validate_data",
                "nodes": [
                    {"name": "validate_data", "implementation": "validate_data"},
                    {"name": "check_quality", "implementation": "check_quality"},
                    {"name": "discover_causality", "implementation": "discover_causality"},
                    {"name": "error_handler", "implementation": "error_handler"},
                    {"name": "validate_model", "implementation": "validate_model"},
                    {"name": "generate_report", "implementation": "generate_report"}
                ],
                "edges": [
                    {"from": "validate_data", "to": "check_quality"},
                    {"from": "check_quality", "to": "discover_causality"},
                    {"from": "discover_causality", "to": "validate_model", "condition": "check_success"},
                    {"from": "discover_causality", "to": "error_handler", "condition": "check_error"},
                    {"from": "error_handler", "to": "discover_causality"},
                    {"from": "validate_model", "to": "generate_report"}
                ]
            },
            "confidence": 0.85,
            "key_innovations": [
                "Dodanie pętli retry dla discover_causality",
                "Walidacja jakości przed analizą przyczynową"
            ],
            "risk_mitigation": {
                "data_quality": "Podwójna walidacja przed analizą",
                "algorithm_failure": "Error handler z retry mechanism"
            }
        }
        return json.dumps(response)
    
    @staticmethod
    def _creative_planner_response() -> str:
        """Odpowiedź kreatywnego planera"""
        response = {
            "thought_process": [
                "Myślę nieszablonowo - co gdyby pipeline sam się optymalizował?",
                "Inspiracja z natury: mrówki znajdują optymalną ścieżkę",
                "Dodaję element adaptacyjności i uczenia się"
            ],
            "plan": {
                "entry_point": "load_data",
                "nodes": [
                    {"name": "load_data", "implementation": "load_data"},
                    {"name": "clean_data", "implementation": "clean_data"},
                    {"name": "optimize_performance", "implementation": "optimize_performance"},
                    {"name": "discover_causality", "implementation": "discover_causality"},
                    {"name": "train_model", "implementation": "train_model"},
                    {"name": "notify_user", "implementation": "notify_user"}
                ],
                "edges": [
                    {"from": "load_data", "to": "clean_data"},
                    {"from": "clean_data", "to": "optimize_performance"},
                    {"from": "optimize_performance", "to": "discover_causality"},
                    {"from": "discover_causality", "to": "train_model"},
                    {"from": "train_model", "to": "notify_user"}
                ]
            },
            "confidence": 0.75,
            "key_innovations": [
                "Samooptymalizacja pipeline'u",
                "Proaktywne powiadomienia użytkownika",
                "Adaptacyjne dostosowanie do typu danych"
            ],
            "risk_mitigation": {
                "performance": "Continuous optimization",
                "user_experience": "Real-time notifications"
            }
        }
        return json.dumps(response)
    
    @staticmethod
    def _risk_analyst_response() -> str:
        """Odpowiedź analityka ryzyka"""
        response = {
            "thought_process": [
                "Identyfikuję wszystkie możliwe punkty awarii",
                "Analizuję cascading failures",
                "Projektuję redundancję i fallback paths"
            ],
            "plan": {
                "entry_point": "validate_data",
                "nodes": [
                    {"name": "validate_data", "implementation": "validate_data"},
                    {"name": "clean_data", "implementation": "clean_data"},
                    {"name": "check_quality", "implementation": "check_quality"},
                    {"name": "discover_causality", "implementation": "discover_causality"},
                    {"name": "error_handler", "implementation": "error_handler"},
                    {"name": "rollback", "implementation": "rollback"},
                    {"name": "validate_model", "implementation": "validate_model"},
                    {"name": "generate_report", "implementation": "generate_report"}
                ],
                "edges": [
                    {"from": "validate_data", "to": "clean_data"},
                    {"from": "clean_data", "to": "check_quality"},
                    {"from": "check_quality", "to": "discover_causality", "condition": "quality_ok"},
                    {"from": "check_quality", "to": "rollback", "condition": "quality_fail"},
                    {"from": "discover_causality", "to": "validate_model", "condition": "success"},
                    {"from": "discover_causality", "to": "error_handler", "condition": "error"},
                    {"from": "error_handler", "to": "rollback", "condition": "cannot_recover"},
                    {"from": "error_handler", "to": "discover_causality", "condition": "can_retry"},
                    {"from": "validate_model", "to": "generate_report"},
                    {"from": "rollback", "to": "generate_report"}
                ]
            },
            "confidence": 0.90,
            "key_innovations": [
                "Comprehensive error handling",
                "Multiple fallback paths",
                "Quality gates at critical points"
            ],
            "risk_mitigation": {
                "data_corruption": "Rollback mechanism",
                "algorithm_failure": "Multiple retry with degradation",
                "quality_issues": "Early detection and abort"
            }
        }
        return json.dumps(response)
    
    @staticmethod
    def _aggregator_response(prompt: str) -> str:
        """Odpowiedź agregatora - synteza propozycji"""
        # Sprawdź iterację jeśli jest w prompcie
        iteration = 1
        if "ITERATION:" in prompt:
            try:
                iteration = int(prompt.split("ITERATION:")[1].split("/")[0].strip())
            except:
                pass
        
        response = {
            "thought_process": [
                "Analizuję siły każdej propozycji",
                "Identyfikuję synergie między podejściami",
                "Łączę najlepsze elementy w spójną całość"
            ],
            "final_plan": {
                "entry_point": "validate_data",
                "nodes": [
                    {"name": "validate_data", "implementation": "validate_data"},
                    {"name": "clean_data", "implementation": "clean_data"},
                    {"name": "check_quality", "implementation": "check_quality"},
                    {"name": "optimize_performance", "implementation": "optimize_performance"},
                    {"name": "discover_causality", "implementation": "discover_causality"},
                    {"name": "error_handler", "implementation": "error_handler"},
                    {"name": "rollback", "implementation": "rollback"},
                    {"name": "train_model", "implementation": "train_model"},
                    {"name": "validate_model", "implementation": "validate_model"},
                    {"name": "generate_report", "implementation": "generate_report"},
                    {"name": "notify_user", "implementation": "notify_user"}
                ],
                "edges": [
                    {"from": "validate_data", "to": "clean_data"},
                    {"from": "clean_data", "to": "check_quality"},
                    {"from": "check_quality", "to": "optimize_performance", "condition": "quality_ok"},
                    {"from": "check_quality", "to": "rollback", "condition": "quality_fail"},
                    {"from": "optimize_performance", "to": "discover_causality"},
                    {"from": "discover_causality", "to": "train_model", "condition": "success"},
                    {"from": "discover_causality", "to": "error_handler", "condition": "error"},
                    {"from": "error_handler", "to": "rollback", "condition": "max_retries"},
                    {"from": "error_handler", "to": "discover_causality", "condition": "can_retry"},
                    {"from": "train_model", "to": "validate_model"},
                    {"from": "validate_model", "to": "generate_report"},
                    {"from": "generate_report", "to": "notify_user"}
                ]
            },
            "synthesis_reasoning": "Połączyłem solidną obsługę błędów od Risk Analyst, innowacyjną optymalizację od Creative Planner, i rygorystyczną walidację od Causal Analyst",
            "component_sources": {
                "Causal Analyst": ["validate_data", "check_quality", "validate_model"],
                "Creative Planner": ["optimize_performance", "notify_user"],
                "Risk Analyst": ["error_handler", "rollback", "conditional_edges"]
            },
            "confidence_score": 0.80 + iteration * 0.05,  # Rośnie z iteracjami
            "improvements": [
                "Dodanie cache dla powtarzalnych operacji",
                "Implementacja progressive enhancement",
                "Monitoring w czasie rzeczywistym"
            ]
        }
        return json.dumps(response)
    
    @staticmethod
    def _critic_response(prompt: str) -> str:
        """Odpowiedź krytyka - ocena planu"""
        # Sprawdź iterację
        iteration = 1
        if "ITERATION:" in prompt:
            try:
                iteration = int(prompt.split("ITERATION:")[1].split("/")[0].strip())
            except:
                pass
        
        # Dostosuj ocenę do iteracji
        base_score = 60 + iteration * 8
        approved = base_score >= 75 or iteration >= 4
        
        response = {
            "approved": approved,
            "score": min(base_score + random.randint(-5, 10), 95),
            "strengths": [
                "Comprehensive error handling",
                "Good balance between robustness and efficiency",
                "Clear separation of concerns",
                "Innovative optimization approach"
            ][:2 + iteration],  # Więcej mocnych stron w późniejszych iteracjach
            "weaknesses": [
                "Missing parallelization opportunities",
                "No caching mechanism",
                "Limited monitoring capabilities",
                "Could benefit from more granular error types"
            ][iteration-1:],  # Mniej słabości w późniejszych iteracjach
            "feedback": f"Plan shows {'significant' if iteration > 2 else 'good'} improvement. {'Ready for deployment.' if approved else 'Further refinement needed.'}",
            "improvements": [
                "Add parallel processing for independent steps",
                "Implement result caching",
                "Add detailed logging and monitoring",
                "Consider adding A/B testing capability"
            ][iteration-1:] if not approved else []
        }
        
        # KLUCZOWA ZMIANA: Dodaj frazę "PLAN_ZATWIERDZONY" jeśli zatwierdzamy
        response_json = json.dumps(response)
        
        if approved:
            # Dodaj magiczną frazę PO JSONie
            response_json += "\n\nPLAN_ZATWIERDZONY"
        
        return response_json
    
    @staticmethod
    def _default_response() -> str:
        """Domyślna odpowiedź"""
        response = {
            "thought_process": ["Analyzing task", "Creating plan"],
            "plan": {
                "entry_point": "load_data",
                "nodes": [
                    {"name": "load_data", "implementation": "load_data"},
                    {"name": "process", "implementation": "clean_data"},
                    {"name": "output", "implementation": "generate_report"}
                ],
                "edges": [
                    {"from": "load_data", "to": "process"},
                    {"from": "process", "to": "output"}
                ]
            },
            "confidence": 0.7
        }
        return json.dumps(response)

# Zastąp oryginalną klasę LLMWrapper
import llm_wrapper
original_call = llm_wrapper.LLMWrapper.__call__

def enhanced_call(self, prompt: str) -> str:
    """Rozszerzone wywołanie z lepszymi dummy responses"""
    if self.provider == "dummy":
        return ExtendedLLMWrapper.generate_dummy_response(self.model_name, prompt)
    else:
        return original_call(self, prompt)

# Monkey-patch oryginalnej klasy
llm_wrapper.LLMWrapper.__call__ = enhanced_call