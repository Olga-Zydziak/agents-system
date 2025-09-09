import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from process_logger import log as process_log

class ExplainabilityLayer:
    """
    Dodaje wyjaśnialność do istniejącego systemu MOA bez większych zmian
    """
    
    @staticmethod
    def inject_explainability_prompt(base_prompt: str, agent_role: str) -> str:
        """
        Dodaje instrukcje wyjaśnialności do promptu - MODEL PRZEZ API TO ZROZUMIE
        """
        
        explainability_suffix = """

## EXPLAINABILITY REQUIREMENTS (MANDATORY)
You must include an additional JSON field "cognitive_trace" in your response with:

```json
"cognitive_trace": {
    "trigger_words": [/* actual words from prompt that influenced you */],
    "reasoning_chain": [/* your actual reasoning steps */],
    "confidence_per_decision": {
        /* your actual confidence values between 0-1 for each decision type */
        "node_selection": /* your confidence */,
        "edge_conditions": /* your confidence */,
        "overall_structure": /* your confidence */
    },
    "alternatives_considered": {
        "rejected_nodes": [/* nodes you considered but rejected with reasons */],
        "rejected_patterns": [/* patterns you didn't use and why */]
    },
    "key_influences": {
        "from_prompt": /* what specific instruction shaped this most */,
        "from_context": /* what contextual factor was crucial */,
        "from_role": /* how your assigned role affected this */
    },
    "uncertainty_points": [/* where you're least certain and why */],
    "word_choices": {
        /* actual words you chose and why */
    }
}
This is REQUIRED - include it after your main plan/response.
"""
        return base_prompt + explainability_suffix
    @staticmethod
    def extract_cognitive_trace(response: str) -> Optional[Dict]:
        """
        Wyciąga cognitive_trace z odpowiedzi modelu
        """
        try:
            # Szukaj cognitive_trace w odpowiedzi
            if isinstance(response, dict):
                return response.get("cognitive_trace")

            # Jeśli string, parsuj JSON
            if isinstance(response, str):
                # Usuń markdown jeśli jest
                cleaned = re.sub(r'```json?\s*|\s*```', '', response)
                parsed = json.loads(cleaned)
                return parsed.get("cognitive_trace")
        except:
            # Fallback - spróbuj znaleźć wzorzec
            pattern = r'"cognitive_trace":\s*\{[^}]+\}'
            match = re.search(pattern, str(response))
            if match:
                try:
                    return json.loads("{" + match.group() + "}")["cognitive_trace"]
                except:
                    pass
            return None

    @staticmethod
    def analyze_response_semantics(prompt: str, response: str, agent_name: str) -> Dict:
        """
        Analizuje związki semantyczne między promptem a odpowiedzią
        """
        analysis = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "semantic_markers": {}
        }

        # Kluczowe słowa z promptu które pojawiły się w odpowiedzi
        prompt_keywords = set(re.findall(r'\b[A-Za-z_]+\b', prompt.lower()))
        response_keywords = set(re.findall(r'\b[A-Za-z_]+\b', response.lower()))

        analysis["semantic_markers"]["keyword_overlap"] = list(prompt_keywords & response_keywords)[:20]
        analysis["semantic_markers"]["new_concepts"] = list(response_keywords - prompt_keywords)[:20]

        # Wykryj wzorce decyzyjne
        if "error" in response.lower():
            analysis["semantic_markers"]["error_handling_focus"] = True
        if "rollback" in response.lower():
            analysis["semantic_markers"]["rollback_strategy"] = True
        if "optimiz" in response.lower():
            analysis["semantic_markers"]["optimization_focus"] = True

        return analysis

    @staticmethod
    def create_explainability_report(all_analyses: List[Dict]) -> Dict:
        """
        Tworzy raport wyjaśnialności dla całej debaty
        """
        report = {
            "debate_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_turns": len(all_analyses),
            "agents_involved": list(set(a["agent"] for a in all_analyses if "agent" in a)),
            "cognitive_patterns": {},
            "decision_evolution": [],
            "key_influences": {},
            "uncertainty_map": {}
        }

        # Agreguj cognitive traces
        for analysis in all_analyses:
            if "cognitive_trace" in analysis:
                trace = analysis["cognitive_trace"]
                agent = analysis.get("agent", "unknown")

                if agent not in report["cognitive_patterns"]:
                    report["cognitive_patterns"][agent] = []

                report["cognitive_patterns"][agent].append({
                    "triggers": trace.get("trigger_words", []),
                    "confidence": trace.get("confidence_per_decision", {}),
                    "alternatives": trace.get("alternatives_considered", {})
                })

                # Mapuj niepewności
                for uncertainty in trace.get("uncertainty_points", []):
                    if uncertainty not in report["uncertainty_map"]:
                        report["uncertainty_map"][uncertainty] = []
                    report["uncertainty_map"][uncertainty].append(agent)

        return report



class ExplainabilityHooks:
        """
        Hooki do wstrzyknięcia w istniejący kod z minimalną ingerencją
        """
    def __init__(self):
        self.layer = ExplainabilityLayer()
        self.session_analyses = []
    
    def on_prompt_build(self, base_prompt: str, agent_role: str) -> str:
        """
        Hook wywoływany przy budowaniu promptu - DODAJ TO DO TWOJEJ METODY BUDOWANIA PROMPTÓW
        """
        return self.layer.inject_explainability_prompt(base_prompt, agent_role)

    def on_response_received(self, prompt: str, response: str, agent_name: str) -> Dict:
        """
        Hook wywoływany po otrzymaniu odpowiedzi - DODAJ TO PO OTRZYMANIU ODPOWIEDZI Z API
        """
        # Ekstraktuj cognitive trace
        cognitive_trace = self.layer.extract_cognitive_trace(response)

        # Analiza semantyczna
        semantic_analysis = self.layer.analyze_response_semantics(prompt, response, agent_name)

        # Połącz analizy
        full_analysis = {
            **semantic_analysis,
            "cognitive_trace": cognitive_trace,
            "raw_response_sample": response[:500] if isinstance(response, str) else str(response)[:500]
        }

        # Zapisz do sesji
        self.session_analyses.append(full_analysis)

        # Loguj kluczowe informacje
        if cognitive_trace:
            process_log(f"[EXPLAINABILITY] {agent_name} confidence: {cognitive_trace.get('confidence_per_decision', {})}")
            process_log(f"[EXPLAINABILITY] Key triggers: {cognitive_trace.get('trigger_words', [])[:5]}")

        return full_analysis

    def generate_final_report(self) -> Dict:
        """
        Generuje końcowy raport wyjaśnialności
        """
        report = self.layer.create_explainability_report(self.session_analyses)

        # Zapisz do pliku
        report_file = f"explainability_reports/report_{report['debate_id']}.json"
        import os
        os.makedirs("explainability_reports", exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        process_log(f"[EXPLAINABILITY] Report saved to: {report_file}")
        return report