"""
Structured parser oparty na Pydantic.  Zamiast heurystycznych prób parsowania
ręcznego, wykorzystuje schematy Pydantic do walidacji odpowiedzi LLM.  Ten
moduł zastępuje dotychczasowy `response_parser` w nowej konfiguracji.

Model `ProposerResponse` definiuje minimalną strukturę planu wygenerowanego
przez agentów‑proposerów.  Model `AggregatorResponse` rozszerza go o pole
`final_plan` oraz metadane używane przez agregatora.  Model `CriticResponse`
zawiera ocenę, listę mocnych i słabych stron oraz ewentualne sugestie
poprawek, zgodnie z założonym formatem JSON.

Jeśli odpowiedź nie jest poprawnym JSON‑em (np. zawiera `````markdown````
fences) lub nie spełnia schematu, parser zwraca `None`.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Dict, Any
from process_logger import log as process_log
from pydantic import BaseModel, ValidationError, Field


class ProposerPlan(BaseModel):
    """Reprezentuje plan proponowany przez agenta‐proposera."""

    entry_point: str = Field(..., description="Nazwa pierwszego węzła w planie")
    nodes: List[Dict[str, Any]] = Field(..., description="Lista węzłów planu")
    edges: List[Dict[str, Any]] = Field(..., description="Lista krawędzi planu")


class ProposerResponse(BaseModel):
    """Struktura odpowiedzi agenta proponującego."""

    thought_process: List[str] = Field(..., description="Opis kroków rozumowania")
    plan: ProposerPlan = Field(..., description="Plan w formacie grafu")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pewność (0–1)")
    key_innovations: Optional[List[str]] = Field(default_factory=list)
    risk_mitigation: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AggregatorResponse(BaseModel):
    """Struktura odpowiedzi agregatora.  Rozszerza odpowiedź proponera o finalny plan."""

    thought_process: List[str]
    final_plan: ProposerPlan
    synthesis_reasoning: Optional[str]
    component_sources: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    improvements: Optional[List[str]] = Field(default_factory=list)


class CriticResponse(BaseModel):
    """Struktura odpowiedzi krytyka."""

    approved: bool
    score: float = Field(..., ge=0.0, le=100.0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    feedback: Optional[str]
    improvements: Optional[List[str]] = Field(default_factory=list)


class StructuredResponseParser:
    """
    Parser, który wykorzystuje modele Pydantic do walidacji i konwersji odpowiedzi
    na słowniki.  Oczekuje, że agent zwraca poprawny JSON zgodny z jednym z
    powyższych schematów.  Można łatwo rozszerzyć o kolejne typy odpowiedzi.
    """

    def __init__(self) -> None:
        pass

    def _strip_code_fences(self, response: str) -> str:
        """Usuwa bloki kodu (```json ... ```) z odpowiedzi."""
        # Usuń bloki ```json ... ``` lub ``` ... ```
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response

    def parse_agent_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Przetwarza odpowiedź agenta i próbuje ją zmapować na jeden z
        zdefiniowanych modeli.  Zwraca zserializowaną postać słownikową,
        lub None, jeśli nie można sparsować.
        """
        if not response:
            return None

        # Usuń otaczające bloki kodu
        cleaned = self._strip_code_fences(response.strip())

        # Spróbuj sparsować jako JSON
        try:
            data = json.loads(cleaned)
        except Exception:
            return None

        # Kolejno próbuj dopasować do modeli
        for model_cls in (ProposerResponse, AggregatorResponse, CriticResponse):
            try:
                obj = model_cls.parse_obj(data)
                return obj.dict()
            except ValidationError:
                continue

        # Jeśli nic nie pasuje, zwróć oryginalne dane
        return data
    
    def parse_critic_response(self, text: str):
        """
        Parsuje odpowiedź krytyka - zwraca CAŁY JSON
        """
        import json
        import re

        if not text:
            return None

        try:
            # Usuń markdown code blocks
            clean_text = text.strip()

            # Usuń ```json i ```
            clean_text = re.sub(r'```json\s*', '', clean_text)
            clean_text = re.sub(r'```\s*', '', clean_text)

            # Usuń PLAN_ZATWIERDZONY z końca
            if "PLAN_ZATWIERDZONY" in clean_text:
                # Znajdź ostatnie wystąpienie i usuń wszystko po nim
                parts = clean_text.rsplit("PLAN_ZATWIERDZONY", 1)
                clean_text = parts[0].strip()

            # Teraz po prostu sparsuj JSON
            result = json.loads(clean_text)

            # Debug - wypisz co znalazłeś
            process_log(f"[PARSER] Znaleziono klucze: {list(result.keys())}")

            return result

        except json.JSONDecodeError as e:
            process_log(f"[PARSER] JSON decode error: {e}")

            # Plan B - znajdź JSON manualnie
            try:
                # Znajdź od pierwszego { do ostatniego }
                start = text.find('{')
                end = text.rfind('}')

                if start >= 0 and end > start:
                    json_str = text[start:end+1]
                    return json.loads(json_str)
            except:
                pass

        return None