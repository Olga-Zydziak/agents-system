"""
Wrapper modeli LLM. Umożliwia łatwą zamianę źródła modelu (np. OpenAI, lokalny model itp.).
W tym przykładzie implementujemy klasę `LLMWrapper`, która w trybie demonstracyjnym
generuje sztuczną odpowiedź. Aby użyć prawdziwego modelu (np. GPT‑5), należy
uzupełnić implementację wywołania API w metodzie `__call__`.
"""

import os
import json


class LLMWrapper:
    def __init__(self, provider: str, model_name: str, api_key_env: str = None, temperature: float = 0.5):
        """
        :param provider: dostawca modelu, np. "openai" lub "dummy" dla demonstracji
        :param model_name: nazwa modelu u dostawcy
        :param api_key_env: nazwa zmiennej środowiskowej z kluczem API
        :param temperature: parametr kreatywności dla modeli typu GPT
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.environ.get(api_key_env) if api_key_env else None

    def __call__(self, prompt: str) -> str:
        """
        Zwraca odpowiedź modelu na dany prompt. W wersji demonstracyjnej,
        jeśli provider to "dummy", generuje prosty plan w formacie JSON.
        W przeciwnym razie wymaga zaimplementowania wywołania API.
        """
        if self.provider == "dummy":
            # Zwróć przykładowy JSON jako ciąg znaków
            response = {
                "thought_process": ["Analiza zadania", "Propozycja rozwiązania"],
                "plan": {
                    "entry_point": "start",
                    "nodes": [
                        {"name": "start", "implementation": "init_task"},
                        {"name": "finish", "implementation": "end_task"}
                    ],
                    "edges": [
                        {"from": "start", "to": "finish"}
                    ]
                },
                "confidence": 0.85
            }
            return json.dumps(response)
        elif self.provider == "openai":
            # Przykład wywołania OpenAI ChatCompletion – wymaga biblioteki openai i klucza API
            try:
                import openai  # zaimportuj wewnątrz, aby uniknąć zależności dla dummy
            except ImportError:
                raise RuntimeError("Biblioteka openai nie jest zainstalowana. Zainstaluj ją lub użyj provider='dummy'.")
            if not self.api_key:
                raise RuntimeError("Brak klucza API. Ustaw zmienną środowiskową lub przekaż api_key_env.")
            openai.api_key = self.api_key
            # Buduj listę wiadomości zgodnie z API ChatCompletion
            messages = [
                {"role": "system", "content": "You are an advanced planning agent."},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message["content"]
        else:
            raise NotImplementedError(f"Provider '{self.provider}' nie jest obsługiwany.")