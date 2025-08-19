"""
Inteligentny parser odpowiedzi agentów z auto-korekcją
"""
import json
import re
from typing import Dict, Any, Optional
import ast

# Lokalny logger procesu
from process_logger import log as process_log

class ResponseParser:
    """
    Zaawansowany parser który radzi sobie z różnymi formatami odpowiedzi
    """
    
    def parse_agent_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parsuje odpowiedź agenta próbując różnych strategii
        """
        if not response:
            return None
        # Zaloguj otrzymaną odpowiedź (obcinamy do 200 znaków, aby log nie rósł nadmiernie)
        process_log(f"Received response: {response[:200]}")
        
        # Strategia 1: Czysty JSON
        parsed = self._try_pure_json(response)
        if parsed:
            process_log(f"Parsed using pure JSON: {parsed}")
            return parsed
        
        # Strategia 2: JSON z dodatkami (markdown, komentarze)
        parsed = self._try_extract_json(response)
        if parsed:
            process_log(f"Parsed using extract JSON: {parsed}")
            return parsed
        
        # Strategia 3: Python dict jako string (bez wykonywania kodu)
        parsed = self._try_python_dict(response)
        if parsed:
            process_log(f"Parsed using python-like dict: {parsed}")
            return parsed
        
        # Strategia 4: Strukturalna ekstrakcja
        parsed = self._try_structural_extraction(response)
        if parsed:
            process_log(f"Parsed using structural extraction: {parsed}")
            return parsed
        
        # Strategia 5: AI-based repair (używa regex i heurystyk)
        parsed = self._try_ai_repair(response)
        if parsed:
            process_log(f"Parsed using AI repair: {parsed}")
            return parsed
        
        process_log(f"Parse failed: {response[:200]}")
        print(f"⚠ Nie udało się sparsować odpowiedzi: {response[:100]}...")
        return None
    
    def _try_pure_json(self, response: str) -> Optional[Dict]:
        """Próbuje parsować jako czysty JSON"""
        try:
            return json.loads(response.strip())
        except:
            return None
    
    def _try_extract_json(self, response: str) -> Optional[Dict]:
        """Ekstraktuje JSON z tekstu"""
        # Szukamy JSON w blokach kodu
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Szukamy pierwszego { i ostatniego }
        start = response.find('{')
        end = response.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end+1])
            except:
                pass
        
        return None
    
    def _try_python_dict(self, response: str) -> Optional[Dict]:
        """
        Próbuje sparsować słownik zapisany w notacji Pythona bez użycia eval. Wyszukuje
        pierwszą strukturę w nawiasach klamrowych, następnie zamienia pojedyncze cudzysłowy
        na podwójne i dodaje cudzysłowy do kluczy, aby użyć json.loads. Jeśli napotka błąd,
        zwraca None.
        """
        try:
            # Wyszukaj fragment przypominający słownik
            dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(dict_pattern, response)
            if not match:
                return None
            obj_str = match.group(0)
            # Zamień pojedyncze cudzysłowy na podwójne
            json_like = obj_str.replace("'", '"')
            # Dodaj cudzysłowy do kluczy, jeśli ich brakuje
            json_like = re.sub(r'(?<!\")\b([A-Za-z_][A-Za-z0-9_]*)\b\s*:', r'"\1":', json_like)
            return json.loads(json_like)
        except Exception:
            return None
    
    def _try_structural_extraction(self, response: str) -> Optional[Dict]:
        """Ekstraktuje strukturę na podstawie kluczowych słów"""
        result = {}
        
        # Szukamy kluczowych sekcji
        patterns = {
            "thought_process": r'(?:thought_process|thinking|reasoning)[:\s]+([^\n]+(?:\n(?!\w+:)[^\n]+)*)',
            "entry_point": r'(?:entry_point|start)[:\s]+["\']?(\w+)["\']?',
            "confidence": r'(?:confidence|certainty)[:\s]+(\d*\.?\d+)',
            "nodes": r'nodes[:\s]+\[(.*?)\]',
            "edges": r'edges[:\s]+\[(.*?)\]'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                
                if key == "confidence":
                    try:
                        result[key] = float(value)
                    except:
                        result[key] = 0.5
                elif key in ["nodes", "edges"]:
                    # Próbuj sparsować jako listę
                    try:
                        result[key] = ast.literal_eval(f"[{value}]")
                    except:
                        result[key] = []
                elif key == "thought_process":
                    # Podziel na kroki
                    steps = [s.strip() for s in value.split('\n') if s.strip()]
                    result[key] = steps
                else:
                    result[key] = value
        
        return result if result else None
    
    def _try_ai_repair(self, response: str) -> Optional[Dict]:
        """Próbuje naprawić JSON używając heurystyk"""
        # Usuń komentarze
        response = re.sub(r'//.*?\n', '', response)
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
        
        # Napraw typowe błędy
        repairs = [
            (r',\s*}', '}'),  # Usuń trailing commas
            (r',\s*]', ']'),
            (r'"\s*:\s*"([^"]*)"(?=[,}])', r'": "\1"'),  # Napraw cudzysłowy
            (r'(\w+)(?=\s*:)', r'"\1"'),  # Dodaj cudzysłowy do kluczy
            (r':\s*([^",\[\{}\]]+)(?=[,}])', r': "\1"'),  # Dodaj cudzysłowy do wartości
        ]
        
        for pattern, replacement in repairs:
            response = re.sub(pattern, replacement, response)
        
        # Spróbuj ponownie
        return self._try_pure_json(response)