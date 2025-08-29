import os
import json
import vertexai
from vertexai import agent_engines
from typing import Dict, List, Optional



# --------------------

def query_mission_memory(
    client: vertexai.Client, 
    engine_name: str, 
    query_text: str, 
    scope: Optional[Dict] = None, 
    top_k: int = 5
) -> List[Dict]:
    """
    Odpytuje pamięć semantycznie na podstawie zapytania i zakresu (scope),
    a następnie zwraca listę sparsowanych wspomnień jako słowniki.
    """
    if scope is None:
        scope = {"source": "learned_strategies_json"}
        
    retrieved_mems = []
    print(f"\n--- 🚀 Odpytuję pamięć z zapytaniem: '{query_text}' | Zakres: {scope} ---")

    try:
        # Parametry wyszukiwania semantycznego
        search_params = {
            "search_query": query_text,
            "top_k": top_k
        }

        # Wywołanie API do pobrania wspomnień
        memories_iterator = client.agent_engines.retrieve_memories(
            name=engine_name,
            scope=scope,
            similarity_search_params=search_params
        )

        # Iteracja po wynikach i parsowanie
        for i, mem in enumerate(memories_iterator):
            try:
                # Dostęp do surowego faktu (który jest stringiem JSON)
                json_string_fact = mem.memory.fact
                
                # Parsowanie stringa JSON do słownika Pythonowego
                parsed_record = json.loads(json_string_fact)
                retrieved_mems.append(parsed_record)

            except json.JSONDecodeError as e:
                print(f"⚠️ OSTRZEŻENIE: Rekord {i} jest uszkodzony (błąd JSON). Błąd: {e}")
            except Exception as e:
                print(f"⚠️ OSTRZEŻENIE: Pominięto niekompatybilny rekord {i}. Błąd: {e}")
        
        print(f"✅ Znaleziono i poprawnie przetworzono {len(retrieved_mems)} pasujących wspomnień.")
        return retrieved_mems

    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD ODCZYTU PAMIĘCI: Nie udało się wykonać zapytania. Błąd: {e}")
        return []