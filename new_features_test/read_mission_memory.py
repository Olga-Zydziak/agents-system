import json
import vertexai
from typing import Dict, List, Optional

# --- 1. KONFIGURACJA ---
# Upewnij się, że te dane są poprawne.
PROJECT_ID = "dark-data-discovery"
LOCATION = "us-central1"
ENGINE_NAME = (
    "projects/815755318672/locations/us-central1/reasoningEngines/6370486808450433024"
)


# --- 2. DEFINICJA FUNKCJI ODCZYTU ---
def query_mission_memory(
    client: vertexai.Client,
    engine_name: str,
    query_text: str,
    scope: Optional[Dict] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Odpytuje pamięć agenta. Używa domyślnego scope, jeśli żaden nie zostanie podany.
    """
    # === KLUCZOWA ZMIANA: Zapewnienie domyślnego, poprawnego scope ===
    if scope is None:
        scope = {"source": "learned_strategies_json"}

    retrieved_facts = []
    print(
        f"\n--- 🚀 Odpytuję pamięć z zapytaniem: '{query_text}' | Zakres (Scope): {scope} ---"
    )

    try:
        search_params = {"search_query": query_text, "top_k": top_k}
        memories_iterator = client.agent_engines.retrieve_memories(
            name=engine_name, scope=scope, similarity_search_params=search_params
        )

        for i, mem in enumerate(memories_iterator):
            json_string_fact = mem.memory.fact
            parsed_fact = json.loads(json_string_fact)
            retrieved_facts.append(parsed_fact)

        print(
            f"✅ Znaleziono i poprawnie przetworzono {len(retrieved_facts)} pasujących wspomnień."
        )
        return retrieved_facts

    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD ODCZYTU PAMIĘCI: {e}")
        return []
