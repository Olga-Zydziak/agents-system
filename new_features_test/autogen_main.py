"""
Główny plik demonstracyjny dla systemu MOA używającego AutoGen
"""

import os
import json
from autogen_orchestrator import AutoGenMOAOrchestrator
from process_logger import log as process_log

# Przykładowa biblioteka węzłów
NODE_LIBRARY = {
    "load_data": {"description": "Wczytuje dane z różnych źródeł"},
    "clean_data": {"description": "Czyści dane"},
    "validate_data": {"description": "Waliduje dane"},
    "discover_causality": {"description": "Odkrywa relacje przyczynowe (może zawieść)"},
    "error_handler": {"description": "Obsługuje błędy"},
    "rollback": {"description": "Cofa zmiany"},
    "generate_report": {"description": "Generuje raport"},
    "validate_model": {"description": "Waliduje model"},
    "optimize_performance": {"description": "Optymalizuje wydajność"},
}


def ensure_dummy_wrapper():
    """Upewnia się, że LLMWrapper działa w trybie dummy"""
    # Import extended wrapper żeby dodać lepsze dummy responses
    try:
        import extended_llm_wrapper

        print("✓ Extended LLM wrapper loaded (better dummy responses)")
    except:
        print("ℹ Using basic LLM wrapper")


def run_autogen_demo():
    """Uruchamia demo z AutoGen"""

    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║           🤖 AUTOGEN MOA DEBATE SYSTEM                              ║
║                                                                      ║
║  • Multi-agent debate using AutoGen GroupChat                       ║
║  • Dynamic context injection from memory                            ║
║  • Iterative improvements with critic feedback                      ║
║  • Automatic termination on "PLAN_ZATWIERDZONY"                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    )

    # Przykładowe misje
    missions = {
        "1": "Stwórz prosty pipeline do analizy danych CSV",
        "2": "Zaprojektuj ODPORNY NA BŁĘDY przepływ odkrywania przyczynowości z mechanizmem retry",
        "3": "Zbuduj adaptacyjny system ML z continuous learning",
    }

    print("\nChoose a mission:")
    for key, mission in missions.items():
        print(f"  {key}. {mission[:60]}...")
    print("  4. Custom mission")
    print("  0. Exit")

    choice = input("\nYour choice: ").strip()

    if choice == "0":
        return
    elif choice in missions:
        mission = missions[choice]
    elif choice == "4":
        mission = input("\nEnter your custom mission:\n> ").strip()
        if not mission:
            print("Mission cannot be empty!")
            return
    else:
        print("Invalid choice!")
        return

    print(f"\n📋 MISSION: {mission}")
    print("-" * 70)

    # Inicjalizuj orchestrator
    orchestrator = AutoGenMOAOrchestrator(
        mission=mission, node_library=NODE_LIBRARY, config_file="agents_config.json"
    )

    # Uruchom pełny cykl debaty
    final_plan = orchestrator.run_full_debate_cycle()

    if final_plan:
        print("\n" + "=" * 70)
        print("📊 FINAL APPROVED PLAN:")
        print(json.dumps(final_plan, indent=2))
    else:
        print("\n❌ No plan was approved")

    print("\n" + "=" * 70)
    print("📁 Check outputs/ for saved plans")
    print("📝 Check logs/conversation_log.txt for detailed logs")
    print("🧠 Check memory/ for learned patterns")


def main():
    """Główna funkcja"""
    # Upewnij się że katalogi istnieją
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("memory", exist_ok=True)

    # Załaduj extended wrapper dla lepszych dummy responses
    ensure_dummy_wrapper()

    # Log start
    process_log("=== AutoGen MOA System Started ===")

    try:
        run_autogen_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        process_log("=== AutoGen MOA System Ended ===")


if __name__ == "__main__":
    main()
