"""
Prosty logger procesu generowania planu i rozmów między agentami.
Wszystkie komunikaty są dopisywane do pliku tekstowego z sygnaturą czasu.
"""

import os
from datetime import datetime

# Ścieżka do pliku logów
LOG_FILE = "logs/conversation_log.txt"


def log(message: str) -> None:
    """
    Dodaje komunikat do pliku logów z aktualnym znacznikiem czasu. Tworzy katalog
    `logs` jeśli nie istnieje.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().isoformat()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {message}\n")
    except Exception as e:
        # Jeżeli logowanie się nie powiodło, wypisz ostrzeżenie
        print(f"⚠ Błąd logowania: {e}")