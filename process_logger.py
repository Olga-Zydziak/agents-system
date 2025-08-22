"""
Prosty logger procesu generowania planu i rozmów między agentami.
Wszystkie komunikaty są dopisywane do pliku tekstowego z sygnaturą czasu.
"""

import sys
from datetime import datetime  # zamiast: import datetime

LOG_FILE = "process_log.txt"
STREAM_STDOUT = True

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # zamiast: datetime.datetime.now()
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if STREAM_STDOUT:
        print(line, file=sys.stdout, flush=True)