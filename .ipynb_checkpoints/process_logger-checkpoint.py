"""
Prosty logger procesu generowania planu i rozmów między agentami.
Wszystkie komunikaty są dopisywane do pliku tekstowego z sygnaturą czasu.
"""

import sys
from datetime import datetime  # zamiast: import datetime

LOG_FILE = "process_log.txt"
STREAM_STDOUT = True



def log_exception(msg: str, exc: BaseException):
    import traceback
    tb = traceback.TracebackException.from_exception(exc)
    last = tb.stack[-1] if tb and tb.stack else None
    where = f"{last.filename}:{last.lineno} in {last.name}" if last else "unknown location"
    log(f"{msg}: {type(exc).__name__}: {exc} @ {where}")
    # pełny traceback w kolejnej linii:
    log("".join(tb.format()))



def log(msg: str):
    ts = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # zamiast: datetime.datetime.now()
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if STREAM_STDOUT:
        print(line, file=sys.stdout, flush=True)
