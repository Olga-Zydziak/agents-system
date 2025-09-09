# exporter_missions_lib.py
# Eksporter misji – czysta funkcja do użycia w notebooku lub pipeline

from __future__ import annotations
import json, re, hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)

def _strip_fences(text: str) -> str:
    return FENCE_RE.sub("", text or "").strip()

def _to_str_content(content: Any) -> str:
    if content is None: return ""
    s = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
    return _strip_fences(s)

def _iso_utc(ts: str | None) -> str:
    if not ts: return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    t = ts.replace(" ", "T")
    if "." in t: t = t.split(".")[0]
    return t if t.endswith("Z") else t + "Z"

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _extract_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(snapshot.get("final_plan"), dict):
        return snapshot["final_plan"]
    for it in snapshot.get("iterations") or []:
        for key in ("aggregator","critic"):
            block = it.get(key, {}) or {}
            try:
                data = json.loads(_to_str_content(block.get("content","")))
                if isinstance(data, dict) and "final_plan" in data: return data["final_plan"]
                if isinstance(data, dict) and "plan_approved" in data: return data["plan_approved"]
            except Exception:
                pass
    return {"entry_point": "", "nodes": [], "edges": []}

def _count_nodes_edges(plan: Dict[str, Any]) -> Tuple[int, int]:
    return len(plan.get("nodes") or []), len(plan.get("edges") or [])

def _approved(snapshot: Dict[str, Any]) -> bool:
    verdicts = [str(snapshot.get("verdict",""))]
    for it in snapshot.get("iterations") or []:
        critic = it.get("critic",{}) or {}
        verdicts.append(str(critic.get("verdict","")))
        if "zatwierdzony" in _to_str_content(critic.get("content","")).lower():
            verdicts.append("ZATWIERDZONY")
    if "plan_zatwierdzony" in str(snapshot.get("decision_marker","")).lower():
        verdicts.append("ZATWIERDZONY")
    return any("zatwierdzony" in v.lower() for v in verdicts)

def _derive_flags(snapshot: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str,bool]:
    blob = "\n".join([
        _to_str_content(snapshot.get("mission_prompt","")),
        _to_str_content(snapshot.get("llm_generated_summary","")),
        _to_str_content(snapshot.get("aggregator_reasoning","")),
        _to_str_content(snapshot.get("identified_patterns","")),
        json.dumps(plan, ensure_ascii=False)
    ]).lower()
    return {
        "has_retry": ("retry" in blob or "ponow" in blob),
        "has_rollback": ("rollback" in blob or "wycof" in blob),
        "has_optimization": ("optimiz" in blob or "optymal" in blob),
    }

def _build_txt(snapshot: Dict[str, Any], plan: Dict[str, Any]) -> str:
    mission_id = snapshot.get("memory_id") or snapshot.get("mission_id") or f"mission_{_hash(json.dumps(snapshot, ensure_ascii=False))}"
    timestamp = _iso_utc(snapshot.get("timestamp"))
    mtype = snapshot.get("mission_type","")
    tags = ", ".join(snapshot.get("tags") or [])
    outcome = str(snapshot.get("outcome",""))
    score = snapshot.get("final_score", snapshot.get("score",""))
    verdict = "ZATWIERDZONY" if _approved(snapshot) else ""
    nodes_cnt, edges_cnt = _count_nodes_edges(plan)
    prompt = _to_str_content(snapshot.get("mission_prompt",""))
    llm_summary = _to_str_content(snapshot.get("llm_generated_summary",""))

    txt = []
    txt.append(f"# Mission: {prompt[:80] or '—'}")
    txt.append(f"ID: {mission_id}")
    txt.append(f"Timestamp: {timestamp}")
    txt.append(f"Type: {mtype}")
    txt.append(f"Tags: {tags}")
    txt.append(f"Outcome | Score | Verdict: {outcome} | {score} | {verdict}\n")
    txt.append("## Executive Summary")
    txt.append(llm_summary or "Brak skrótu; patrz szczegóły planu i ryzyka.\n")
    txt.append("## Final Plan (skrót)")
    txt.append(f'Entry: {plan.get("entry_point","")}')
    node_names = [n.get("name") for n in plan.get("nodes",[]) if isinstance(n,dict)]
    txt.append(f"Węzły ({nodes_cnt}): " + ", ".join(node_names))
    return "\n".join(txt).strip() + "\n"

def _build_transcript(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    def norm(m: Dict[str, Any]) -> Dict[str, Any]:
        m2 = dict(m); m2["content"] = _to_str_content(m.get("content","")); return m2
    out = {
        "mission_id": snapshot.get("memory_id") or snapshot.get("mission_id"),
        "iterations": [],
        "full_transcript": []
    }
    for it in snapshot.get("iterations") or []:
        it_out = {}
        for k,v in it.items():
            if k == "proposers":
                it_out[k] = [{"agent": p.get("agent"), "content": _to_str_content(p.get("content",""))} for p in (v or [])]
            elif k in ("aggregator","critic"):
                block = v or {}
                it_out[k] = {"content": _to_str_content(block.get("content",""))}
            else:
                it_out[k] = v
        out["iterations"].append(it_out)
    out["full_transcript"] = [norm(m) for m in (snapshot.get("full_transcript") or [])]
    return out

def _build_metrics(snapshot: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    mission_id = snapshot.get("memory_id") or snapshot.get("mission_id")
    flags = _derive_flags(snapshot, plan)
    nodes_cnt, edges_cnt = _count_nodes_edges(plan)
    return {
        "mission_id": mission_id,
        "timestamp": _iso_utc(snapshot.get("timestamp")),
        "mission_type": snapshot.get("mission_type"),
        "tags": snapshot.get("tags") or [],
        "outcome": snapshot.get("outcome"),
        "final_score": snapshot.get("final_score", snapshot.get("score")),
        "approved": _approved(snapshot),
        "nodes_count": nodes_cnt,
        "edges_count": edges_cnt,
        **flags,
        "lang": snapshot.get("lang","pl"),
    }

def _ndjson_line(mission_id: str, txt_uri: str, plan_uri: str, transcript_uri: str, metrics_uri: str, metrics: Dict[str, Any]) -> str:
    return json.dumps({
        "id": mission_id,
        "structData": {**metrics, "links": {
            "plan_uri": plan_uri, "transcript_uri": transcript_uri, "metrics_uri": metrics_uri
        }},
        "content": {"mimeType": "text/plain", "uri": txt_uri}
    }, ensure_ascii=False)

def process_file(src_json: Path, input_dir: Path, out_dir: Path, gcs_prefix: str) -> List[str]:
    rel = src_json.relative_to(input_dir)
    snap = json.loads(src_json.read_text(encoding="utf-8"))
    mission_id = snap.get("memory_id") or snap.get("mission_id") or f"mission_{_hash(json.dumps(snap, ensure_ascii=False))}"
    plan = _extract_plan(snap)
    txt = _build_txt(snap, plan)
    transcript = _build_transcript(snap)
    metrics = _build_metrics(snap, plan)
    rel_dir = rel.parent
    base = out_dir / rel_dir / mission_id
    _ensure_dir(base)
    (base / f"{mission_id}.txt").write_text(txt, encoding="utf-8")
    (base / f"{mission_id}.plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / f"{mission_id}.transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / f"{mission_id}.metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    gcs_base = f"{gcs_prefix}/{rel_dir.as_posix()}/{mission_id}"
    return [_ndjson_line(mission_id,
                         f"{gcs_base}/{mission_id}.txt",
                         f"{gcs_base}/{mission_id}.plan.json",
                         f"{gcs_base}/{mission_id}.transcript.json",
                         f"{gcs_base}/{mission_id}.metrics.json",
                         metrics)]

def export_missions(input_dir: str, out_dir: str, gcs_prefix: str,
                    pattern="**/*.json", ndjson_out="metadata.ndjson"):
    input_dir = Path(input_dir).resolve()
    out_dir = Path(out_dir).resolve()
    _ensure_dir(out_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        print("Brak plików JSON do przetworzenia."); return
    lines: List[str] = []
    for f in files:
        try:
            lines.extend(process_file(f, input_dir, out_dir, gcs_prefix))
        except Exception as e:
            print(f"[WARN] Pomiń {f}: {e}")
    ndjson_path = out_dir / ndjson_out
    ndjson_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK. Artefakty w: {out_dir}\nNDJSON: {ndjson_path}")
