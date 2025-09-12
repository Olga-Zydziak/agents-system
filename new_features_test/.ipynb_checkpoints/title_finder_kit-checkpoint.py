# title_finder_kit.py
# ------------------------------------------------------------
# Vertex AI Search "Title Finder" toolkit for Jupyter notebooks
# - Works with Vertex AI Search (Discovery Engine) if serving_config is provided
# - Also works locally from metadata NDJSON + missions folder
# - Fuzzy title search (RapidFuzz optional) + re-ranking
# - Safe protobuf/struct coercion (no JSON serialization crashes)
# - Local filters (final_score, approved, tags ANY; fuzzy tag matching optional)
# - Plan graph plotting (matplotlib + networkx) with default styling
# - Convenience: open_best_hit + analysis helpers
#
# Usage:
#   %pip install -q google-cloud-discoveryengine rapidfuzz pandas networkx matplotlib
#   from title_finder_kit import *
#   set_paths('/mnt/data/metadata_20250829.ndjson', '/mnt/data/missions_extracted')
#   rows = search_vertex('', 'adaptacyjny system ml', page_size=30)
#   rows = rerank_by_title('adaptacyjny system ml', rows)
#   df = as_dataframe(rows); display(df)
#   best = rows[0]
#   bundle = open_best_hit(best, show_graph=True)
#   summary = analyze_best(best, show_graph=True)
# ------------------------------------------------------------

import os, re, json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

# Optional fuzzy
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    def _simple_ratio(a: str, b: str) -> float:
        a = (a or '').lower(); b = (b or '').lower()
        if not a or not b: return 0.0
        if a in b or b in a: return 100.0 * min(len(a), len(b))/max(len(a), len(b))
        return 0.0
    class fuzz:  # type: ignore
        @staticmethod
        def partial_ratio(a, b): return _simple_ratio(a, b)
        @staticmethod
        def token_set_ratio(a, b): return _simple_ratio(a, b)

# Optional Discovery Engine
try:
    from google.cloud import discoveryengine_v1 as de  # type: ignore
    from google.api_core.client_options import ClientOptions  # type: ignore
except Exception:
    de = None
    ClientOptions = None  # type: ignore

# ---- GLOBAL STATE ----

# ---- GCS SUPPORT ----
# Optional Google Cloud Storage client (used when URIs start with gs://)
try:
    from google.cloud import storage as _gcs  # type: ignore
except Exception:
    _gcs = None

_GCS_CLIENT = None  # lazy-initialized
def enable_gcs(project: str | None = None) -> None:
    
    global _GCS_CLIENT
    if _gcs is None:
        raise ImportError("google-cloud-storage not installed. Run: %pip install google-cloud-storage")
    _GCS_CLIENT = _gcs.Client(project=project)

def _ensure_gcs() -> None:
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        if _gcs is None:
            raise ImportError("google-cloud-storage not installed. Run: %pip install google-cloud-storage")
        _GCS_CLIENT = _gcs.Client()

def _parse_gs_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    path = uri[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob

def gcs_download_text(uri: str, encoding: str = "utf-8") -> str:
    _ensure_gcs()
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _GCS_CLIENT.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(encoding=encoding)

def gcs_download_json(uri: str, encoding: str = "utf-8"):
    import json as _json
    text = gcs_download_text(uri, encoding=encoding)
    return _json.loads(text)

def gcs_exists(uri: str) -> bool:
    try:
        _ensure_gcs()
        bucket_name, blob_name = _parse_gs_uri(uri)
        bucket = _GCS_CLIENT.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception:
        return False

def gcs_list(prefix_uri: str, max_items: int = 50) -> list[str]:
    _ensure_gcs()
    bucket_name, blob_prefix = _parse_gs_uri(prefix_uri)
    bucket = _GCS_CLIENT.bucket(bucket_name)
    it = _GCS_CLIENT.list_blobs(bucket_or_name=bucket, prefix=blob_prefix, max_results=max_items)
    return [f"gs://{bucket_name}/{b.name}" for b in it]

META_PATH: Optional[Path] = None
MISSIONS_DIR: Optional[Path] = None
_LOCAL_INDEX: Dict[str, Path] = {}
DF_META: pd.DataFrame = pd.DataFrame()

# --------------------- Core utils ---------------------

def set_paths(metadata_path: str, missions_dir: str) -> None:
    
    global META_PATH, MISSIONS_DIR, _LOCAL_INDEX, DF_META
    META_PATH = Path(metadata_path) if not str(metadata_path).startswith("gs://") else None
    MISSIONS_DIR = Path(missions_dir) if missions_dir else None

    # metadata: local or GCS
    import json as _json
    if isinstance(metadata_path, str) and metadata_path.startswith("gs://"):
        # read NDJSON from GCS
        try:
            text = gcs_download_text(metadata_path)
            rows = [_json.loads(line) for line in text.splitlines() if line.strip()]
            DF_META = pd.json_normalize(rows)
        except Exception:
            DF_META = pd.DataFrame()
    else:
        if META_PATH and META_PATH.exists():
            rows = [_json.loads(line) for line in META_PATH.open(encoding='utf-8') if line.strip()]
            DF_META = pd.json_normalize(rows)
        else:
            DF_META = pd.DataFrame()

    # index local files by basename (optional; not used for GCS)
    _LOCAL_INDEX = {}
    if MISSIONS_DIR and MISSIONS_DIR.exists():
        for p in MISSIONS_DIR.rglob('*'):
            if p.is_file():
                _LOCAL_INDEX[p.name] = p


def infer_api_endpoint(serving_config: str) -> str:
    m = re.search(r'/locations/([^/]+)/', serving_config)
    loc = m.group(1) if m else 'global'
    return f'{loc}-discoveryengine.googleapis.com'

def coerce(obj: Any) -> Any:
    """Coerce protobuf/MapComposite/RepeatedComposite to JSON-serializable types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # protobuf Message?
    if hasattr(obj, 'DESCRIPTOR'):
        try:
            d = {}
            for desc, value in obj.ListFields():  # type: ignore[attr-defined]
                d[desc.name] = coerce(value)
            return d
        except Exception:
            pass
    # mapping-like (MapComposite)
    if hasattr(obj, 'items'):
        try:
            return {str(k): coerce(v) for k, v in obj.items()}  # type: ignore[attr-defined]
        except Exception:
            pass
    # iterable-like (RepeatedComposite)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return [coerce(v) for v in list(obj)]
        except Exception:
            pass
    return obj

def safe_dumps(o: Any) -> str:
    return json.dumps(coerce(o), ensure_ascii=False, indent=2)

def _resolve_local_from_uri(uri: Optional[str]) -> Optional[Path]:
    if not uri: return None
    base = os.path.basename(uri)
    return _LOCAL_INDEX.get(base)

def _guess_local_artifacts(display_id: Optional[str]) -> Dict[str, Optional[str]]:
    """If links.* are missing, guess by display_id + suffix."""
    out = {'plan_uri': None, 'transcript_uri': None, 'metrics_uri': None, 'txt_uri': None}
    if not display_id: return out
    suffixes = {
        'plan_uri': '.plan.json',
        'transcript_uri': '.transcript.json',
        'metrics_uri': '.metrics.json',
        'txt_uri': '.txt',
    }
    for k, suf in suffixes.items():
        cand = (display_id or '') + suf
        p = _LOCAL_INDEX.get(cand)
        if p: out[k] = str(p)
    return out

def load_local_json(uri: Optional[str]) -> Optional[Dict[str, Any]]:
    p = Path(uri) if uri and '://' not in (uri or '') else _resolve_local_from_uri(uri)
    if not p or not Path(p).exists(): return None
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return None

def load_local_text(uri: Optional[str]) -> Optional[str]:
    p = Path(uri) if uri and '://' not in (uri or '') else _resolve_local_from_uri(uri)
    if not p or not Path(p).exists(): return None
    try:
        return Path(p).read_text(encoding='utf-8')
    except Exception:
        return None

# --------------------- Search + filters ---------------------
def search_vertex(
    serving_config: Optional[str],
    query: str,
    page_size: int = 30,
    api_endpoint: Optional[str] = None,
    filter_expression: Optional[str] = None,
    *,
    local_min_final_score: Optional[float] = None,
    local_approved: Optional[bool] = None,
    local_tags_any: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Query Vertex Search if serving_config provided; else local mode from DF_META.
       Local filters: min final_score, approved (bool), tags ANY.
    """
    rows: List[Dict[str, Any]] = []

    # --- Online (Vertex) ---
    if serving_config and de is not None:
        endpoint = api_endpoint or infer_api_endpoint(serving_config)
        client = de.SearchServiceClient(client_options=ClientOptions(api_endpoint=endpoint))
        req = de.SearchRequest(serving_config=serving_config, query=query, page_size=page_size)
        try:
            req.query_expansion_spec = de.SearchRequest.QueryExpansionSpec(
                condition=de.SearchRequest.QueryExpansionSpec.Condition.AUTO
            )
        except Exception:
            pass
        try:
            req.spell_correction_spec = de.SearchRequest.SpellCorrectionSpec(
                mode=de.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            )
        except Exception:
            pass
        if filter_expression:
            req.filter = filter_expression

        for r in client.search(request=req):
            doc = r.document
            struct = coerce(getattr(doc, 'struct_data', {}))
            links = struct.get('links') if isinstance(struct.get('links'), dict) else {}
            tags = struct.get('tags', [])
            if isinstance(tags, dict):
                tags = list(tags.values())
            elif isinstance(tags, str):
                tags = [tags]

            display_id = struct.get('display_id')
            link_fb = _guess_local_artifacts(display_id)

            rec = {
                'engine_score': float(getattr(r, 'score', 0.0) or 0.0),
                'mission_id': struct.get('mission_id'),
                'display_id': display_id,
                'mission_type': struct.get('mission_type'),
                'outcome': struct.get('outcome'),
                'tags': tags,
                'approved': struct.get('approved'),
                'final_score': struct.get('final_score'),
                'lang': struct.get('lang'),
                'nodes_count': struct.get('nodes_count'),
                'edges_count': struct.get('edges_count'),
                'plan_uri': links.get('plan_uri') or link_fb['plan_uri'],
                'transcript_uri': links.get('transcript_uri') or link_fb['transcript_uri'],
                'metrics_uri': links.get('metrics_uri') or link_fb['metrics_uri'],
                'txt_uri': links.get('txt_uri') or link_fb['txt_uri'],
                '_struct': struct,
            }

            if _passes_local_filters(rec, local_min_final_score, local_approved, local_tags_any):
                rows.append(rec)

        return rows

    # --- Local mode ---
    if DF_META.empty:
        return rows

    cols = [c for c in ('structData.display_id','structData.mission_id','structData.outcome') if c in DF_META.columns]
    if not cols:
        cols = [c for c in DF_META.columns if c.startswith('structData.')]

    q = (query or '').strip().lower()
    for _, meta in DF_META.iterrows():
        text = ' '.join(str(meta.get(c, '')) for c in cols).lower()
        if q and not any(tok in text for tok in q.split()):
            continue

        display_id = meta.get('structData.display_id')
        link_fb = _guess_local_artifacts(display_id)
        tags = meta.get('structData.tags')
        if isinstance(tags, dict):
            tags = list(tags.values())
        if isinstance(tags, str):
            tags = [tags]

        rec = {
            'engine_score': 0.0,
            'mission_id': meta.get('structData.mission_id'),
            'display_id': display_id,
            'mission_type': meta.get('structData.mission_type'),
            'outcome': meta.get('structData.outcome'),
            'tags': tags,
            'approved': meta.get('structData.approved'),
            'final_score': meta.get('structData.final_score'),
            'lang': meta.get('structData.lang'),
            'nodes_count': meta.get('structData.nodes_count'),
            'edges_count': meta.get('structData.edges_count'),
            'plan_uri': meta.get('structData.links.plan_uri') or link_fb['plan_uri'],
            'transcript_uri': meta.get('structData.links.transcript_uri') or link_fb['transcript_uri'],
            'metrics_uri': meta.get('structData.links.metrics_uri') or link_fb['metrics_uri'],
            'txt_uri': meta.get('structData.links.txt_uri') or link_fb['txt_uri'],
            '_struct': meta.to_dict(),
        }

        if _passes_local_filters(rec, local_min_final_score, local_approved, local_tags_any):
            rows.append(rec)

    return rows

def _passes_local_filters(rec: Dict[str, Any],
                          min_score: Optional[float],
                          approved: Optional[bool],
                          tags_any: Optional[List[str]]) -> bool:
    # final_score
    if min_score is not None and rec.get('final_score') is not None:
        try:
            if float(rec['final_score']) < float(min_score):
                return False
        except Exception:
            pass
    # approved
    if approved is not None and rec.get('approved') is not None:
        try:
            if bool(rec['approved']) != bool(approved):
                return False
        except Exception:
            pass
    # tags ANY (substring + RapidFuzz if available)
    if tags_any:
        want = [str(t).strip().lower() for t in tags_any if str(t).strip()]
        have = [str(t).lower() for t in (rec.get('tags') or [])]
        if want:
            def _match_any(w):
                if any(w in h for h in have):
                    return True
                try:
                    return any(fuzz.partial_ratio(w, h) >= 70 for h in have)  # type: ignore
                except Exception:
                    return False
            if not any(_match_any(w) for w in want):
                return False
    return True

# --------------------- Re-ranking + DataFrame ---------------------
def fuzzy_score(query: str, row: Dict[str, Any]) -> float:
    fields = [row.get('display_id') or '', row.get('mission_id') or '', row.get('outcome') or '']
    scores = [
        fuzz.token_set_ratio(query, fields[0]),
        fuzz.partial_ratio(query, fields[0]),
        fuzz.token_set_ratio(query, fields[1]),
        fuzz.partial_ratio(query, fields[1]),
        fuzz.token_set_ratio(query, fields[2]),
        fuzz.partial_ratio(query, fields[2]),
    ]
    return max(float(s or 0.0) for s in scores)

def rerank_by_title(query: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in rows:
        r['fuzzy_score'] = fuzzy_score(query, r)
        es = r.get('engine_score') or 0.0
        try:
            es = float(es)
        except Exception:
            es = 0.0
        r['combined_score'] = (0.85 * r['fuzzy_score']) + (0.15 * es)
    rows.sort(key=lambda x: (x.get('combined_score', 0.0), x.get('engine_score') or 0.0), reverse=True)
    return rows

def as_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = [
        'combined_score','fuzzy_score','engine_score',
        'display_id','mission_id','mission_type','outcome',
        'tags','approved','final_score','lang',
        'nodes_count','edges_count',
        'plan_uri','transcript_uri','metrics_uri','txt_uri'
    ]
    return pd.DataFrame([{k: r.get(k) for k in cols} for r in rows])

# --------------------- Plan graph plotting ---------------------
def plot_plan_graph(plan: Dict[str, Any]) -> None:
    """Draw a simple directed graph from plan['nodes'] / plan['edges'] using networkx + matplotlib.
       Constraints: single-plot, matplotlib only, no custom colors/styles.
    """
    try:
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print('Install first: %pip install networkx matplotlib')
        return
    nodes = plan.get('nodes', []) or []
    edges = plan.get('edges', []) or []
    G = nx.DiGraph()
    for n in nodes:
        nid = n.get('id') or n.get('name') or str(n)
        G.add_node(nid)
    for e in edges:
        src = e.get('source') or e.get('from')
        dst = e.get('target') or e.get('to')
        if src and dst:
            G.add_edge(src, dst)
    import matplotlib.pyplot as plt  # ensure single-plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    plt.show()

# --------------------- Convenience: open + quick preview ---------------------
def open_best_hit(best: dict, *, show_graph: bool = True, print_samples: bool = True, max_txt: int = 800) -> dict:
    """Open local artifacts for a 'best' row and print quick previews.
    Returns a dict with loaded objects for further analysis.
    """
    plan = load_local_json(best.get("plan_uri"))
    transcript = load_local_json(best.get("transcript_uri"))
    metrics = load_local_json(best.get("metrics_uri"))
    txt = load_local_text(best.get("txt_uri"))

    out = {"plan": plan, "transcript": transcript, "metrics": metrics, "txt": txt}

    # Quick prints
    if print_samples:
        if plan:
            print("plan.json: keys:", list(plan.keys())[:12], " total:", len(plan.keys()))
            nodes = plan.get("nodes", []) or []
            edges = plan.get("edges", []) or []
            print(f"  nodes={len(nodes)}  edges={len(edges)}")
        else:
            print("(no local plan.json)")

        if transcript:
            print("transcript.json: keys:", list(transcript.keys())[:12], " total:", len(transcript.keys()))
            msgs = transcript.get("messages") or transcript.get("turns") or []
            if isinstance(msgs, list) and msgs:
                sample = msgs[0]
                txt0 = sample.get("text") or sample.get("content") or str(sample)[:160]
                print("  first message:", (txt0[:200] + "...") if len(txt0) > 200 else txt0)
        else:
            print("(no local transcript.json)")

        if metrics:
            print("metrics.json: keys:", list(metrics.keys())[:12], " total:", len(metrics.keys()))
        else:
            print("(no local metrics.json)")

        if txt:
            print("txt: sample:", (txt[:max_txt] + ('...' if len(txt) > max_txt else '')))
        else:
            print("(no local txt)")

    if show_graph and plan:
        try:
            plot_plan_graph(plan)
        except Exception as e:
            print("Graph preview failed:", e)

    return out

# --------------------- Analysis helpers ---------------------
def analyze_plan(plan: dict) -> dict:
    """Lightweight structural analysis of plan graph."""
    result = {"node_count": 0, "edge_count": 0}
    if not isinstance(plan, dict):
        return result
    nodes = plan.get("nodes", []) or []
    edges = plan.get("edges", []) or []
    result["node_count"] = len(nodes)
    result["edge_count"] = len(edges)

    # Build adjacency
    node_ids = set()
    for n in nodes:
        nid = n.get("id") or n.get("name") or str(n)
        node_ids.add(nid)
    indeg = {nid: 0 for nid in node_ids}
    outdeg = {nid: 0 for nid in node_ids}
    for e in edges:
        src = e.get("source") or e.get("from")
        dst = e.get("target") or e.get("to")
        if src is None or dst is None:
            continue
        outdeg[src] = outdeg.get(src, 0) + 1
        indeg[dst] = indeg.get(dst, 0) + 1
        node_ids.add(src); node_ids.add(dst)

    result["sources"] = [n for n in node_ids if indeg.get(n, 0) == 0]
    result["sinks"] = [n for n in node_ids if outdeg.get(n, 0) == 0]
    result["max_out_degree"] = max(outdeg.values()) if outdeg else 0
    result["max_in_degree"] = max(indeg.values()) if indeg else 0

    # Try DAG checks and longest path via networkx (optional)
    try:
        import networkx as nx  # type: ignore
        G = nx.DiGraph()
        for n in node_ids: G.add_node(n)
        for e in edges:
            src = e.get("source") or e.get("from")
            dst = e.get("target") or e.get("to")
            if src and dst: G.add_edge(src, dst)
        result["is_dag"] = nx.is_directed_acyclic_graph(G)
        if result["is_dag"]:
            try:
                lp = nx.dag_longest_path(G)
                result["longest_path_nodes"] = lp
                result["longest_path_length"] = len(lp) - 1 if lp else 0
            except Exception:
                result["longest_path_length"] = None
        else:
            try:
                cyc = next(iter(nx.simple_cycles(G)), [])
                result["sample_cycle"] = cyc[:12]
            except Exception:
                result["sample_cycle"] = []
    except Exception:
        result["is_dag"] = None

    # Category counts (node.type, node.kind, node.category)
    from collections import Counter
    cat_fields = ["type", "kind", "category"]
    cats = Counter()
    for n in nodes:
        for f in cat_fields:
            if f in n:
                cats[f"{f}:{n[f]}"] += 1
                break
    result["category_counts"] = dict(cats)
    return result


def analyze_transcript(transcript: dict) -> dict:
    """
    Obsługuje wiele schematów:
    - transcript['messages'] / ['turns']
    - transcript['full_transcript'] (str lub list)
    - transcript['iterations'][i]['messages'/'turns'/'full_transcript']
    """
    res = {"turns": 0, "speakers": {}, "sample_first": None, "sample_last": None,
           "word_count_first": 0, "word_count_last": 0}

    if not isinstance(transcript, dict):
        return res

    # 1) Spróbuj bezpośrednio
    msgs = None
    if isinstance(transcript.get("messages"), list):
        msgs = transcript["messages"]
    elif isinstance(transcript.get("turns"), list):
        msgs = transcript["turns"]

    # 2) Jeśli brak, spróbuj iterations / full_transcript
    if msgs is None:
        acc = []
        # full_transcript może być stringiem albo listą obiektów
        ft = transcript.get("full_transcript")
        if isinstance(ft, str):
            # rozbij na linie pseudo-wiadomości
            for line in ft.splitlines():
                line = line.strip()
                if not line:
                    continue
                acc.append({"speaker": "unknown", "text": line})
        elif isinstance(ft, list):
            # spodziewamy się listy obiektów typu {"speaker":..,"text":..} itp.
            for it in ft:
                if isinstance(it, dict):
                    acc.append(it)
                else:
                    acc.append({"speaker": "unknown", "text": str(it)})
        # iterations
        its = transcript.get("iterations")
        if isinstance(its, list):
            for it in its:
                if not isinstance(it, dict):
                    continue
                for key in ("messages", "turns", "full_transcript"):
                    val = it.get(key)
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, dict):
                                acc.append(v)
                            else:
                                acc.append({"speaker": "unknown", "text": str(v)})
                    elif isinstance(val, str):
                        for line in val.splitlines():
                            line = line.strip()
                            if line:
                                acc.append({"speaker": "unknown", "text": line})
        msgs = acc if acc else []

    # 3) Wyliczenia
    from collections import Counter
    def _txt(m):
        # heurystyka: typowe klucze tekstu
        return m.get("text") or m.get("content") or m.get("message") or m.get("utterance") or ""
    def _spk(m):
        return m.get("speaker") or m.get("role") or m.get("author") or "unknown"

    # przefiltruj puste
    msgs = [m for m in msgs if isinstance(m, dict) and (_txt(m) or _spk(m))]

    res["turns"] = len(msgs)
    speakers = Counter()
    for m in msgs:
        speakers[_spk(m)] += 1
    res["speakers"] = dict(speakers)

    if msgs:
        res["sample_first"] = _txt(msgs[0])[:300]
        res["sample_last"]  = _txt(msgs[-1])[:300]
        res["word_count_first"] = len(str(res["sample_first"]).split())
        res["word_count_last"]  = len(str(res["sample_last"]).split())

    return res



def analyze_metrics(metrics: dict) -> dict:
    res = {"numerics": {}, "keys": [], "min": None, "max": None, "mean": None}
    if not isinstance(metrics, dict):
        return res
    res["keys"] = list(metrics.keys())
    numerics = {}
    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (int, float)):
            numerics[prefix] = float(obj)
        else:
            pass
    walk("", metrics)
    res["numerics"] = numerics
    if numerics:
        vals = list(numerics.values())
        res["min"] = min(vals); res["max"] = max(vals)
        res["mean"] = sum(vals) / len(vals)
    return res

def analyze_txt(text: str | None) -> dict:
    res = {"chars": 0, "words": 0, "lines": 0, "top_terms": []}
    if not text:
        return res
    res["chars"] = len(text)
    # solidne liczenie linii niezależnie od \n / \r\n
    res["lines"] = len(text.splitlines()) if text else 0

    import re
    words = re.findall(r"[A-Za-zÀ-ž0-9_]+", text.lower())
    res["words"] = len(words)
    from collections import Counter
    stop = set("""a an the and or if then else for of to in on with without we you i he she it they them is are was were be been being do does did this that these those as by from will would can could should may might must not nie oraz i w z do na że to jest są""".split())
    terms = [w for w in words if w not in stop and len(w) > 2]
    res["top_terms"] = Counter(terms).most_common(15)
    return res

def analyze_best(best: dict, *, show_graph: bool = False, print_summary: bool = True) -> dict:
    """Load artifacts for 'best' and return a compact analysis dict."""
    plan = load_local_json(best.get("plan_uri"))
    transcript = load_local_json(best.get("transcript_uri"))
    metrics = load_local_json(best.get("metrics_uri"))
    txt = load_local_text(best.get("txt_uri"))

    out = {
        "plan": analyze_plan(plan) if plan else None,
        "transcript": analyze_transcript(transcript) if transcript else None,
        "metrics": analyze_metrics(metrics) if metrics else None,
        "txt": analyze_txt(txt) if txt else None,
    }
    if print_summary:
        print("=== ANALYZE: plan ==="); print(safe_dumps(out["plan"]))
        print("=== ANALYZE: transcript ==="); print(safe_dumps(out["transcript"]))
        print("=== ANALYZE: metrics ==="); print(safe_dumps(out["metrics"]))
        print("=== ANALYZE: txt ==="); print(safe_dumps(out["txt"]))
    if show_graph and plan:
        try:
            plot_plan_graph(plan)
        except Exception as e:
            print("Graph preview failed:", e)
    return out


# ===================== Diagnostics & Robust Mapping =====================
def print_local_index_stats(sample: int = 20) -> None:
    """Print how many local files are indexed and show a sample of filenames."""
    total = len(_LOCAL_INDEX)
    print(f"Indexed local files: {total}")
    if total:
        names = list(_LOCAL_INDEX.keys())[:sample]
        print("Sample:", names)

def rebuild_local_index() -> None:
    """Re-scan the missions directory and rebuild the local index by basename."""
    global _LOCAL_INDEX
    _LOCAL_INDEX = {}
    if MISSIONS_DIR and MISSIONS_DIR.exists():
        for pth in MISSIONS_DIR.rglob('*'):
            if pth.is_file():
                _LOCAL_INDEX[pth.name] = pth

def resolve_uri_to_local(uri: Optional[str], *, display_id: Optional[str] = None) -> Optional[Path]:
    """More robust resolver for gs:// URIs -> local files.
    Steps:
      1) exact basename match
      2) substring match using display_id + extension
      3) fuzzy filename match with RapidFuzz (score >= 70)
    """
    if not uri:
        return None
    base = os.path.basename(uri)
    # 1) exact basename
    p = _LOCAL_INDEX.get(base)
    if p:
        return p

    # 2) substring using display_id
    if display_id:
        ext = ""
        if "." in base:
            ext = base[base.find("."):]
        candidates = [v for k, v in _LOCAL_INDEX.items() if (display_id in k) and (not ext or k.endswith(ext))]
        if candidates:
            return sorted(candidates, key=lambda q: len(q.name))[0]

    # 3) fuzzy
    try:
        from rapidfuzz import process as rf_process  # type: ignore
        choices = list(_LOCAL_INDEX.keys())
        match = rf_process.extractOne(base, choices, score_cutoff=70)
        if match:
            return _LOCAL_INDEX.get(match[0])
    except Exception:
        pass

    return None

def debug_locate_artifacts(best: dict) -> None:
    """Show how URIs map to local files, with hints if not found."""
    disp = best.get("display_id")
    for key in ["plan_uri","transcript_uri","metrics_uri","txt_uri"]:
        uri = best.get(key)
        local = resolve_uri_to_local(uri, display_id=disp)
        base = os.path.basename(uri) if uri else None
        print(f"{key}:")
        print("  uri:", uri)
        print("  basename:", base)
        print("  -> local:", str(local) if local else None)
        if not local:
            if disp:
                hits = [k for k in _LOCAL_INDEX.keys() if disp in k]
                if hits:
                    print("  candidates (contain display_id):", hits[:5])
            print("  (tip) Run print_local_index_stats(10) to inspect indexed filenames.")


# ---- Unified loaders (GCS or local) ----
def load_json(uri: str | None):
    if not uri:
        return None
    if isinstance(uri, str) and uri.startswith("gs://"):
        try:
            return gcs_download_json(uri)
        except Exception:
            return None
    # local path
    return load_local_json(uri)  # fallback to existing local resolver

def load_text(uri: str | None):
    if not uri:
        return None
    if isinstance(uri, str) and uri.startswith("gs://"):
        try:
            return gcs_download_text(uri)
        except Exception:
            return None
    return load_local_text(uri)

# Backward-compat aliasing for existing code that calls load_local_* directly with gs://
# These wrappers will delegate to GCS if needed.
_original_load_local_json = load_local_json
_original_load_local_text = load_local_text

def load_local_json(uri: str | None):
    if isinstance(uri, str) and uri.startswith("gs://"):
        return load_json(uri)
    # else call original local resolver
    return _original_load_local_json(uri)

def load_local_text(uri: str | None):
    if isinstance(uri, str) and uri.startswith("gs://"):
        return load_text(uri)
    return _original_load_local_text(uri)



_VERBOSE_ERRORS: bool = False

def set_error_verbosity(verbose: bool = True) -> None:
    """Przełącznik logów błędów loaderów (GCS/local)."""
    global _VERBOSE_ERRORS
    _VERBOSE_ERRORS = bool(verbose)

def _print_err(ctx: str, e: Exception) -> None:
    if _VERBOSE_ERRORS:
        print(f"[{ctx}] {e.__class__.__name__}: {e}")

# --- Niskopoziomowe narzędzia GCS (fallback, jeśli nie masz w pliku) ---
try:
    _gcs  # noqa: F821
except NameError:
    try:
        from google.cloud import storage as _gcs  # type: ignore
    except Exception:
        _gcs = None
try:
    _GCS_CLIENT  # noqa: F821
except NameError:
    _GCS_CLIENT = None

def _ensure_gcs():
    """Lazy-init klienta GCS (wymaga google-cloud-storage + ADC)."""
    global _GCS_CLIENT
    if _gcs is None:
        raise ImportError("Brak pakietu google-cloud-storage. Uruchom: %pip install google-cloud-storage")
    if _GCS_CLIENT is None:
        _GCS_CLIENT = _gcs.Client()

def _parse_gs_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"Nie-poprawny gs:// URI: {uri}"
    path = uri[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob

def strict_read_text(uri: str, encoding: str = "utf-8") -> str:
    """Twardy odczyt tekstu: rzuca konkretne wyjątki, nic nie połyka."""
    if isinstance(uri, str) and uri.startswith("gs://"):
        _ensure_gcs()
        bkt, blob = _parse_gs_uri(uri)
        return _GCS_CLIENT.bucket(bkt).blob(blob).download_as_text(encoding=encoding)
    # lokalnie
    from pathlib import Path as _P
    return _P(uri).read_text(encoding=encoding)

def strict_read_json(uri: str, encoding: str = "utf-8"):
    import json as _json
    return _json.loads(strict_read_text(uri, encoding=encoding))

# --- Verbose-wrap dla istniejących loaderów (jeśli je masz w pliku) ---
def load_json_verbose(uri: str | None, *, strict: bool = False):
    """Czyta JSON z gs:// lub lokalnie, wypisuje błąd. strict=True -> podnosi wyjątek."""
    try:
        if not uri:
            return None
        if strict:
            return strict_read_json(uri)
        # jeśli masz w module load_json -> użyj
        try:
            return load_json(uri)  # type: ignore[name-defined]
        except NameError:
            # fallback: strict
            return strict_read_json(uri)
    except Exception as e:
        _print_err("load_json", e)
        if strict:
            raise
        return None

def load_text_verbose(uri: str | None, *, strict: bool = False):
    """Czyta TXT z gs:// lub lokalnie, wypisuje błąd. strict=True -> podnosi wyjątek."""
    try:
        if not uri:
            return None
        if strict:
            return strict_read_text(uri)
        try:
            return load_text(uri)  # type: ignore[name-defined]
        except NameError:
            return strict_read_text(uri)
    except Exception as e:
        _print_err("load_text", e)
        if strict:
            raise
        return None

# --- Diagnoza gs:// (istnienie obiektu i listing rodzica) ---
def diagnose_gcs(uri_or_prefix: str, *, list_parent: bool = True, max_items: int = 10) -> dict:
    info = {
        "has_storage_lib": _gcs is not None,
        "client_init_ok": None,
        "project": None,
        "uri": uri_or_prefix,
        "is_gs": isinstance(uri_or_prefix, str) and uri_or_prefix.startswith("gs://"),
        "exists": None,
        "error": None,
        "parent_list_sample": [],
    }
    if not info["is_gs"]:
        info["error"] = "Not a gs:// URI"
        _print_err("diagnose_gcs", Exception(info["error"]))
        return info

    if _gcs is None:
        info["error"] = "google-cloud-storage not installed"
        _print_err("diagnose_gcs", Exception(info["error"]))
        return info

    try:
        _ensure_gcs()
        info["client_init_ok"] = True
        try:
            info["project"] = _GCS_CLIENT.project
        except Exception:
            info["project"] = None
    except Exception as e:
        info["client_init_ok"] = False
        info["error"] = f"{e.__class__.__name__}: {e}"
        _print_err("diagnose_gcs.client", e)
        return info

    try:
        from pathlib import PurePosixPath
        bkt, blob = _parse_gs_uri(uri_or_prefix)
        if blob and not uri_or_prefix.endswith("/"):
            # plik
            try:
                obj = _GCS_CLIENT.bucket(bkt).blob(blob)
                info["exists"] = obj.exists()
            except Exception as e:
                info["error"] = f"{e.__class__.__name__}: {e}"
                _print_err("diagnose_gcs.exists", e)
            if not info["exists"] and list_parent:
                parent = f"gs://{bkt}/{str(PurePosixPath(blob).parent)}/"
                try:
                    it = _GCS_CLIENT.list_blobs(bkt, prefix=str(PurePosixPath(blob).parent)+"/", max_results=max_items)
                    info["parent_list_sample"] = [f"gs://{bkt}/{b.name}" for b in it]
                except Exception as e:
                    _print_err("diagnose_gcs.parent_list", e)
        else:
            # prefix/katalog
            try:
                it = _GCS_CLIENT.list_blobs(bkt, prefix=blob, max_results=max_items)
                info["parent_list_sample"] = [f"gs://{bkt}/{b.name}" for b in it]
            except Exception as e:
                info["error"] = f"{e.__class__.__name__}: {e}"
                _print_err("diagnose_gcs.list", e)
    except Exception as e:
        info["error"] = f"{e.__class__.__name__}: {e}"
        _print_err("diagnose_gcs", e)

    return info

# --- Bezpieczny podgląd artefaktów z logami i trybem "strict" ---
def open_best_hit_verbose(best: dict, *, show_graph: bool = True, strict: bool = False, max_txt: int = 800) -> dict:
    """Jak open_best_hit, ale z logami błędów i możliwością strict read."""
    plan = None; transcript = None; metrics = None; txt = None
    try:
        plan = load_json_verbose(best.get("plan_uri"), strict=strict)
    except Exception as e:
        _print_err("open_best_hit.plan_uri", e)
    try:
        transcript = load_json_verbose(best.get("transcript_uri"), strict=strict)
    except Exception as e:
        _print_err("open_best_hit.transcript_uri", e)
    try:
        metrics = load_json_verbose(best.get("metrics_uri"), strict=strict)
    except Exception as e:
        _print_err("open_best_hit.metrics_uri", e)
    try:
        txt = load_text_verbose(best.get("txt_uri"), strict=strict)
    except Exception as e:
        _print_err("open_best_hit.txt_uri", e)

    out = {"plan": plan, "transcript": transcript, "metrics": metrics, "txt": txt}

    # szybki skrót
    if plan:
        try:
            nodes = plan.get("nodes", []) or []
            edges = plan.get("edges", []) or []
            print(f"plan.json: nodes={len(nodes)} edges={len(edges)}")
        except Exception as e:
            _print_err("open_best_hit.plan_summary", e)

    if show_graph and plan:
        try:
            plot_plan_graph(plan)  # zakładam, że masz w pliku
        except Exception as e:
            _print_err("open_best_hit.plot_plan_graph", e)

    if txt:
        sample = txt[:max_txt]
        print("txt: sample:", sample + ("..." if len(txt) > max_txt else ""))

    return out