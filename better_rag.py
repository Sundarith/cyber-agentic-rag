"""
Hybrid RAG (BM25 + embeddings) over ATT&CK chunks + local LLM via Ollama.
"""
import json
import math
import os
import re
import sys
import time
import difflib
import threading
import numpy as np
import requests
from collections import Counter, defaultdict
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

ATTACK_CHUNKS = Path("data/processed/attack_chunks.jsonl")
CAPEC_CHUNKS  = Path("data/processed/capec_chunks.jsonl")
CWE_CHUNKS    = Path("data/processed/cwe_chunks.jsonl")
CVE_CHUNKS    = Path("data/processed/cve_chunks.jsonl")
RELATIONS     = Path("data/processed/entity_relations.json")
CAPEC_RELS    = Path("data/processed/capec_attack_relations.json")
CAPEC_CWE_RELS = Path("data/processed/capec_cwe_relations.json")
CWE_PHRASE_INDEX = Path("data/processed/cwe_phrase_index.json")
LLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"   # vLLM OpenAI-compatible API
MODEL        = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDER     = "BAAI/bge-small-en-v1.5"
BM25_K1      = 1.5
BM25_B       = 0.75
HYBRID_ALPHA = 0.5   # 0 = pure BM25, 1 = pure embedding
TOP_K        = 8
EMB_CACHE    = Path("data/processed/chunk_embs.npy")
KNN_CWE_NEIGHBORS = int(os.environ.get("CTI_RAG_KNN_CWE_NEIGHBORS", "5"))
KNN_CONFIDENCE_THRESHOLD = float(os.environ.get("CTI_RAG_KNN_CONFIDENCE_THRESHOLD", "0.75"))
KNN_CONFIDENCE_MARGIN = float(os.environ.get("CTI_RAG_KNN_CONFIDENCE_MARGIN", "1.25"))
CWE_KEYWORD_ANCHORS_ENABLED = os.environ.get("CTI_RAG_CWE_KEYWORD_ANCHORS", "0") == "1"
CWE_HIERARCHY_EXPANSION_ENABLED = os.environ.get("CTI_RAG_CWE_HIERARCHY_EXPANSION", "0") == "1"
CWE_SELECTOR_ENABLED = os.environ.get("CTI_RAG_CWE_SELECTOR", "0") == "1"
CWE_PHRASE_SELECTOR_ENABLED = os.environ.get("CTI_RAG_CWE_PHRASE_SELECTOR", "0") == "1"
CWE_CROSSENCODER_ENABLED = os.environ.get("CTI_RAG_CWE_CROSSENCODER", "0") == "1"
CWE_RESCUE_ENABLED = os.environ.get("CTI_RAG_CWE_RESCUE", "0") == "1"
CWE_RESCUE_POOL_K = int(os.environ.get("CTI_RAG_CWE_RESCUE_POOL_K", "50"))
CWE_RESCUE_MAX_ADD = int(os.environ.get("CTI_RAG_CWE_RESCUE_MAX_ADD", "1"))
CWE_RESCUE_MAX_RANK = int(os.environ.get("CTI_RAG_CWE_RESCUE_MAX_RANK", "2"))
CWE_RESCUE_MIN_SCORE = float(os.environ.get("CTI_RAG_CWE_RESCUE_MIN_SCORE", "0.70"))
CWE_RESCUE_MIN_LEXICAL_SCORE = float(os.environ.get("CTI_RAG_CWE_RESCUE_MIN_LEXICAL_SCORE", "4.0"))
CWE_RESCUE_MIN_LEXICAL_TERMS = int(os.environ.get("CTI_RAG_CWE_RESCUE_MIN_LEXICAL_TERMS", "2"))
PROFILE = "--profile" in sys.argv or os.environ.get("CTI_RAG_PROFILE", "0") == "1"
CAPTURE_TIMING = PROFILE or os.environ.get("CTI_RAG_CAPTURE_TIMING", "0") == "1"


def _profile_print(*args, **kwargs) -> None:
    if PROFILE:
        print(*args, **kwargs)

# High-precision CVE-description phrases. These are soft hints: they inject matching
# CWE chunks, but do not strip competing evidence the way bridge mappings do.
CWE_KEYWORD_ANCHORS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(use[- ]after[- ]free|uaf)\b", re.IGNORECASE), "CWE-416"),
    (re.compile(r"\b(path|directory) traversal\b", re.IGNORECASE), "CWE-22"),
    (re.compile(r"\bnull pointer dereference\b|\bnull dereference\b", re.IGNORECASE), "CWE-476"),
    (re.compile(r"\bstack[- ]based buffer overflow\b", re.IGNORECASE), "CWE-121"),
    (re.compile(r"\bheap[- ]based buffer overflow\b", re.IGNORECASE), "CWE-122"),
    (re.compile(r"\b(divide|division) by zero\b", re.IGNORECASE), "CWE-369"),
    (re.compile(r"\bout[- ]of[- ]bounds read\b", re.IGNORECASE), "CWE-125"),
    (re.compile(r"\bout[- ]of[- ]bounds write\b", re.IGNORECASE), "CWE-787"),
    (re.compile(r"\bunrestricted (file )?upload\b|\barbitrary file upload\b", re.IGNORECASE), "CWE-434"),
)

# Conservative post-generation selector for cases where the correct CWE is already
# in the retrieved context but the small LLM chooses a near-miss.
CWE_SELECTOR_RULES: tuple[tuple[str, re.Pattern, str], ...] = (
    ("null_pointer_deref", re.compile(r"\bnull pointer dereference\b|\bnull dereference\b", re.IGNORECASE), "CWE-476"),
    ("use_after_free", re.compile(r"\buse[- ]after[- ]free\b|\buaf\b", re.IGNORECASE), "CWE-416"),
    ("out_of_bounds_read", re.compile(r"\bout[- ]of[- ]bounds read\b", re.IGNORECASE), "CWE-125"),
    ("out_of_bounds_write", re.compile(r"\bout[- ]of[- ]bounds write\b", re.IGNORECASE), "CWE-787"),
    ("path_traversal", re.compile(r"\b(path|directory) traversal\b|\blocal file inclusion\b|\bLFI\b", re.IGNORECASE), "CWE-22"),
    ("dangerous_file_upload", re.compile(r"\bunrestricted (file )?upload\b|\barbitrary file upload\b|\bupload[^.]{0,80}dangerous file\b", re.IGNORECASE), "CWE-434"),
    ("command_injection", re.compile(r"\bcommand injection\b|\bshell commands?\b|\barbitrary commands?\b", re.IGNORECASE), "CWE-77"),
    ("missing_authorization", re.compile(r"\bmissing authorization\b|\bwithout authorization\b|\bdoes not (check|have) authorization\b|\bdoes not have authorisation\b|\bunauthori[sz]ed [^.]{0,60}\b(access|action|function|operation|users?)\b", re.IGNORECASE), "CWE-862"),
    ("improper_certificate_validation", re.compile(r"\b(improper|missing|incorrect|fails? to|does not) [^.]{0,60}certificate validation\b|\bvalidate [^.]{0,40}certificate\b", re.IGNORECASE), "CWE-295"),
    ("csrf_missing_check", re.compile(r"\b(lacking|lack of|no|missing|without|does not have) (?:a )?CSRF check\b|\bCSRF check (?:is )?(missing|not in place)\b", re.IGNORECASE), "CWE-352"),
    ("xss", re.compile(r"\bcross[- ]site scripting\b|\bXSS\b", re.IGNORECASE), "CWE-79"),
    ("csrf", re.compile(r"\bcross[- ]site request forgery\b|\bCSRF\b", re.IGNORECASE), "CWE-352"),
)

# ── data loading ──────────────────────────────────────────────────────────────

print("Loading chunks...")
chunks = [json.loads(line) for line in ATTACK_CHUNKS.open()]
if CAPEC_CHUNKS.exists():
    chunks += [json.loads(line) for line in CAPEC_CHUNKS.open()]
if CWE_CHUNKS.exists():
    chunks += [json.loads(line) for line in CWE_CHUNKS.open()]
if CVE_CHUNKS.exists():
    chunks += [json.loads(line) for line in CVE_CHUNKS.open()]
print(f"  {len(chunks)} chunks loaded")

print("Building reverse indexes for groups / malware / tools / campaigns...")
_OBS_RE      = re.compile(r'\n## Observed in the Wild\n(.+?)(?=\n## |\Z)', re.DOTALL)
_GROUPS_RE   = re.compile(r'\*\*Groups:\*\*\s*(.+)')
_CAMPAIGNS_RE = re.compile(r'\*\*Campaigns:\*\*\s*(.+)')
_MALWARE_RE  = re.compile(r'\*\*Malware:\*\*\s*(.+)')
_TOOLS_RE    = re.compile(r'\*\*Tools:\*\*\s*(.+)')

group_to_techs:    dict[str, list] = {}
campaign_to_techs: dict[str, list] = {}
malware_to_techs:  dict[str, list] = {}
tool_to_techs:     dict[str, list] = {}
display_name:      dict[str, str]  = {}   # lower → original casing

def _index(line_re, text, idx_dict, c):
    m = line_re.search(text)
    if not m:
        return
    for raw in m.group(1).split(", "):
        name = raw.strip()
        if name:
            key = name.lower()
            display_name[key] = name
            idx_dict.setdefault(key, []).append(c)

for c in chunks:
    obs = _OBS_RE.search(c["text"])
    if not obs:
        continue
    body = obs.group(1)
    _index(_GROUPS_RE,    body, group_to_techs,    c)
    _index(_CAMPAIGNS_RE, body, campaign_to_techs, c)
    _index(_MALWARE_RE,   body, malware_to_techs,  c)
    _index(_TOOLS_RE,     body, tool_to_techs,     c)

print(f"  {len(group_to_techs)} groups, {len(campaign_to_techs)} campaigns, {len(malware_to_techs)} malware, {len(tool_to_techs)} tools")

# group ↔ malware/tool relations (loaded from build step)
group_to_malware: dict[str, list] = {}
group_to_tools:   dict[str, list] = {}
malware_to_groups: dict[str, list] = {}
tool_to_groups:    dict[str, list] = {}

if RELATIONS.exists():
    _rel = json.loads(RELATIONS.read_text())
    for g, d in _rel.get("group_uses", {}).items():
        g_lower = g.lower()
        display_name.setdefault(g_lower, g)
        group_to_malware[g_lower] = d.get("malware", [])
        group_to_tools[g_lower]   = d.get("tools",   [])
        for m in d.get("malware", []):
            display_name.setdefault(m.lower(), m)
            malware_to_groups.setdefault(m.lower(), []).append(g)
        for t in d.get("tools", []):
            display_name.setdefault(t.lower(), t)
            tool_to_groups.setdefault(t.lower(), []).append(g)
    print(f"  Group↔malware: {sum(len(v) for v in group_to_malware.values())} edges; group↔tools: {sum(len(v) for v in group_to_tools.values())} edges")

# Resolve Aliases: Add aliases to the entity-to-technique indices
_ALSO_KNOWN_RE = re.compile(r'\*\*Also known as:\*\*\s*(.+)')
alias_count = 0
for c in chunks:
    ctype = c.get("type")
    if ctype in ("group", "malware", "tool", "campaign"):
        primary_name = c["name"].lower()
        idx_dict = {
            "group": group_to_techs,
            "campaign": campaign_to_techs,
            "malware": malware_to_techs,
            "tool": tool_to_techs
        }.get(ctype)
        if not idx_dict: continue
        
        # Extract aliases from text
        m = _ALSO_KNOWN_RE.search(c["text"])
        if m:
            aliases = [a.strip().lower() for a in m.group(1).split(", ")]
            primary_techs = idx_dict.get(primary_name, [])
            for alias in aliases:
                if alias and alias != primary_name:
                    # Map alias to same techniques as primary name
                    alias_techs = idx_dict.setdefault(alias, [])
                    for pt in primary_techs:
                        if pt not in alias_techs:
                            alias_techs.append(pt)
                    display_name.setdefault(alias, c["name"])
                    alias_count += 1
print(f"  Indexed {alias_count} entity aliases")

capec_to_attack: dict[str, list] = defaultdict(list)
attack_to_capec: dict[str, list] = {}
if CAPEC_RELS.exists():
    _cr = json.loads(CAPEC_RELS.read_text())
    attack_to_capec = _cr.get("tech_to_capec", {})
    for tid, capec_ids in attack_to_capec.items():
        for cid in capec_ids:
            capec_to_attack[cid].append(tid)

capec_to_cwe: dict[str, list] = {}
if CAPEC_CWE_RELS.exists():
    _cwr = json.loads(CAPEC_CWE_RELS.read_text())
    capec_to_cwe = _cwr.get("capec_to_cwe", {})
    print(f"  CAPEC→CWE bridge: {len(capec_to_cwe)} mappings loaded")

# ID lookup index for instant retrieval
id_to_chunk: dict[str, dict] = {}
id_to_idx: dict[str, int] = {}
for i, c in enumerate(chunks):
    if c.get("identifier"):
        cid = c["identifier"].upper()
        id_to_chunk[cid] = c
        id_to_idx[cid] = i

# Universal Knowledge Graph for multi-hop expansion
print("Building Knowledge Graph...")
chunk_graph = defaultdict(set)
_id_re = re.compile(r'\b(T\d{4}(?:\.\d{3})?|M\d{4}|G\d{4}|S\d{4}|C\d{4}|DS\d{4}|CAPEC-\d+|CWE-\d+|CVE-\d{4}-\d+)\b', re.IGNORECASE)

for c in chunks:
    cid = c.get("identifier", "").upper()
    if not cid: continue
    if c.get("source") == "CVE":
        # Only wire explicit NVD CWE assignments — not prose text (too many false edges)
        for cwe_id in c.get("cwe_ids", []):
            cwe_upper = cwe_id.upper()
            if cwe_upper in id_to_chunk:
                chunk_graph[cid].add(cwe_upper)
                chunk_graph[cwe_upper].add(cid)
        continue
    # Extract explicit edges from chunk text
    for m in _id_re.finditer(c["text"]):
        m_upper = m.group(1).upper()
        if m_upper != cid and m_upper in id_to_chunk:
            chunk_graph[cid].add(m_upper)
            chunk_graph[m_upper].add(cid)

# Fold in CAPEC relations
for tid, capec_ids in attack_to_capec.items():
    tid_upper = tid.upper()
    for cid in capec_ids:
        cid_upper = cid.upper()
        if tid_upper in id_to_chunk and cid_upper in id_to_chunk:
            chunk_graph[tid_upper].add(cid_upper)
            chunk_graph[cid_upper].add(tid_upper)
print(f"  Graph built with {len(chunk_graph)} connected nodes")

# Build CWE child index for bidirectional hierarchy expansion
child_cwe_ids: dict[str, list[str]] = defaultdict(list)
for c in chunks:
    if c.get("source") == "CWE":
        cid = c.get("identifier", "").upper()
        for pid in c.get("parent_cwe_ids", []):
            child_cwe_ids[pid.upper()].append(cid)
print(f"  CWE child index built for {len(child_cwe_ids)} parent CWEs")

# technique-name index — sorted longest-first so "Scanning IP Blocks" wins over "Active Scanning"
_tech_name_index: list[tuple[str, int]] = sorted(
    [(c["name"].lower(), i) for i, c in enumerate(chunks) if c.get("name")],
    key=lambda x: len(x[0]), reverse=True,
)

print("Building BM25 index...")
_tok_re = re.compile(r'\w+')

def _tokenize(text: str) -> list[str]:
    return _tok_re.findall(text.lower())


_CWE_ANCHOR_STOPWORDS = {
    "the", "and", "for", "with", "without", "from", "into", "onto", "that", "this",
    "when", "where", "which", "while", "before", "after", "within", "using", "used",
    "uses", "use", "user", "users", "product", "software", "application", "system",
    "component", "resource", "resources", "data", "input", "output", "value", "values",
    "element", "elements", "special", "proper", "properly", "incorrect", "incorrectly",
    "improper", "improperly", "neutralization", "neutralize", "neutralizes", "weakness",
    "vulnerability", "vulnerabilities", "attacker", "attackers", "allows", "allow",
    "could", "would", "should", "might", "able", "certain", "specific", "external",
    "internal", "related", "intended", "attempts", "attempt", "perform", "performs",
    "performed", "common", "different", "various", "malicious", "crafted", "arbitrary",
    "a", "an", "as", "at", "be", "been", "being", "by", "can", "cannot", "do", "does",
    "due", "etc", "have", "has", "having", "if", "in", "is", "it", "its", "may", "must",
    "not", "of", "on", "once", "only", "or", "other", "same", "see", "some", "such",
    "than", "then", "there", "these", "they", "their", "them", "those", "to", "was",
    "were", "will", "cwe", "ref", "description", "following", "contain", "contains",
    "containing", "ensure", "provide", "provides", "provided", "make", "makes", "made",
    "get", "set", "called", "call", "cause", "causes", "causing", "lead", "leads",
    "result", "results", "occur", "occurs", "issue", "issues", "case", "cases",
}


def _anchor_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _normalize_anchor_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"['`\"]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _anchor_terms(text: str) -> set[str]:
    terms = set()
    for token in _normalize_anchor_text(text).split():
        term = _anchor_token(token)
        if len(term) < 3 or term in _CWE_ANCHOR_STOPWORDS:
            continue
        terms.add(term)
    return terms


def _anchor_phrase(text: str) -> str:
    return " ".join(_anchor_token(t) for t in _normalize_anchor_text(text).split())


def _extract_cwe_anchor_profile(c: dict) -> dict:
    name = c.get("name", "")

    phrases = set()
    for raw in [name]:
        phrase = _anchor_phrase(raw)
        if len(phrase.split()) >= 2:
            phrases.add(phrase)
    for raw in re.findall(r"\(([^)]+)\)", name):
        raw = raw.strip(" '\"")
        phrase = _anchor_phrase(raw)
        if len(phrase) >= 4 and (len(phrase.split()) >= 2 or raw.isupper()):
            phrases.add(phrase)
    for raw in re.findall(r"'([^']+)'", name):
        phrase = _anchor_phrase(raw)
        if len(phrase.split()) >= 2:
            phrases.add(phrase)

    return {
        "phrases": phrases,
        "name_terms": _anchor_terms(name),
        "desc_terms": set(),
    }

_N = len(chunks)
_doc_len: np.ndarray = np.zeros(_N, dtype=np.int32)
_doc_freq: Counter = Counter()
_inv_idx: dict[str, dict[int, int]] = {}   # term → {doc_idx: tf}

for i, c in enumerate(chunks):
    toks = _tokenize(c["text"])
    _doc_len[i] = len(toks)
    for t in set(toks):
        _doc_freq[t] += 1
    for t, tf in Counter(toks).items():
        if t not in _inv_idx:
            _inv_idx[t] = {}
        _inv_idx[t][i] = tf

_avgdl = float(_doc_len.mean())
print(f"  {len(_doc_freq)} unique terms, avgdl={_avgdl:.0f}")

# Pre-convert postings to numpy arrays so BM25 becomes vectorized.
# Avoids per-term Python loops that hold the GIL and serialize between eval threads.
_inv_idx_np: dict[str, tuple[np.ndarray, np.ndarray]] = {
    t: (np.fromiter(p.keys(), dtype=np.int32, count=len(p)),
        np.fromiter(p.values(), dtype=np.float32, count=len(p)))
    for t, p in _inv_idx.items()
}

_thread_local = threading.local()

print("Loading embedder...")
embedder = SentenceTransformer(EMBEDDER, device="cuda" if torch.cuda.is_available() else "cpu")
try:
    _embedder_1 = SentenceTransformer(EMBEDDER, device="cuda:1")
    print("  Second embedder on cuda:1 ready for parallel eval")
except Exception as _e:
    _embedder_1 = None
    print(f"  cuda:1 not available ({_e}), parallel eval will share cuda:0")

def _get_embedder() -> SentenceTransformer:
    return getattr(_thread_local, 'embedder', embedder)

print("Embedding all chunks (checking cache)...")
if EMB_CACHE.exists():
    chunk_embs = np.load(EMB_CACHE)
    if len(chunk_embs) != len(chunks):
        print(f"  Cache size {len(chunk_embs)} != {len(chunks)} chunks. Rebuilding...")
        chunk_embs = embedder.encode(
            [c["text"] for c in chunks], batch_size=512,
            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
        )
        np.save(EMB_CACHE, chunk_embs)
    else:
        print("  Loaded from cache.")
else:
    chunk_embs = embedder.encode(
        [c["text"] for c in chunks], batch_size=512,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
    )
    np.save(EMB_CACHE, chunk_embs)
print(f"  Shape: {chunk_embs.shape}\n")

# Pre-compute boolean masks for scoring — used in retrieve() as numpy array ops
# to avoid O(N) Python loops that hold the GIL and block parallel eval threads.
_detection_mask   = np.array([c.get("type") == "detection" for c in chunks], dtype=bool)
_nameless_mask    = np.array([not c.get("identifier") for c in chunks], dtype=bool)
_cve_source_mask  = np.array([c.get("source") == "CVE" for c in chunks], dtype=bool)
_cwe_source_mask  = np.array([c.get("source") == "CWE" for c in chunks], dtype=bool)
_root_cause_mask  = np.array([
    c.get("source") == "CWE" or c.get("type") in ("weakness", "category", "view")
    for c in chunks
], dtype=bool)

_cwe_anchor_profiles: dict[str, dict] = {
    c.get("identifier", "").upper(): _extract_cwe_anchor_profile(c)
    for c in chunks
    if c.get("source") == "CWE" and c.get("identifier")
}
_cwe_anchor_df: Counter = Counter()
for profile in _cwe_anchor_profiles.values():
    for term in profile["name_terms"] | profile["desc_terms"]:
        _cwe_anchor_df[term] += 1
_cwe_anchor_n = max(1, len(_cwe_anchor_profiles))
for profile in _cwe_anchor_profiles.values():
    terms = profile["name_terms"] | profile["desc_terms"]
    profile["term_weights"] = {
        term: math.log((_cwe_anchor_n + 1) / (_cwe_anchor_df[term] + 1)) + 1.0
        for term in terms
    }
print(f"  CWE lexical anchors built for {len(_cwe_anchor_profiles)} CWEs")

_cwe_phrase_index: dict[str, list[dict]] = {}
if CWE_PHRASE_SELECTOR_ENABLED:
    if CWE_PHRASE_INDEX.exists():
        _raw_phrase_index = json.loads(CWE_PHRASE_INDEX.read_text(encoding="utf-8"))
        _cwe_phrase_index = {
            cwe_id.upper(): row.get("phrases", [])
            for cwe_id, row in _raw_phrase_index.items()
        }
        print(f"  CWE phrase selector index loaded for {len(_cwe_phrase_index)} CWEs")
    else:
        print(f"  CWE phrase selector requested but {CWE_PHRASE_INDEX} is missing")

# Name → chunk index for O(1) entity lookups (replaces inner loops over all chunks)
_name_to_idx: dict[str, int] = {
    c["name"].lower(): i for i, c in enumerate(chunks) if c.get("name")
}

# k-NN CWE bridge: indices of CVE chunks with NVD-assigned cwe_ids. Used as a fallback
# when the direct CVE→NVD bridge doesn't fire (unmapped CVE descriptions). The N most
# similar mapped CVEs vote on the CWE, and the winner is injected as a soft hint.
_mapped_cve_indices = np.array(
    [i for i, c in enumerate(chunks) if c.get("source") == "CVE" and c.get("cwe_ids")],
    dtype=np.int32,
)
print(f"  {_mapped_cve_indices.size:,} mapped CVE chunks indexed for k-NN CWE voting")


# ── retrieval ─────────────────────────────────────────────────────────────────

def _bm25_scores(query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(_N)
    for t in set(query_tokens):
        entry = _inv_idx_np.get(t)
        if entry is None:
            continue
        idx_arr, tf_arr = entry
        df = idx_arr.size
        idf = math.log((_N - df + 0.5) / (df + 0.5) + 1)
        dl_arr = _doc_len[idx_arr]
        scores[idx_arr] += idf * (tf_arr * (BM25_K1 + 1)) / (tf_arr + BM25_K1 * (1 - BM25_B + BM25_B * dl_arr / _avgdl))
    return scores


def retrieve(query: str, k: int = TOP_K) -> list:
    _r0 = time.time()
    q_emb  = _get_embedder().encode([query], normalize_embeddings=True)[0]
    _r1 = time.time()
    emb_sc = chunk_embs @ q_emb
    _r2 = time.time()

    bm25_sc = _bm25_scores(_tokenize(query))
    if bm25_sc.max() > 0:
        bm25_sc = bm25_sc / bm25_sc.max()
    _r3 = time.time()

    combined = HYBRID_ALPHA * emb_sc + (1 - HYBRID_ALPHA) * bm25_sc

    # 4. Intent-based boosting (e.g., if query asks for detection, boost detection chunks)
    if DETECTION_RE.search(query):
        combined[_detection_mask] += 2.0

    # 5. Fuzzy name boost (handle typos like 'blusmacking')
    # Skip for long queries (CVE descriptions, prompts) — common English words like
    # "vulnerability", "server", "exploit" match technique names and flood the top-K.
    query_words = _tokenize(query)
    if len(query_words) <= 15:
        for word in query_words:
            if len(word) < 5: continue
            matches = difflib.get_close_matches(word, [tn[0] for tn in _tech_name_index], n=1, cutoff=0.85)
            if matches:
                best_match = matches[0]
                for name, idx in _tech_name_index:
                    if name == best_match:
                        combined[idx] += 3.0
                        print(f"[Fuzzy match: '{word}' → '{best_match}']")
                        break

    # Boost exact ID matches to rank 1 — small nudge only, BM25 IDF already handles
    # unique IDs well. A massive boost monopolizes all slots with noise.
    id_matches = re.findall(r'\b(?:[TM]\d{4}(?:\.\d{3})?|CAPEC-\d+|CWE-\d+|CVE-\d{4}-\d+)\b', query, re.IGNORECASE)
    for match in id_matches:
        idx = id_to_idx.get(match.upper())
        if idx is not None:
            combined[idx] += 2.0

    # if no ID, boost the chunk whose technique name appears verbatim in the query
    # longest name wins so "Scanning IP Blocks" beats "Active Scanning" when both fit
    if not id_matches:
        q_lower = query.lower()
        for name, idx in _tech_name_index:
            # fast substring pre-check — substring is a necessary condition for
            # the word-boundary regex match, so this skips ~all 30k iterations
            # for queries (e.g. CVE descriptions) that contain no chunk names.
            if name in q_lower and re.search(rf'\b{re.escape(name)}\b', q_lower):
                combined[idx] = combined.max() + 0.1
                break

    # Entity-Technique Bridge: If query mentions a group/malware/tool, boost its associated chunks
    # This helps retrieval "see" the techniques so the graph can reach CAPECs/CWEs in 2 hops.
    found_entities = _find_all_entities_in_query(query)
    for ent_name, ent_type in found_entities:
        idx_dict = {"group": group_to_techs, "campaign": campaign_to_techs, "malware": malware_to_techs, "tool": tool_to_techs}[ent_type]
        related_chunks = idx_dict.get(ent_name, [])

        # Boost the entity chunk itself so "What is Sandworm Team?" returns G0034 directly.
        # Score-based (not keyword routing) so multi-hop queries still work correctly.
        ent_idx = _name_to_idx.get(ent_name)
        if ent_idx is not None and chunks[ent_idx].get("type") in ("group", "malware", "tool", "campaign"):
            combined[ent_idx] += 2.0

        for rc in related_chunks:
            rc_idx = id_to_idx.get(rc["identifier"].upper())
            if rc_idx is not None:
                boost = 1.5
                if chunks[rc_idx]["identifier"] in attack_to_capec:
                    boost += 1.0
                combined[rc_idx] += boost

        # Also boost relations (group -> malware etc)
        if ent_type == "group":
            for m_name in group_to_malware.get(ent_name, []):
                m_idx = _name_to_idx.get(m_name.lower())
                if m_idx is not None:
                    combined[m_idx] += 1.0

    # Intent Routing: Boost CWE chunks if query focuses on root causes.
    # Keep this small — technique chunks must stay in top-K so the graph can reach the
    # correct CWEs via traversal. The +2.0 graph-neighbor CWE boost (in ask()) does the
    # heavy lifting for root-cause queries.
    if ROOT_CAUSE_RE.search(query):
        combined[_root_cause_mask] += 0.3

    # Penalize nameless chunks — they have no identifier so they can't participate
    # in graph traversal and only add noise to the context.
    combined[_nameless_mask] -= 1.0

    # Penalize CVE chunks for non-CVE queries — they're semantically similar to
    # ATT&CK/CAPEC content and flood the top-K when no CVE ID is in the query.
    # Exception: for long description-style queries (CVE description prompts),
    # let CVE chunks compete naturally — the matching CVE wins via verbatim text
    # similarity, and the NVD bridge in ask() then uses its cwe_ids as authoritative.
    if not re.search(r'\bCVE-\d{4}-\d+\b', query, re.IGNORECASE):
        if len(_tokenize(query)) <= 15:
            combined[_cve_source_mask] -= 0.5
    else:
        # For CVE queries: suppress all CWE chunks — they're generic noise when the CVE
        # chunk already contains the CWE info (or "n/a"). The LLM should reason from the
        # CVE description, not pick a random quality CWE like CWE-1116.
        combined[_cwe_source_mask] -= 1.5

    _r4 = time.time()
    top_idx  = np.argsort(-combined)[:k]
    _r5 = time.time()
    _profile_print(f"  [R] enc={_r1-_r0:.2f} matmul={_r2-_r1:.2f} bm25={_r3-_r2:.2f} boosts={_r4-_r3:.2f} sort={_r5-_r4:.2f}", flush=True)
    return [(chunks[i], float(combined[i])) for i in top_idx]


# ── LLM ───────────────────────────────────────────────────────────────────────

def _history_str(history: list) -> str:
    out = ""
    for turn in history[-3:]:
        a = turn["a"][:600] + "...[truncated]" if len(turn["a"]) > 600 else turn["a"]
        out += f"User: {turn['q']}\nAssistant: {a}\n\n"
    return out


def _llm(prompt: str) -> str:
    try:
        r = requests.post(LLM_ENDPOINT, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256,
            "stream": False,
        })
        r.raise_for_status()
        data = r.json()
        if "choices" not in data or not data["choices"]:
            print(f"vLLM Error (no choices): {data}")
            return "Error: vLLM returned an unexpected response."
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"vLLM Connection Error: {e}")
        if 'r' in locals():
            print(f"Status Code: {r.status_code}, Body: {r.text}")
        return "Error: Could not connect to vLLM."


ENTITY_REVERSE_RE = re.compile(r'\b(what|which|list|show).{0,50}\b(techniques?|attacks?|tactics?|malwares?|software|tools?|groups?|actors?|do|does|use[sd]?|cause[sd]?)\b', re.IGNORECASE)
WHAT_IS_RE        = re.compile(r'\bwhat\s+is\b', re.IGNORECASE)
TARGET_MALWARE_RE = re.compile(r'\b(malwares?|software)\b', re.IGNORECASE)
TARGET_TOOLS_RE   = re.compile(r'\btools?\b', re.IGNORECASE)
TARGET_GROUPS_RE  = re.compile(r'\b(groups?|actors?|threat\s+actors?)\b', re.IGNORECASE)
ROOT_CAUSE_RE     = re.compile(
    r'\b(?:'
    r'CWE-\d+|'
    r'CWE(?:s)?|'
    r'Common\s+Weakness\s+Enumeration|'
    r'(?:root\s+cause|weakness|vulnerabilit(?:y|ies)|flaw|bug)\s+(?:CWE|mapping|classification|ID)|'
    r'(?:map|classify|assign)\b.{0,80}\bCWE|'
    r'CWE\b.{0,80}\b(?:map|mapping|classif(?:y|ication)|root\s+cause|weakness|underl(?:y|ies|ying))'
    r')\b',
    re.IGNORECASE,
)
DETECTION_RE      = re.compile(r'\b(detect|monitor|analytic|log|sensor|data component)\b', re.IGNORECASE)
MITIGATION_RE     = re.compile(r'\b(mitigation|mitigate|prevent|protect|remediate)\b', re.IGNORECASE)
CAPEC_RE          = re.compile(r'\b(attack patterns?|capec|exploit patterns?|attack techniques?)\b', re.IGNORECASE)


def _techs_for_entity(name: str, entity_type: str, idx: dict) -> str:
    techs = idx.get(name.lower(), [])
    if not techs:
        return f"No {entity_type} named '{name}' found in the dataset."
    display = display_name.get(name.lower(), name)
    lines = [f"**{display}** ({entity_type}) is observed using {len(techs)} technique(s):\n"]
    for c in sorted(techs, key=lambda x: x.get("identifier", "")):
        sub = " (sub-technique)" if c.get("is_subtechnique", False) else ""
        lines.append(f"- {c['identifier']} — {c['name']}{sub}")
    return "\n".join(lines)


def _find_entity_in_query(question: str):
    """Return (name, entity_type, index) if a known group/campaign/malware/tool name is in the query."""
    q_lower = question.lower()
    for name in sorted(group_to_techs, key=len, reverse=True):
        if re.search(rf'\b{re.escape(name)}\b', q_lower):
            return name, "group", group_to_techs
    for name in sorted(campaign_to_techs, key=len, reverse=True):
        if re.search(rf'\b{re.escape(name)}\b', q_lower):
            return name, "campaign", campaign_to_techs
    for name in sorted(malware_to_techs, key=len, reverse=True):
        if re.search(rf'\b{re.escape(name)}\b', q_lower):
            return name, "malware", malware_to_techs
    for name in sorted(tool_to_techs, key=len, reverse=True):
        if re.search(rf'\b{re.escape(name)}\b', q_lower):
            return name, "tool", tool_to_techs
    return None


def _find_all_entities_in_query(question: str) -> list:
    """Return all (name, entity_type) pairs found in the query, avoiding overlapping matches."""
    q_lower = question.lower()
    found = []
    matched_spans = []
    for idx_dict, etype in [(group_to_techs, "group"), (campaign_to_techs, "campaign"), (malware_to_techs, "malware"), (tool_to_techs, "tool")]:
        for name in sorted(idx_dict, key=len, reverse=True):
            m = re.search(rf'\b{re.escape(name)}\b', q_lower)
            if m and not any(m.start() < end and m.end() > start for start, end in matched_spans):
                found.append((name, etype))
                matched_spans.append((m.start(), m.end()))
    return found


def _cwe_ids_from_answer(answer: str) -> list[str]:
    seen = []
    for match in re.findall(r"\bCWE-\d+\b", answer or "", re.IGNORECASE):
        cwe_id = match.upper()
        if cwe_id not in seen:
            seen.append(cwe_id)
    return seen


def _select_context_cwe(question: str, answer: str, retrieved_chunks: list[dict]) -> dict | None:
    if not CWE_SELECTOR_ENABLED:
        return None
    if not (ROOT_CAUSE_RE.search(question) or re.search(r"\bCVE-\d{4}-\d+\b", question, re.IGNORECASE)):
        return None

    context_cwes = {
        c.get("identifier", "").upper()
        for c in retrieved_chunks
        if c.get("identifier", "").upper().startswith("CWE-")
    }
    if not context_cwes:
        return None

    predicted = (_cwe_ids_from_answer(answer) or [""])[-1]
    for reason, pattern, cwe_id in CWE_SELECTOR_RULES:
        cwe_upper = cwe_id.upper()
        if cwe_upper not in context_cwes or not pattern.search(question):
            continue
        if cwe_upper == predicted:
            return None
        if cwe_upper != predicted:
            cwe_chunk = id_to_chunk.get(cwe_upper, {})
            return {
                "selected": cwe_upper,
                "previous": predicted,
                "reason": reason,
                "name": cwe_chunk.get("name", ""),
            }
    return None


def _select_phrase_index_cwe(question: str, answer: str, retrieved_chunks: list[dict]) -> dict | None:
    if not CWE_PHRASE_SELECTOR_ENABLED or not _cwe_phrase_index:
        return None
    if not (ROOT_CAUSE_RE.search(question) or re.search(r"\bCVE-\d{4}-\d+\b", question, re.IGNORECASE)):
        return None

    context_cwes = {
        c.get("identifier", "").upper()
        for c in retrieved_chunks
        if c.get("identifier", "").upper().startswith("CWE-")
    }
    if not context_cwes:
        return None

    query_norm = _normalize_anchor_text(question)
    matches = []
    for cwe_id in context_cwes:
        for phrase_row in _cwe_phrase_index.get(cwe_id, []):
            phrase = phrase_row.get("phrase", "")
            sources = phrase_row.get("sources", [])
            if "alternate_term" not in sources:
                continue
            if not phrase or not re.search(rf"\b{re.escape(phrase)}\b", query_norm):
                continue
            matches.append({
                "selected": cwe_id,
                "phrase": phrase,
                "sources": sources,
                "score": (len(phrase.split()), len(phrase)),
            })

    if not matches:
        return None
    matches.sort(key=lambda item: item["score"], reverse=True)
    selected = matches[0]
    predicted = (_cwe_ids_from_answer(answer) or [""])[-1]
    if selected["selected"] == predicted:
        return None
    predicted_chunk = id_to_chunk.get(predicted, {})
    if selected["selected"] in {p.upper() for p in predicted_chunk.get("parent_cwe_ids", [])}:
        return None
    cwe_chunk = id_to_chunk.get(selected["selected"], {})
    return {
        "selected": selected["selected"],
        "previous": predicted,
        "reason": "cwe_phrase_index",
        "name": cwe_chunk.get("name", ""),
        "phrase": selected["phrase"],
        "sources": selected["sources"],
    }


def _rewrite_cwe_answer(answer: str, selection: dict) -> str:
    selected = selection["selected"]
    name = selection.get("name") or selected
    return f"The vulnerability description matches {selected}: {name}.\n{selected}"


def _cwe_only_candidates(query: str, k: int = CWE_RESCUE_POOL_K) -> list[dict]:
    q_emb = _get_embedder().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    emb_sc = chunk_embs @ q_emb
    bm25_sc = _bm25_scores(_tokenize(query))
    if bm25_sc.max() > 0:
        bm25_sc = bm25_sc / bm25_sc.max()
    combined = HYBRID_ALPHA * emb_sc + (1 - HYBRID_ALPHA) * bm25_sc
    if ROOT_CAUSE_RE.search(query):
        combined[_root_cause_mask] += 0.3

    cwe_indices = np.flatnonzero(_cwe_source_mask)
    if cwe_indices.size == 0:
        return []
    top_local = np.argsort(-combined[cwe_indices])[:k]
    top_idx = cwe_indices[top_local]
    return [
        {
            "identifier": chunks[int(idx)].get("identifier", "").upper(),
            "name": chunks[int(idx)].get("name", ""),
            "score": float(combined[int(idx)]),
            "chunk": chunks[int(idx)],
        }
        for idx in top_idx
    ]


def _select_cwe_rescue_candidates(
    question: str,
    candidates: list[dict],
    existing_cwes: set[str],
    knn_weights: dict[str, float],
) -> list[dict]:
    if not candidates:
        return []

    selected: list[dict] = []
    query_norm = _normalize_anchor_text(question)
    query_terms = _anchor_terms(question)
    ranked: list[dict] = []

    for rank, candidate in enumerate(candidates, start=1):
        if rank > CWE_RESCUE_MAX_RANK:
            break

        cwe_id = candidate["identifier"]
        if cwe_id in existing_cwes:
            continue

        score = candidate["score"]
        if score < CWE_RESCUE_MIN_SCORE:
            continue

        profile = _cwe_anchor_profiles.get(cwe_id)
        if not profile:
            continue

        phrase_hits = sorted(
            phrase for phrase in profile["phrases"]
            if phrase and re.search(rf"\b{re.escape(phrase)}\b", query_norm)
        )
        term_weights = profile.get("term_weights", {})
        name_hits = sorted(
            term for term in (query_terms & profile["name_terms"])
            if term_weights.get(term, 0.0) >= 2.0
        )
        desc_hits = []
        lexical_terms = sorted(set(name_hits))
        lexical_score = (
            8.0 * len(phrase_hits)
            + 1.5 * sum(term_weights.get(term, 0.0) for term in name_hits)
        )
        if not phrase_hits and len(lexical_terms) < CWE_RESCUE_MIN_LEXICAL_TERMS:
            continue
        single_rare_name_hit = (
            len(name_hits) == 1
            and term_weights.get(name_hits[0], 0.0) >= 4.5
        )
        has_specific_overlap = (
            bool(phrase_hits)
            or len(name_hits) >= 2
            or single_rare_name_hit
        )
        if not has_specific_overlap:
            continue
        if lexical_score < CWE_RESCUE_MIN_LEXICAL_SCORE and not phrase_hits:
            continue

        ranked.append({
            **candidate,
            "rank": rank,
            "reason": "cwe_auto_lexical_anchor",
            "lexical_score": lexical_score,
            "phrase_hits": phrase_hits,
            "name_hits": name_hits,
            "desc_hits": desc_hits,
            "knn_weight": knn_weights.get(cwe_id, 0.0),
        })

    ranked.sort(key=lambda c: (-c["lexical_score"], c["rank"], -c["score"]))
    for candidate in ranked:
        selected.append(candidate)
        if len(selected) >= CWE_RESCUE_MAX_ADD:
            break
    return selected




def ask(question: str, history: list, eval_mode: bool = False, debug_info: dict | None = None) -> tuple[str, list, str]:
    # Reverse entity lookup — fires when both a reverse-question pattern AND a known entity name appear.
    # Skip when 2+ entities detected — let retrieval + LLM handle comparison/relationship queries.
    # SKIP/BYPASS if query also asks for CWEs, detection, or mitigations (complex synthesis needed).
    if ENTITY_REVERSE_RE.search(question) and not (ROOT_CAUSE_RE.search(question) or DETECTION_RE.search(question) or MITIGATION_RE.search(question) or WHAT_IS_RE.search(question)):
        all_entities = _find_all_entities_in_query(question)
        match = _find_entity_in_query(question) if len(all_entities) < 2 else None
        if match:
            name, entity_type, _ = match
            display = display_name.get(name, name)

            # determine target: techniques (default), malware, tools, or groups
            if TARGET_MALWARE_RE.search(question) and entity_type == "group":
                items = group_to_malware.get(name, [])
                if items:
                    text = f"**{display}** (group) is observed using {len(items)} malware:\n" + "\n".join(f"- {m}" for m in sorted(items))
                else:
                    text = f"No malware attributed to {display} in MITRE ATT&CK."
                print(f"[Reverse lookup: {display} → malware ({len(items)})]\n")
                return text, [], text

            if TARGET_TOOLS_RE.search(question) and entity_type == "group":
                items = group_to_tools.get(name, [])
                if items:
                    text = f"**{display}** (group) is observed using {len(items)} tool(s):\n" + "\n".join(f"- {t}" for t in sorted(items))
                else:
                    text = f"No tools attributed to {display} in MITRE ATT&CK."
                print(f"[Reverse lookup: {display} → tools ({len(items)})]\n")
                return text, [], text

            if TARGET_GROUPS_RE.search(question) and entity_type in ("malware", "tool"):
                rev_idx = malware_to_groups if entity_type == "malware" else tool_to_groups
                items = rev_idx.get(name, [])
                if items:
                    text = f"**{display}** ({entity_type}) is used by {len(items)} group(s):\n" + "\n".join(f"- {g}" for g in sorted(set(items)))
                else:
                    text = f"No groups attributed to using {display} in MITRE ATT&CK."
                print(f"[Reverse lookup: {display} → groups ({len(items)})]\n")
                return text, [], text

            # default: techniques (existing behavior)
            idx = {"group": group_to_techs, "campaign": campaign_to_techs, "malware": malware_to_techs, "tool": tool_to_techs}[entity_type]
            text = _techs_for_entity(name, entity_type, idx)
            print(f"[Reverse entity lookup: {display} ({entity_type}) → {len(idx[name])} techniques]\n")
            return text, [], text

    _t = {} if CAPTURE_TIMING else None
    if _t is not None: _t["start"] = time.time()

    retrieved = retrieve(question)
    retrieved_chunks = [c for c, _ in retrieved]
    if debug_info is not None:
        debug_info["initial_retrieved"] = [
            {
                "identifier": c.get("identifier"),
                "name": c.get("name"),
                "source": c.get("source"),
                "type": c.get("type"),
                "score": score,
                "cwe_ids": c.get("cwe_ids", []),
            }
            for c, score in retrieved
        ]

    if _t is not None: _t["retrieve"] = time.time()

    score_info = ", ".join(f"{c['identifier'] or c.get('name', '?')} ({s:.2f})" for c, s in retrieved)
    if not eval_mode:
        print(f"[Hybrid search: {score_info}]\n")

    # For CVE-specific queries: keep only the matched CVE chunk as context.
    # Other chunks (random CVEs, generic CWEs) add noise — the CVE chunk already
    # contains the description and any NVD-assigned CWE the LLM should use.
    _cve_id_m = re.search(r'\bCVE-\d{4}-\d+\b', question, re.IGNORECASE)
    if _cve_id_m:
        _queried_cve = _cve_id_m.group(0).upper()
        _cve_specific = [c for c in retrieved_chunks if c.get("identifier", "").upper() == _queried_cve]
        if _cve_specific:
            retrieved_chunks = _cve_specific
        if debug_info is not None:
            debug_info["cve_id_filter"] = {
                "queried_cve": _queried_cve,
                "matched": bool(_cve_specific),
            }

    # Direct bridge injection: if query asks about CAPEC for a specific technique ID,
    # inject bridge-mapped CAPECs directly — bridge mappings are explicit facts, not
    # semantic guesses, so they should not compete in the graph traversal lottery.
    _tech_m = re.search(r'\bT\d{4}(?:\.\d{3})?\b', question)
    if _tech_m and CAPEC_RE.search(question):
        _tech_id = _tech_m.group(0).upper()
        _bridge_capecs = attack_to_capec.get(_tech_id, [])
        if debug_info is not None:
            debug_info["tech_capec_bridge"] = {"tech_id": _tech_id, "capec_ids": _bridge_capecs}
        for _cid in _bridge_capecs:
            if _cid.upper() in id_to_chunk:
                retrieved_chunks.append(id_to_chunk[_cid.upper()])
                print(f"[Bridge injection: {_tech_id} → {_cid}]\n")

    # Direct bridge injection: if query asks about CWE for a specific CAPEC ID,
    # inject bridge-mapped CWEs directly.
    _capec_m = re.search(r'\bCAPEC-\d+\b', question, re.IGNORECASE)
    if _capec_m and ROOT_CAUSE_RE.search(question):
        _capec_id = _capec_m.group(0).upper()
        _bridge_cwes = capec_to_cwe.get(_capec_id, [])
        if debug_info is not None:
            debug_info["capec_cwe_bridge"] = {"capec_id": _capec_id, "cwe_ids": _bridge_cwes}
        if _bridge_cwes:
            # Bridge is ground truth — strip semantically-retrieved CWE chunks so the
            # LLM can't pick a false-positive (e.g. CWE-203 for CAPEC-203 query).
            _bridge_cwe_ids = {cid.upper() for cid in _bridge_cwes}
            retrieved_chunks = [c for c in retrieved_chunks
                                if not (c.get("source") == "CWE"
                                        and c.get("identifier", "").upper() not in _bridge_cwe_ids)]
        for _cid in _bridge_cwes:
            if _cid.upper() in id_to_chunk:
                retrieved_chunks.append(id_to_chunk[_cid.upper()])
                print(f"[Bridge injection: {_capec_id} → {_cid}]\n")

    # Mitigation filter: strip M#### chunks that are not graph-neighbors of the queried
    # technique. Prevents unrelated mitigations (e.g. M1055 "Do Not Mitigate") from
    # polluting the answer when the technique has a specific mitigation (e.g. M1024).
    if _tech_m and MITIGATION_RE.search(question):
        _tech_id_mit = _tech_m.group(0).upper()
        _tech_neighbors = chunk_graph.get(_tech_id_mit, set())
        retrieved_chunks = [c for c in retrieved_chunks
                            if not (c.get("type") == "mitigation"
                                    and c.get("identifier", "").upper() not in _tech_neighbors)]

    if _t is not None: _t["filters"] = time.time()

    # Universal Knowledge Graph Traversal
    # Determine depth: 2-hop for complex queries (Deep Search), 1-hop otherwise.
    is_deep = bool(ROOT_CAUSE_RE.search(question) or DETECTION_RE.search(question) or MITIGATION_RE.search(question) or CAPEC_RE.search(question))
    max_neighbors = 5 if is_deep else 2

    neighbor_ids = set()
    added_ids = set(c.get("identifier", "").upper() for c in retrieved_chunks if c.get("identifier"))

    # 1st Hop
    hop1_ids = set()
    for c in retrieved_chunks:
        cid = c.get("identifier", "").upper()
        if cid in chunk_graph:
            for neighbor in chunk_graph[cid]:
                if neighbor not in added_ids:
                    hop1_ids.add(neighbor)
    
    # 2nd Hop (Deep Search only)
    all_candidate_ids = set(hop1_ids)
    if is_deep:
        for nid in hop1_ids:
            if nid in chunk_graph:
                for n2 in chunk_graph[nid]:
                    if n2 not in added_ids:
                        all_candidate_ids.add(n2)
    
    if _t is not None:
        _t["graph"] = time.time()
        _t["candidate_count"] = len(all_candidate_ids)
    if debug_info is not None:
        debug_info["graph_candidates"] = {
            "is_deep": is_deep,
            "hop1_count": len(hop1_ids),
            "candidate_count": len(all_candidate_ids),
        }

    # Re-rank all candidate neighbors by similarity to the query.
    # Use pre-computed chunk_embs instead of re-encoding — identical vectors, zero GPU cost.
    neighbor_chunks = []
    if all_candidate_ids:
        n_list   = [nid for nid in all_candidate_ids if nid in id_to_idx]
        n_chunks = [id_to_chunk[nid] for nid in n_list if nid in id_to_chunk]
        n_list   = [nid for nid in n_list if nid in id_to_chunk]
        if n_chunks:
            n_embs = chunk_embs[[id_to_idx[nid] for nid in n_list]]
            q_emb  = _get_embedder().encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
            scores = np.dot(n_embs, q_emb)

            # For root-cause queries, prefer CWE chunks over technique/CAPEC chunks
            if ROOT_CAUSE_RE.search(question):
                for idx_n, nc in enumerate(n_chunks):
                    if nc.get("source") == "CWE" or nc.get("identifier", "").upper().startswith("CWE-"):
                        scores[idx_n] += 2.0

            # For attack-pattern queries, prefer CAPEC chunks in neighbor re-ranking
            if CAPEC_RE.search(question):
                for idx_n, nc in enumerate(n_chunks):
                    if nc.get("identifier", "").upper().startswith("CAPEC-"):
                        scores[idx_n] += 2.0

            # For mitigation queries, prefer M#### chunks in neighbor re-ranking
            if MITIGATION_RE.search(question):
                for idx_n, nc in enumerate(n_chunks):
                    if nc.get("type") == "mitigation":
                        scores[idx_n] += 2.0

            # Take top N
            top_n_idx = np.argsort(scores)[-max_neighbors:][::-1]
            neighbor_chunks = [n_chunks[i] for i in top_n_idx]
            if debug_info is not None:
                debug_info["graph_neighbors"] = [
                    {
                        "identifier": n_chunks[i].get("identifier"),
                        "name": n_chunks[i].get("name"),
                        "source": n_chunks[i].get("source"),
                        "type": n_chunks[i].get("type"),
                        "score": float(scores[i]),
                    }
                    for i in top_n_idx
                ]
            
            for nc in neighbor_chunks:
                if nc.get("identifier", "").upper() not in added_ids:
                    retrieved_chunks.append(nc)
                    added_ids.add(nc.get("identifier", "").upper())
            
            mode_str = "Deep Search (2-hop)" if is_deep else "1-hop"
            if not eval_mode:
                print(f"[{mode_str}: added top {len(neighbor_chunks)} graph neighbors to context]\n")

    if _t is not None: _t["neighbors"] = time.time()

    # CVE description bridge: if a CVE chunk made it into context (likely the matching
    # CVE for a description-style query), its NVD CWE assignments are authoritative.
    # Strip all other CWE chunks and inject the NVD CWEs. Same pattern as CAPEC→CWE bridge.
    bridge_fired = False
    if not _cve_id_m:
        cve_with_cwes = next((c for c in retrieved_chunks
                              if c.get("source") == "CVE" and c.get("cwe_ids")), None)
        if cve_with_cwes:
            nvd_cwes = {cid.upper() for cid in cve_with_cwes["cwe_ids"]}
            retrieved_chunks = [c for c in retrieved_chunks
                                if not (c.get("source") == "CWE"
                                        and c.get("identifier", "").upper() not in nvd_cwes)]
            existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
            for cwe_id in nvd_cwes:
                if cwe_id not in existing_ids and cwe_id in id_to_chunk:
                    retrieved_chunks.append(id_to_chunk[cwe_id])
            bridge_fired = True
            if debug_info is not None:
                debug_info["cve_description_bridge"] = {
                    "cve_id": cve_with_cwes.get("identifier"),
                    "cwe_ids": sorted(nvd_cwes),
                }
            if not eval_mode:
                print(f"[CVE description bridge: {cve_with_cwes['identifier']} → {sorted(nvd_cwes)}]\n")

    knn_high_confidence = False
    knn_cwe_weights: dict[str, float] = {}

    # k-NN CWE fallback: for unmapped CVE descriptions (direct bridge didn't fire),
    # find the N most similar mapped CVEs and vote on their CWEs, weighted by similarity.
    # High-confidence votes (top CWE wins enough weighted signal) are treated as
    # authoritative — strip competing CWE chunks like the direct bridge does. Lower
    # confidence stays as a soft hint (inject without stripping).
    if not bridge_fired and _mapped_cve_indices.size:
        q_emb_knn = _get_embedder().encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        mapped_sims = chunk_embs[_mapped_cve_indices] @ q_emb_knn
        top_n = max(1, min(KNN_CWE_NEIGHBORS, _mapped_cve_indices.size))
        top_local = np.argpartition(-mapped_sims, top_n - 1)[:top_n]
        top_local = top_local[np.argsort(-mapped_sims[top_local])]
        cwe_weights: dict[str, float] = defaultdict(float)
        total_weight = 0.0
        knn_neighbors = []
        for local_i in top_local:
            sim = float(mapped_sims[local_i])
            global_i = int(_mapped_cve_indices[local_i])
            neighbor_cwes = chunks[global_i].get("cwe_ids", [])
            knn_neighbors.append({
                "identifier": chunks[global_i].get("identifier"),
                "name": chunks[global_i].get("name"),
                "similarity": sim,
                "cwe_ids": neighbor_cwes,
            })
            for cwe_id in chunks[global_i].get("cwe_ids", []):
                cwe_weights[cwe_id.upper()] += sim
            total_weight += sim
        if cwe_weights:
            sorted_cwes = sorted(cwe_weights.items(), key=lambda kv: -kv[1])
            knn_weights = {cid: weight for cid, weight in sorted_cwes}
            knn_cwe_weights = knn_weights
            top_cwe, top_weight = sorted_cwes[0]
            second_weight = sorted_cwes[1][1] if len(sorted_cwes) > 1 else 0.0
            top_share = top_weight / total_weight if total_weight else 0.0
            margin_ratio = (top_weight / second_weight) if second_weight else float("inf")
            high_confidence = (
                top_share >= KNN_CONFIDENCE_THRESHOLD
                and margin_ratio >= KNN_CONFIDENCE_MARGIN
            )
            if debug_info is not None:
                debug_info["knn_cwe"] = {
                    "neighbors": knn_neighbors,
                    "weights": knn_weights,
                    "top_cwe": top_cwe,
                    "top_share": top_share,
                    "second_weight": second_weight,
                    "margin_ratio": margin_ratio,
                    "margin_threshold": KNN_CONFIDENCE_MARGIN,
                    "threshold": KNN_CONFIDENCE_THRESHOLD,
                    "neighbor_count": top_n,
                    "mode": "high_confidence" if high_confidence else "soft_hint",
                }

            if high_confidence:
                knn_high_confidence = True
                # High confidence: strip non-voted CWE chunks (authoritative signal)
                retrieved_chunks = [c for c in retrieved_chunks
                                    if not (c.get("source") == "CWE"
                                            and c.get("identifier", "").upper() != top_cwe)]
                existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
                if top_cwe not in existing_ids and top_cwe in id_to_chunk:
                    retrieved_chunks.append(id_to_chunk[top_cwe])
                if not eval_mode:
                    print(f"[k-NN CWE high-confidence: {top_cwe} share={top_share:.0%}]\n")
            else:
                # Lower confidence: soft hint, inject top 3 without stripping
                top_voted = [cid for cid, _ in sorted_cwes[:3]]
                existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
                for cwe_id in top_voted:
                    if cwe_id in id_to_chunk and cwe_id not in existing_ids:
                        retrieved_chunks.append(id_to_chunk[cwe_id])
                if not eval_mode:
                    # Contextual Imputation: Add brief descriptions of the nearest neighbors
                    # to help the LLM "impute" the missing context for the unmapped CVE.
                    neighbor_context_added = False
                    for neighbor in knn_neighbors[:2]: # Only top 2 to save tokens
                        n_id = neighbor["identifier"]
                        n_chunk = id_to_chunk.get(n_id)
                        if n_chunk:
                            desc_snippet = n_chunk["text"][:300].replace("\n", " ").strip() + "..."
                            imputed_chunk = {
                                "type": "SIMILAR_VULNERABILITY",
                                "identifier": n_id,
                                "name": n_chunk.get("name", ""),
                                "text": f"Similar vulnerability {n_id} is mapped to {neighbor.get('cwe_ids')}. Description: {desc_snippet}",
                                "source": "k-NN Imputation"
                            }
                            retrieved_chunks.append(imputed_chunk)
                            neighbor_context_added = True

                    msg = f"[k-NN CWE soft hint: {top_voted} top share={top_share:.0%}]"
                    if neighbor_context_added:
                        msg += " (Context imputed)"
                    print(msg + "\n")

    if CWE_KEYWORD_ANCHORS_ENABLED and not _cve_id_m and not bridge_fired:
        anchored_cwes = []
        existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
        for pattern, cwe_id in CWE_KEYWORD_ANCHORS:
            cwe_upper = cwe_id.upper()
            if pattern.search(question) and cwe_upper in id_to_chunk and cwe_upper not in existing_ids:
                retrieved_chunks.append(id_to_chunk[cwe_upper])
                anchored_cwes.append(cwe_upper)
                existing_ids.add(cwe_upper)
        if anchored_cwes and not eval_mode:
            print(f"[CWE keyword anchors: {anchored_cwes}]\n")
        if debug_info is not None:
            debug_info["keyword_anchors"] = anchored_cwes

    # Experimental CWE-only rescue for long CVE-description prompts. This is a soft
    # add based on retrieval agreement: no phrase-to-CWE mapping, and no stripping.
    if (
        CWE_RESCUE_ENABLED
        and not _cve_id_m
        and not bridge_fired
        and ROOT_CAUSE_RE.search(question)
        and len(_tokenize(question)) > 15
    ):
        cwe_candidates = _cwe_only_candidates(question)
        existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
        existing_cwes = {cid for cid in existing_ids if cid.startswith("CWE-")}
        rescue_selected = _select_cwe_rescue_candidates(question, cwe_candidates, existing_cwes, knn_cwe_weights)
        rescue_added = []
        for candidate in rescue_selected:
            cwe_id = candidate["identifier"]
            if cwe_id in id_to_chunk and cwe_id not in existing_ids:
                retrieved_chunks.append(id_to_chunk[cwe_id])
                existing_ids.add(cwe_id)
                rescue_added.append(candidate)

        def _debug_candidate(candidate: dict) -> dict:
            return {
                "identifier": candidate["identifier"],
                "name": candidate["name"],
                "score": candidate["score"],
                "rank": candidate.get("rank"),
                "reason": candidate.get("reason"),
                "lexical_score": candidate.get("lexical_score"),
                "phrase_hits": candidate.get("phrase_hits"),
                "name_hits": candidate.get("name_hits"),
                "desc_hits": candidate.get("desc_hits"),
                "knn_weight": candidate.get("knn_weight"),
            }

        if debug_info is not None:
            debug_info["cwe_rescue"] = {
                "added": [_debug_candidate(c) for c in rescue_added],
                "selected": [_debug_candidate(c) for c in rescue_selected],
                "top_candidates": [_debug_candidate({**c, "rank": i + 1}) for i, c in enumerate(cwe_candidates[:10])],
                "pool_k": CWE_RESCUE_POOL_K,
                "max_add": CWE_RESCUE_MAX_ADD,
                "max_rank": CWE_RESCUE_MAX_RANK,
                "min_score": CWE_RESCUE_MIN_SCORE,
                "min_lexical_score": CWE_RESCUE_MIN_LEXICAL_SCORE,
                "min_lexical_terms": CWE_RESCUE_MIN_LEXICAL_TERMS,
                "knn_weights_available": bool(knn_cwe_weights),
            }
        if rescue_added and not eval_mode:
            added_ids = [c["identifier"] for c in rescue_added]
            print(f"[CWE rescue: added {added_ids}]\n")

    # Experimental hierarchy expansion for unmapped CVE cases. When CWE evidence is
    # soft, add both parent and child CWE candidates so the full hierarchy is visible
    # to the model. Do not run after authoritative NVD or high-confidence k-NN bridges.
    if CWE_HIERARCHY_EXPANSION_ENABLED and not bridge_fired and not knn_high_confidence:
        existing_ids = {c.get("identifier", "").upper() for c in retrieved_chunks}
        hierarchy_added = []
        for c in list(retrieved_chunks):
            cid = c.get("identifier", "").upper()
            if not cid.startswith("CWE-"):
                continue
            
            # Add parents (upwards)
            for parent_id in c.get("parent_cwe_ids", []):
                parent_upper = parent_id.upper()
                if parent_upper in id_to_chunk and parent_upper not in existing_ids:
                    retrieved_chunks.append(id_to_chunk[parent_upper])
                    existing_ids.add(parent_upper)
                    hierarchy_added.append(parent_upper)
            
            # Add children (downwards)
            for child_id in child_cwe_ids.get(cid, []):
                child_upper = child_id.upper()
                if child_upper in id_to_chunk and child_upper not in existing_ids:
                    retrieved_chunks.append(id_to_chunk[child_upper])
                    existing_ids.add(child_upper)
                    hierarchy_added.append(child_upper)

        if hierarchy_added and not eval_mode:
            print(f"[CWE hierarchy expansion: {hierarchy_added}]\n")
        if debug_info is not None:
            debug_info["cwe_hierarchy_expansion"] = hierarchy_added

    # Cross-encoder re-ranking of CWE chunks. Pure reordering, no stripping.
    # Skipped on authoritative paths (CVE description bridge, high-confidence k-NN),
    # since those decisions are factual lookups rather than retrieval guesses.
    if CWE_CROSSENCODER_ENABLED and not bridge_fired and not knn_high_confidence:
        cwe_positions = [i for i, c in enumerate(retrieved_chunks)
                         if (c.get("source") == "CWE"
                             or c.get("identifier", "").upper().startswith("CWE-"))]
        if len(cwe_positions) >= 2:
            from cwe_reranker import score_cwe_chunks
            cwe_chunks_to_score = [retrieved_chunks[i] for i in cwe_positions]
            scored = score_cwe_chunks(question, cwe_chunks_to_score)
            reordered_cwe_iter = iter(sc[0] for sc in scored)
            cwe_position_set = set(cwe_positions)
            rebuilt = []
            for i, c in enumerate(retrieved_chunks):
                if i in cwe_position_set:
                    rebuilt.append(next(reordered_cwe_iter))
                else:
                    rebuilt.append(c)
            retrieved_chunks = rebuilt
            if debug_info is not None:
                debug_info["cwe_crossencoder"] = [
                    {"identifier": c.get("identifier"),
                     "name": c.get("name"),
                     "score": s}
                    for c, s in scored
                ]
            if not eval_mode and scored:
                top = scored[0][0].get("identifier")
                print(f"[CWE cross-encoder reranked: top={top} n={len(scored)}]\n")

    hist = _history_str(history)
    
    context_parts = []
    for c in retrieved_chunks:
        ctype = c.get("type", "chunk").replace("_", " ").upper()
        if not c.get("identifier"):
            context_parts.append(f"[{ctype} — {c['name']}]\n{c['text']}")
        else:
            context_parts.append(f"[{ctype} {c['identifier']} — {c['name']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)
    if debug_info is not None:
        debug_info["final_context"] = [
            {
                "identifier": c.get("identifier"),
                "name": c.get("name"),
                "source": c.get("source"),
                "type": c.get("type"),
                "cwe_ids": c.get("cwe_ids", []),
            }
            for c in retrieved_chunks
        ]
        debug_info["final_context_cwes"] = [
            c.get("identifier", "").upper()
            for c in retrieved_chunks
            if c.get("identifier", "").upper().startswith("CWE-")
        ]
    
    eval_rule = (
        "10. Be concise: one justification sentence then the CWE ID on the last line. No more."
        if eval_mode else ""
    )

    # is_cve_query fires when CVE ID is in query OR a CVE chunk is in context (description match).
    # The cve_rule then tells the LLM to trust the explicit CWE listed in the CVE chunk.
    is_cve_query = (bool(re.search(r'\bCVE-\d{4}-\d+\b', question, re.IGNORECASE))
                    or any(c.get("source") == "CVE" for c in retrieved_chunks))
    cve_rule = (
        "9. For CVE root-cause (CWE) questions: if the context lists an explicit CWE ID, state it. "
        "If the weakness shows 'n/a' or no CWE ID, you MUST analyze the vulnerability description "
        "and determine the most likely CWE (e.g. CWE-79 for XSS, CWE-89 for SQL injection, "
        "CWE-416 for use-after-free, CWE-787 for out-of-bounds write, CWE-77/78 for command injection). "
        "Always end your answer with the CWE ID on its own line. Never say 'I don't have enough information' "
        "for CVE weakness questions."
        if is_cve_query else ""
    )

    hierarchy_rule = (
        "9b. CWE abstraction level: Prefer Base-level CWEs over Variant-level (too specific) "
        "or Class/Pillar-level (too abstract). If context shows a Variant such as CWE-121 "
        "(Stack-based Buffer Overflow) alongside its Base parent CWE-787 (Out-of-bounds Write), "
        "report the Base-level CWE. If context shows a Class such as CWE-77 (Command Injection) "
        "alongside its Base child CWE-78 (OS Command Injection), report the Base-level CWE. "
        "The abstraction level is shown in parentheses in each CWE block header."
        if is_cve_query else ""
    )

    prompt = f"""You are a Cyber Threat Intelligence expert. Answer the question based ONLY on the provided context.

CRITICAL RULES:
1. If the question asks for "Detection" or "How to detect", use the "Detection Strategy" or "Analytics" chunks.
2. Do NOT use the "Mitigations" section to answer detection questions. Mitigations are for prevention; Detection is for monitoring/log-analysis.
3. If an answer cannot be found in the context, say "I don't have enough information in my database."
4. Answer ONLY what is asked. Be concise and technical.
5. Do NOT invent, infer, or fabricate technique IDs (T####), group names, or relationships not explicitly stated in the context above.
6. Answer ONLY the current question. Do not bring in entities or facts from conversation history unless the current question references them.
7. The [TYPE IDENTIFIER — NAME] header in each context block is the authoritative name and ID for that entry. Use those exact values. Never rename or re-identify based on training knowledge.
8. Do not add shell commands, scripts, vendor product names, or tool recommendations that are not verbatim in the context blocks above.
{cve_rule}
{hierarchy_rule}
{eval_rule}

Conversation history (for context only):
{hist}

Context:
{context}

QUESTION: {question}

ANSWER:"""
    if _t is not None: _t["prompt_built"] = time.time()
    answer = _llm(prompt)
    cwe_selection = _select_phrase_index_cwe(question, answer, retrieved_chunks)
    if not cwe_selection:
        cwe_selection = _select_context_cwe(question, answer, retrieved_chunks)
    if cwe_selection:
        answer = _rewrite_cwe_answer(answer, cwe_selection)
        if not eval_mode:
            print(
                f"[CWE selector: {cwe_selection['previous'] or 'none'} → "
                f"{cwe_selection['selected']} ({cwe_selection['reason']})]\n"
            )
    if debug_info is not None:
        debug_info["cwe_selector"] = cwe_selection
    if _t is not None:
        _t["llm"] = time.time()
        if debug_info is not None:
            debug_info["timing"] = {
                "retrieve_s": _t["retrieve"] - _t["start"],
                "filters_s": _t["filters"] - _t["retrieve"],
                "graph_s": _t["graph"] - _t["filters"],
                "candidate_count": _t["candidate_count"],
                "neighbors_s": _t["neighbors"] - _t["graph"],
                "prompt_s": _t["prompt_built"] - _t["neighbors"],
                "llm_s": _t["llm"] - _t["prompt_built"],
                "total_s": _t["llm"] - _t["start"],
            }
        _profile_print(
            f"[T] retrieve={_t['retrieve']-_t['start']:.2f}s "
            f"filters={_t['filters']-_t['retrieve']:.2f}s "
            f"graph={_t['graph']-_t['filters']:.2f}s ({_t['candidate_count']} cands) "
            f"neighbors={_t['neighbors']-_t['graph']:.2f}s "
            f"prompt={_t['prompt_built']-_t['neighbors']:.2f}s "
            f"llm={_t['llm']-_t['prompt_built']:.2f}s "
            f"total={_t['llm']-_t['start']:.2f}s",
            flush=True
        )
    return answer, retrieved_chunks, answer


if __name__ == "__main__":
    history = []
    while True:
        q = input("Ask (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        print()
        answer, retrieved_chunks, shown_text = ask(q, history)
        history.append({"q": q, "a": answer, "retrieved": retrieved_chunks, "shown_text": shown_text})
        print(answer)
        print()
        print("─" * 60)
        print()
