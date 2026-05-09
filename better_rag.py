"""
Hybrid RAG (BM25 + embeddings) over ATT&CK chunks + local LLM via Ollama.
"""
import json
import math
import re
import difflib
import numpy as np
import requests
from collections import Counter, defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer

ATTACK_CHUNKS = Path("data/processed/attack_chunks.jsonl")
CAPEC_CHUNKS  = Path("data/processed/capec_chunks.jsonl")
CWE_CHUNKS    = Path("data/processed/cwe_chunks.jsonl")
CVE_CHUNKS    = Path("data/processed/cve_chunks.jsonl")
RELATIONS     = Path("data/processed/entity_relations.json")
CAPEC_RELS    = Path("data/processed/capec_attack_relations.json")
CAPEC_CWE_RELS = Path("data/processed/capec_cwe_relations.json")
OLLAMA        = "http://localhost:11434/api/generate"
MODEL        = "qwen2.5:7b"
EMBEDDER     = "BAAI/bge-small-en-v1.5"
BM25_K1      = 1.5
BM25_B       = 0.75
HYBRID_ALPHA = 0.5   # 0 = pure BM25, 1 = pure embedding
TOP_K        = 8
EMB_CACHE    = Path("data/processed/chunk_embs.npy")

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
    if c.get("source") == "CVE":
        continue  # CVE text references many CAPEC/CWE IDs that aren't meaningful graph edges
    cid = c.get("identifier", "").upper()
    if not cid: continue
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

# technique-name index — sorted longest-first so "Scanning IP Blocks" wins over "Active Scanning"
_tech_name_index: list[tuple[str, int]] = sorted(
    [(c["name"].lower(), i) for i, c in enumerate(chunks) if c.get("name")],
    key=lambda x: len(x[0]), reverse=True,
)

print("Building BM25 index...")
_tok_re = re.compile(r'\w+')

def _tokenize(text: str) -> list[str]:
    return _tok_re.findall(text.lower())

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

print("Loading embedder...")
embedder = SentenceTransformer(EMBEDDER, device="cuda")

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


# ── retrieval ─────────────────────────────────────────────────────────────────

def _bm25_scores(query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(_N)
    for t in set(query_tokens):
        postings = _inv_idx.get(t)
        if not postings:
            continue
        df = len(postings)
        idf = math.log((_N - df + 0.5) / (df + 0.5) + 1)
        for i, tf in postings.items():
            dl = _doc_len[i]
            scores[i] += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / _avgdl))
    return scores


def retrieve(query: str, k: int = TOP_K) -> list:
    q_emb   = embedder.encode([query], normalize_embeddings=True)[0]
    emb_sc  = chunk_embs @ q_emb

    bm25_sc = _bm25_scores(_tokenize(query))
    if bm25_sc.max() > 0:
        bm25_sc = bm25_sc / bm25_sc.max()

    combined = HYBRID_ALPHA * emb_sc + (1 - HYBRID_ALPHA) * bm25_sc

    # 4. Intent-based boosting (e.g., if query asks for detection, boost detection chunks)
    if DETECTION_RE.search(query):
        for i, c in enumerate(chunks):
            if c.get("type") == "detection":
                combined[i] += 2.0

    # 5. Fuzzy name boost (handle typos like 'blusmacking')
    query_words = _tokenize(query)
    for word in query_words:
        if len(word) < 5: continue
        # Find close matches in our name index
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
        for i, c in enumerate(chunks):
            if c.get("identifier", "").upper() == match.upper():
                combined[i] += 2.0
                break

    # if no ID, boost the chunk whose technique name appears verbatim in the query
    # longest name wins so "Scanning IP Blocks" beats "Active Scanning" when both fit
    if not id_matches:
        q_lower = query.lower()
        for name, idx in _tech_name_index:
            if re.search(rf'\b{re.escape(name)}\b', q_lower):
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
        for i, c in enumerate(chunks):
            if c.get("name", "").lower() == ent_name and c.get("type") in ("group", "malware", "tool", "campaign"):
                combined[i] += 2.0
                break

        for rc in related_chunks:
            # find index of this chunk
            for i, c in enumerate(chunks):
                if c["identifier"] == rc["identifier"]:
                    boost = 1.5
                    # Connectivity Boost: if this technique has a CAPEC mapping, boost it more
                    if c["identifier"] in attack_to_capec:
                        boost += 1.0
                    combined[i] += boost
                    break
        # Also boost relations (group -> malware etc)
        if ent_type == "group":
            for m_name in group_to_malware.get(ent_name, []):
                for i, c in enumerate(chunks):
                    if c["name"].lower() == m_name.lower():
                        combined[i] += 1.0
                        break

    # Intent Routing: Boost CWE chunks if query focuses on root causes.
    # Keep this small — technique chunks must stay in top-K so the graph can reach the
    # correct CWEs via traversal. The +2.0 graph-neighbor CWE boost (in ask()) does the
    # heavy lifting for root-cause queries.
    if ROOT_CAUSE_RE.search(query):
        for i, c in enumerate(chunks):
            if c.get("source") == "CWE" or c.get("type") in ("weakness", "category", "view"):
                combined[i] += 0.3

    # Penalize nameless chunks — they have no identifier so they can't participate
    # in graph traversal and only add noise to the context.
    for i, c in enumerate(chunks):
        if not c.get("identifier"):
            combined[i] -= 1.0

    # Penalize CVE chunks for non-CVE queries — they're semantically similar to
    # ATT&CK/CAPEC content and flood the top-K when no CVE ID is in the query.
    if not re.search(r'\bCVE-\d{4}-\d+\b', query, re.IGNORECASE):
        for i, c in enumerate(chunks):
            if c.get("source") == "CVE":
                combined[i] -= 0.5
    else:
        # For CVE queries: suppress all CWE chunks — they're generic noise when the CVE
        # chunk already contains the CWE info (or "n/a"). The LLM should reason from the
        # CVE description, not pick a random quality CWE like CWE-1116.
        for i, c in enumerate(chunks):
            if c.get("source") == "CWE":
                combined[i] -= 1.5

    top_idx  = np.argsort(-combined)[:k]
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
        r = requests.post(OLLAMA, json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        })
        r.raise_for_status()
        data = r.json()
        if "response" not in data:
            print(f"Ollama Error (Missing 'response'): {data}")
            return "Error: Ollama returned an unexpected response."
        return data["response"]
    except Exception as e:
        print(f"Ollama Connection Error: {e}")
        if 'r' in locals():
            print(f"Status Code: {r.status_code}, Body: {r.text}")
        return "Error: Could not connect to Ollama."


ENTITY_REVERSE_RE = re.compile(r'\b(what|which|list|show).{0,50}\b(techniques?|attacks?|tactics?|malwares?|software|tools?|groups?|actors?|do|does|use[sd]?|cause[sd]?)\b', re.IGNORECASE)
TARGET_MALWARE_RE = re.compile(r'\b(malwares?|software)\b', re.IGNORECASE)
TARGET_TOOLS_RE   = re.compile(r'\btools?\b', re.IGNORECASE)
TARGET_GROUPS_RE  = re.compile(r'\b(groups?|actors?|threat\s+actors?)\b', re.IGNORECASE)
ROOT_CAUSE_RE     = re.compile(r'\b(root cause|vulnerability|flaw|cwe|bug|code|architectural|improper|validation)\b', re.IGNORECASE)
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




def ask(question: str, history: list) -> tuple[str, list, str]:
    # Reverse entity lookup — fires when both a reverse-question pattern AND a known entity name appear.
    # Skip when 2+ entities detected — let retrieval + LLM handle comparison/relationship queries.
    # SKIP/BYPASS if query also asks for CWEs, detection, or mitigations (complex synthesis needed).
    if ENTITY_REVERSE_RE.search(question) and not (ROOT_CAUSE_RE.search(question) or DETECTION_RE.search(question) or MITIGATION_RE.search(question)):
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

    retrieved = retrieve(question)
    retrieved_chunks = [c for c, _ in retrieved]

    score_info = ", ".join(f"{c['identifier'] or c.get('name', '?')} ({s:.2f})" for c, s in retrieved)
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

    # Direct bridge injection: if query asks about CAPEC for a specific technique ID,
    # inject bridge-mapped CAPECs directly — bridge mappings are explicit facts, not
    # semantic guesses, so they should not compete in the graph traversal lottery.
    _tech_m = re.search(r'\bT\d{4}(?:\.\d{3})?\b', question)
    if _tech_m and CAPEC_RE.search(question):
        _tech_id = _tech_m.group(0).upper()
        _bridge_capecs = attack_to_capec.get(_tech_id, [])
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
    
    # Re-rank all candidate neighbors by similarity to the query
    neighbor_chunks = []
    if all_candidate_ids:
        n_list = list(all_candidate_ids)
        n_chunks = [id_to_chunk[nid] for nid in n_list if nid in id_to_chunk]
        if n_chunks:
            n_embs = embedder.encode([nc["text"] for nc in n_chunks], convert_to_numpy=True, normalize_embeddings=True)
            q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
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

            # Take top N
            top_n_idx = np.argsort(scores)[-max_neighbors:][::-1]
            neighbor_chunks = [n_chunks[i] for i in top_n_idx]
            
            for nc in neighbor_chunks:
                if nc.get("identifier", "").upper() not in added_ids:
                    retrieved_chunks.append(nc)
                    added_ids.add(nc.get("identifier", "").upper())
            
            mode_str = "Deep Search (2-hop)" if is_deep else "1-hop"
            print(f"[{mode_str}: added top {len(neighbor_chunks)} graph neighbors to context]\n")

    hist = _history_str(history)
    
    context_parts = []
    for c in retrieved_chunks:
        ctype = c.get("type", "chunk").replace("_", " ").upper()
        if not c.get("identifier"):
            context_parts.append(f"[{ctype} — {c['name']}]\n{c['text']}")
        else:
            context_parts.append(f"[{ctype} {c['identifier']} — {c['name']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)
    
    is_cve_query = bool(re.search(r'\bCVE-\d{4}-\d+\b', question, re.IGNORECASE))
    cve_rule = (
        "9. For CVE root-cause (CWE) questions: if the context lists an explicit CWE ID, state it. "
        "If the weakness shows 'n/a' or no CWE ID, you MUST analyze the vulnerability description "
        "and determine the most likely CWE (e.g. CWE-79 for XSS, CWE-89 for SQL injection, "
        "CWE-416 for use-after-free, CWE-787 for out-of-bounds write, CWE-77/78 for command injection). "
        "Always end your answer with the CWE ID on its own line. Never say 'I don't have enough information' "
        "for CVE weakness questions."
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

Conversation history (for context only):
{hist}

Context:
{context}

QUESTION: {question}

ANSWER:"""
    answer = _llm(prompt)
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
