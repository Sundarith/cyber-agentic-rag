"""
Convert MITRE ATT&CK Enterprise STIX bundle into retrievable chunks.
Chunk types produced:
  - technique      : one per active ATT&CK technique/sub-technique
  - group/malware/tool : one per active actor entity
  - mitigation     : one per course-of-action (standalone, searchable by name/ID)
  - tactic         : one per ATT&CK tactic (14 total)
  - data_source    : one per MITRE data source (detection coverage)
"""
try:
    import orjson as json_lib
    def _load(path): return json_lib.loads(path.read_bytes())
except ImportError:
    import json as json_lib
    def _load(path): return json_lib.loads(path.read_text())
import json  # still needed for JSONL output (orjson returns bytes not str)
import re
from collections import defaultdict
from pathlib import Path

RAW = Path("data/raw/enterprise-attack.json")
OUT = Path("data/processed/attack_chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\(Citation:[^)]+\)", "", text)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_attack_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def get_attack_url(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url", "")
    return ""


def is_active(obj):
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def get_tactics(obj):
    phases = obj.get("kill_chain_phases", [])
    return [
        p["phase_name"].replace("-", " ").title()
        for p in phases
        if p.get("kill_chain_name") == "mitre-attack"
    ]


def main():
    bundle = _load(RAW)
    objs = bundle["objects"]

    capec_to_cwe = {}
    cwe_rel_path = Path("data/processed/capec_cwe_relations.json")
    if cwe_rel_path.exists():
        rel_data = json.loads(cwe_rel_path.read_text())
        capec_to_cwe = rel_data.get("capec_to_cwe", {})

    # ── Collect objects by type ──────────────────────────────────────────────
    techniques   = {}
    mitigations  = {}
    actors       = {}   # intrusion-set + malware + tool
    tactics_objs = {}   # x-mitre-tactic
    data_comps   = {}   # x-mitre-data-component
    det_strats   = {}   # x-mitre-detection-strategy
    analytics    = {}   # x-mitre-analytic

    for o in objs:
        t = o["type"]
        if t == "attack-pattern" and is_active(o):
            techniques[o["id"]] = o
        elif t == "course-of-action" and is_active(o):
            mitigations[o["id"]] = o
        elif t in ("intrusion-set", "malware", "tool", "campaign") and is_active(o):
            actors[o["id"]] = o
        elif t == "x-mitre-tactic":
            tactics_objs[o["id"]] = o
        elif t == "x-mitre-data-component":
            data_comps[o["id"]] = o
        elif t == "x-mitre-detection-strategy" and is_active(o):
            det_strats[o["id"]] = o
        elif t == "x-mitre-analytic" and is_active(o):
            analytics[o["id"]] = o

    # ── Parse relationships ──────────────────────────────────────────────────
    mits_for_tech  = defaultdict(list)   # tech_uuid  → [(mit_uuid, rel_desc)]
    tech_for_mit   = defaultdict(list)   # mit_uuid   → [tech_uuid]
    parent_of_sub  = {}                  # sub_uuid   → parent_uuid
    subs_of_parent = defaultdict(list)   # parent_uuid→ [sub_uuid]
    users_of_tech  = {}                  # tech_uuid  → {groups, malware, tools, campaigns}
    group_uses     = {}                  # group_name → {malware, tools, campaigns}
    detects_tech   = defaultdict(list)   # tech_uuid  → [comp_name]
    det_strat_to_tech = defaultdict(list) # strat_uuid → [tech_uuid]

    for o in objs:
        if o["type"] != "relationship" or o.get("x_mitre_deprecated"):
            continue
        rt  = o.get("relationship_type")
        src = o.get("source_ref", "")
        tgt = o.get("target_ref", "")

        if rt == "mitigates" and tgt in techniques and src in mitigations:
            rel_desc = clean_text(o.get("description", ""))
            mits_for_tech[tgt].append((src, rel_desc))
            tech_for_mit[src].append(tgt)

        elif rt == "subtechnique-of" and src in techniques and tgt in techniques:
            parent_of_sub[src] = tgt
            subs_of_parent[tgt].append(src)

        elif rt == "uses" and src in actors:
            actor = actors[src]
            atype = actor["type"]
            name  = actor.get("name", "")
            if tgt in techniques:
                entry = users_of_tech.setdefault(tgt, {"groups": [], "malware": [], "tools": [], "campaigns": []})
                if atype == "intrusion-set":
                    entry["groups"].append(name)
                elif atype == "malware":
                    entry["malware"].append(name)
                elif atype == "tool":
                    entry["tools"].append(name)
                elif atype == "campaign":
                    entry["campaigns"].append(name)
            elif tgt in actors and atype == "intrusion-set":
                target = actors[tgt]
                ge = group_uses.setdefault(name, {"malware": [], "tools": [], "campaigns": []})
                if target["type"] == "malware":
                    ge["malware"].append(target.get("name", ""))
                elif target["type"] == "tool":
                    ge["tools"].append(target.get("name", ""))
                elif target["type"] == "campaign":
                    ge["campaigns"].append(target.get("name", ""))

        elif rt == "detects" and src in det_strats and tgt in techniques:
            det_strat_to_tech[src].append(tgt)
            strat = det_strats[src]
            for analytic_ref in strat.get("x_mitre_analytic_refs", []):
                if analytic_ref in analytics:
                    analytic = analytics[analytic_ref]
                    for ref in analytic.get("x_mitre_log_source_references", []):
                        c_uuid = ref.get("x_mitre_data_component_ref")
                        if c_uuid in data_comps:
                            detects_tech[tgt].append(data_comps[c_uuid].get("name", ""))

    # ── Map ATT&CK to CAPEC ──────────────────────────────────────────────────
    tech_to_capec = defaultdict(list)
    for uuid, t in techniques.items():
        for ref in t.get("external_references", []):
            if ref.get("source_name") == "capec":
                tech_to_capec[uuid].append(ref.get("external_id"))
    print(f"  Found {len(tech_to_capec)} techniques with CAPEC references")

    # ── 1. Technique chunks ──────────────────────────────────────────────────
    chunks = []

    for uuid, t in techniques.items():
        tid = get_attack_id(t)
        if not tid:
            continue

        parts = [f"# {tid} — {t.get('name', '')}"]

        tactics = get_tactics(t)
        if tactics:
            parts.append(f"*Tactic: {', '.join(tactics)}*")

        parent_uuid = parent_of_sub.get(uuid)
        if parent_uuid:
            parent = techniques[parent_uuid]
            parts.append(f"*Sub-technique of {get_attack_id(parent)} — {parent.get('name', '')}*")

        desc = clean_text(t.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")

        platforms = t.get("x_mitre_platforms", [])
        if platforms:
            parts.append(f"\n**Platforms:** {', '.join(platforms)}")

        sub_uuids = subs_of_parent.get(uuid, [])
        if sub_uuids:
            sub_lines = [
                f"- {get_attack_id(techniques[s])} — {techniques[s].get('name', '')}"
                for s in sorted(sub_uuids, key=lambda x: get_attack_id(techniques[x]))
            ]
            parts.append("\n## Sub-techniques\n" + "\n".join(sub_lines))

        # observed in the wild (aggregate parent + sub-techniques)
        def _collect(u):
            w = users_of_tech.get(u, {})
            g = set(w.get("groups", []))
            m = set(w.get("malware", []))
            tl = set(w.get("tools", []))
            c = set(w.get("campaigns", []))
            for s in subs_of_parent.get(u, []):
                ws = users_of_tech.get(s, {})
                g  |= set(ws.get("groups", []))
                m  |= set(ws.get("malware", []))
                tl |= set(ws.get("tools", []))
                c  |= set(ws.get("campaigns", []))
            return g, m, tl, c

        groups, malware, tools, campaigns = _collect(uuid)
        wild_lines = []
        if groups:
            wild_lines.append("**Groups:** " + ", ".join(sorted(groups)))
        if campaigns:
            wild_lines.append("**Campaigns:** " + ", ".join(sorted(campaigns)))
        if malware:
            wild_lines.append("**Malware:** " + ", ".join(sorted(malware)))
        if tools:
            wild_lines.append("**Tools:** " + ", ".join(sorted(tools)))
        if wild_lines:
            parts.append("\n## Observed in the Wild\n" + "\n".join(wild_lines))

        # detection (data components that detect this technique)
        det_comps = sorted(set(detects_tech.get(uuid, [])))
        if det_comps:
            parts.append("\n## Detection\n**Data Sources:** " + ", ".join(det_comps))

        # mitigations
        mit_entries = mits_for_tech.get(uuid, [])
        if mit_entries:
            parts.append("\n## Mitigations")
            for mit_uuid, rel_desc in mit_entries:
                mit = mitigations[mit_uuid]
                mit_id   = get_attack_id(mit)
                mit_desc = clean_text(mit.get("description", ""))
                line = f"\n- **{mit.get('name', '')} ({mit_id})**: {mit_desc}"
                if rel_desc:
                    line += f"\n  *Specific to this technique: {rel_desc}*"
                parts.append(line)
        else:
            parts.append("\n## Mitigations\n\n*No specific mitigations listed in MITRE ATT&CK.*")

        # related CAPEC
        related_capecs = tech_to_capec.get(uuid, [])
        if related_capecs:
            parts.append("\n## Related CAPEC Attack Patterns")
            for cid in related_capecs:
                parts.append(f"- {cid}")
                
            related_cwes = set()
            for cid in related_capecs:
                if cid in capec_to_cwe:
                    related_cwes.update(capec_to_cwe[cid])
            if related_cwes:
                parts.append("\n## Underlying Root Causes (CWE)")
                for cwe in sorted(related_cwes):
                    parts.append(f"- {cwe}")

        chunk_text = "\n".join(parts).strip()
        chunks.append({
            "id":              f"attack-{tid}",
            "text":            chunk_text,
            "source":          "MITRE ATT&CK",
            "type":            "technique",
            "identifier":      tid,
            "name":            t.get("name", ""),
            "tactics":         tactics,
            "url":             get_attack_url(t),
            "is_subtechnique": bool(parent_uuid),
            "n_mitigations":   len(mit_entries),
            "n_groups":        len(groups),
            "n_malware":       len(malware),
            "n_tools":         len(tools),
        })

    # ── 2. Entity chunks (groups, malware, tools) ────────────────────────────
    # Build O(1) name lookup: (type, name) → uuid
    actor_name_idx: dict[tuple, str] = {}
    for a_uuid, a in actors.items():
        actor_name_idx[(a["type"], a.get("name", ""))] = a_uuid

    # Invert users_of_tech: actor_uuid → [(tid, tname, is_sub)]
    entity_techs: dict[str, list] = defaultdict(list)
    for tech_uuid, usage in users_of_tech.items():
        if tech_uuid not in techniques:
            continue
        t   = techniques[tech_uuid]
        tid = get_attack_id(t)
        if not tid:
            continue
        is_sub = tech_uuid in parent_of_sub
        entry  = (tid, t.get("name", ""), is_sub)
        for name in usage.get("groups", []):
            a_uuid = actor_name_idx.get(("intrusion-set", name))
            if a_uuid:
                entity_techs[a_uuid].append(entry)
        for name in usage.get("malware", []):
            a_uuid = actor_name_idx.get(("malware", name))
            if a_uuid:
                entity_techs[a_uuid].append(entry)
        for name in usage.get("tools", []):
            a_uuid = actor_name_idx.get(("tool", name))
            if a_uuid:
                entity_techs[a_uuid].append(entry)
        for name in usage.get("campaigns", []):
            a_uuid = actor_name_idx.get(("campaign", name))
            if a_uuid:
                entity_techs[a_uuid].append(entry)

    malware_used_by: dict[str, list] = defaultdict(list)
    tool_used_by:    dict[str, list] = defaultdict(list)
    for g, d in group_uses.items():
        for m in d.get("malware", []):
            malware_used_by[m].append(g)
        for tl in d.get("tools", []):
            tool_used_by[tl].append(g)

    entity_chunks = []
    type_label = {"intrusion-set": "group", "malware": "malware", "tool": "tool", "campaign": "campaign"}

    for a_uuid, actor in actors.items():
        atype = actor["type"]
        label = type_label.get(atype)
        if not label:
            continue
        name = actor.get("name", "")
        aid  = get_attack_id(actor)
        if not aid or not name:
            continue

        parts = [f"# {name} ({aid}) — {label}"]

        other_aliases = [a for a in actor.get("aliases", []) if a != name]
        if other_aliases:
            parts.append(f"**Also known as:** {', '.join(other_aliases)}")

        desc = clean_text(actor.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")

        MAX_TECHS = 10
        techs_sorted = sorted(set(entity_techs.get(a_uuid, [])), key=lambda x: x[0])
        if techs_sorted:
            lines = []
            for tid, tname, is_sub in techs_sorted[:MAX_TECHS]:
                sub = " (sub-technique)" if is_sub else ""
                lines.append(f"- {tid} — {tname}{sub}")
            if len(techs_sorted) > MAX_TECHS:
                lines.append(f"\n*...and {len(techs_sorted) - MAX_TECHS} more. Ask for the full list by name or ID.*")
            parts.append(f"\n## Techniques Used ({len(techs_sorted)})\n" + "\n".join(lines))

        if atype == "intrusion-set":
            gu  = group_uses.get(name, {})
            mal = sorted(set(gu.get("malware", [])))
            tls = sorted(set(gu.get("tools", [])))
            assoc = []
            if mal:
                assoc.append("**Malware:** " + ", ".join(mal))
            if tls:
                assoc.append("**Tools:** " + ", ".join(tls))
            if assoc:
                parts.append("\n## Associated Software\n" + "\n".join(assoc))

        if atype == "malware":
            glist = sorted(set(malware_used_by.get(name, [])))
            if glist:
                parts.append("\n## Used By\n" + "\n".join(f"- {g}" for g in glist))
        elif atype == "tool":
            glist = sorted(set(tool_used_by.get(name, [])))
            if glist:
                parts.append("\n## Used By\n" + "\n".join(f"- {g}" for g in glist))
        elif atype == "campaign":
            # Campaigns may be attributed to groups — find from group_uses if possible
            attributed = sorted([g for g in group_uses if name in group_uses[g].get("campaigns", [])])
            if not attributed:
                # fallback: check if any intrusion-set references this campaign
                pass
            if attributed:
                parts.append("\n## Attributed To\n" + "\n".join(f"- {g}" for g in attributed))

        entity_chunks.append({
            "id":              f"entity-{aid}",
            "text":            "\n".join(parts).strip(),
            "source":          "MITRE ATT&CK",
            "type":            label,
            "identifier":      aid,
            "name":            name,
            "tactics":         [],
            "url":             get_attack_url(actor),
            "is_subtechnique": False,
            "n_mitigations":   0,
            "n_groups":        0,
            "n_malware":       0,
            "n_tools":         0,
        })

    # ── 3. Mitigation chunks (standalone) ────────────────────────────────────
    mitigation_chunks = []
    for mit_uuid, mit in mitigations.items():
        mid  = get_attack_id(mit)
        name = mit.get("name", "")
        if not mid or not name:
            continue

        parts = [f"# {name} ({mid}) — mitigation"]

        desc = clean_text(mit.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")

        covered = []
        for t_uuid in tech_for_mit.get(mit_uuid, []):
            if t_uuid in techniques:
                t   = techniques[t_uuid]
                tid = get_attack_id(t)
                if tid:
                    covered.append((tid, t.get("name", "")))
        covered.sort()
        if covered:
            lines = [f"- {tid} — {tname}" for tid, tname in covered]
            parts.append(f"\n## Mitigates Techniques ({len(covered)})\n" + "\n".join(lines))

        mitigation_chunks.append({
            "id":              f"mitigation-{mid}",
            "text":            "\n".join(parts).strip(),
            "source":          "MITRE ATT&CK",
            "type":            "mitigation",
            "identifier":      mid,
            "name":            name,
            "tactics":         [],
            "url":             get_attack_url(mit),
            "is_subtechnique": False,
            "n_mitigations":   0,
            "n_groups":        0,
            "n_malware":       0,
            "n_tools":         0,
        })

    # ── 4. Tactic chunks ─────────────────────────────────────────────────────
    tactic_chunks = []
    tactic_techs: dict[str, list] = defaultdict(list)
    for uuid, t in techniques.items():
        tid = get_attack_id(t)
        if not tid:
            continue
        for tac in get_tactics(t):
            tactic_techs[tac].append((tid, t.get("name", ""), uuid in parent_of_sub))

    for tac_obj in tactics_objs.values():
        tac_name   = tac_obj.get("name", "")
        short_name = tac_obj.get("x_mitre_shortname", "").replace("-", " ").title()
        tac_id     = get_attack_id(tac_obj)
        key        = short_name if short_name in tactic_techs else tac_name

        parts = [f"# {tac_name} Tactic ({tac_id})"]

        desc = clean_text(tac_obj.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")

        all_tac_techs = sorted(tactic_techs.get(key, []), key=lambda x: x[0])
        parents = [(tid, tname) for tid, tname, is_sub in all_tac_techs if not is_sub]
        subs    = [(tid, tname) for tid, tname, is_sub in all_tac_techs if is_sub]
        if all_tac_techs:
            lines = [f"- {tid} — {tname}" for tid, tname in parents]
            if subs:
                lines.append(f"\n*Sub-techniques ({len(subs)}):*")
                for tid, tname in subs[:25]:
                    lines.append(f"  - {tid} — {tname}")
                if len(subs) > 25:
                    lines.append(f"  *...and {len(subs)-25} more*")
            parts.append(f"\n## Techniques ({len(all_tac_techs)} total)\n" + "\n".join(lines))

        slug = tac_id if tac_id else tac_name.lower().replace(" ", "-")
        tactic_chunks.append({
            "id":              f"tactic-{slug}",
            "text":            "\n".join(parts).strip(),
            "source":          "MITRE ATT&CK",
            "type":            "tactic",
            "identifier":      tac_id or tac_name,
            "name":            tac_name,
            "tactics":         [],
            "url":             get_attack_url(tac_obj),
            "is_subtechnique": False,
            "n_mitigations":   0,
            "n_groups":        0,
            "n_malware":       0,
            "n_tools":         0,
        })

    # ── 5. Detection Strategy chunks ─────────────────────────────────────────
    detect_chunks = []

    for strat_uuid, strat in det_strats.items():
        sid = get_attack_id(strat)
        name = strat.get("name", "")
        if not name:
            continue

        parts = [f"# {name} ({sid}) — Detection Strategy"]

        desc = clean_text(strat.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")

        ana_refs = strat.get("x_mitre_analytic_refs", [])
        for ana_ref in ana_refs:
            ana = analytics.get(ana_ref)
            if not ana: continue
            ana_name = ana.get("name", "")
            ana_desc = clean_text(ana.get("description", ""))
            if ana_name:
                parts.append(f"\n### Analytic: {ana_name}")
            if ana_desc:
                parts.append(f"{ana_desc}")

            comps = set()
            for ls in ana.get("x_mitre_log_source_references", []):
                cref = ls.get("x_mitre_data_component_ref")
                if cref and cref in data_comps:
                    comps.add(data_comps[cref].get("name", ""))
            if comps:
                parts.append(f"**Data Components:** {', '.join(sorted(comps))}")

        detected_techs = sorted(set(det_strat_to_tech.get(strat_uuid, [])))
        if detected_techs:
            lines = []
            for tu in detected_techs:
                t = techniques[tu]
                tid = get_attack_id(t)
                lines.append(f"- {tid} — {t.get('name', '')}")
            parts.append("\n## Detects Techniques\n" + "\n".join(lines))

        slug = sid if sid else name.lower().replace(" ", "-")
        detect_chunks.append({
            "id":              f"detection-{slug}",
            "text":            "\n".join(parts).strip(),
            "source":          "MITRE ATT&CK",
            "type":            "detection",
            "identifier":      sid or name,
            "name":            name,
            "tactics":         [],
            "url":             get_attack_url(strat),
            "is_subtechnique": False,
            "n_mitigations":   0,
            "n_groups":        0,
            "n_malware":       0,
            "n_tools":         0,
        })

    # ── Write output ─────────────────────────────────────────────────────────
    all_chunks = chunks + entity_chunks + mitigation_chunks + tactic_chunks + detect_chunks

    with OUT.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    relations_path = OUT.parent / "entity_relations.json"
    deduped = {
        g: {"malware": sorted(set(d["malware"])), "tools": sorted(set(d["tools"])), "campaigns": sorted(set(d["campaigns"]))}
        for g, d in group_uses.items()
    }
    relations_path.write_text(json.dumps({"group_uses": deduped}, indent=2))

    # Save CAPEC relations map mapping ATT&CK technique IDs -> list of CAPEC IDs
    capec_rel_path = OUT.parent / "capec_attack_relations.json"
    tid_to_capec = {}
    for uuid, capec_ids in tech_to_capec.items():
        if uuid in techniques:
            tid = get_attack_id(techniques[uuid])
            if tid:
                tid_to_capec[tid] = capec_ids
    print(f"  Mapping {len(tid_to_capec)} TIDs to CAPECs for {capec_rel_path.name}")
    capec_rel_path.write_text(json.dumps({"tech_to_capec": tid_to_capec}, indent=2))

    print(f"Wrote {len(all_chunks)} chunks to {OUT}")
    print(f"  Technique chunks:      {len(chunks)}")
    print(f"  Entity chunks:         {len(entity_chunks)}")
    n_grp = sum(1 for c in entity_chunks if c["type"] == "group")
    n_mal = sum(1 for c in entity_chunks if c["type"] == "malware")
    n_tl  = sum(1 for c in entity_chunks if c["type"] == "tool")
    n_cmp = sum(1 for c in entity_chunks if c["type"] == "campaign")
    print(f"    Groups: {n_grp}, Campaigns: {n_cmp}, Malware: {n_mal}, Tools: {n_tl}")
    print(f"  Mitigation chunks:     {len(mitigation_chunks)}")
    print(f"  Tactic chunks:         {len(tactic_chunks)}")
    print(f"  Detection Strategy chunks: {len(detect_chunks)}")
    print(f"  Sub-techniques:        {sum(1 for c in chunks if c['is_subtechnique'])}")
    print(f"  With group usage:      {sum(1 for c in chunks if c['n_groups'] > 0)}")
    print(f"  With detection data:   {sum(1 for c in chunks if '## Detection' in c['text'])}")
    avg_tech = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    avg_ent  = sum(len(c["text"]) for c in entity_chunks) / len(entity_chunks) if entity_chunks else 0
    print(f"  Avg technique chunk:   {avg_tech:.0f} chars")
    print(f"  Avg entity chunk:      {avg_ent:.0f} chars")


if __name__ == "__main__":
    main()
