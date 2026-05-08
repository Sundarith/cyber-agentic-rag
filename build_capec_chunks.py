import json
import re
from collections import defaultdict
from pathlib import Path

RAW = Path("data/raw/stix-capec.json")
OUT = Path("data/processed/capec_chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove citation-like references if any
    text = re.sub(r"\(Citation:[^)]+\)", "", text)
    
    # Handle HTML/XHTML tags often found in MITRE STIX
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<xhtml:p>", "", text)
    text = re.sub(r"</xhtml:p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)  # Strip remaining tags
    
    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_capec_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "capec":
            return ref.get("external_id", "")
    return ""

def get_capec_url(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "capec":
            return ref.get("url", "")
    return ""

def main():
    print("Loading CAPEC bundle...")
    bundle = json.loads(RAW.read_text())
    objs = bundle.get("objects", [])

    attack_patterns = {}
    mitigations = {}
    
    mits_for_ap = defaultdict(list)
    ap_for_mit = defaultdict(list)

    print("Parsing objects...")
    for o in objs:
        t = o.get("type")
        if t == "attack-pattern":
            attack_patterns[o["id"]] = o
        elif t == "course-of-action":
            mitigations[o["id"]] = o
        elif t == "relationship":
            rt = o.get("relationship_type")
            src = o.get("source_ref", "")
            tgt = o.get("target_ref", "")
            if rt == "mitigates" and tgt in attack_patterns and src in mitigations:
                mits_for_ap[tgt].append(src)
                ap_for_mit[src].append(tgt)

    chunks = []
    
    capec_to_cwe = defaultdict(list)
    print("Building Attack Pattern chunks...")
    for uuid, ap in attack_patterns.items():
        cid = get_capec_id(ap)
        if not cid:
            continue
        if not ap.get("name", "").strip():
            continue  # skip attack-pattern objects with no displayable name

        parts = [f"# {cid} — {ap.get('name', '')}"]
        
        desc = clean_text(ap.get("description", ""))
        if desc:
            parts.append(f"\n{desc}")
            
        prereqs = ap.get("x_capec_prerequisites", [])
        if prereqs:
            parts.append("\n## Prerequisites\n" + "\n".join(f"- {clean_text(p)}" for p in prereqs))
            
        exec_flow = clean_text(ap.get("x_capec_execution_flow", ""))
        if exec_flow:
            parts.append(f"\n## Execution Flow\n{exec_flow}")
            
        consequences = ap.get("x_capec_consequences", {})
        if consequences:
            parts.append("\n## Consequences")
            for scope, impacts in consequences.items():
                parts.append(f"- **{scope}**: {', '.join(impacts)}")
                
        severity = ap.get("x_capec_typical_severity")
        if severity:
            parts.append(f"\n**Typical Severity:** {severity}")
            
        # mitigations
        mit_entries = mits_for_ap.get(uuid, [])
        if mit_entries:
            parts.append("\n## Mitigations")
            for mit_uuid in mit_entries:
                if mit_uuid in mitigations:
                    mit = mitigations[mit_uuid]
                    mit_desc = clean_text(mit.get("description", ""))
                    parts.append(f"\n- {mit_desc}")
        else:
            parts.append("\n## Mitigations\n\n*No specific mitigations listed.*")

        # extract CWEs
        cwes = []
        for ref in ap.get("external_references", []):
            if ref.get("source_name") == "cwe":
                cwes.append(ref.get("external_id"))
        
        if cwes:
            capec_to_cwe[cid].extend(cwes)
            parts.append("\n## Related Root Causes (CWE)")
            for cwe in sorted(set(cwes)):
                parts.append(f"- {cwe}")
            
        # related ATT&CK techniques
        related_attack_ids = []
        for ref in ap.get("external_references", []):
            if ref.get("source_name") == "ATTACK":
                related_attack_ids.append(ref.get("external_id"))
                
        if related_attack_ids:
            parts.append("\n## Related MITRE ATT&CK Techniques")
            for att_id in related_attack_ids:
                parts.append(f"- {att_id}")
            
        chunk_text = "\n".join(parts).strip()
        chunks.append({
            "id": f"capec-{cid}",
            "text": chunk_text,
            "source": "CAPEC",
            "type": "attack_pattern",
            "identifier": cid,
            "name": ap.get("name", ""),
            "url": get_capec_url(ap),
            "n_mitigations": len(mit_entries),
        })

    print("Building Mitigation chunks...")
    mitigation_chunks = []
    for mit_uuid, mit in mitigations.items():
        desc = clean_text(mit.get("description", ""))
        if not desc:
            continue
            
        name = desc.split('.')[0][:60]
        if not name:
            name = "Mitigation"
            
        parts = [f"# {name} — mitigation"]
        parts.append(f"\n{desc}")
        
        covered = []
        for ap_uuid in ap_for_mit.get(mit_uuid, []):
            if ap_uuid in attack_patterns:
                ap = attack_patterns[ap_uuid]
                cid = get_capec_id(ap)
                if cid:
                    covered.append((cid, ap.get("name", "")))
        covered.sort()
        if covered:
            lines = [f"- {cid} — {cname}" for cid, cname in covered]
            parts.append(f"\n## Mitigates Attack Patterns ({len(covered)})\n" + "\n".join(lines))
            
        mitigation_chunks.append({
            "id": f"mitigation-{mit_uuid}",
            "text": "\n".join(parts).strip(),
            "source": "CAPEC",
            "type": "mitigation",
            "identifier": "",
            "name": name,
            "url": "",
            "n_mitigations": 0,
        })
        
    all_chunks = chunks + mitigation_chunks
    
    print(f"Writing to {OUT}...")
    with OUT.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
            
    # Save CAPEC -> CWE relations
    capec_cwe_path = OUT.parent / "capec_cwe_relations.json"
    capec_cwe_path.write_text(json.dumps({"capec_to_cwe": capec_to_cwe}, indent=2))

    # Save Tech -> CAPEC relations (The Bridge)
    tech_to_capec = defaultdict(list)
    for uuid, ap in attack_patterns.items():
        cid = get_capec_id(ap)
        if not cid: continue
        for ref in ap.get("external_references", []):
            if ref.get("source_name") == "ATTACK":
                tid = ref.get("external_id")
                if tid:
                    tech_to_capec[tid].append(cid)
                    
    capec_attack_path = OUT.parent / "capec_attack_relations.json"
    capec_attack_path.write_text(json.dumps({"tech_to_capec": tech_to_capec}, indent=2))
            
    print(f"Done. Wrote {len(all_chunks)} chunks to {OUT}")
    print(f"  Attack Pattern chunks: {len(chunks)}")
    print(f"  Mitigation chunks:     {len(mitigation_chunks)}")
    print(f"  Relations: {len(capec_to_cwe)} CAPEC->CWE, {len(tech_to_capec)} Tech->CAPEC")

if __name__ == '__main__':
    main()
