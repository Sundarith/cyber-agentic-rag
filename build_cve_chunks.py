import json
from pathlib import Path
import re

RAW_DIR = Path("data/raw/cves/cves")
OUT = Path("data/processed/cve_chunks.jsonl")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def process_cve(file_path: Path):
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return None

    if data.get("dataType") != "CVE_RECORD":
        return None
        
    meta = data.get("cveMetadata", {})
    state = meta.get("state")
    if state != "PUBLISHED":
        return None
        
    cve_id = meta.get("cveId")
    if not cve_id:
        return None

    containers = data.get("containers", {})
    cna = containers.get("cna", {})
    
    # Description
    desc = ""
    for d in cna.get("descriptions", []):
        if d.get("lang") == "en":
            desc = clean_text(d.get("value", ""))
            break
            
    parts = [f"# {cve_id}"]
    
    # Metrics (CVSS)
    cvss = None
    severity = ""
    for m in cna.get("metrics", []):
        if "cvssV3_1" in m:
            cvss = m["cvssV3_1"].get("baseScore")
            severity = m["cvssV3_1"].get("baseSeverity", "")
            break
        elif "cvssV3_0" in m:
            cvss = m["cvssV3_0"].get("baseScore")
            severity = m["cvssV3_0"].get("baseSeverity", "")
            break
            
    if cvss is not None:
        parts.append(f"*CVSS v3 Base Score: {cvss} ({severity})*")
        
    if desc:
        parts.append(f"\n{desc}")
        
    # Affected Products
    affected_lines = []
    for a in cna.get("affected", []):
        vendor = a.get("vendor", "Unknown Vendor").strip()
        product = a.get("product", "Unknown Product").strip()
        
        versions = []
        for v in a.get("versions", []):
            ver = v.get("version", "")
            status = v.get("status", "")
            lt = v.get("lessThan", "")
            lte = v.get("lessThanOrEqual", "")
            
            v_str = ver
            if lte:
                v_str += f" <= {lte}"
            elif lt:
                v_str += f" < {lt}"
                
            if status:
                v_str += f" ({status})"
                
            if v_str:
                versions.append(v_str)
                
        line = f"- {vendor} {product}"
        if versions:
            line += f": {', '.join(versions)}"
        affected_lines.append(line)
        
    if affected_lines:
        parts.append("\n## Affected Products\n" + "\n".join(affected_lines))
        
    # Problem Types (CWE)
    cwes = []
    for pt in cna.get("problemTypes", []):
        for d in pt.get("descriptions", []):
            cwe_id = d.get("cweId")
            cwe_desc = d.get("description", "")
            if cwe_id:
                cwes.append(f"{cwe_id}: {cwe_desc}")
            elif cwe_desc:
                cwes.append(cwe_desc)
    
    if cwes:
        parts.append("\n## Weaknesses (CWE)\n" + "\n".join(f"- {c}" for c in cwes))
        
    # Impacts (CAPEC)
    capecs = []
    for imp in cna.get("impacts", []):
        capec_id = imp.get("capecId")
        if capec_id:
            for d in imp.get("descriptions", []):
                if d.get("lang") == "en":
                    capecs.append(f"{capec_id}: {d.get('value', '')}")
                    break
            else:
                capecs.append(capec_id)
                
    if capecs:
        parts.append("\n## Exploitable By (CAPEC)\n" + "\n".join(f"- {c}" for c in capecs))
        
    # Solutions
    solutions = []
    for s in cna.get("solutions", []):
        if s.get("lang") == "en":
            solutions.append(clean_text(s.get("value", "")))
            
    if solutions:
        parts.append("\n## Solutions\n" + "\n\n".join(solutions))

    chunk_text = "\n".join(parts).strip()
    
    return {
        "id": f"cve-{cve_id.lower()}",
        "text": chunk_text,
        "source": "CVE",
        "type": "vulnerability",
        "identifier": cve_id,
        "name": cve_id,
        "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
    }

def main():
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR} does not exist.")
        return
        
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    print("Finding JSON files...")
    files = list(RAW_DIR.rglob("*.json"))
    files = [f for f in files if "delta" not in f.name]
    print(f"Found {len(files)} CVE files.")
    
    chunks = []
    processed = 0
    
    for f in files:
        chunk = process_cve(f)
        if chunk:
            chunks.append(chunk)
        processed += 1
        if processed % 10000 == 0:
            print(f"Processed {processed} files...")
            
    print(f"Writing {len(chunks)} chunks to {OUT}...")
    with OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
