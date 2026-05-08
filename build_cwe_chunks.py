import json
import re
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
ZIP_PATH = Path("data/raw/cwec_latest.xml.zip")
XML_PATH = Path("data/raw/cwec_latest.xml")
OUT = Path("data/processed/cwe_chunks.jsonl")

def download_and_extract():
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not XML_PATH.exists():
        print(f"Downloading {URL}...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Assume there is only one XML file in the zip
            xml_filename = [n for n in zip_ref.namelist() if n.endswith('.xml')][0]
            zip_ref.extract(xml_filename, ZIP_PATH.parent)
            Path(ZIP_PATH.parent / xml_filename).rename(XML_PATH)

def clean_text(text):
    if not text:
        return ""
    # strip namespaces from tags if they were stringified
    text = re.sub(r'\{http[^\}]+\}', '', text)
    # clean extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_from_node(node):
    if node is None:
        return ""
    # recursively get text
    texts = []
    if node.text:
        texts.append(node.text)
    for child in node:
        texts.append(get_text_from_node(child))
        if child.tail:
            texts.append(child.tail)
    return " ".join(texts).strip()

def main():
    download_and_extract()
    print("Parsing XML...")
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    
    # CWE XML namespaces
    ns = {'cwe': 'http://cwe.mitre.org/cwe-6'}
    # Try finding Weaknesses directly without namespace if needed, or with.
    # The actual namespace can vary by version, e.g. cwe-6, cwe-7.
    # A safer way is to just ignore namespaces using local-name() or by regex.
    # For ElementTree, we can iterate over all elements and check their tag end.
    
    weaknesses = []
    for elem in root.iter():
        if elem.tag.endswith('Weakness') or elem.tag.endswith('Category') or elem.tag.endswith('View'):
            weaknesses.append(elem)

    print(f"Found {len(weaknesses)} CWE elements.")
    chunks = []
    
    for w in weaknesses:
        w_id = w.attrib.get("ID")
        w_name = w.attrib.get("Name")
        if not w_id or not w_name:
            continue
            
        cwe_id = f"CWE-{w_id}"
        tag_type = w.tag.split('}')[-1] # Weakness, Category, View
        
        parts = [f"# {cwe_id} — {w_name}"]
        parts.append(f"*(CWE {tag_type})*")
        
        # Description
        desc_node = None
        for child in w:
            if child.tag.endswith("Description"):
                desc_node = child
                break
        
        desc = get_text_from_node(desc_node)
        if desc:
            parts.append(f"\n{clean_text(desc)}")
            
        # Extended Description
        ext_desc_node = None
        for child in w:
            if child.tag.endswith("Extended_Description"):
                ext_desc_node = child
                break
        ext_desc = get_text_from_node(ext_desc_node)
        if ext_desc:
            parts.append(f"\n{clean_text(ext_desc)}")
            
        # Mitigations / Potential Mitigations
        mitigations = []
        for child in w:
            if child.tag.endswith("Potential_Mitigations"):
                for mit in child:
                    if mit.tag.endswith("Mitigation"):
                        mit_desc_node = None
                        for md in mit:
                            if md.tag.endswith("Description"):
                                mit_desc_node = md
                        md_text = get_text_from_node(mit_desc_node)
                        if md_text:
                            mitigations.append(clean_text(md_text))
                            
        if mitigations:
            parts.append("\n## Mitigations")
            for m in mitigations:
                parts.append(f"- {m}")
        else:
            parts.append("\n## Mitigations\n\n*No specific mitigations listed.*")
            
        # Applicable Platforms
        platforms = []
        for child in w:
            if child.tag.endswith("Applicable_Platforms"):
                for p in child:
                    name = p.attrib.get("Name")
                    if name:
                        platforms.append(name)
        if platforms:
            parts.append("\n## Applicable Platforms\n" + ", ".join(platforms))
            
        # Modes of Introduction
        modes = []
        for child in w:
            if child.tag.endswith("Modes_Of_Introduction"):
                for intro in child:
                    phase = None
                    note = None
                    for info in intro:
                        if info.tag.endswith("Phase"):
                            phase = get_text_from_node(info)
                        if info.tag.endswith("Note"):
                            note = get_text_from_node(info)
                    if phase:
                        txt = phase
                        if note:
                            txt += f": {note}"
                        modes.append(clean_text(txt))
        if modes:
            parts.append("\n## Modes of Introduction")
            for m in modes:
                parts.append(f"- {m}")
        
        chunk_text = "\n".join(parts).strip()
        chunks.append({
            "id": f"cwe-{w_id}",
            "text": chunk_text,
            "source": "CWE",
            "type": tag_type.lower(),
            "identifier": cwe_id,
            "name": w_name,
            "url": f"https://cwe.mitre.org/data/definitions/{w_id}.html"
        })
        
    print(f"Writing to {OUT}...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Done. Wrote {len(chunks)} chunks.")

if __name__ == '__main__':
    main()
