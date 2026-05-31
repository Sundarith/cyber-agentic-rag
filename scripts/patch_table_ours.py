#!/usr/bin/env python3
import os

body_path = '/home/sheng/cyber-ft/docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/body.tex'

print("Starting ours label patch in Table I...")

if os.path.exists(body_path):
    with open(body_path, 'r') as f:
        content = f.read()

    target = "Ours & 3.8B & open & \\textbf{90.9\\%} & this work \\\\"
    replacement = "Ours (Phi-4-mini-reasoning + RAG) & 3.8B & open & \\textbf{90.9\\%} & this work \\\\"

    if target in content:
        patched_content = content.replace(target, replacement)
        with open(body_path, 'w') as f:
            f.write(patched_content)
        print("Table I successfully patched!")
    else:
        print("Error: Target Ours row not found in body.tex!")
else:
    print(f"Error: {body_path} not found!")
