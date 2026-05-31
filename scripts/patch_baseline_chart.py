#!/usr/bin/env python3
import os
import subprocess

html_path = '/home/sheng/cyber-ft/docs/figures/sources/baseline_chart.html'
out_png_1 = '/home/sheng/cyber-ft/docs/figures/baseline_chart.png'
out_png_2 = '/home/sheng/cyber-ft/docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/figures/baseline_chart.png'

print("Starting baseline chart patch...")

if os.path.exists(html_path):
    with open(html_path, 'r') as f:
        content = f.read()

    # Targets for replacements
    target_comment = "<!-- 1. Ours: 90.6%, width=(90.6-40)*15.2 = 769.12 -->"
    replace_comment = "<!-- 1. Ours: 90.9%, width=(90.9-40)*15.2 = 773.68 -->"

    target_rect = '<rect x="405" y="112" width="769" height="42" fill="#047857" stroke="#ffd24d" stroke-width="3.2"></rect>'
    replace_rect = '<rect x="405" y="112" width="774" height="42" fill="#047857" stroke="#ffd24d" stroke-width="3.2"></rect>'

    target_sub = '<text class="method-sub" x="390" y="149" text-anchor="end" style="font-weight:750;">DeepSeek-R1-Distill-Llama-8B · CVE/CWE RAG</text>'
    replace_sub = '<text class="method-sub" x="390" y="149" text-anchor="end" style="font-weight:750;">Phi-4-mini-reasoning (3.8B) · CVE/CWE RAG</text>'

    target_val = '<text class="bar-value-ours" x="1184" y="140">90.6%</text>'
    replace_val = '<text class="bar-value-ours" x="1184" y="140">90.9%</text>'

    patched_content = content
    
    if target_comment in patched_content:
        patched_content = patched_content.replace(target_comment, replace_comment)
        print("Patched comment.")
    if target_rect in patched_content:
        patched_content = patched_content.replace(target_rect, replace_rect)
        print("Patched bar rect width.")
    if target_sub in patched_content:
        patched_content = patched_content.replace(target_sub, replace_sub)
        print("Patched model sub-label.")
    if target_val in patched_content:
        patched_content = patched_content.replace(target_val, replace_val)
        print("Patched bar value text.")

    if patched_content != content:
        with open(html_path, 'w') as f:
            f.write(patched_content)
        print("baseline_chart.html written successfully.")

        # Run headless Chrome screenshot command
        cmd = [
            "google-chrome",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--screenshot={out_png_1}",
            "--window-size=3840,2520",
            "file://" + html_path
        ]
        print(f"Running screenshot render: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Generated {out_png_1} successfully!")

        # Copy to the LaTeX document figures folder
        subprocess.run(["cp", out_png_1, out_png_2], check=True)
        print(f"Copied to {out_png_2}")
    else:
        print("No changes made to baseline_chart.html.")
else:
    print(f"Error: {html_path} not found!")

print("Baseline chart patch completed.")
