#!/usr/bin/env python3
import os

body_path = '/home/sheng/cyber-ft/docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/body.tex'
main_path = '/home/sheng/cyber-ft/docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/main.tex'

print("Starting patching script...")

# 1. Patch body.tex
if os.path.exists(body_path):
    with open(body_path, 'r') as f:
        body_content = f.read()

    # Align DeepSeek clean prompt total score in Table II
    target_table_ds = r"  & Context-only clean prompt & 775/903 (85.8\%) & 46/97 (47.4\%) & 820/1000 (82.0\%) \\"
    replace_table_ds = r"  & Context-only clean prompt & 775/903 (85.8\%) & 46/97 (47.4\%) & 821/1000 (82.1\%) \\"
    
    # Align cluster range description
    target_range = r"RAG rows cluster in a much narrower band (82.0\% to 90.9\%)."
    replace_range = r"RAG rows cluster in a much narrower band (82.1\% to 90.9\%)."

    # Align DeepSeek discussion section score and gain
    target_ds_disc = r"reaches 82.0\%, a +56.3 percentage-point lift."
    replace_ds_disc = r"reaches 82.1\%, a +56.4 percentage-point lift."

    # Restructure Table III (Mapped failures)
    target_table_iii = r"""\begin{tabularx}{0.98\textwidth}{@{}Ycccc@{}}
\toprule
\textbf{Failure category} &
\textbf{DeepSeek (128 fails)} &
\textbf{Granite (105 fails)} &
\textbf{Phi-4-mini (66 fails)} &
\textbf{Gemma 4 (84 fails)} \\
\midrule
NVD disagrees with CTI-Bench, model follows NVD"""

    replace_table_iii = r"""\begin{tabularx}{0.98\textwidth}{@{}Ycccc@{}}
\toprule
\textbf{Failure category} &
\textbf{DeepSeek-R1-8B} &
\textbf{IBM Granite-8B} &
\textbf{Phi-4-mini} &
\textbf{Gemma 4 E4B} \\
\midrule
\textbf{Total Mapped Failures} & \textbf{128} & \textbf{105} & \textbf{66} & \textbf{84} \\
\midrule
NVD disagrees with CTI-Bench, model follows NVD"""

    # Restructure Table IV (Unmapped failures)
    target_table_iv = r"""\begin{tabularx}{0.98\textwidth}{@{}Ycccc@{}}
\toprule
\textbf{Failure category} &
\textbf{DeepSeek (51 fails)} &
\textbf{Granite (39 fails)} &
\textbf{Phi-4-mini (25 fails)} &
\textbf{Gemma 4 (13 fails)} \\
\midrule
No CWE-ID parsed from answer"""

    replace_table_iv = r"""\begin{tabularx}{0.98\textwidth}{@{}Ycccc@{}}
\toprule
\textbf{Failure category} &
\textbf{DeepSeek-R1-8B} &
\textbf{IBM Granite-8B} &
\textbf{Phi-4-mini} &
\textbf{Gemma 4 E4B} \\
\midrule
\textbf{Total Unmapped Failures} & \textbf{51} & \textbf{39} & \textbf{25} & \textbf{13} \\
\midrule
No CWE-ID parsed from answer"""

    patched_body = body_content
    
    if target_table_ds in patched_body:
        patched_body = patched_body.replace(target_table_ds, replace_table_ds)
        print("Patched DeepSeek score in Table II.")
    else:
        print("Warning: DeepSeek score in Table II not found!")

    if target_range in patched_body:
        patched_body = patched_body.replace(target_range, replace_range)
        print("Patched RAG cluster range in body.")
    else:
        print("Warning: RAG cluster range in body not found!")

    if target_ds_disc in patched_body:
        patched_body = patched_body.replace(target_ds_disc, replace_ds_disc)
        print("Patched DeepSeek discussion section score/gain.")
    else:
        print("Warning: DeepSeek discussion section score/gain not found!")

    if target_table_iii in patched_body:
        patched_body = patched_body.replace(target_table_iii, replace_table_iii)
        print("Patched Table III structure.")
    else:
        print("Warning: Table III target structure not found!")

    if target_table_iv in patched_body:
        patched_body = patched_body.replace(target_table_iv, replace_table_iv)
        print("Patched Table IV structure.")
    else:
        print("Warning: Table IV target structure not found!")

    if patched_body != body_content:
        with open(body_path, 'w') as f:
            f.write(patched_body)
        print("body.tex written successfully.")
    else:
        print("No changes made to body.tex.")
else:
    print(f"Error: {body_path} not found!")

# 2. Patch main.tex
if os.path.exists(main_path):
    with open(main_path, 'r') as f:
        main_content = f.read()

    target_abstract = r"they all reach 82.0\% to 90.9\%,"
    replace_abstract = r"they all reach 82.1\% to 90.9\%,"

    if target_abstract in main_content:
        patched_main = main_content.replace(target_abstract, replace_abstract)
        with open(main_path, 'w') as f:
            f.write(patched_main)
        print("main.tex written successfully.")
    else:
        print("Warning: Target abstract range not found in main.tex!")
else:
    print(f"Error: {main_path} not found!")

print("Patching completed.")
