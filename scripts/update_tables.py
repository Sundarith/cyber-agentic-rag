import re
import sys
import difflib

filepath = "/home/sheng/cyber-ft/docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/body.tex"

with open(filepath, "r", encoding="utf-8") as f:
    original = f.read()

content = original

# Define replacements
replacements = [
    # Table III (also remove the tabcolsep command) - MUST RUN BEFORE TABLE II
    (r'\\setlength\{\\tabcolsep\}\{4pt\}\s*\\label\{tab:final-results\}\s*\\begin\{tabularx\}\{\\columnwidth\}\{Xccc\}', 
     r'\\label{tab:final-results}\n\\begin{tabular}{lccc}'),
    # Table I
    (r'\\begin\{tabularx\}\{\\columnwidth\}\{Xcccc\}', r'\\begin{tabular}{lcccc}'),
    # Table II
    (r'\\begin\{tabularx\}\{\\columnwidth\}\{Xccc\}', r'\\begin{tabular}{lccc}'),
    # Table IV
    (r'\\begin\{tabularx\}\{\\columnwidth\}\{Xccp\{0.18\\columnwidth\}\}', r'\\begin{tabular}{lccl}'),
    # Table V
    (r'\\begin\{tabularx\}\{\\columnwidth\}\{Xcc\}', r'\\begin{tabular}{lcc}'),
    # Table VI
    (r'\\begin\{tabularx\}\{\\columnwidth\}\{Xrrr\}', r'\\begin{tabular}{lrrr}'),
    # Closing tags
    (r'\\end\{tabularx\}', r'\\end{tabular}')
]

for pattern, repl in replacements:
    content, count = re.subn(pattern, repl, content)
    print(f"Pattern {pattern} replaced {count} times.")

diff = list(difflib.unified_diff(
    original.splitlines(keepends=True),
    content.splitlines(keepends=True),
    fromfile='body.tex (original)',
    tofile='body.tex (modified)'
))

if diff:
    print("Diff of changes:")
    sys.stdout.writelines(diff)
else:
    print("No changes made!")

if len(sys.argv) > 1 and sys.argv[1] == "--write":
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Changes written to file successfully.")
else:
    print("Dry run complete. Run with --write to apply changes.")
