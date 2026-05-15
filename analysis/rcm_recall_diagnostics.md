# CTI-RCM Retrieval Recall Diagnostics

Rows analyzed: **97**

## Recall By Stage

- Initial retrieval contains GT CWE: 47/97 (48.5%)
- CWE-only top 50 contains GT CWE: 77/97 (79.4%)
- k-NN votes contain GT CWE: 61/97 (62.9%)
- k-NN top vote is GT CWE: 43/97 (44.3%)
- Graph neighbors contain GT CWE: 8/97 (8.2%)
- Final context contains GT CWE: 72/97 (74.2%)
- CWE rescue selected GT CWE: 3/97 (3.1%)
- CWE rescue added candidates: 12

## Final Context Split

- GT present in final context: 72/97
- GT missing from final context: 25/97

## Examples Missing From Final Context

- CVE-2023-5905 GT=CWE-862 | initial=False cwe_top50=True knn=CWE-22 final=['CWE-144', 'CWE-146', 'CWE-151', 'CWE-22', 'CWE-433']
- CVE-2023-4797 GT=CWE-77 | initial=False cwe_top50=True knn=CWE-79 final=['CWE-79', 'CWE-89']
- CVE-2023-6029 GT=CWE-862 | initial=False cwe_top50=True knn=CWE-352 final=['CWE-144', 'CWE-146', 'CWE-152', 'CWE-166', 'CWE-352']
- CVE-2023-7199 GT=CWE-639 | initial=False cwe_top50=False knn=CWE-79 final=['CWE-125', 'CWE-144', 'CWE-148', 'CWE-150', 'CWE-151']
- CVE-2021-46928 GT=CWE-755 | initial=False cwe_top50=False knn=CWE-400 final=['CWE-119', 'CWE-122', 'CWE-1239', 'CWE-1257', 'CWE-1260']
- CVE-2021-46935 GT=CWE-668 | initial=False cwe_top50=False knn=CWE-416 final=['CWE-120', 'CWE-131', 'CWE-20', 'CWE-416']
- CVE-2021-46934 GT=CWE-754 | initial=False cwe_top50=False knn=CWE-20 final=['CWE-158', 'CWE-20', 'CWE-269', 'CWE-356', 'CWE-441']
- CVE-2021-46943 GT=CWE-131 | initial=False cwe_top50=True knn=CWE-119 final=['CWE-119', 'CWE-120', 'CWE-121', 'CWE-122', 'CWE-1274']
- CVE-2022-48654 GT=CWE-908 | initial=False cwe_top50=False knn=CWE-824 final=['CWE-119', 'CWE-120', 'CWE-121', 'CWE-1257', 'CWE-1311']
- CVE-2021-46923 GT=CWE-668 | initial=False cwe_top50=True knn=CWE-401 final=['CWE-114', 'CWE-1386', 'CWE-280', 'CWE-281', 'CWE-282']
- CVE-2024-26909 GT=CWE-416 | initial=False cwe_top50=False knn=CWE-476 final=['CWE-476']
- CVE-2021-46906 GT=CWE-668 | initial=False cwe_top50=False knn=CWE-120 final=['CWE-116', 'CWE-119', 'CWE-120', 'CWE-121', 'CWE-122']
