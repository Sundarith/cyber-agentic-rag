# Manual Evaluation Queries — Cyber Threat RAG

Run interactively against `better_rag.py`. For each query, record **pass** or **fail**
and a one-line note on what was wrong (if fail).

**Failure modes to watch for:**
- Hallucinated technique IDs (T-numbers not in MITRE ATT&CK)
- Wrong CWE links (CWE cited but not connected to the actual technique chain)
- Missing information (answer says "I don't have enough information" when data exists)
- Wrong entity names (e.g., describes VBScript when asked about PowerShell)

---

## Category 1 — Technique Definition (5 queries)

### Q1
**Query:** `What is T1059.001?`
**Must contain:** "PowerShell", tactic "Execution", parent T1059
**Must NOT contain:** "VBScript", "Visual Basic", "WMI"
**Pass:** Answer names PowerShell as the scripting environment, no other interpreter confused for it.

---

### Q2
**Query:** `What is T1566?`
**Must contain:** "Phishing", tactic "Initial Access"
**Must NOT contain:** fabricated sub-technique IDs not in ATT&CK (e.g., T1566.004 does not exist)
**Pass:** Correct tactic, correct technique name, sub-techniques cited (if any) are real (T1566.001, T1566.002, T1566.003, T1566.004).

---

### Q3
**Query:** `What is T1486?`
**Must contain:** "Data Encrypted for Impact", "ransomware" or "encrypt", tactic "Impact"
**Must NOT contain:** unrelated techniques presented as sub-techniques of T1486 (it has none)
**Pass:** Correctly described as ransomware/destructive encryption, no fabricated sub-techniques.

---

### Q4
**Query:** `What is T1190?`
**Must contain:** "Exploit Public-Facing Application", "vulnerability", tactic "Initial Access"
**Must NOT contain:** hallucinated CVE IDs not present in the chunk
**Pass:** Correct name and tactic, any CVEs cited exist in context.

---

### Q5
**Query:** `What is T1003.001?`
**Must contain:** "LSASS", "credential", parent T1003, tactic "Credential Access"
**Must NOT contain:** "Mimikatz" attributed to the wrong technique
**Pass:** Correctly identifies LSASS Memory dumping as the mechanism.

---

## Category 2 — Entity Reverse Lookup (4 queries)

### Q6
**Query:** `What techniques does APT29 use?`
**Must contain:** technique list, IDs starting with T, count > 50
**Must NOT contain:** techniques from a different group (e.g., APT28-specific techniques incorrectly attributed)
**Pass:** Fast-path fires (`[Reverse entity lookup: APT29...]`), list returned without LLM call.

---

### Q7
**Query:** `What techniques does Prestige use?`
**Must contain:** T1112, T1083, T1486, T1490, count = 9
**Must NOT contain:** techniques not in the S1058 chunk
**Pass:** Exactly the 9 techniques listed in the Prestige (S1058) chunk.

---

### Q8
**Query:** `What malware does Sandworm Team use?`
**Must contain:** "Prestige" or "S1058", at least one other malware name
**Must NOT contain:** malware attributed to a different group
**Pass:** Fast-path fires for group→malware lookup, Prestige appears in the list.

---

### Q9
**Query:** `What groups use T1566?`
**Must contain:** at least APT28 or APT29 (both are known T1566 users)
**Must NOT contain:** group names fabricated or not in ATT&CK
**Pass:** Returns a list of groups; APT28 or APT29 present.

---

## Category 3 — CVE Lookup (3 queries)

### Q10
**Query:** `Tell me about CVE-2021-44228`
**Must contain:** "Log4j", "JNDI", "LDAP", "arbitrary code" or "remote code execution"
**Log check:** `[Hybrid search: CVE-2021-44228 (score ≥ 10.0)]`
**Must NOT contain:** wrong product names (e.g., "Apache Struts" instead of Log4j)
**Pass:** Score ≥ 10.0 in log, correct product and mechanism described.

---

### Q11
**Query:** `Tell me about CVE-2017-0144`
**Must contain:** "SMB", "EternalBlue" or "Windows", "remote code execution"
**Log check:** `[Hybrid search: CVE-2017-0144 (score ≥ 10.0)]`
**Pass:** Score ≥ 10.0, SMB/EternalBlue connection present.

---

### Q12
**Query:** `Tell me about CVE-2014-0160`
**Must contain:** "OpenSSL", "Heartbleed" or "memory", "information disclosure"
**Log check:** `[Hybrid search: CVE-2014-0160 (score ≥ 10.0)]`
**Pass:** Score ≥ 10.0, OpenSSL and memory leak mechanism described.

---

## Category 4 — CWE Root Cause Chain (4 queries)

### Q13
**Query:** `What are the CWE root causes for the techniques used by Prestige?`
**Must contain:** CWE-15, CWE-276 or CWE-285 (from T1083→CAPEC-127 chain)
**Log check:** `[Deep Search (2-hop)...]` must fire
**Must NOT contain:** CWE IDs with fabricated descriptions
**Pass:** At least 2 CWEs correctly named and linked through the chain; descriptions match real CWE names.

---

### Q14
**Query:** `What CWEs are associated with T1112?`
**Must contain:** CWE-15 (T1112→CAPEC-203→CWE-15)
**Log check:** `[Deep Search (2-hop)...]` must fire
**Pass:** CWE-15 ("External Control of System or Configuration Setting") appears with correct name.

---

### Q15
**Query:** `What is CAPEC-66?`
**Must contain:** "SQL injection" or "SQL", CAPEC-66 score ≥ 10 in log
**Must contain (CWE):** CWE-89 (CAPEC-66 maps to CWE-89 Improper Neutralization of SQL)
**Pass:** Score ≥ 10, SQL injection correctly described, CWE-89 present in answer or context.

---

### Q16
**Query:** `What CWE underlies CAPEC-98?`
**Must contain:** CWE-451 ("User Interface Misrepresentation of Critical Information")
**Log check:** `[Deep Search (2-hop)...]` must fire
**Pass:** CWE-451 cited with correct or close description.

---

## Category 5 — Detection (3 queries)

### Q17
**Query:** `How is T1059.001 detected?`
**Must contain:** PowerShell, logging, monitoring — NOT mitigation advice
**Must NOT contain:** "disable PowerShell" presented as a detection method (that's mitigation)
**Log check:** `MITIGATION_RE` must NOT fire; answer should reference analytics/monitoring
**Pass:** Detection-focused answer (logs, events, monitoring) without confusing detection with prevention.

---

### Q18
**Query:** `How is T1566.001 detected?`
**Must contain:** email, attachment, monitoring or analytics
**Must NOT contain:** mitigation steps presented as detection
**Pass:** Answer focuses on identifying spearphishing attachments through monitoring.

---

### Q19
**Query:** `How do I detect T1190?`
**Must contain:** web logs, IDS/IPS, anomaly or vulnerability scanning references
**Pass:** Detection-oriented answer for public-facing application exploitation.

---

## Category 6 — Mitigation (4 queries)

### Q20
**Query:** `How do I mitigate T1566?`
**Must contain:** M1017 (User Training) or M1049 (Antivirus/Antimalware) or M1054
**Must NOT contain:** detection steps presented as mitigations
**Pass:** At least one valid ATT&CK mitigation ID (M####) cited correctly.

---

### Q21
**Query:** `How do I mitigate CAPEC-98?`
**Must contain:** at least 2 concrete mitigation strategies (MFA, training, filtering, etc.)
**Pass:** Mitigation chunks drive the answer; strategies are actionable and grounded in context.

---

### Q22
**Query:** `What are the mitigations for T1059?`
**Must contain:** execution prevention, application whitelisting, or code signing references
**Must NOT contain:** M-numbers that don't exist in ATT&CK
**Pass:** Cited M-IDs (if any) are real ATT&CK mitigation IDs.

---

### Q23
**Query:** `How do I defend against T1486?`
**Must contain:** backups, or "data backup", or M1053
**Must NOT contain:** advice that only applies to detection, not prevention
**Pass:** Backup/recovery-oriented mitigations present; grounded in ATT&CK data.

---

## Category 7 — Cross-Domain Chain (2 queries)

### Q24
**Query:** `What attack patterns are associated with T1083?`
**Must contain:** CAPEC-127 or CAPEC-497
**Log check:** `[Deep Search (2-hop)...]` or `[1-hop...]` must fire
**Pass:** At least one CAPEC ID from the bridge (CAPEC-127 "Directory Indexing" or CAPEC-497 "File Discovery") present.

---

### Q25
**Query:** `What CWEs does APT28 exploit through its techniques?`
**Must contain:** at least 2 CWE IDs; descriptions must match real CWE names
**Log check:** `[Deep Search (2-hop)...]` must fire
**Must NOT contain:** CWE IDs with hallucinated descriptions
**Pass:** Multi-hop chain fires (APT28→techniques→CAPECs→CWEs), CWE names are accurate.

---

## Scoring

| # | Query | Pass/Fail | Notes |
|---|---|---|---|
| 1 | T1059.001 definition | | |
| 2 | T1566 definition | | |
| 3 | T1486 definition | | |
| 4 | T1190 definition | | |
| 5 | T1003.001 definition | | |
| 6 | APT29 techniques | | |
| 7 | Prestige techniques | | |
| 8 | Sandworm malware | | |
| 9 | Groups using T1566 | | |
| 10 | CVE-2021-44228 | | |
| 11 | CVE-2017-0144 | | |
| 12 | CVE-2014-0160 | | |
| 13 | Prestige CWE chain | | |
| 14 | T1112 CWE chain | | |
| 15 | CAPEC-66 (SQL inj) | | |
| 16 | CAPEC-98 CWE | | |
| 17 | Detect T1059.001 | | |
| 18 | Detect T1566.001 | | |
| 19 | Detect T1190 | | |
| 20 | Mitigate T1566 | | |
| 21 | Mitigate CAPEC-98 | | |
| 22 | Mitigate T1059 | | |
| 23 | Defend T1486 | | |
| 24 | CAPEC for T1083 | | |
| 25 | APT28 CWE chain | | |

**Target: 22/25 (88%) before considering embedding model upgrade or further tuning.**
