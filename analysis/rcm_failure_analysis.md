# CTI-RCM Failure Analysis

Input failures analyzed: **25**

## High-Level Split

- NVD-mapped failures: 0/25 (0.0%)
- NVD-unmapped failures: 25/25 (100.0%)
- Gold CWE present in final context but answer still wrong: 10/25 (40.0%)
- Gold CWE missing from final context: 15/25 (60.0%)
- Retrieval miss with no gold CWE in initial, graph, or k-NN evidence: 14/25 (56.0%)

## k-NN CWE Behavior

- k-NN top was correct: 1/25 (4.0%)
- k-NN top was wrong: 24/25 (96.0%)
- k-NN contained gold but ranked another CWE higher: 6/25 (24.0%)
- High-confidence k-NN was wrong: 4/25 (16.0%)

## CWE Relationship Pattern

- unrelated_or_distant: 15/25 (60.0%)
- predicted_child_of_gold: 7/25 (28.0%)
- sibling_same_parent: 3/25 (12.0%)

## Most Actionable Failure Examples

### llm_selection_error

- **CVE-2023-52435** GT `CWE-119` (Improper Restriction of Operations within the Bounds of a Memory Buffer) -> predicted `CWE-190` (Integer Overflow or Wraparound); evidence: graph_neighbors, final_context, knn_votes; k-NN top: `CWE-190` soft_hint
- **CVE-2024-0853** GT `CWE-295` (Improper Certificate Validation) -> predicted `CWE-299` (Improper Check for Certificate Revocation); evidence: final_context, knn_votes; k-NN top: `CWE-295` soft_hint
- **CVE-2022-48661** GT `CWE-404` (Improper Resource Shutdown or Release) -> predicted `CWE-401` (Missing Release of Memory after Effective Lifetime); evidence: initial_retrieval, final_context, knn_votes; k-NN top: `CWE-401` soft_hint
- **CVE-2023-6029** GT `CWE-862` (Missing Authorization) -> predicted `CWE-352` (Cross-Site Request Forgery (CSRF)); evidence: final_context; k-NN top: `CWE-352` soft_hint
- **CVE-2022-48657** GT `CWE-120` (Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')) -> predicted `CWE-680` (Integer Overflow to Buffer Overflow); evidence: initial_retrieval, final_context, knn_votes; k-NN top: `CWE-190` soft_hint

### wrong_high_confidence_knn

- **CVE-2023-4797** GT `CWE-77` (Improper Neutralization of Special Elements used in a Command ('Command Injection')) -> predicted `CWE-78` (Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')); evidence: none; k-NN top: `CWE-79` high_confidence
- **CVE-2023-52452** GT `CWE-665` (Improper Initialization) -> predicted `CWE-125` (Out-of-bounds Read); evidence: none; k-NN top: `CWE-125` high_confidence
- **CVE-2023-6529** GT `CWE-79` (Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')) -> predicted `CWE-352` (Cross-Site Request Forgery (CSRF)); evidence: initial_retrieval, knn_votes; k-NN top: `CWE-352` high_confidence
- **CVE-2023-46805** GT `CWE-287` (Improper Authentication) -> predicted `CWE-288` (Authentication Bypass Using an Alternate Path or Channel); evidence: knn_votes; k-NN top: `CWE-288` high_confidence

### knn_had_gold_but_ranked_other

- **CVE-2023-6383** GT `CWE-862` (Missing Authorization) -> predicted `CWE-548` (Exposure of Information Through Directory Listing); evidence: final_context, knn_votes; k-NN top: `CWE-79` soft_hint

### retrieval_miss

- **CVE-2021-46928** GT `CWE-755` (Improper Handling of Exceptional Conditions) -> predicted `CWE-787` (Out-of-bounds Write); evidence: none; k-NN top: `CWE-400` soft_hint
- **CVE-2021-46935** GT `CWE-668` (Exposure of Resource to Wrong Sphere) -> predicted `CWE-131` (Incorrect Calculation of Buffer Size); evidence: none; k-NN top: `CWE-416` soft_hint
- **CVE-2021-46934** GT `CWE-754` (Improper Check for Unusual or Exceptional Conditions) -> predicted `CWE-782` (Exposed IOCTL with Insufficient Access Control); evidence: none; k-NN top: `CWE-20` soft_hint
- **CVE-2021-46943** GT `CWE-131` (Incorrect Calculation of Buffer Size) -> predicted `CWE-789` (Memory Allocation with Excessive Size Value); evidence: none; k-NN top: `CWE-119` soft_hint
- **CVE-2022-48654** GT `CWE-908` (Use of Uninitialized Resource) -> predicted `CWE-457` (Use of Uninitialized Variable); evidence: none; k-NN top: `CWE-824` soft_hint

## Recommended Next Tuning Targets

1. If `llm_selection_error` is high, add a deterministic CWE selector after retrieval instead of relying only on generated text.
2. If `wrong_high_confidence_knn` appears, raise the k-NN confidence threshold or require a wider margin over the second CWE.
3. If `retrieval_miss` is high, improve CWE candidate retrieval for unmapped CVEs before changing prompts.
4. If parent/child/sibling errors dominate, use CWE hierarchy during candidate reranking.
