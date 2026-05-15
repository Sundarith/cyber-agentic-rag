# CTI-RCM Failure Analysis

Input failures analyzed: **29**

## High-Level Split

- NVD-mapped failures: 0/29 (0.0%)
- NVD-unmapped failures: 29/29 (100.0%)
- Gold CWE present in final context but answer still wrong: 10/29 (34.5%)
- Gold CWE missing from final context: 19/29 (65.5%)
- Retrieval miss with no gold CWE in initial, graph, or k-NN evidence: 16/29 (55.2%)

## k-NN CWE Behavior

- k-NN top was correct: 2/29 (6.9%)
- k-NN top was wrong: 26/29 (89.7%)
- k-NN contained gold but ranked another CWE higher: 7/29 (24.1%)
- High-confidence k-NN was wrong: 4/29 (13.8%)

## CWE Relationship Pattern

- unrelated_or_distant: 13/29 (44.8%)
- sibling_same_parent: 7/29 (24.1%)
- predicted_child_of_gold: 6/29 (20.7%)
- predicted_parent_of_gold: 3/29 (10.3%)

## Most Actionable Failure Examples

### llm_selection_error

- **CVE-2023-52435** GT `CWE-119` (Improper Restriction of Operations within the Bounds of a Memory Buffer) -> predicted `CWE-190` (Integer Overflow or Wraparound); evidence: final_context, knn_votes; k-NN top: `CWE-190` soft_hint
- **CVE-2024-0853** GT `CWE-295` (Improper Certificate Validation) -> predicted `CWE-299` (Improper Check for Certificate Revocation); evidence: final_context, knn_votes; k-NN top: `CWE-295` soft_hint
- **CVE-2022-48657** GT `CWE-120` (Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')) -> predicted `CWE-190` (Integer Overflow or Wraparound); evidence: initial_retrieval, final_context, knn_votes; k-NN top: `CWE-190` soft_hint
- **CVE-2019-25160** GT `CWE-125` (Out-of-bounds Read) -> predicted `CWE-787` (Out-of-bounds Write); evidence: initial_retrieval, final_context, knn_votes; k-NN top: `CWE-787` soft_hint
- **CVE-2024-26583** GT `CWE-362` (Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')) -> predicted `CWE-364` (Signal Handler Race Condition); evidence: initial_retrieval, final_context, knn_votes; k-NN top: `CWE-362` soft_hint

### wrong_high_confidence_knn

- **CVE-2023-4797** GT `CWE-77` (Improper Neutralization of Special Elements used in a Command ('Command Injection')) -> predicted `CWE-78` (Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')); evidence: none; k-NN top: `CWE-79` high_confidence
- **CVE-2023-52452** GT `CWE-665` (Improper Initialization) -> predicted `CWE-125` (Out-of-bounds Read); evidence: none; k-NN top: `CWE-125` high_confidence
- **CVE-2023-6529** GT `CWE-79` (Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')) -> predicted `CWE-352` (Cross-Site Request Forgery (CSRF)); evidence: initial_retrieval, knn_votes; k-NN top: `CWE-352` high_confidence
- **CVE-2023-46805** GT `CWE-287` (Improper Authentication) -> predicted `CWE-288` (Authentication Bypass Using an Alternate Path or Channel); evidence: knn_votes; k-NN top: `CWE-288` high_confidence

### knn_had_gold_but_ranked_other

- **CVE-2021-46949** GT `CWE-476` (NULL Pointer Dereference) -> predicted `CWE-754` (Improper Check for Unusual or Exceptional Conditions); evidence: final_context, knn_votes; k-NN top: `CWE-416` soft_hint
- **CVE-2021-46948** GT `CWE-476` (NULL Pointer Dereference) -> predicted `CWE-754` (Improper Check for Unusual or Exceptional Conditions); evidence: final_context, knn_votes; k-NN top: `CWE-416` soft_hint

### retrieval_miss

- **CVE-2023-6029** GT `CWE-862` (Missing Authorization) -> predicted `CWE-352` (Cross-Site Request Forgery (CSRF)); evidence: none; k-NN top: `CWE-352` soft_hint
- **CVE-2021-46928** GT `CWE-755` (Improper Handling of Exceptional Conditions) -> predicted `CWE-733` (Compiler Optimization Removal or Modification of Security-critical Code); evidence: none; k-NN top: `CWE-400` soft_hint
- **CVE-2021-46935** GT `CWE-668` (Exposure of Resource to Wrong Sphere) -> predicted `CWE-787` (Out-of-bounds Write); evidence: none; k-NN top: `CWE-416` soft_hint
- **CVE-2023-6064** GT `CWE-532` (Insertion of Sensitive Information into Log File) -> predicted `CWE-534` (DEPRECATED: Information Exposure Through Debug Log Files); evidence: none; k-NN top: `CWE-79` soft_hint
- **CVE-2021-46934** GT `CWE-754` (Improper Check for Unusual or Exceptional Conditions) -> predicted `CWE-781` (Improper Address Validation in IOCTL with METHOD_NEITHER I/O Control Code); evidence: none; k-NN top: `CWE-20` soft_hint

## Recommended Next Tuning Targets

1. If `llm_selection_error` is high, add a deterministic CWE selector after retrieval instead of relying only on generated text.
2. If `wrong_high_confidence_knn` appears, raise the k-NN confidence threshold or require a wider margin over the second CWE.
3. If `retrieval_miss` is high, improve CWE candidate retrieval for unmapped CVEs before changing prompts.
4. If parent/child/sibling errors dominate, use CWE hierarchy during candidate reranking.
