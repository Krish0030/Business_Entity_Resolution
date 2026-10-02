# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Swastika Sinha  
**Team Members:** Swastika Sinha  
**Submission Date:** 25 September 2026

## 1. Executive Summary

This solution combines compact, multi-pass string blocking with a supervised LightGBM pair classifier. It uses only the supplied data, generates an auditable candidate set, and selects a decision threshold using macro F0.5, including singletons. The independently held-out macro F0.5 is **0.896357**, before refitting the final model on all sampled training anchors.

## 2. Methodology

### 2.1 Problem Analysis

| Split | Source 1 | Source 2 | Source 3 |
|---|---:|---:|---:|
| Training | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

The test references include 663,106 US, 809,986 India, and 259,452 France records. There are no labeled France training records. Observed corruptions include legal-suffix changes, punctuation, accents, reordered words, abbreviated addresses, changed house numbers, shortened names, missing addresses, native-script names, and unrelated trade names at similar addresses. Similar or identical names can belong to different labeled businesses, making name-only acceptance risky.

### 2.2 Solution Strategy

**Approach Type:** Multi-pass blocking plus a supervised pair classifier.  
**Core Design:** Packed hash indexes allow comparison against millions of reference entities without materializing a Cartesian product. Validation uses full-reference bucket frequencies and scans every training target, while collecting labels/features for a manageable sample of reference anchors.

Read all files with an explicit tab separator. Normalize names and addresses, generate blocked pairs, apply a broad similarity prefilter, calculate pair features, score every remaining pair, and accept probabilities at or above the selected threshold. Emit all reference IDs, including empty predictions. Predictions are independent pair decisions; multiple records from either target source are allowed, and no one-to-one constraint is imposed.

## 3. Candidate Generation (Blocking)

Country is an unrestricted string equality partition. All observed labels are processed automatically. No country-specific classifier or country one-hot encoding is used. A country mismatch prevents a pair from being considered.

Normalization uses Unicode NFKD, case folding, removal of combining accents, alphanumeric tokenization, a fixed list of common legal suffixes, and a small fixed address-abbreviation map. Original token order and sorted unique-token forms are both retained. Purely numeric address tokens lose leading zeroes. Other scripts are preserved as tokens; no translation, external address normalization, or business lookup occurs.

For each normalized record, union the following keys:

- Exact sorted name and exact sorted address.
- Pairs of five-character name-token prefixes, using at most eight distinct prefixes.
- Individual five-character name prefixes.
- Name prefixes crossed with up to four sorted distinct address numbers.
- Up to ten long address-token prefixes crossed with up to three address numbers.

Keys use CRC32 and are packed with reference row offsets in sorted uint64 arrays. A key is ignored if it retrieves more than 64 references. Hash collisions can merge buckets; they are never themselves evidence for a match. Several independent keys help preserve recall despite noisy fields and bucket caps.

Deduplicate reference-target pairs across passes. The last blocking filter retains a pair if sorted-name similarity is at least 78, sorted-address similarity is at least 78, or both are at least 48, on a 0–100 RapidFuzz ratio scale. Every surviving pair is passed to the final classifier and appears in `candidate_pairs.tsv`. Thresholding only removes pairs from the matching output, never from the candidate output.

**Training candidates:** 239,807 pairs for 6,570 sampled reference anchors, including 20,848 true links out of 22,748 labeled links. The blocking index contains every training reference, and all 10,320,219 training target records are scanned. Full bucket caps are computed before projecting the index onto sampled anchors, preserving exactly the full-index candidate set for those anchors.

**Independent holdout blocking recall:** 91.5066% (4,288 / 4,686 true links). A perfect classifier restricted to these candidates would achieve macro F0.5 of 0.968018. Blocking losses are measured, not assumed to be zero.

**Test candidate count:** 88,037,235 pairs. **Accepted test links:** 6,365,528. Both TSVs contain exactly 1,732,544 reference rows. Against 17,272,751,604,416 unrestricted Cartesian pairs, the candidate reduction ratio is 99.99949031%. This is not a recall estimate; test labels are unavailable. Exact country counts and validation results are in `output/run_summary.json`, `output/validation_report.json`, and `output/official_validation_report.json`. Both the independent streaming checks and the unmodified supplied validator functions (run in bounded batches with ID checking) passed.

## 4. Matching Model

**Features:** 28 numerical features computed from names and addresses. They include ordinary, sorted-token, token-set, and partial string ratios; token Jaccard and containment overlap; numeric Jaccard and containment; numeric disagreement; exact normalized equality; minimum and maximum field lengths; length ratios; missing-field indicators; and minimum token counts. Entity IDs and country labels are not classifier features.

**Model:** LightGBM binary boosted trees, 23 maximum leaves per tree, learning rate 0.04, minimum 40 examples per leaf, L2 regularization 5, and feature fraction 0.9. The development fit uses early stopping on tuning binary loss, with a 350-tree limit and patience 35. It selected 348 trees. The final model has 348 trees and is approximately 0.9 MB; it is far below the 8-billion-parameter limit. LightGBM and the supplied trained model are MIT licensed. There is no pretrained language model.

**Sampling and splitting:** Select anchors with a stable CRC32 hash fraction below 0.003, independent of file ordering. A separately prefixed hash assigns 3,978 anchors to fitting, 1,270 to tuning, and 1,322 to holdout. Splits are disjoint by Source 1 anchor. A target may occur as a negative candidate for more than one anchor, so this is not a strict connected-component split across every negative pair. No IDs enter model features. Full reference strings contribute only unsupervised blocking statistics outside the sampled fit groups.

**Threshold selection:** Evaluate thresholds 0.05 through 0.99 in steps of 0.01 on tuning anchors. Select **0.58** by maximum macro F0.5. Singletons score one only for an empty prediction. For a non-singleton, use `1.25 * TP / (predicted_count + 0.25 * true_count)`. Score the untouched holdout once, then refit the model on all 6,570 sampled anchors with the iteration count and threshold fixed. The reported score comes from the pre-refit development model, not training predictions from the final model.

## 5. Results & Error Analysis

| Metric | Independent holdout |
|---|---:|
| Macro F0.5 | **0.896357** |
| Pair precision | 0.963816 |
| Pair recall | 0.812847 |
| True positives | 3,809 |
| False positives | 143 |
| False negatives | 877 |
| Singleton accuracy | 58 / 64 = 0.906250 |
| US macro F0.5 (785 anchors) | 0.918611 |
| India macro F0.5 (537 anchors) | 0.863826 |

These are local validation results, not leaderboard scores. Test labels were not supplied, and no portal submission or test-score claim is made. In particular, the labeled validation does not establish performance on France. The holdout has only 64 singletons, so the singleton estimate has substantial sampling uncertainty.

Of the 877 missed holdout links, 398 were absent from blocking and 479 were rejected by the classifier. Misses include native-script names combined with partial addresses, altered house numbers, unrelated trade names, and heavily abbreviated address components. Some false positives have nearly identical normalized names and closely overlapping addresses despite distinct ground-truth assignments. Legal-suffix removal and token-set containment can make those difficult negatives look like strong matches.

`model/holdout_errors.tsv` contains pair IDs and error categories; `model/error_examples.json` includes selected source records for inspection. The original held-out results and complete threshold grid are in `model/metadata.json`. Error inspection was descriptive; it did not change the blocker, features, selected threshold, or fitted model after the holdout was scored.

Limitations include fixed bucket caps, country equality blocking, no transliteration model, field truncation within blocking keys, independent pair decisions without global conflict resolution, and supervised fitting on a reference sample. The supplied code supports increasing the sample rate, but doing so requires a fresh cache and a new validation run.

## 6. Conclusion

The pipeline produces complete test predictions and an exact audit of the candidate pairs scored by its classifier. Local validation demonstrates a high-precision operating point, with measured recall losses from both blocking and classification. Generalization to France remains unmeasured until the challenge evaluates test predictions.

## Appendix

### A. Code Artefacts

All pipeline and validation source is under `code/business_entity_resolution/src/`. The accompanying README provides commands for fresh end-to-end training, inference from the included model, and format validation. `requirements.txt` pins dependencies; `LICENSE` and `THIRD_PARTY_LICENSES.txt` record model/code licensing. The challenge data is not duplicated in the package.

Inference uses disjoint country/source/hash partitions, bounded target batches, and separate SQLite pair stores. Completed partitions are reusable after interruption. Output assembly merges sorted partitions and writes one row for every test reference. Parallelization changes execution order but not candidate semantics or decisions. Caches must be replaced after dataset, feature, blocker, or model changes.

### B. Additional Results and Fair Play

The run uses Python 3.12.14 on Windows. Dependencies were obtained as software packages; no external business data, APIs, geocoders, registration databases, or identity services were accessed. Original input files were read without modification. The package includes the trained model, local validation metrics, error examples, and test-output validation reports for reproduction and audit.
