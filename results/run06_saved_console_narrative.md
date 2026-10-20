This is verbatim recorded console output from executed notebook 06, cell 23 (zero-based), not a new evaluation.

```text
# Results — generated only after real evaluation

Model: Qwen/Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775.
Training/validation/test passages: 140/33/35.
Selected epoch: 1, chosen using validation data only.
Labeling method: {'train': 'AI-assisted: passage-by-passage audit by Codex after original human review; NOT independent human labels', 'validation': 'AI-assisted: passage-by-passage audit by Codex after original human review; NOT independent human labels', 'test': 'AI-assisted: passage-by-passage audit by Codex after original human review; NOT independent human labels'}.
Membership was frozen before the initial training hints; later AI audit reviewed every split. No re-splitting occurred.
All labels are AI-assisted, not independent human gold. These metrics measure agreement with audited labels.
Audit manifest SHA-256: 80292047e3857b7150e438dbc01922750dcbc6dbca779b25689b0c5251ec65f3.
GPU smoke test: PASS; it used zero test rows.
Adapted minus original test macro-F1: -0.0404.

See test_metrics.json for per-class precision/recall/F1 and confusion matrices;
test_predictions.csv for provenance-linked outputs; adapted_test_errors.csv for errors.

Interpret the actual direction and magnitude of the difference; adaptation need not improve results.
Small grouped test sets have substantial uncertainty. These are descriptive point estimates,
not a statistical significance claim or evidence of performance for all music listeners.

Complete with Aaron's own interpretation before submission:
- Which categories were confused, using concrete errors?
- How did limited/subjective labels, context loss, oldest-first sampling, class balance,
  release-group concentration, and unknown pretraining overlap affect the conclusions?
- How were missing license/attribution fields and any redistribution restrictions handled?
- What did AI actually help with, and what did Aaron independently run, verify, or change?
- What would a larger, independent evaluation test next?


```
