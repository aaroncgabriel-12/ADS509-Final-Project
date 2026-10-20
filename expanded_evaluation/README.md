# Expanded evaluation

This phase extends the prototype in the same project. Start with [readable notebook 11](../notebooks/11_music_review_EXPANDED_B_RESULTS_READABLE.ipynb). It preserves the seven executed code cells and their outputs from the original downloaded notebook; only Markdown explanations were rewritten.

## Data and results

| Split | Passages | Review/release groups |
|---|---:|---:|
| Training | 400 | 180 |
| Validation | 200 | 99 |
| Test | 200 | 99 |

The 140 prototype training rows were preserved. Aaron reviewed 260 additional training labels, 200 validation labels and 200 test labels after AI proposals. Thirty-seven ambiguous candidates were skipped and replaced within their assigned queues. The labels are AI-assisted with human review, not independent human gold.

The fresh three-epoch B run selected epoch 3 using validation macro-F1 (0.562990). A reload preflight preceded the single final test comparison. On the same 200 test passages from 99 groups, accuracy was 31.5% unadapted and 71.0% adapted; macro-F1 was 0.238963 and 0.662959.

The adapted confusion matrix has actual rows and predicted columns, ordered specific musical observation, general opinion, background/other:

| Actual / predicted | Specific musical observation | General opinion | Background/other |
|---|---:|---:|---:|
| Specific musical observation | 90 | 12 | 11 |
| General opinion | 16 | 27 | 11 |
| Background/other | 4 | 4 | 25 |

True supports are 113/54/33. Adapted prediction counts are 110/43/47; unadapted counts are 83/1/116. General-opinion recall is 50%. More varied training examples and clearer mixed-passage guidelines are sensible future improvements. No further training was performed after viewing these results.

## File map

| Location | Purpose |
|---|---|
| [frozen_data](frozen_data/) | Exact train, validation and test CSVs, original 140 rows, frozen manifest, review decisions and skipped candidates. |
| [inputs](inputs/) | Exact development and test archives consumed by notebook 11. |
| [helpers](helpers/) | Matching contract, training, preflight and final-evaluation code, plus environment pins. |
| [results](results/) | Actual saved settings, validation predictions/history, optimizer logs, selection receipt, reload preflight and final comparison/predictions. Adapter weights are excluded. |
| [VERIFIED_RESULTS_SUMMARY.json](VERIFIED_RESULTS_SUMMARY.json) | Independent local checks of the saved outputs; no model rerun. |
| [MARKDOWN_ONLY_VALIDATION.json](MARKDOWN_ONLY_VALIDATION.json) | Evidence that readable notebook edits preserved executed code and outputs. |
| [RESULTS_DELIVERY_SHA256.json](RESULTS_DELIVERY_SHA256.json) | Historical full-run snapshot. It references the original archive layout, including checkpoints retained locally; it is not the current package index. |
| [../presentation_evidence/expanded_evaluation](../presentation_evidence/expanded_evaluation/) | Three figures in PNG/SVG, editable metrics tables, full original source paths and hashes. |
| [../packaging/EXPANSION_INTEGRATION.json](../packaging/EXPANSION_INTEGRATION.json) | Current package provenance, preserved notebook hashes and source copy records. |

The original execution archive and original executed notebook are preserved in the sibling `LOCAL_ARCHIVE_DO_NOT_UPLOAD/expanded_evaluation` folder. The archive contains all three adapter checkpoints and tokenizer files. Source originals remain in the expansion project.

## Review and reproduction

Run `python scripts/verify_expanded_evaluation.py` from the repository root to check saved data, predictions, package hashes and notebook preservation. It does not run models or contact the network.

Notebook 11 is an executed Colab record. For a separately approved reproduction, use a fresh free T4 session with the recorded runtime and package pins. Upload `inputs/music_expansion_train_validation.zip` at the first prompt. Train using development data, select the checkpoint and complete reload verification. Only then upload `inputs/music_expansion_test_after_selection.zip`. The notebook checks the freeze and helper hashes before use. The archives contain matching manifests and review evidence.

The frozen manifest records the state before execution, including false training/test flags. Those immutable preparation flags do not describe the later completed run; use the saved executed notebook and results for execution status. Original machine/runtime paths and source hashes are historical provenance. Current relative navigation is listed here.

The test data and outcomes are now exposed. A reproduction would not create a fresh independent evaluation. Any new development needs a new untouched test set. Prototype scores and expanded scores use different evaluation data, so their difference cannot establish a causal effect of more training data alone.
