Music Review Insight Finder — completed expanded B experiment

ONE completed fresh B-settings run on a Tesla T4. No retraining or tuning after test.
Frozen data:400 training /200 validation /200 test.
Freeze:reviewed_export_20261019T234617178637Z
Frozen manifest SHA-256:8332b6ad842b1608fe760af1aaf9f13beae9de9ed6ebddadd1cf9aadaff342a5
The existing140 training rows were unchanged. All660 new usable labels had recorded
Aaron review after AI proposals. Labels are AI-assisted, not independent human gold.

SELECTED CHECKPOINT
adapter_epoch_3. Validation accuracy0.645, macro-F10.5629897528631705.
Selection used validation macro-F1, then lower unweighted CE, then earlier epoch.
Validation macro-F1 by epoch:0.3321138795337132 /0.4802897615055936 /0.5629897528631705.
150 optimizer attempts:148 succeeded,2 were skipped at attempts1and2 due to nonfinite
gradients under mixed precision.147 successful updates used a nonzero learning rate.
Recorded base-weight fingerprints matched before/after. Only fresh LoRA trained.

ONE-TIME TEST COMPARISON — SAME200 ROWS,99 REVIEW/RELEASE GROUPS
Unadapted:accuracy0.315,macro-F10.2389626991736292.
Adapted:accuracy0.71,macro-F10.6629586396067372.
Macro-F1 difference:+0.42399594043310795;accuracy difference:+39.5percentage points.
Adapted confusion matrix, actual rows/predicted columns A/B/C:
90 12 11
16 27 11
4  4 25
A=specific musical observation;B=general opinion;C=background/other.
No test scores affected checkpoint selection. Saved tokenizer/adapter preflight
passed on development data before test upload, and the test-attempt ledger records
one completed comparison. No model was rerun during local evidence verification.

FILES
11_music_review_EXPANDED_B_EXECUTED_ORIGINAL.ipynb preserves the downloaded executed
notebook byte-for-byte. All7 code cells executed; no error outputs.
11_music_review_EXPANDED_B_RESULTS_READABLE.ipynb rewrites only its6 Markdown cells,
with concise past-tense methods/results and a comprehensive final AI disclosure.
Code, outputs, execution counts, cell IDs and metadata are identical to the original.
MARKDOWN_ONLY_VALIDATION.json records that comparison.
music_expanded_B_actual_results.zip is the downloaded runtime archive unchanged.
runtime_results/ contains its safely extracted contents, including all3 adapters,
settings, diagnostics, prediction CSVs, history, selection/preflight/test receipts.
VERIFIED_RESULTS_SUMMARY.json records full metrics, versions, hashes and checks.
RESULTS_DELIVERY_SHA256.json hashes the delivered files.

INTEGRITY / INTERPRETATION
ZIP CRC passed;59 members. Selected checkpoint and sealed evidence hashes passed.
All validation epochs and both test metrics were independently recomputed from saved
predictions and matched. Sources, IDs and true labels matched frozen CSV rows.
The original executed notebook retains its preparation-time custom metadata tag
execution_status=not_run; actual execution is evidenced by counts1through7, saved
outputs and execution timing. Metadata was preserved as requested in the readable copy.
Passages are clustered, labels are subjective, and pretraining overlap is unknown.
These results do not establish business time savings or general performance outside
this dataset. Previous project files, frozen labels, notebooks and results were preserved.
