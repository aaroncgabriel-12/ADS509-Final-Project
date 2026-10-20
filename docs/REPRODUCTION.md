# Expanded evaluation navigation

The project now includes a second phase. For frozen 400/200/200 data, exact notebook-11 input archives, matching helpers and saved predictions, use [the expanded guide](../expanded_evaluation/README.md). For an offline saved-result check, run `python scripts/verify_expanded_evaluation.py`. No training is needed to inspect these results.

The instructions below describe the historical prototype.

# Reproduction and evidence boundaries

## Review without training

Open readable notebook 10 to inspect the same stored A–D JSON metrics. Readable
09 reproduces VADER on only the frozen 140-row training file, and checks all
compound scores against the saved score CSV. A local VADER lexicon is required.
The existing script can produce a fresh result directory:

```bash
python 09_training_sentiment_EDA_VADER/analyze_training_vader.py --output-dir outputs/vader_recheck
```

The output directory must be new. The script refuses a changed training hash.
No new output is claimed merely by supplying these commands.

## Collection and preparation

`music_review_pipeline.py` is the actual collector, not the empty notebook 01.
It normalizes API responses to source/license/author/text fields, caches pages,
rejects duplicate IDs and checks metadata. Twenty included normalized cache pages
allow the original 500-review collection to be checked offline. Full hidden/draft
records and user profiles were not cached. All original source fields/attributions
are retained. The collection config/audit describe oldest-first bounded sampling.

Notebook 02 keeps the original collection/preparation computations. Its project
path now resolves to this repository. Cached pages are included; using a new
session or removing a cache can cause public API GET requests. New collection is
not guaranteed to recreate a source that may change. Preparation retains case,
punctuation and negation, splits long sentences into 100-word chunks, and removes
short/exact duplicate passages. Sections 5–8 are the earlier manual annotation
workflow and should not be mistaken for the final audited labels.

## Recorded GPU notebook and immutable input pairing

| Executed notebook | Exact input from `inputs/` |
|---|---|
| 06_music_review_lora_AI_AUDITED_COLAB.ipynb | music_insight_AI_AUDITED_training_bundle.zip |
| 07_music_review_VALIDATION_ONLY_AB_COLAB-2.ipynb | music_insight_TRAIN_VALIDATION_ONLY_AB_bundle.zip |
| 08_music_review_C_LOWER_LR_COLAB-5.ipynb | music_insight_TRAIN_VALIDATION_ONLY_C_lr1e4_bundle.zip |

Each notebook verifies the exact archive/hash and matching helper files. Bundle
code is also expanded under `historical_preparation/run06`, `run07`, `run08` for
review. The original ZIPs must remain unchanged. They contain nonsecret original
local provenance paths; these were retained to preserve the notebook integrity
checks. Their hashes and scan findings are recorded in `packaging/`.

Training used Python 3.12.13, torch 2.11.0+cu128, Tesla T4, Transformers 4.57.1,
PEFT 0.17.1, Accelerate 1.10.1, Safetensors 0.6.2, NumPy 1.26.4, pandas 2.2.3,
scikit-learn 1.6.1. Choose the matching CUDA environment, follow each notebook's
setup/restart cells, and upload its matching bundle only. CPU/Mac kernels are not
drop-in substitutes. GPU nondeterminism, runtime availability and package sources
can prevent bit-for-bit reproduction; no new training was run for this package.

Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`. Notebooks 07/08 pin it
in their helpers. Notebook 06 historically resolved `MODEL_REVISION = None` and
saved that exact commit; for a separately created reproduction copy, set that
recorded revision rather than silently following the current default branch.
The archived executed notebook is preserved byte-for-byte.

08 contains completed C and the fresh five-epoch D experiment. It uses bundled
verified B summary/settings by default; the large B results ZIP is optional. D's
private functions and the D3 diagnostic repair are retained in execution order.
C and D were not resumed from one another. The repair did not rerun training.

## Old test and checkpoints

06 includes a previously exposed historical test set and recorded test evaluation.
Do not present rerunning that test as a new independent confirmation. Follow-up
bundles contain training and validation only. The package verifier does not parse
or score old-test rows; whole-archive hashes/CRC scans do not evaluate a model.

No model weights, adapter checkpoints, or bulk C/D diagnostic prediction tables
are included in the upload folder. Three complete result ZIPs are preserved in
the sibling local archive and mapped in its manifest. Restoring/reading their
checkpoints requires the recorded model/runtime. No claim is made that an initial
run-06 result ZIP was found; its executed notebook remains the available evidence.

## Labels and historical preparation

Frozen AI-audited references are inputs, not independently generated ground truth.
Earlier local labeling helpers are supplied for transparency, not as a new task
to relabel or re-split. The historical hint/session helpers had only local folder
bindings made portable. Their old independent-label intentions were superseded
by the later nonblinded audit, whose provenance remains in final CSVs/manifests.
One-off chat audit/preparation backups remain in the original local workspace;
the stable frozen decisions/labels consumed by the GPU runs are in the bundles.

## Planning and remaining deliverables

`presentation_outline_PLANNING_v1.docx` is the current Library version 1,
materialized October 19, including completed VADER coverage. It is an outline,
not completed slides or a finished script. No video, hosting URL, GitHub URL,
instructor access confirmation or Canvas submission exists in this package.
