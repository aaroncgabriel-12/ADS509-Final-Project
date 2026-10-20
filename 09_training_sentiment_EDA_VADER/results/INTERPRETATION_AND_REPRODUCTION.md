# Descriptive sentiment EDA: frozen training passages only

This addition uses the VADER method taught in ADS 509 Module 2, cells 23-25:
`SentimentIntensityAnalyzer().polarity_scores(text)` and a 40-bin compound-score histogram.
Module 3 cells 6-7 carry forward and discuss the `sent_compound` feature. This is
descriptive sentiment EDA alongside the musical-insight classification task.

## Actual result

Population: **140 frozen training passages**, each from a distinct saved review/release group.
Negative compound scores: **26 (18.6%)**;
exactly zero: **36 (25.7%)**;
positive compound scores: **78 (55.7%)**.
Mean compound score: **0.2582**; median: **0.2369**;
range: **-0.9224 to 0.9413**.

Read this as the distribution of a lexicon's scores in the selected training subset.
Do not extrapolate it to all 500 reviews, all music reviewers, or all listeners.
Sign groups mean compound < 0, == 0, and > 0; they are not validated sentiment
classes and do not use the optional conventional +/-0.05 categorical thresholds.
Zero can reflect no recognized sentiment or balanced evidence, rather than neutral opinion.
Short excerpts, context loss, sarcasm, and music-specific language may affect scores.

## Scope and preservation

Only the exact training CSV with SHA256 `3500dcd7e53b39a09fd5f78cd28eee94e9cb9ebc1d3e5e56d7fa69e784841096` is accepted. Its existing
text is scored unchanged. No validation/test rows are loaded, no network requests
or installations occur, and no task labels, splits, model parameters, checkpoints,
or evaluation results are written or changed. The task labels remain AI-assisted;
VADER scores are separate descriptors, not replacement targets or human gold.
This analysis was added after model selection and does not justify retuning.

`training_vader_scores.csv` retains IDs and existing category labels with actual
VADER scores, without duplicating passage text. `sentiment_summary.json` records
counts, histogram bins, versions, source/lexicon hashes, and limitations.
`training_vader_sentiment.png` and `.svg` show the same computed distribution.
All source/license/attribution records remain in the unchanged source dataset.

## Reproduce locally

Use the existing Python environment with NLTK and the local VADER lexicon:

```bash
/opt/miniconda3/envs/tf_env/bin/python analyze_training_vader.py --output-dir results_recheck
```

Run from this analysis folder. The source path defaults to the existing project
training CSV. Alternatively use `--input /absolute/path/to/the/exact/training.csv`.
The chosen output directory must not already exist; prior outputs are never overwritten.
No `nltk.download` call is made. A missing dependency/resource stops the run.

## Course source

`coursework-root/ADS 509/ADS-509-mod2/M2_CleaningAndEDA_Student.ipynb`,
cells 23-25 (zero-based notebook indices), with saved executions 15 and 16.
`coursework-root/ADS 509/ADS-509-mod3/M3_EmbeddingComparison_Student.ipynb`, cells 6-7.

This supplies actual sentiment-distribution evidence alongside the existing text
length/frequency EDA. It is not an instructor approval or a claim that all final
project requirements, presentation requirements, or repository requirements are complete.
