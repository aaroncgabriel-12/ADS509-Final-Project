"""Descriptive Module-2 VADER EDA on the exact frozen training CSV only.

No model training, network calls, split changes, or sentiment target labels.
Requires existing nltk, numpy and matplotlib plus the local VADER lexicon.
An existing output directory is refused to preserve every prior run.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

EXPECTED_TRAIN_SHA256 = '3500dcd7e53b39a09fd5f78cd28eee94e9cb9ebc1d3e5e56d7fa69e784841096'
TRAIN_RELATIVE_PATH = 'music_review_runs/2026-10-16_music_insight_AI_audit_v1/audited_splits/ai_audited_train.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent.parent/TRAIN_RELATIVE_PATH)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    source = args.input.resolve()
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError('Input must be the exact frozen 140-row training CSV; no validation/test input accepted.')
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 140 and {r['split'] for r in rows} == {'train'}
    assert len({r['passage_id'] for r in rows}) == 140
    assert all(isinstance(r['text'], str) and r['text'].strip() for r in rows)

    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='music_vader_matplotlib_'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np

    # Same analyzer and compound-score approach as Module 2 cells 24-25.
    # Use the existing passage text, retaining punctuation/case/negation; no stopword filtering.
    analyzer = SentimentIntensityAnalyzer()  # Requires the already installed local resource; never downloads.
    scored = []
    for row in rows:
        values = analyzer.polarity_scores(row['text'])
        assert all(math.isfinite(v) for v in values.values())
        assert -1 <= values['compound'] <= 1
        sign = 'negative_score' if values['compound'] < 0 else 'positive_score' if values['compound'] > 0 else 'zero_score'
        scored.append({'passage_id': row['passage_id'], 'review_id': row['review_id'],
            'group_id': row['group_id'], 'existing_task_label': row['label'],
            'vader_neg': values['neg'], 'vader_neu': values['neu'],
            'vader_pos': values['pos'], 'vader_compound': values['compound'], 'score_sign': sign})
    values = [row['vader_compound'] for row in scored]
    signs = ['negative_score', 'zero_score', 'positive_score']
    counts = {name: sum(row['score_sign'] == name for row in scored) for name in signs}
    assert sum(counts.values()) == len(rows)
    edges = np.linspace(-1, 1, 41)
    bin_counts, bin_edges = np.histogram(values, bins=edges)
    assert int(bin_counts.sum()) == len(rows)
    lexicon_path = Path(str(nltk.data.find('sentiment/vader_lexicon.zip')))
    summary = {
        'analysis': 'Descriptive VADER sentiment distribution; frozen training passages only',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'n': len(rows),
        'input_path': str(source), 'input_sha256': source_hash,
        'population': '140 selected AI-audited training passages; not the 500-review corpus or all listeners',
        'text_preprocessing': 'Existing passage text unchanged; no additional lowercasing, stopword filtering or token truncation',
        'sign_group_definitions': {'negative_score': 'compound < 0', 'zero_score': 'compound == 0', 'positive_score': 'compound > 0'},
        'sign_counts': counts, 'sign_percentages': {k: 100*v/len(rows) for k,v in counts.items()},
        'compound': {'mean': statistics.mean(values), 'median': statistics.median(values),
                     'minimum': min(values), 'maximum': max(values), 'population_std': statistics.pstdev(values)},
        'histogram': {'bins': 40, 'range': [-1, 1], 'counts': bin_counts.tolist(), 'edges': bin_edges.tolist()},
        'existing_task_label_counts': dict(Counter(r['label'] for r in rows)),
        'versions': {'python': __import__('sys').version.split()[0], 'nltk': nltk.__version__,
                     'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'lexicon_path': str(lexicon_path), 'lexicon_sha256': hashlib.sha256(lexicon_path.read_bytes()).hexdigest(),
        'course_evidence': {'module_2': 'ADS-509-mod2/M2_CleaningAndEDA_Student.ipynb: cells 23-25; VADER compound scores and 40-bin distribution',
                            'module_3': 'ADS-509-mod3/M3_EmbeddingComparison_Student.ipynb: cells 6-7 retain/discuss sent_compound'},
        'scope': {'training_rows_scored': 140, 'validation_rows_scored': 0, 'historical_test_rows_scored': 0,
                  'model_training_performed': False, 'network_calls': False, 'labels_or_splits_changed': False},
        'interpretation_limits': ['Lexicon estimates, not human-validated sentiment labels or task-classifier performance.',
            'Zero score may mean no recognized valence or balanced evidence; it does not prove neutral opinion.',
            'Score-sign counts are numerical summaries, not conventional +/-0.05 VADER sentiment classes.',
            'Short passages, missing context, sarcasm and music-domain language can affect interpretation.',
            'Oldest-first source sampling and selected training membership limit population inference.',
            'Performed after model selection for descriptive reporting; not used to retune or select models.']}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output/'training_vader_scores.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0])); writer.writeheader(); writer.writerows(scored)
    (output/'sentiment_summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios':[1.65,1]})
    fig.suptitle('VADER scores for the frozen training passages', x=.07, ha='left', fontsize=18, fontweight='bold')
    axes[0].hist(values, bins=edges, color='#168477', edgecolor='white', linewidth=.6)
    axes[0].axvline(0, color='#4b5563', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Compound sentiment score (-1 to +1)', ylabel='Passages', xlim=(-1,1),
                title='Compound-score distribution (40 bins)')
    bars = axes[1].bar(['Negative\nscore','Exactly\nzero','Positive\nscore'], [counts[k] for k in signs],
                       color=['#ad6259','#889399','#168477'], width=.64)
    axes[1].set(ylabel='Passages', title='Numerical sign counts')
    axes[1].set_ylim(0, max(counts.values())*1.3)
    for bar, key in zip(bars, signs):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     f'{counts[key]}\n({100*counts[key]/len(rows):.1f}%)', ha='center', va='bottom', fontsize=11)
    fig.text(.07,.025,'n = 140 | training passages only | descriptive lexicon estimates; no human sentiment validation',
             fontsize=10, color='#4b5563')
    fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.2,wspace=.3)
    fig.savefig(output/'training_vader_sentiment.png', dpi=200)
    fig.savefig(output/'training_vader_sentiment.svg')
    plt.close(fig)
    interpretation = f'''# Descriptive sentiment EDA: frozen training passages only

This addition uses the VADER method taught in ADS 509 Module 2, cells 23-25:
`SentimentIntensityAnalyzer().polarity_scores(text)` and a 40-bin compound-score histogram.
Module 3 cells 6-7 carry forward and discuss the `sent_compound` feature. This is
descriptive sentiment EDA alongside the musical-insight classification task.

## Actual result

Population: **140 frozen training passages**, each from a distinct saved review/release group.
Negative compound scores: **{counts['negative_score']} ({100*counts['negative_score']/140:.1f}%)**;
exactly zero: **{counts['zero_score']} ({100*counts['zero_score']/140:.1f}%)**;
positive compound scores: **{counts['positive_score']} ({100*counts['positive_score']/140:.1f}%)**.
Mean compound score: **{statistics.mean(values):.4f}**; median: **{statistics.median(values):.4f}**;
range: **{min(values):.4f} to {max(values):.4f}**.

Read this as the distribution of a lexicon's scores in the selected training subset.
Do not extrapolate it to all 500 reviews, all music reviewers, or all listeners.
Sign groups mean compound < 0, == 0, and > 0; they are not validated sentiment
classes and do not use the optional conventional +/-0.05 categorical thresholds.
Zero can reflect no recognized sentiment or balanced evidence, rather than neutral opinion.
Short excerpts, context loss, sarcasm, and music-specific language may affect scores.

## Scope and preservation

Only the exact training CSV with SHA256 `{source_hash}` is accepted. Its existing
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

`coursework-root/ADS-509/ADS-509-mod2/M2_CleaningAndEDA_Student.ipynb`,
cells 23-25 (zero-based notebook indices), with saved executions 15 and 16.
`coursework-root/ADS-509/ADS-509-mod3/M3_EmbeddingComparison_Student.ipynb`, cells 6-7.

This supplies actual sentiment-distribution evidence alongside the existing text
length/frequency EDA. It is not an instructor approval or a claim that all final
project requirements, presentation requirements, or repository requirements are complete.
'''
    (output/'INTERPRETATION_AND_REPRODUCTION.md').write_text(interpretation)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    files = [p for p in output.iterdir() if p.is_file()]
    (output/'ARTIFACT_SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in files}, indent=2)+'\n')
    print(json.dumps({'output':str(output),'n':len(rows),'counts':counts,'compound':summary['compound'],
                      'input_unchanged':True,'no_model_training':True}, indent=2))


if __name__ == '__main__':
    main()
