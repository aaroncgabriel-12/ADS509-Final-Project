# Expanded data and assistance

The expanded evaluation uses additional passages from the same attributed CritiqueBrainz collection. The frozen CSVs and prediction files retain record-level source and license metadata. No new source collection or model weights were added to the upload folder. Notebook 11 records expanded-run AI assistance; notebook 10 retains the prototype disclosure.

The existing notices below continue to apply.

# Sources and third-party notices

## Review text and attribution

Source: [CritiqueBrainz](https://critiquebrainz.org/) review API, documented at
[reviews endpoint](https://critiquebrainz.readthedocs.io/api/endpoints.html#reviews).
Saved collection settings: English text reviews of release groups, page size 50,
oldest-first bounded sample, collected October 15, 2026. A release group may be
an album, EP, single or another release type.

All 500 included normalized reviews and all 7,664 passages report **CC BY-NC-SA
3.0** in their saved record metadata. `data/ATTRIBUTION.csv` lists all 500 review
IDs, author names, original source/review links and license links. The CSVs and
frozen bundles retain the record-level attribution. Names/IDs in those records
identify public source authors/reviews, not private accounts or credentials.
No full user profiles are included. Model-prediction labels and figures do not
erase the original source rights.

The license's [official deed and legal-code link](https://creativecommons.org/licenses/by-nc-sa/3.0/)
describe attribution, noncommercial and ShareAlike terms. HTML/whitespace cleanup,
passage splitting, labels and new visual layouts are modifications described in
the notebooks/guide. This metadata inspection is not rights adjudication or
blanket redistribution clearance. Review the terms and any other applicable
rights before public reuse; do not relabel the review dataset with a permissive
code license. No external publication has occurred during packaging.

## Model

Qwen/Qwen2.5-0.5B-Instruct, recorded revision
`7ae557604adf67be50417f59c2c2f167def9a775`.
The [pinned repository LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)
identifies Apache-2.0. No base weights/tokenizer cache or adapter weights are
included in this upload folder. Adapter checkpoints remain in the separate local
result ZIPs; their potential data-related restrictions are not cleared here.

## VADER and dependencies

NLTK VADER is used via `SentimentIntensityAnalyzer`, following the saved Module 2
coursework method. The external lexicon ZIP is not vendored. Obtain it through
NLTK's downloader if absent, and retain its applicable notices. The exact resource
hash used is recorded in `09_training_sentiment_EDA_VADER/results/sentiment_summary.json`.
Python package pins identify third-party libraries; their own licenses continue
to apply. No new license is assigned to those libraries.

## Project code

This package records the existing code and accurate AI-assistance provenance.
It does not invent an author-approved project-wide open-source license. Choose a
code license deliberately if making a public reusable software release, while
keeping third-party text/model notices separate. The consolidated code-assistance
discussion is at the end of readable notebook 10.
