# Evidence by project phase

The [expanded evaluation figures](expanded_evaluation/) compare both models on the same 200 passages from 99 groups. They include accuracy/macro-F1, a named count confusion matrix, and class metrics/prediction counts. All identify the AI-assisted labels with human review. [The expanded guide](../expanded_evaluation/README.md) provides context.

The five figures and captions below document the historical prototype. Keep their 33-passage validation and 140-passage VADER sample labels visible. Do not present them as expanded-test results.

# Presentation evidence: Music Review Insight Finder

Five numbered PNGs are ready to place in the 11-slide outline. These are actual
data/model results. Files 01–04 preserve existing plot bytes; file 05 is a new
rendering from saved prediction rows. None are raw screenshots. No models or old
test data were rerun. The full project/training notebooks remain elsewhere.

## Captions, populations, and slide map

| PNG | Outline slide | Suggested caption / speaking point | Population and source |
|---|---|---|---|
| `01_EDA_7664_passages_lengths_and_term_frequencies.png` | 3: Preparation and exploratory analysis | Prepared passages vary in length; frequent English-stopword-filtered terms describe this corpus. Word counts are EDA, not learned topics or model performance. | 7,664 prepared passages derived from 500 collected reviews. The length plot is passage length, not the length of 500 whole reviews. Existing session `2026-10-15_music_insight_v1/charts/01_lengths_and_frequencies.png`; notebook 02 uses CountVectorizer with English stopwords, max_features=2000, top 15. |
| `02_VADER_140_training_passages_sentiment.png` | 3: Preparation and exploratory analysis | VADER scores lean positive in the selected training subset: 26 below zero, 36 exactly zero, 78 above zero. | Exactly 140 frozen AI-audited training passages; existing `09_training_sentiment_EDA_VADER/results/` PNG, summary, and scores. Not the entire 500-review corpus. |
| `03_A_to_D_33_validation_passages_accuracy_and_macroF1.png` | 7: Controlled experiments A through D | B has the highest observed validation macro F1 (0.4649); D ties B's accuracy (19/33) but has slightly lower macro F1. | Same 33 development-validation passages across A–D; new readable results-review figure from saved aggregate JSON. A: .5455/.3054, B: .5758/.4649, C: .5152/.4230, D: .5758/.4619 (accuracy/macro F1). |
| `04_B_33_validation_passages_confusion_matrix.png` | 8: Best observed result and error patterns | B still overpredicts musical observation: 10 general-opinion/background passages are assigned to that category. | Selected B epoch 2; 33 validation passages, supports 18/7/8. Rows are AI-audited references; columns predictions. Matrix [[15,2,1],[4,2,1],[6,0,2]]. |
| `05_B_saved_validation_errors_two_examples.png` | 8: Best observed result and error patterns | Two actual saved errors show a content-free transition mistaken for musical detail and a short musical comparison mistaken for general opinion. | Two deliberately selected examples among B's 14 errors on 33 validation passages. Exact text, audited labels, predictions, ambiguity, audit reasons, IDs and attribution are retained in `SAVED_B_ERROR_EXAMPLES.json`. |

Slides 1 (question), 2 (collection), 4 (labels/splits), 5 (pretrained model/LoRA),
6 (historical baseline evaluation), 9 (limitations), 10 (GenAI reflection), and
11 (conclusion/handoff) can use the outline's concise text and the original
evidence. This package does not fabricate a historical-baseline image or LoRA
measurement. No illustration is presented as experimental evidence.

## Read the results carefully

- Task classes describe musical content, not positive/negative sentiment.
- All final reference labels received a nonblinded AI audit; they are not
  independent human gold. The saved human-accepted-audit flag is false.
- A–D repeatedly use the same small validation set. The plots support a
  development comparison, not independent generalization or a promised accuracy.
- The original test set was already exposed in run 06. It was not opened or
  rescored while making this package. The .3054 original-adapted reference is
  not the unadapted baseline.
- VADER sign groups use <0, ==0, >0, not the conventional ±.05 thresholds.
  Zero does not prove neutral sentiment. Scores are lexicon estimates affected
  by domain language, context and sarcasm. Added after model selection, the EDA
  was not used to retune the model.
- The review sample is bounded/oldest-first, not representative of all listeners.
- Error examples are selected illustrations. Their saved ambiguity flags and
  audit reasons are AI-audit annotations, not new human findings.

## Attribution and sharing

Error example 1: Colin Buttimer, source `http://www.bbc.co.uk/music/reviews/rjw6`,
CritiqueBrainz review `422a7626-fbe0-4af0-a729-d2243af4ad9d`, passage
`90b5a6bb99708b8221fff138`.

Error example 2: Daniel Pike, source `http://www.bbc.co.uk/music/reviews/2m54`,
CritiqueBrainz review `1ae83e70-f70b-4108-980a-72194089911f`, passage
`1369097bc02852daf847044b`.

Both records carry CC BY-NC-SA 3.0 metadata, with the saved license URL
`https://creativecommons.org/licenses/by-nc-sa/3.0/`. Quotes are unchanged;
plot layout and captions are new. Metadata retention is not license clearance.
Review attribution/ShareAlike/NonCommercial obligations before public reuse.
Delivery here is back to Aaron, not external publication.

## Files and verification

`EVIDENCE_SOURCES_AND_CHECKS.json` maps exact local source paths, archive entries,
SHA256 hashes and verification. `SAVED_B_ERROR_EXAMPLES.json` stores only the two
selected examples with their saved provenance. `ARTIFACT_SHA256.json` lists the
package file hashes. The ZIP uses a single `presentation_evidence/` top folder.
All PNGs were decoded and visually checked; ZIP CRCs were checked. Protected
original files remain unchanged. Only the separately authorized duplicated
Markdown disclosures were removed from readable notebooks 02 and 09; the final
accurate consolidated code-assistance section remains in readable notebook 10.
