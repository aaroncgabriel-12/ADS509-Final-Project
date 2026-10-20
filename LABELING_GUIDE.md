# Music Review Insight Finder — labeling guide, version 1

Question: Can an adapted open-source LLM distinguish specific musical observations from general opinions and background information in music-review passages?

This continues the existing project. It is **not sentiment classification**. Positive and negative observations may receive the same label.

| CSV label | Code | Use when | Invented teaching example |
|---|---|---|---|
| `specific_musical_observation` | A | The passage describes/evaluates an identifiable musical or performance element: voice, instrument, rhythm, arrangement, lyrics, recording, mixing, etc. | “The bass overwhelms the vocals in the chorus.” |
| `general_opinion` | B | It expresses overall liking, dislike, quality, or recommendation without identifying a musical element. | “This is my favorite album.” |
| `background_other` | C | It gives history, biography, personal context, unrelated content, or another statement without musical assessment or overall opinion. | “The band moved to London in 2004.” |

## Decision order

1. If there is a specific musical observation, choose A, even when background or general praise is also present.
2. Otherwise, if there is a general evaluative opinion, choose B.
3. Otherwise choose C.

Merely naming an instrument in a factual personnel list is background, not necessarily musical assessment. “The drummer is Sam” is C; “The drummer keeps a precise, driving rhythm” is A. “Great production” is A because it evaluates a musical/recording element, although it is vague. “Great record” is B.

Read the passage, not the rating. Do not infer labels from stars, keywords, model predictions, or the artist's reputation. Keep negation, slang, and typos in the source text. Label the text's claim; do not treat historical statements as verified facts.

## Editing `annotation.csv`

- Edit only `label`, `status`, and `notes`. Leave source text, IDs, and provenance intact.
- Set `status` to `done` after assigning one exact label above.
- For genuinely unresolvable, corrupted, non-English, or context-dependent passages, set `status` to `skip` and write a reason in `notes`. Do not force a label to balance classes.
- Keep unfinished rows as `todo`. Import stops while any remain.
- Keep all rows; never delete rows to hide missing labels.
- Record difficult decisions in `notes`. Recheck a small sample later for your own consistency. This does not substitute for independent annotator agreement.

Sentence splitting is approximate: abbreviations, quotations, and shortened fragments may need an explicit skip. Excluding unclear passages can make the task easier than real-world use; report counts and reasons.

The defaults of up to 300 passages, at most three passages per review, and minimum class coverage are proposed workflow settings, not instructor requirements. If rare classes lack independent reviews, collect/label more examples and document the sampling change; never fabricate labels.
