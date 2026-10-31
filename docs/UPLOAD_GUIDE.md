# Publication and maintenance

The project is published at
[aaroncgabriel-12/ADS509-Final-Project](https://github.com/aaroncgabriel-12/ADS509-Final-Project).
The earlier first-upload instructions are superseded by the published repository.

Before publishing later changes:

1. Review the diff and the included source-data, model, license, and provenance
   notices. Keep credentials, local environments, weights, runtime caches, and
   private archives outside tracked content.
2. Preserve executed notebooks, frozen data, staged input archives, and saved
   predictions during documentation updates. Historical integrity records
   continue to describe those original artifacts.
3. Update `packaging/FILE_SHA256.json` for intentionally changed tracked files.
   This manifest covers the current package; preparation-time reports remain
   historical evidence and should not imply a new experiment.
4. Run `python scripts/verify_submission.py` and
   `python scripts/verify_expanded_evaluation.py` from the repository root.
   These inspect saved evidence without model training or network calls.
5. Publish the reviewed change, then verify the rendered README, notebook links,
   and figure links on GitHub.

The small prototype and expanded input ZIPs preserve the exact recorded
experiments. Full model checkpoints and bulk result archives are not distributed
here. Original preparation reports record measured package sizes at that time.

Publication is separate from video delivery, instructor access, and Canvas
submission. Those activities are not completed by a documentation update.
