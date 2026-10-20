"""Render the two actual saved B errors without model inference or network access."""
from pathlib import Path
import argparse
import json
import os
import tempfile
import textwrap
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="music_error_figure_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_figure"))
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
selected = json.loads((ROOT / "presentation_evidence/SAVED_B_ERROR_EXAMPLES.json").read_text())["examples"]
assert len(selected) == 2 and all(row["label"] != row["predicted_label"] for row in selected)
STAGE = args.output_dir
STAGE.mkdir(parents=True, exist_ok=False)

labels = {'specific_musical_observation': 'Specific musical observation',
          'general_opinion': 'General opinion', 'background_other': 'Background / other'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
fig = plt.figure(figsize=(16, 9), facecolor='#f5f8f7')
fig.text(.055, .94, 'Two saved validation errors from experiment B', fontsize=25, fontweight='bold', color='#123e39')
fig.text(.055, .895, 'Selected epoch 2 | 19 / 33 correct | 14 errors | these are two selected examples', fontsize=14, color='#475569')
for i, row in enumerate(selected):
    left = .045 + i * .475
    card = FancyBboxPatch((left, .15), .455, .69, boxstyle='round,pad=0.012,rounding_size=0.01',
                          transform=fig.transFigure, facecolor='white', edgecolor='#c8d8d2')
    fig.patches.append(card)
    x = left + .022
    fig.text(x, .79, f'Example {i+1}', fontsize=13, fontweight='bold', color='#168477')
    fig.text(x, .735, '\u201c' + row['text'] + '\u201d', fontsize=18, color='#123e39')
    fig.text(x, .645, 'AI-audited reference', fontsize=12, color='#64748b')
    fig.text(x, .61, labels[row['label']], fontsize=16, fontweight='bold', color='#123e39')
    fig.text(x, .55, 'Saved model prediction', fontsize=12, color='#64748b')
    fig.text(x, .515, labels[row['predicted_label']], fontsize=16, fontweight='bold', color='#9a4f43')
    fig.text(x, .45, 'Saved AI-audit ambiguity: ' + row['ambiguity'], fontsize=13)
    reason = '\n'.join(textwrap.wrap('Saved audit reason: ' + row['audit_reason'], width=62))
    fig.text(x, .40, reason, fontsize=12, linespacing=1.4, va='top', color='#334155')
    fig.text(x, .275, 'Passage ID: ' + row['passage_id'], fontsize=10)
    fig.text(x, .245, 'Review ID: ' + row['review_id'], fontsize=10)
    fig.text(x, .215, row['author'] + ' | ' + row['source_url'], fontsize=10, color='#475569')
    fig.text(x, .185, 'Saved license metadata: ' + row['license_id'], fontsize=10, color='#475569')
fig.text(.055, .095, 'New rendering from saved outputs; not a raw screenshot. References are AI-assisted, not independent human gold.', fontsize=12, color='#475569')
fig.text(.055, .06, 'Quotes unchanged. Audit reasons and ambiguity flags are preserved from the saved CSV; they are not new human adjudications.', fontsize=11, color='#475569')
fig.savefig(STAGE / '05_B_saved_validation_errors_two_examples.png', dpi=150)
plt.close(fig)
print(STAGE / "05_B_saved_validation_errors_two_examples.png")
