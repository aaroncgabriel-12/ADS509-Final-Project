"""Plain-Python configuration, reference checks and safe output handling for C."""
import csv
import hashlib
import io
import json
import math
import traceback
import zipfile
from pathlib import Path
import c_contract as contract

C_NAME = 'C_train_weighted_ABC_lr1e4'
C_LEARNING_RATE = 1e-4
EXPECTED_CSV_SHA256 = {
    'train': '3500dcd7e53b39a09fd5f78cd28eee94e9cb9ebc1d3e5e56d7fa69e784841096',
    'validation': '61b79776c0471b46039be0367998d71031c62c45eb25a7d93c0ca3adbd0ef088',
}
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
B_NAME = 'B_train_weighted_ABC'


def check_c_request(name, learning_rate, manifest):
    contract.require(name == C_NAME, 'This package permits exactly one C experiment.')
    contract.require(learning_rate == C_LEARNING_RATE, 'C changes only the learning rate to 1e-4.')
    contract.require(manifest['csv_sha256'] == EXPECTED_CSV_SHA256, 'Use the unchanged notebook 07 inputs.')
    contract.require(manifest['rows'] == {'train': 140, 'validation': 33}, 'Changed input sizes.')
    contract.require(manifest['seed'] == 1509509, 'Changed initialization seed.')


def create_output(work_dir):
    """Exclusive directory creation prevents accidentally repeating C in one work folder."""
    output = Path(work_dir) / 'C_results'
    output.mkdir(exist_ok=False)
    return output


def validate_metrics(metrics):
    """Check a real saved report against its confusion matrix, without reading examples."""
    contract.require(metrics['label_order'] == list(contract.LABELS), 'Unexpected B label mapping.')
    matrix = metrics['confusion_matrix']
    contract.require(len(matrix) == 3 and all(len(r) == 3 for r in matrix), 'Invalid confusion matrix.')
    contract.require(all(type(x) is int and x >= 0 for r in matrix for x in r), 'Invalid counts.')
    contract.require(sum(sum(r) for r in matrix) == 33, 'B must contain exactly 33 validation decisions.')
    actual = []; guesses = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            actual.extend([contract.LABELS[i]] * count)
            guesses.extend([contract.LABELS[j]] * count)
    contract.require(len(actual) == 33, 'B must use the same 33 validation passages.')
    recomputed = contract.classification_metrics(actual, guesses)
    for key in ['n', 'per_class', 'prediction_counts', 'confusion_matrix', 'label_order']:
        contract.require(metrics[key] == recomputed[key], 'Inconsistent B metrics: ' + key)
    for key in ['accuracy', 'macro_f1']:
        contract.require(math.isclose(metrics[key], recomputed[key], abs_tol=1e-12), 'Inconsistent B ' + key)


def _reference(settings_blob, summary_blob, source):
    settings = json.loads(settings_blob); summary = json.loads(summary_blob)
    contract.require(settings['experiment'] == summary['experiment'] == B_NAME, 'Not experiment B.')
    contract.require(summary['status'] == 'completed_validation_only', 'B did not finish.')
    contract.require(summary['frozen_base_verified'] is True, 'B frozen-base check missing.')
    contract.require(summary['base_sha256_before'] == summary['base_sha256_after'], 'B base changed.')
    contract.require(settings['base_sha256_before'] == summary['base_sha256_before'], 'B base metadata differs.')
    contract.require(settings['adapter_initial_sha256'] == summary['adapter_initial_sha256'], 'B adapter metadata differs.')
    contract.require(settings['learning_rate'] == 2e-4, 'B must have learning rate 2e-4.')
    contract.require(settings['manifest']['csv_sha256'] == EXPECTED_CSV_SHA256, 'B used different data.')
    contract.require(summary['selected']['epoch'] in [1, 2, 3], 'Invalid B selected epoch.')
    validate_metrics(summary['selected']['validation'])
    return {'settings': settings, 'summary': summary, 'source': source,
            'settings_sha256': hashlib.sha256(settings_blob).hexdigest(),
            'summary_sha256': hashlib.sha256(summary_blob).hexdigest()}


def read_b_directory(directory):
    directory = Path(directory)
    paths = [directory / name for name in ['settings.json', 'experiment_summary.json']]
    contract.require(all(p.is_file() and not p.is_symlink() and p.stat().st_size < 2_000_000 for p in paths),
                     'Missing or oversized B metadata. Preserve/download the completed B archive first.')
    return _reference(*(p.read_bytes() for p in paths), source=str(directory))


def read_b_zip(blob, source='user-supplied completed B results ZIP'):
    """Read just two JSON members. No extraction, code execution or test-file access."""
    names = [B_NAME + '/' + n for n in ['settings.json', 'experiment_summary.json']]
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in names:
            matches = [i for i in archive.infolist() if i.filename == name]
            contract.require(len(matches) == 1 and matches[0].file_size < 2_000_000, 'Missing/duplicate/oversized B JSON.')
        return _reference(*(archive.read(n) for n in names), source=source)


def verify_initialization(reference, base_sha256, adapter_sha256):
    b = reference['summary']
    contract.require(base_sha256 == b['base_sha256_before'], 'C/B initial base weights differ; stop.')
    contract.require(adapter_sha256 == b['adapter_initial_sha256'], 'C/B initial adapters differ; stop.')


def verify_controlled_settings(b, c):
    contract.require(b['learning_rate'] == 2e-4 and c['learning_rate'] == C_LEARNING_RATE, 'Wrong learning-rate comparison.')
    contract.require(c['model'] == MODEL_ID and c['revision'] == MODEL_REVISION, 'Changed base model.')
    contract.require(set(b) == set(c), 'Settings schema differs from B; inspect before training.')
    for key in set(b) - {'experiment', 'learning_rate'}:
        contract.require(b[key] == c[key], 'C changed more than learning rate: ' + key)


def compare_prediction_files(original, reloaded, score_tolerance=0.002):
    """Require the saved checkpoint to reproduce every saved validation label."""
    def rows(path):
        with Path(path).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    a = rows(original); b = rows(reloaded)
    contract.require(len(a) == len(b) and bool(a), 'Reload row count differs.')
    largest = 0.0
    for x, y in zip(a, b):
        for key in ['passage_id', 'text', 'label', 'predicted_label']:
            contract.require(x[key] == y[key], 'Reload predictions/row alignment differ: ' + key)
        for code in contract.CODES:
            u, v = float(x['restricted_score_' + code]), float(y['restricted_score_' + code])
            contract.require(math.isfinite(u) and math.isfinite(v), 'Nonfinite reload probability.')
            largest = max(largest, abs(u-v))
    contract.require(largest <= score_tolerance, 'Reload probabilities differ beyond FP16 tolerance.')
    return {'status': 'PASS', 'rows': len(a), 'all_predicted_labels_identical': True,
            'largest_probability_difference': largest, 'probability_atol': score_tolerance}


def save_errors(predictions, output):
    with Path(predictions).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    errors = [r for r in rows if r['label'] != r['predicted_label']]
    with Path(output).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(errors)
    return len(errors)


def save_failure(output, stage, error):
    """Keep partial artifacts and traceback; never label a failed run completed."""
    path = Path(output) / ('failure_' + stage + '.json')
    if not path.exists():
        contract.save_json(path, {'status': 'FAILED', 'stage': stage,
            'error_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})


def compare_with_b(reference, c_summary):
    contract.require(c_summary['post_training_diagnostics_complete'] is True, 'Finish C diagnostics first.')
    contract.require(c_summary['b_initialization_verified'] is True, 'B/C initialization comparison unavailable.')
    b = reference['summary']['selected']['validation']; c = c_summary['selected']['validation']
    return {'status': 'actual_validation_results_only', 'b_reference': reference,
            'B': b, 'C': c, 'delta_accuracy': c['accuracy']-b['accuracy'],
            'delta_macro_f1': c['macro_f1']-b['macro_f1'],
            'independent_human_evaluation': False, 'confirmatory_final_comparison_performed': False}


def export_results(output, work_dir):
    output = Path(output); work_dir = Path(work_dir)
    contract.require(output.is_dir() and output.parent == work_dir, 'Unexpected C output directory.')
    completion = output / 'completion.json'
    completed = completion.exists() and json.loads(completion.read_text()).get('status') == 'C_completed_with_reload_checks'
    state = 'COMPLETE' if completed else 'INCOMPLETE'
    archive_path = work_dir / ('music_validation_only_C_lr1e4_' + state + '_results.zip')
    members = sorted(p for p in output.rglob('*') if p.is_file())
    contract.require(bool(members) and all(not p.is_symlink() for p in members), 'Empty/symlink output rejected.')
    checksums = {str(p.relative_to(output)): contract.sha(p) for p in members}
    # Exclusive ZIP mode: running the export cell again cannot overwrite an archive.
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in members: z.write(p, str(p.relative_to(output)))
        z.writestr('ARCHIVE_SHA256.json', json.dumps(checksums, indent=2))
    return archive_path, state
