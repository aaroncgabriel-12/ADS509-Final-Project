"""Verify frozen inputs and saved expansion results offline; never load a model."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'expanded_evaluation'
LABELS = ['specific_musical_observation', 'general_opinion', 'background_other']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def check():
    import nbformat
    provenance = json.loads((ROOT / 'packaging/EXPANSION_INTEGRATION.json').read_text())
    for name, digest in provenance['existing_notebook_sha256'].items():
        assert sha(ROOT / name) == digest, name
    notebook = ROOT / 'notebooks/11_music_review_EXPANDED_B_RESULTS_READABLE.ipynb'
    assert sha(notebook) == provenance['expanded_notebook_sha256']
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert len(code) == 7
    assert [cell.execution_count for cell in code] == list(range(1, 8))
    for cell in code:
        ast.parse(cell.source)
        assert all(out.output_type != 'error' for out in cell.outputs)
    assert provenance['expanded_notebook_code_outputs_ids_metadata_preserved'] is True
    settings_source = '\n'.join(c.source for c in code)
    helpers = None
    freeze_sha = None
    for cell in code:
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Assign):
                names = [n.id for n in node.targets if isinstance(n, ast.Name)]
                if 'EXPECTED_HELPERS' in names:
                    helpers = ast.literal_eval(node.value)
                if 'EXPECTED_FREEZE_SHA256' in names:
                    freeze_sha = ast.literal_eval(node.value)
    assert helpers and len(helpers) == 5
    assert sha(EXP / 'frozen_data/frozen_manifest.json') == freeze_sha
    package_files = provenance['expanded_package_sha256']
    for name, digest in package_files.items():
        path = EXP / 'inputs' / name
        assert sha(path) == digest
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            hashes = json.loads(z.read('package_sha256.json'))
            assert set(hashes) == set(z.namelist()) - {'package_sha256.json'}
            for member, want in hashes.items():
                assert hashlib.sha256(z.read(member)).hexdigest() == want
            for member, want in helpers.items():
                assert hashlib.sha256(z.read(member)).hexdigest() == want
                assert sha(EXP / 'helpers' / member) == want
            assert hashlib.sha256(z.read('frozen_manifest.json')).hexdigest() == freeze_sha
            receipt = json.loads(z.read('package_receipt.json'))
            for split in receipt['splits']:
                assert z.read('data/' + split + '.csv') == (EXP / 'frozen_data' / (split + '.csv')).read_bytes()
            if receipt['stage'] == 'train_validation':
                assert 'data/test.csv' not in z.namelist()
            else:
                assert receipt['stage'] == 'test_after_selection'
                assert 'data/train.csv' not in z.namelist()
    parts = {s: read_rows(EXP / 'frozen_data' / (s + '.csv')) for s in ['train', 'validation', 'test']}
    assert [len(parts[s]) for s in parts] == [400, 200, 200]
    groups = {s: {r['group_id'] for r in rows} for s, rows in parts.items()}
    assert [len(groups[s]) for s in parts] == [180, 99, 99]
    for field in ['passage_id', 'review_id', 'group_id']:
        sets = [{r[field] for r in rows} for rows in parts.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)), field
    old = read_rows(EXP / 'frozen_data/old_training_140_unchanged.csv')
    assert len(old) == 140
    assert sha(EXP / 'frozen_data/old_training_140_unchanged.csv') == sha(ROOT / 'music_review_runs/2026-10-16_music_insight_AI_audit_v1/audited_splits/ai_audited_train.csv')
    train = {r['passage_id']: r for r in parts['train']}
    assert all(all(train[r['passage_id']][key] == value for key, value in r.items()) for r in old)
    assert all(r['license_id'] == 'CC BY-NC-SA 3.0' for rows in parts.values() for r in rows)
    comparison = json.loads((EXP / 'results/one_time_test_comparison/comparison.json').read_text())
    assert comparison['freeze_sha256'] == freeze_sha
    assert comparison['test_used_for_selection'] is False
    assert comparison['selected_checkpoint'] == 'adapter_epoch_3'
    assert comparison['independent_human_evaluation'] is False
    assert comparison['test_groups'] == 99
    assert sha(EXP / 'frozen_data/test.csv') == comparison['test_csv_sha256']
    assert sha(EXP / 'results/selection_receipt.json') == comparison['selection_receipt_sha256']
    preflight = next((EXP / 'results').glob('preflight_*/preflight.json'))
    assert sha(preflight) == comparison['preflight_sha256']
    test = {r['passage_id']: r for r in parts['test']}
    saved_metrics = {}
    for model in ['unadapted', 'adapted']:
        rows = read_rows(EXP / 'results/one_time_test_comparison' / (model + '_test_predictions.csv'))
        assert len(rows) == 200 and {r['passage_id'] for r in rows} == set(test)
        matrix = [[0] * 3 for _ in range(3)]
        ce = 0.0
        for row in rows:
            assert all(row[k] == v for k, v in test[row['passage_id']].items())
            i, j = LABELS.index(row['label']), LABELS.index(row['predicted_label'])
            matrix[i][j] += 1
            scores = [float(row['restricted_score_' + letter]) for letter in 'ABC']
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in scores)
            assert math.isclose(sum(scores), 1, abs_tol=1e-5)
            ce -= math.log(max(scores[i], 1e-300))
        m = comparison[model]['metrics']
        assert matrix == m['confusion_matrix']
        assert sum(matrix[i][i] for i in range(3)) / 200 == m['accuracy']
        f1s = []
        for i, label in enumerate(LABELS):
            support, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
            precision = matrix[i][i] / predicted if predicted else 0.0
            recall = matrix[i][i] / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1s.append(f1)
            assert m['per_class'][label]['support'] == support
            assert m['prediction_counts'][label] == predicted
            for key, value in [('precision', precision), ('recall', recall), ('f1', f1)]:
                assert math.isclose(value, m['per_class'][label][key], abs_tol=1e-12)
        assert math.isclose(sum(f1s) / 3, m['macro_f1'], abs_tol=1e-12)
        # CSV probabilities are rounded, so CE uses a small tolerance.
        assert math.isclose(ce / 200, comparison[model]['unweighted_ce'], abs_tol=1e-5)
        saved_metrics[model] = {'accuracy': m['accuracy'], 'macro_f1': m['macro_f1']}
    evidence = ROOT / 'presentation_evidence/expanded_evaluation'
    hashes = json.loads((evidence / 'EVIDENCE_SHA256.json').read_text())
    for name, want in hashes.items():
        assert sha(evidence / name) == want, name
    # Original source paths remain provenance. Current portable paths are in the guide.
    source_map = {'comparison_json': EXP / 'results/one_time_test_comparison/comparison.json',
                  'verified_summary_json': EXP / 'VERIFIED_RESULTS_SUMMARY.json',
                  'adapted_predictions_csv': EXP / 'results/one_time_test_comparison/adapted_test_predictions.csv',
                  'unadapted_predictions_csv': EXP / 'results/one_time_test_comparison/unadapted_test_predictions.csv',
                  'frozen_test_csv': EXP / 'frozen_data/test.csv',
                  'selection_receipt_json': EXP / 'results/selection_receipt.json'}
    ep = json.loads((evidence / 'presentation_metrics_and_provenance.json').read_text())
    for name, path in source_map.items():
        assert sha(path) == ep['source_files'][name]['sha256']
    links = []
    for doc in [ROOT / 'README.md', EXP / 'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (doc.parent / target.split('#')[0]).exists(), (str(doc), target)
                links.append(target)
    return {'status': 'PASS', 'network_calls': False, 'model_rerun': False,
            'prototype_notebooks_match_current_integrity_records': len(provenance['existing_notebook_sha256']),
            'expanded_notebook_schema_and_execution_record': 'PASS: seven code cells, counts 1–7, no error outputs',
            'input_archives_and_helper_hashes': len(package_files),
            'split_passages': {s: len(rows) for s, rows in parts.items()},
            'split_groups': {s: len(g) for s, g in groups.items()},
            'original_training_rows_preserved': 140, 'saved_test_metrics': saved_metrics,
            'evidence_hashes_verified': len(hashes), 'relative_links_checked': len(links)}

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
