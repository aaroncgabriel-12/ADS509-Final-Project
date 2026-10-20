"""Check packaged evidence and cached collection without training or network calls."""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile

import nbformat

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def check():
    report = {'model_training': False, 'network_calls': False, 'old_test_evaluation': False}
    manifest_path = ROOT / 'packaging/FILE_SHA256.json'
    if manifest_path.exists():
        hashes = json.loads(manifest_path.read_text())
        for name, value in hashes.items():
            assert sha(ROOT / name) == value, name
        report['file_hashes_verified'] = len(hashes)

    notebook_checks = []
    for path in sorted((ROOT / 'notebooks').rglob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                assert not any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
        notebook_checks.append(str(path.relative_to(ROOT)))
    assert len(notebook_checks) == 7
    report['notebooks_valid'] = notebook_checks
    for name in ['02_data_preparation_readable.ipynb', '09_vader_sentiment_readable.ipynb', '10_saved_results_review_readable.ipynb']:
        notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        disclosures = [c for c in notebook.cells if c.cell_type == 'markdown' and c.source.startswith('## Code assistance and responsibility')]
        assert len(disclosures) == (1 if name.startswith('10') else 0)
        if disclosures:
            assert notebook.cells[-1] == disclosures[0]
    report['prototype_final_readable_disclosure_preserved'] = True

    python_files = list(ROOT.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text())
    report['python_files_parse'] = len(python_files)

    changes = json.loads((ROOT / 'packaging/PACKAGING_CHANGES.json').read_text())
    for item in changes['bundle_checks']:
        path = ROOT / item['file']
        assert sha(path) == item['sha256']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            hashes = json.loads(archive.read('bundle_sha256.json'))
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in hashes.items())
    report['immutable_bundles_verified'] = len(changes['bundle_checks'])

    # Cross-check that each preserved notebook expects the bundled archive's hash.
    expected = dict(zip([
        '06_music_review_lora_AI_AUDITED_COLAB.ipynb',
        '07_music_review_VALIDATION_ONLY_AB_COLAB-2.ipynb',
        '08_music_review_C_LOWER_LR_COLAB-5.ipynb'], changes['bundle_checks']))
    source_inventory = {item['file']: item for item in changes['source_files']}
    for name, item in expected.items():
        path = ROOT / 'notebooks/training' / name
        integration = json.loads((ROOT / 'packaging/EXPANSION_INTEGRATION.json').read_text())
        assert sha(path) == integration['existing_notebook_sha256']['notebooks/training/' + name]
        notebook = json.loads(path.read_text())
        values = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECTED_BUNDLE_SHA256' for t in node.targets):
                        values.append(ast.literal_eval(node.value))
        assert item['sha256'] in values, (name, 'expected input archive hash does not match')
    report['prototype_notebooks_match_current_integrity_records_and_bundle_pairing_verified'] = True

    with (ROOT / 'data/ATTRIBUTION.csv').open(newline='', encoding='utf-8') as handle:
        attribution = list(csv.DictReader(handle))
    assert len(attribution) == 500
    assert all(row['license_id'] == 'CC BY-NC-SA 3.0' for row in attribution)
    assert all(all(row[k] for k in ['author', 'source_url', 'review_url', 'license_url']) for row in attribution)
    report['attributed_reviews'] = len(attribution)

    # Check only train/validation metadata. Do not load old-test rows.
    split_dir = ROOT / 'music_review_runs/2026-10-16_music_insight_AI_audit_v1/audited_splits'
    for filename, count in [('ai_audited_train.csv', 140), ('ai_audited_validation.csv', 33)]:
        with (split_dir / filename).open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count
        assert all(row['license_id'] == 'CC BY-NC-SA 3.0' for row in rows)
        assert all(row['independently_human_labeled'].lower() == 'false' for row in rows)
    report['training_validation_metadata_checked'] = [140, 33]

    for name in ['A', 'B', 'C', 'D']:
        summary = json.loads((ROOT / 'results/evidence' / f'{name}_summary.json').read_text())
        result = summary['selected']['validation']
        matrix = result['confusion_matrix']
        assert result['n'] == sum(sum(row) for row in matrix) == 33
        assert sum(matrix[i][i] for i in range(3)) / 33 == result['accuracy']
        assert summary['independent_human_evaluation'] is False
    report['saved_metric_consistency'] = True

    spec = importlib.util.spec_from_file_location('music_review_packaged_pipeline', ROOT / 'music_review_pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError('Cached collection attempted a network request')
    session = ROOT / 'music_review_runs/2026-10-15_music_insight_v1'
    with tempfile.TemporaryDirectory(prefix='music_offline_collection_') as tmp:
        out = Path(tmp)
        shutil.copytree(session / 'raw_pages', out / 'raw_pages')
        reviews = pipeline.collect_reviews(out, max_reviews=500, max_pages=30, page_size=50,
                                           session=NoNetwork(), sleep_fn=lambda seconds: None)
        assert len(reviews) == 500
        assert sha(out / 'reviews.csv') == sha(session / 'reviews.csv')
        passages, exclusions = pipeline.make_passages(reviews, max_words=100, min_words=3)
        assert len(passages) == 7664
        assert exclusions == {'under_minimum_words': 217, 'duplicate_passage': 20}
        assert hashlib.sha256(pipeline.csv_bytes(passages, pipeline.PASSAGE_FIELDS)).hexdigest() == sha(session / 'passages.csv')
    report['offline_cache_collection'] = 'PASS: exact 500-review CSV, zero network calls'
    report['offline_passage_preparation'] = 'PASS: exact 7,664-passage CSV and exclusion counts'
    report['status'] = 'PASS'
    return report

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
