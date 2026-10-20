"""Validate the separate AI-audited dataset without network or ML dependencies."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import music_review_pipeline as pipeline


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_audited_splits(root, splits):
    root = Path(root)
    manifest = json.loads((root / 'splits/manifest.json').read_text())
    audit = json.loads((root / 'audit_manifest.json').read_text())
    frozen = json.loads((root / 'provenance/frozen_session_manifest.json').read_text())
    source = json.loads((root / 'provenance/selected_rows_original.json').read_text())
    rows = pipeline.load_csv(root / 'audited_annotations.csv')
    require(manifest['schema'] == 'music-review-ai-audited-training-v1', 'Wrong bundle schema.')
    require(audit['schema'] == 'music-review-insight-ai-audit-v1', 'Wrong audit schema.')
    for key, path in [('audit_manifest_sha256', 'audit_manifest.json'),
                      ('frozen_session_manifest_sha256', 'provenance/frozen_session_manifest.json'),
                      ('selected_source_sha256', 'provenance/selected_rows_original.json'),
                      ('audited_annotations_sha256', 'audited_annotations.csv')]:
        require(manifest[key] == file_sha(root / path), 'Changed provenance file: ' + path)
    require(audit['source_session_manifest_sha256'] == manifest['frozen_session_manifest_sha256'],
            'Original split freeze fingerprint differs.')
    require(manifest['independent_human_evaluation'] is False, 'Do not claim independent evaluation.')
    require('NO re-splitting' in manifest['grouping'], 'Missing fixed-membership declaration.')
    require(manifest['seed'] == frozen['seed'], 'Original seed changed.')
    require(set(manifest['labeling_method']) == {'train', 'validation', 'test'}, 'Missing split methods.')
    require(all('AI-assisted' in value and 'NOT independent human' in value
                for value in manifest['labeling_method'].values()), 'Misstated labeling methods.')
    require(len(rows) == len(source) == 229, 'Expected all 229 selected passages, including skips.')
    by_id = {row['passage_id']: row for row in rows}
    originals = {row['passage_id']: row for row in source}
    selected = {pid for pid, entry in frozen['membership'].items() if entry['candidate']}
    require(len(by_id) == len(rows) and set(by_id) == set(originals) == selected,
            'Selected IDs changed, duplicated, or deleted.')
    for pid, row in by_id.items():
        original = originals[pid]
        member = frozen['membership'][pid]
        require(row['split'] == member['split'] and row['group_id'] == member['group_id'],
                'Frozen membership changed: ' + pid)
        require(audit['membership'][pid] == {'split': row['split'], 'group_id': row['group_id']},
                'Audit membership differs: ' + pid)
        for field in pipeline.PASSAGE_FIELDS:
            require(row[field] == str(original[field]), 'Source field changed: ' + pid + '/' + field)
        for field in ('label', 'status', 'notes'):
            require(row['original_' + field] == original['annotation'][field], 'Original choice changed: ' + pid)
        require('AI-assisted' in row['label_provenance'] and 'NOT independent human' in row['label_provenance'],
                'Missing AI label provenance: ' + pid)
        require(row['independently_human_labeled'] == row['human_accepted_audit'] == 'False',
                'Unverified independent/human-acceptance claim: ' + pid)
        require(row['status'] in ('done', 'skip'), 'Unexpected unfinished/invalid status.')
        if row['status'] == 'skip':
            require(row['label'] == '' and bool(row['notes'].strip()), 'Skipped row needs blank label and reason.')
        else:
            require(row['label'] in pipeline.LABELS, 'Unknown label.')
        require(row['audit_reason'] and row['ambiguity'] in ('low', 'medium', 'high'), 'Missing audit decision.')
    for name, part in splits.items():
        expected = [row for row in rows if row['split'] == name and row['status'] == 'done']
        require(part == expected, 'Split is not the exact selected audited subset: ' + name)
        require(pipeline.digest(part) == manifest['split_sha256'][name], 'Changed split data: ' + name)
        require(len(part) == manifest['rows'][name], 'Wrong split count: ' + name)
        require(dict(Counter(r['label'] for r in part)) == manifest['class_counts'][name], 'Class-count mismatch.')
        require(set(r['label'] for r in part) == set(pipeline.LABELS), 'A category is missing; do not fabricate coverage.')
    require({name: len(part) for name, part in splits.items()} == {'train': 140, 'validation': 33, 'test': 35},
            'Expected the fixed audited counts 140/33/35.')
    require(sum(row['status'] == 'skip' for row in rows) == manifest['skipped_passages'] == 21, 'Skip-count mismatch.')
    pipeline.check_splits(splits)
    return manifest


if __name__ == '__main__':
    import sys
    folder = Path(sys.argv[1])
    parts = {name: pipeline.load_csv(folder / 'splits' / (name + '.csv'))
             for name in ('train', 'validation', 'test')}
    data = validate_audited_splits(folder, parts)
    print('PASS: AI-assisted provenance, original source, fixed groups, hashes, and counts', data['rows'])
