"""Freeze group membership and pilot selection before creating any AI hints."""
import csv
import hashlib
import io
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'music_review_runs/2026-10-15_music_insight_v1/annotation.csv'
SEED = 1509509
# Already discussed in chat, or shown in the initial EDA preview: never evaluation.
EXPOSED = {
 'a0f1a7f4450757d32312a4fa', 'c492dd7d20c783ca9e5e4029', 'c89262a181103c5a7d677fb7',
 '0f77abd5b7541a9ffaeb1a61', '0cd841d26cefc074072692a7', '7988fa423942a101d40f1427',
 'e73acf8bf3ba2a36ff38d5fb', '263a1c1e9b0d46552123e021', '28775b162fdbeedfae1cccc0',
 'f12beae1eb6dab6924b34f19', 'ef7dadd2b3920f3e9eed2722', '72b1b433c2b4f21d8c4dbd66',
 '6722f4d60df2a700162be9b2', '684b95c6f71f456dc5030b11',
}

def h(data): return hashlib.sha256(data).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def initialize(project=PROJECT, source=SOURCE, seed=SEED, exposed=EXPOSED, fixture=False):
    project, source = Path(project), Path(source)
    raw = source.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline='')))
    assert len({r['passage_id'] for r in rows}) == len(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row['group_id'], []).append(row)
    forced_ids = exposed | {r['passage_id'] for r in rows if r['status'] != 'todo' or r['label'] or r['notes']}
    forced_groups = sorted({r['group_id'] for r in rows if r['passage_id'] in forced_ids})
    other_groups = sorted(set(groups) - set(forced_groups))
    random.Random(seed).shuffle(other_groups)
    assert len(groups) >= 120 and len(forced_groups) <= 80, 'Need >=120 groups and <=80 previously reviewed/exposed groups.'
    order = forced_groups + other_groups
    group_split = {}
    for index, group in enumerate(order):
        if index < 80: split = 'train'
        elif index < 100: split = 'validation'
        elif index < 120: split = 'test'
        else: split = ['train', 'train', 'train', 'train', 'validation', 'test'][(index - 120) % 6]
        group_split[group] = split
    candidates = {}
    for group, members in groups.items():
        known = [r['passage_id'] for r in members if r['passage_id'] in forced_ids]
        candidates[group] = known if known else [members[0]['passage_id']]
    name = '2026-10-15_music_insight_assisted_v1'
    suffix = 2
    while (project / 'music_review_runs' / name).exists():
        name = '2026-10-15_music_insight_assisted_v' + str(suffix); suffix += 1
    session = project / 'music_review_runs' / name
    session.mkdir(parents=True, exist_ok=False)
    (session / 'source_annotation_snapshot.csv').write_bytes(raw)
    dump(session / 'reference.json', rows)
    membership = {r['passage_id']: {'group_id': r['group_id'], 'split': group_split[r['group_id']],
                  'candidate': r['passage_id'] in candidates[r['group_id']]} for r in rows}
    manifest = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed,
       'source_path': str(source), 'source_sha256': h(raw), 'reference_sha256': h((session/'reference.json').read_bytes()),
       'group_order': order, 'group_split': group_split, 'membership': membership,
       'initial_active_group_count': 120, 'expansion_group_step': 60,
       'previously_exposed_training_ids': sorted(set(forced_ids) & set(membership)),
       'labeling_method': {'train': 'AI suggestions, individually accepted/corrected by a human; provenance recorded',
                          'validation': 'Independent human labels; no AI suggestions',
                          'test': 'Independent human labels; no AI suggestions'},
       'scope': 'specific musical observation / general opinion / background-other',
       'fixture': fixture}
    dump(session / 'session_manifest.json', manifest)
    annotations = {}
    for row in rows:
        if row['status'] != 'todo' or row['label'] or row['notes']:
            annotations[row['passage_id']] = {'label': row['label'], 'status': row['status'], 'notes': row['notes'],
                'provenance': 'prior_human_chat_review_training_only', 'suggestion_id': None}
    dump(session / 'state.json', {'revision': 0, 'active_group_count': 120, 'annotations': annotations, 'audit': []})
    dump(session / 'training_suggestions.json', {})
    (session / 'human_annotations.csv').write_bytes(raw)
    active_groups = set(order[:120])
    selected = [r for r in rows if r['group_id'] in active_groups and membership[r['passage_id']]['candidate']]
    print('SESSION', session)
    print('PILOT_COUNTS', dict(Counter(membership[r['passage_id']]['split'] for r in selected)))
    print('RESERVE_GROUPS', len(order)-120, 'SOURCE_HASH', h(raw))
    return session

if __name__ == '__main__': initialize()
