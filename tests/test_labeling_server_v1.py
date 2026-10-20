"""Only invented fixtures: no real labels or evaluation passages are inspected."""
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from init_assisted_session import initialize
from labeling_server_v1 import Session, Conflict, FIELDS, LABELS, csv_data, sha, read_json

class LabelingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='label_ui_fixture_')
        self.project=Path(self.temp.name)
        self.source=self.project/'source.csv'
        rows=[]
        for i in range(180):
            row={field:'' for field in FIELDS}
            row.update(passage_id='fixture_%03d'%i,review_id='review_%d'%i,entity_id='entity_%d'%i,
                group_id='entity:fixture_%d'%i,text='INVENTED FIXTURE passage number %d, never project data.'%i,
                word_count='10',status='todo',license_id='fixture')
            rows.append(row)
        self.source.write_bytes(csv_data(rows));self.source_hash=sha(self.source.read_bytes())
        self.folder=initialize(self.project,self.source,exposed=set(),fixture=True)
        self.session=Session(self.folder,project=self.project)
        for name in ('music_review_pipeline.py','music_review_training.py','LABELING_GUIDE.md'):
            (self.project/name).write_text('# Invented test bundle member\n')
    def tearDown(self): self.temp.cleanup()

    def payload(self,pid,label=LABELS[0],status='done',notes='',action='manual',revision=None):
        return dict(passage_id=pid,label=label,status=status,notes=notes,action=action,
            revision=self.session.state()['revision'] if revision is None else revision)

    def add_draft(self,pid):
        row=self.session.by_id[pid]
        draft={pid:{'suggestion_id':'INVENTED_TEST_DRAFT','text_sha256':sha(row['text'].encode()),
                    'suggested_label':LABELS[0],'reason':'Invented test suggestion; not real judgment.'}}
        (self.folder/'training_suggestions.json').write_text(json.dumps(draft))

    def test_save_resume_and_conflict(self):
        pid=self.session.view('train')['rows'][0]['passage_id'];self.add_draft(pid)
        self.session.label(self.payload(pid,action='accept_ai'))
        resumed=Session(self.folder,project=self.project)
        self.assertEqual(resumed.state()['annotations'][pid]['label'],LABELS[0])
        self.assertEqual(resumed.state()['annotations'][pid]['provenance'],'human_accepted_ai_training_suggestion')
        with self.assertRaises(Conflict):self.session.label(self.payload(pid,revision=0))
        self.assertEqual(sha(self.source.read_bytes()),self.source_hash)

    def test_evaluation_hints_and_acceptance_rejected(self):
        training=self.session.view('train')['rows'][0]['passage_id'];self.add_draft(training)
        for split in ('validation','test'):
            rows=self.session.view(split)['rows'];self.assertTrue(all(r['suggestion'] is None for r in rows))
            with self.assertRaises(ValueError):self.session.label(self.payload(rows[0]['passage_id'],action='accept_ai'))
        evaluation=self.session.view('test')['rows'][0]['passage_id'];self.add_draft(evaluation)
        with self.assertRaisesRegex(ValueError,'non-training'):self.session.suggestions()

    def test_skip_requires_reason_and_export_requires_review(self):
        pid=self.session.view('train')['rows'][0]['passage_id']
        with self.assertRaises(ValueError):self.session.label(self.payload(pid,label='',status='skip'))
        self.session.label(self.payload(pid,label='',status='skip',notes='INVENTED fixture lacks context'))
        with self.assertRaisesRegex(ValueError,'human review'):self.session.export(self.session.state()['revision'])

    def test_export_uses_frozen_splits_and_expansion_never_moves_groups(self):
        assignment_before=json.dumps(self.session.manifest['membership'],sort_keys=True)
        for row in self.session.active(self.session.state()):
            label=LABELS[int(row['passage_id'].split('_')[1])%3]
            self.session.label(self.payload(row['passage_id'],label=label))
        exported=self.session.export(self.session.state()['revision'])
        split_dir=Path(exported['bundle']).parent/'splits'
        for name in ('train','validation','test'):
            with (split_dir/(name+'.csv')).open(encoding='utf-8-sig',newline='') as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(self.session.membership[row['passage_id']]['split'],name)
        self.assertTrue(Path(exported['bundle']).exists())
        self.session.expand(self.session.state()['revision'])
        self.assertEqual(len(self.session.active(self.session.state())),180)
        self.assertEqual(json.dumps(self.session.manifest['membership'],sort_keys=True),assignment_before)
        self.assertEqual(sha(self.source.read_bytes()),self.source_hash)

if __name__=='__main__':unittest.main()
