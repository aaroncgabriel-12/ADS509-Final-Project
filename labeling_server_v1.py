"""Local-only, human-reviewed annotation screen. Python standard library only.

No remote model calls. AI drafts are a separate, precomputed training-only file.
The frozen session manifest owns split membership; export never re-splits labels.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import tempfile
import threading
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LABELS = ('specific_musical_observation', 'general_opinion', 'background_other')
SPLITS = ('train', 'validation', 'test')
FIELDS = ['passage_id', 'review_id', 'entity_id', 'group_id', 'text', 'word_count',
          'license_id', 'license_name', 'license_url', 'author', 'source_url',
          'review_url', 'collected_at', 'label', 'status', 'notes']

def now(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def structured_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8'))
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def csv_data(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
    return ('\ufeff' + stream.getvalue()).encode('utf-8')

class Conflict(ValueError): pass

class Session:
    def __init__(self, folder, project=None):
        self.folder = Path(folder).resolve()
        self.project = Path(project).resolve() if project else self.folder.parent.parent
        self.manifest_path = self.folder / 'session_manifest.json'
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        assert sha((self.folder/'reference.json').read_bytes()) == self.manifest['reference_sha256']
        self.reference = read_json(self.folder/'reference.json')
        self.by_id = {row['passage_id']: row for row in self.reference}
        assert len(self.by_id) == len(self.reference)
        self.membership = self.manifest['membership']
        self.lock = threading.RLock()
        self._check_membership()

    def _check_membership(self):
        if self.manifest_path.read_bytes() != self.manifest_bytes:
            raise ValueError('Frozen membership was edited. Stop and inspect the session.')
        assert set(self.membership) == set(self.by_id)
        groups = {}
        reviews = {}
        texts = {}
        for pid, row in self.by_id.items():
            split = self.membership[pid]['split']
            assert split in SPLITS
            for key, table in [(row['group_id'], groups), (row['review_id'], reviews), (row['text'].casefold(), texts)]:
                if key in table and table[key] != split:
                    raise ValueError('Group/review/duplicate text crosses frozen splits.')
                table[key] = split
        for pid in self.manifest['previously_exposed_training_ids']:
            assert self.membership[pid]['split'] == 'train'

    def state(self):
        self._check_membership()
        state = read_json(self.folder/'state.json')
        assert 0 <= state['active_group_count'] <= len(self.manifest['group_order'])
        return state

    def active(self, state):
        groups = set(self.manifest['group_order'][:state['active_group_count']])
        return [row for row in self.reference if self.membership[row['passage_id']]['candidate']
                and row['group_id'] in groups]

    def annotation(self, row, state):
        return state['annotations'].get(row['passage_id'], {'label':'', 'status':'todo', 'notes':'',
            'provenance':'not_reviewed', 'suggestion_id':None})

    def suggestions(self):
        drafts = read_json(self.folder/'training_suggestions.json')
        for pid, draft in drafts.items():
            if pid not in self.membership or self.membership[pid]['split'] != 'train':
                raise ValueError('Suggestion file includes a non-training passage; refusing to load.')
            if draft['text_sha256'] != sha(self.by_id[pid]['text'].encode('utf-8')):
                raise ValueError('Suggestion text fingerprint mismatch.')
            if draft['suggested_label'] not in LABELS + (None,):
                raise ValueError('Unknown suggestion label.')
        return drafts

    def materialized(self, state):
        rows = []
        for row in self.reference:
            a = self.annotation(row, state)
            rows.append(dict(row, **{key:a[key] for key in ('label','status','notes')}))
        return rows

    def commit(self, state):
        # The single atomic state file is authoritative. The CSV is a derived view.
        content = (json.dumps(state, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
        atomic_bytes(self.folder/'state.json', content)
        atomic_bytes(self.folder/'human_annotations.csv', csv_data(self.materialized(state)))

    def view(self, split):
        if split not in SPLITS: raise ValueError('Unknown split.')
        with self.lock:
            state = self.state(); active = self.active(state); drafts = self.suggestions()
            counts = {}
            for name in SPLITS:
                statuses = Counter(self.annotation(row,state)['status'] for row in active
                                   if self.membership[row['passage_id']]['split']==name)
                counts[name] = {key:statuses[key] for key in ('todo','done','skip')}
                counts[name]['total'] = sum(statuses.values())
            rows = []
            for row in active:
                pid = row['passage_id']
                if self.membership[pid]['split'] != split: continue
                rows.append({'passage_id':pid, 'text':row['text'], 'review_id':row['review_id'],
                    'word_count':row['word_count'], 'annotation':self.annotation(row,state),
                    'suggestion':drafts.get(pid) if split=='train' else None})
            return {'session':self.folder.name, 'fixture':self.manifest.get('fixture',False),
                    'revision':state['revision'], 'split':split, 'rows':rows, 'counts':counts,
                    'remaining_groups':len(self.manifest['group_order'])-state['active_group_count'],
                    'labels':LABELS}

    def label(self, payload):
        with self.lock:
            state = self.state()
            if payload.get('revision') != state['revision']:
                raise Conflict('Another save changed this session. Refresh before saving again.')
            pid = payload.get('passage_id')
            if pid not in {row['passage_id'] for row in self.active(state)}:
                raise ValueError('Passage is not in the active pilot.')
            split = self.membership[pid]['split']
            draft = self.suggestions().get(pid) if split=='train' else None
            action = payload.get('action', 'manual')
            if action == 'accept_ai':
                if split != 'train' or draft is None:
                    raise ValueError('AI acceptance is allowed only for training rows with a draft.')
                label = draft['suggested_label'] or ''
                status = 'done' if label else 'skip'
                notes = str(payload.get('notes','')).strip() or ('Insufficient context' if not label else '')
                provenance = 'human_accepted_ai_training_suggestion'
            else:
                label, status = payload.get('label',''), payload.get('status','')
                notes = payload.get('notes','')
                if not isinstance(notes,str) or len(notes)>2000: raise ValueError('Notes must be text, at most 2000 characters.')
                notes = notes.strip()
                provenance = ('human_training_review_with_ai_hint' if draft else 'human_training_review_no_hint') if split=='train' else 'independent_human_no_ai_hint'
            if status not in ('todo','done','skip'): raise ValueError('Unknown status.')
            if status == 'done' and label not in LABELS: raise ValueError('Choose one of the three categories.')
            if status in ('todo','skip') and label: raise ValueError('Pending/skipped rows must have a blank label.')
            if status == 'skip' and not notes: raise ValueError('Give a short reason for skipping.')
            previous = self.annotation(self.by_id[pid],state)
            annotation = {'label':label, 'status':status, 'notes':notes, 'provenance':provenance,
                'suggestion_id':draft['suggestion_id'] if draft else None, 'reviewed_at':now()}
            state['annotations'][pid] = annotation
            state['revision'] += 1
            state['audit'].append({'revision':state['revision'], 'at':now(), 'action':action,
                'passage_id':pid, 'split':split, 'before':previous, 'after':annotation})
            self.commit(state)
            return {'saved':True, 'revision':state['revision'], 'annotation':annotation}

    def expand(self, revision):
        with self.lock:
            state=self.state()
            if state['revision'] != revision: raise Conflict('Refresh before expanding.')
            previous=state['active_group_count']
            state['active_group_count']=min(len(self.manifest['group_order']),previous+self.manifest['expansion_group_step'])
            if previous==state['active_group_count']: raise ValueError('All source groups are active.')
            state['revision']+=1
            state['audit'].append({'revision':state['revision'],'at':now(),'action':'expand_existing_frozen_membership',
                                   'from_groups':previous,'to_groups':state['active_group_count']})
            self.commit(state)
            return {'added_groups':state['active_group_count']-previous,'revision':state['revision']}

    def export(self, revision):
        with self.lock:
            state=self.state()
            if state['revision'] != revision: raise Conflict('Refresh before exporting.')
            active=self.active(state)
            pending=[row for row in active if self.annotation(row,state)['status']=='todo']
            if pending: raise ValueError('%d active passages still need human review. Labels are never auto-accepted.'%len(pending))
            parts={name:[] for name in SPLITS}; provenance=[]
            for row in active:
                a=self.annotation(row,state); split=self.membership[row['passage_id']]['split']
                if split!='train' and a['provenance']!='independent_human_no_ai_hint':
                    raise ValueError('Evaluation provenance must be independent human review without AI hints.')
                provenance.append(dict(passage_id=row['passage_id'],split=split,**a))
                if a['status']=='done': parts[split].append(dict(row,**{key:a[key] for key in ('label','status','notes')}))
            coverage={name:dict(Counter(row['label'] for row in rows)) for name,rows in parts.items()}
            minimum={'train':10,'validation':2,'test':2}
            failures=['%s/%s: %d (need %d)'%(name,label,coverage[name].get(label,0),minimum[name])
                      for name in SPLITS for label in LABELS if coverage[name].get(label,0)<minimum[name]]
            if failures:
                raise ValueError('Insufficient class coverage: '+ '; '.join(failures)+'. Expand with preassigned reserve groups; do not force labels.')
            # Membership was frozen before suggestions; validate again, never call a splitter.
            seen_groups={name:{r['group_id'] for r in parts[name]} for name in SPLITS}
            assert all(not seen_groups[a]&seen_groups[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:])
            destination=self.folder/'exports'/('export_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6])
            split_dir=destination/'splits';split_dir.mkdir(parents=True,exist_ok=False)
            saved={}
            for name,rows in parts.items():
                (split_dir/(name+'.csv')).write_bytes(csv_data(rows))
                saved[name]=list(csv.DictReader(io.StringIO(csv_data(rows).decode('utf-8-sig'),newline='')))
            manifest={'seed':self.manifest['seed'],'split_sha256':{n:structured_sha(r) for n,r in saved.items()},
                'rows':{n:len(r) for n,r in saved.items()},'labels':list(LABELS),'created_at':now(),
                'grouping':'release-group disjoint; assigned before suggestions; NO re-splitting',
                'frozen_session_manifest_sha256':sha(self.manifest_bytes),'source_sha256':self.manifest['source_sha256'],
                'labeling_method':self.manifest['labeling_method'],'coverage':coverage,
                'skip_counts':dict(Counter(self.membership[r['passage_id']]['split'] for r in active if self.annotation(r,state)['status']=='skip')),
                'active_group_count':state['active_group_count'],'annotation_revision':state['revision'],
                'scope':self.manifest['scope'],'selection':'One candidate per release group; known prior exposures forced to training; deterministic seeded pilot.'}
            (split_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (destination/'annotation_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
            members={'splits/'+name:split_dir/name for name in ('train.csv','validation.csv','test.csv','manifest.json')}
            for name in ('music_review_pipeline.py','music_review_training.py','LABELING_GUIDE.md'):
                members[name]=self.project/name
            hashes={name:sha(path.read_bytes()) for name,path in members.items()}
            archive=destination/'training_bundle.zip'
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as zipped:
                for name,path in members.items():zipped.writestr(name,path.read_bytes())
                zipped.writestr('bundle_sha256.json',json.dumps(hashes,indent=2))
            atomic_bytes(self.folder/'latest_export.json',(json.dumps({'directory':str(destination),'bundle':str(archive),
                'revision':state['revision'],'frozen_session_manifest_sha256':sha(self.manifest_bytes)},indent=2)+'\n').encode())
            return {'exported':True,'bundle':str(archive),'rows':manifest['rows'],'coverage':coverage}


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music review labeling</title><style>
:root{font:16px system-ui,-apple-system,sans-serif;color:#19352f;background:#f5f6f2}*{box-sizing:border-box}body{margin:0}main{max-width:1080px;margin:auto;padding:30px 28px}h1{font-size:25px;margin:0 0 6px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.muted{color:#5a6a64;font-size:14px}.tools{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}button{font:inherit;border:1px solid #b9c7bf;border-radius:7px;padding:11px 16px;background:white;color:#19352f;cursor:pointer}button:hover{border-color:#25765d;background:#edf5ef}button:disabled{opacity:.45;cursor:default}.selected,.primary{background:#21694f;color:white;border-color:#21694f}.selected:hover,.primary:hover{background:#18553f;color:white}.tabs button{flex:1;min-width:170px}.progress{height:7px;background:#dce6dd;border-radius:5px;overflow:hidden;margin-top:12px}.progress div{height:100%;background:#298463;width:0}.card{background:#fff;border:1px solid #dce3d9;border-radius:10px;padding:28px;margin:20px 0}#passage{font:23px/1.6 Georgia,serif;margin:22px 0 28px;white-space:pre-wrap}#hint{border-left:4px solid #4f7ca3;background:#edf3f7;padding:16px 18px;margin:20px 0}#hint h3{font-size:16px;margin:0 0 8px}#hint p{line-height:1.5;margin:7px 0}.categories{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.categories button{text-align:left;min-height:90px;font-weight:600}.categories small{display:block;font-weight:400;font-size:13px;line-height:1.4;margin-top:6px}.meta{display:flex;gap:20px;justify-content:space-between;font-size:13px;color:#637268;flex-wrap:wrap}textarea{width:100%;font:inherit;border:1px solid #c9d3ca;border-radius:6px;padding:10px;min-height:66px;margin-top:7px}.bottom{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:16px}.notice{padding:12px 14px;background:#e8efe5;border-radius:6px;font-size:14px;line-height:1.45}.error{background:#fdebea;color:#852c24;white-space:pre-wrap}#message{margin:15px 0;min-height:24px;font-size:14px}details{font-size:14px;line-height:1.55;margin:18px 0}summary{cursor:pointer}#empty{font-size:19px;padding:30px 0}kbd{background:#e5eae2;padding:2px 5px;border-radius:3px}.footer{margin-top:24px;border-top:1px solid #d4ddd2;padding-top:20px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.fixture{background:#ffe799;padding:8px;font-weight:600}.saved{color:#1d754b}@media(max-width:700px){main{padding:20px 14px}.categories{grid-template-columns:1fr}.card{padding:20px}header{display:block}#passage{font-size:21px}}
</style></head><body><main><div id="fixture" class="fixture" hidden>TEST FIXTURE — invented data, not project annotations</div>
<header><div><h1>Music Review Insight Finder</h1><div class="muted">Review one passage, choose a category, and save your decision.</div></div><div id="overall" class="muted"></div></header>
<div class="progress"><div id="bar"></div></div>
<nav class="tools tabs"><button data-split="train">Training · AI assisted</button><button data-split="validation">Validation · human only</button><button data-split="test">Test · human only</button></nav>
<div id="policy" class="notice"></div><div id="message" role="status" aria-live="polite"></div>
<section class="card"><div class="meta"><span id="position"></span><span id="status"></span></div><div id="empty" hidden>No passages in this view.</div>
<div id="work"><p id="passage"></p><div id="hint" hidden><h3>AI draft — training only</h3><strong id="suggested"></strong><p id="reason"></p><button id="accept" class="primary">Accept draft & save <kbd>A</kbd></button></div>
<div class="categories"><button data-label="specific_musical_observation"><kbd>1</kbd> Specific musical observation<small>Describes or evaluates a musical element or performance.</small></button><button data-label="general_opinion"><kbd>2</kbd> General opinion<small>Overall liking or quality, without musical detail.</small></button><button data-label="background_other"><kbd>3</kbd> Background / other<small>Context or another statement without musical assessment.</small></button></div>
<label for="notes"><div style="margin-top:20px">Notes <span class="muted">(optional, except when skipping)</span></div></label><textarea id="notes" placeholder="Record ambiguity or a reason for excluding this passage."></textarea>
<div class="bottom"><div><button id="skip">Skip: insufficient context <kbd>X</kbd></button> <button id="save-notes">Save notes</button></div><div><button id="previous">← Previous</button> <button id="next">Next →</button></div></div>
<div class="meta" style="margin-top:20px"><span id="pid"></span><span id="provenance"></span></div></div></section>
<div class="tools"><label><input type="checkbox" id="pending" checked> Show unfinished only</label><button id="refresh">Refresh saved state / drafts</button></div>
<details><summary>Category rules and uncertain passages</summary><p>Choose a specific musical observation first, even if the passage also contains a general opinion or background. Otherwise choose general opinion when it gives an overall evaluation; otherwise choose background/other. Positive and negative musical observations share a category.</p><p>“Great production” is specific; “Great record” is general. A personnel list that merely names an instrument is background. Do not use stars, an artist's reputation, or the AI draft as gold truth. If a fragment is genuinely unresolvable, corrupted, non-English, or needs missing context, skip it with a reason. Do not force labels to balance classes.</p><p>Each category button saves only your current decision and advances. Nothing is accepted automatically. AI drafts are separate from accepted labels. Validation and test passages must be labeled independently without asking AI for help.</p></details>
<div class="footer"><button id="export" class="primary">Check coverage & export frozen splits</button><button id="expand">Expand pilot with reserve groups</button><span id="reserve" class="muted"></span></div>
<p class="muted">All saves stay in this new local session. The earlier CSV and notebooks are preserved. No uploads, installs, training, or publication occur here.</p></main>
<script>
const token='__TOKEN__';let scope='train',state=null,position=0,rows=[],busy=false,dirtyNotes=false;
const $=id=>document.getElementById(id);const names={specific_musical_observation:'Specific musical observation',general_opinion:'General opinion',background_other:'Background / other'};
function message(text,error=false){$('message').textContent=text;$('message').className=error?'notice error':'saved'}
async function api(path,payload){const options=payload?{method:'POST',headers:{'Content-Type':'application/json','X-Session-Token':token},body:JSON.stringify(payload)}:{};const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data}
function current(){return rows[position]}
function safeMove(){return !dirtyNotes||confirm('These note edits have not been saved. Leave them behind?')}
async function load(preferred=null){state=await api('/api/state?split='+scope);rows=state.rows.filter(r=>!$('pending').checked||r.annotation.status==='todo');position=preferred?Math.max(0,rows.findIndex(r=>r.passage_id===preferred)):Math.min(position,Math.max(0,rows.length-1));dirtyNotes=false;render()}
function render(){const totals=Object.values(state.counts);let n=totals.reduce((s,c)=>s+c.total,0),done=totals.reduce((s,c)=>s+c.done+c.skip,0);$('overall').textContent=`${done} / ${n} reviewed · ${state.session}`;$('bar').style.width=(100*done/n)+'%';$('fixture').hidden=!state.fixture;
document.querySelectorAll('[data-split]').forEach(b=>{b.classList.toggle('selected',b.dataset.split===scope);const c=state.counts[b.dataset.split];b.textContent=`${b.dataset.split==='train'?'Training · AI assisted':b.dataset.split==='validation'?'Validation · human only':'Test · human only'} (${c.done+c.skip}/${c.total})`});
$('policy').textContent=scope==='train'?'AI drafts are suggestions, not gold labels. Read each passage and accept or correct the draft yourself. Previously saved choices stay saved.':'Independent human evaluation: no AI hints are shown or accepted here. Label from the passage and category guide without consulting AI.';
$('reserve').textContent=`${state.remaining_groups} independent groups available in reserve`;$('expand').disabled=!state.remaining_groups;$('export').disabled=totals.some(c=>c.todo>0);const row=current();$('work').hidden=!row;$('empty').hidden=!!row;$('position').textContent=row?`${position+1} / ${rows.length} in this view`:'';$('status').textContent=row?`Status: ${row.annotation.status}`:'';
if(!row)return;$('passage').textContent=row.text;$('notes').value=row.annotation.notes;$('pid').textContent='Passage '+row.passage_id;$('provenance').textContent=row.annotation.provenance.replaceAll('_',' ');
const hint=scope==='train'?row.suggestion:null;$('hint').hidden=!hint;if(hint){$('suggested').textContent=hint.suggested_label?names[hint.suggested_label]:'Suggested exclusion: insufficient context';$('reason').textContent=hint.reason;$('accept').textContent=hint.suggested_label?'Accept draft & save (A)':'Accept skip suggestion (A)'}
document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('selected',row.annotation.status==='done'&&b.dataset.label===row.annotation.label));}
async function save(label,status,action='manual'){if(busy||!current())return;busy=true;document.querySelectorAll('.categories button,#accept,#skip,#save-notes').forEach(b=>b.disabled=true);const row=current();try{await api('/api/label',{revision:state.revision,passage_id:row.passage_id,label,status,notes:$('notes').value,action});message('Saved locally.');dirtyNotes=false;await load();}catch(error){message(error.message,true);}finally{busy=false;document.querySelectorAll('.categories button,#accept,#skip,#save-notes').forEach(b=>b.disabled=false)}}
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(b.dataset.label,'done'));
$('accept').onclick=()=>{if(scope==='train'&&current()?.suggestion)save('','','accept_ai')};
$('skip').onclick=()=>{if(!$('notes').value.trim())$('notes').value='Insufficient context';save('','skip')};
$('save-notes').onclick=()=>{const r=current();if(r)save(r.annotation.label,r.annotation.status,'save_notes')};
$('notes').oninput=()=>dirtyNotes=true;
$('previous').onclick=()=>{if(safeMove()){position=Math.max(0,position-1);dirtyNotes=false;render()}};$('next').onclick=()=>{if(safeMove()){position=Math.min(rows.length-1,position+1);dirtyNotes=false;render()}};
document.querySelectorAll('[data-split]').forEach(b=>b.onclick=()=>{if(safeMove()){scope=b.dataset.split;position=0;load().catch(e=>message(e.message,true))}});
$('pending').onchange=()=>{position=0;load().catch(e=>message(e.message,true))};$('refresh').onclick=()=>{if(safeMove())load().catch(e=>message(e.message,true))};
$('expand').onclick=async()=>{if(!confirm('Activate up to 60 additional independent groups? Their train/validation/test assignments are already frozen. New training drafts may be pending; evaluation stays human-only.'))return;try{const result=await api('/api/expand',{revision:state.revision});await load();message(`Added ${result.added_groups} groups without changing existing assignments.`)}catch(e){message(e.message,true)}};
$('export').onclick=async()=>{try{const result=await api('/api/export',{revision:state.revision});message('Saved frozen training archive: '+result.bundle)}catch(e){message(e.message,true)}};
document.addEventListener('keydown',e=>{if(busy||e.ctrlKey||e.metaKey||e.altKey||['INPUT','TEXTAREA'].includes(e.target.tagName))return;if(['1','2','3'].includes(e.key)){e.preventDefault();save(Object.keys(names)[Number(e.key)-1],'done')}else if(e.key.toLowerCase()==='a'){e.preventDefault();$('accept').click()}else if(e.key.toLowerCase()==='x'){e.preventDefault();$('skip').click()}});
window.addEventListener('beforeunload',e=>{if(dirtyNotes){e.preventDefault();e.returnValue=''}});load().catch(e=>message(e.message,true));
</script></body></html>'''

def serve(session, port):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,value,status=200,html=False):
            data=value.encode('utf-8') if html else json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                path=urlparse(self.path)
                if self.headers.get('Host') not in ('127.0.0.1:'+str(port),'localhost:'+str(port)):
                    return self.reply({'error':'Invalid host'},403)
                if path.path=='/':return self.reply(PAGE.replace('__TOKEN__',token),html=True)
                if path.path=='/api/state':return self.reply(session.view(parse_qs(path.query).get('split',['train'])[0]))
                return self.reply({'error':'Not found'},404)
            except Exception as error:self.reply({'error':str(error)},400)
        def do_POST(self):
            try:
                if self.headers.get('X-Session-Token')!=token:return self.reply({'error':'Invalid session token'},403)
                origin=self.headers.get('Origin')
                if origin and origin not in ('http://127.0.0.1:'+str(port),'http://localhost:'+str(port)):
                    return self.reply({'error':'Invalid origin'},403)
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<20000:raise ValueError('Invalid request size.')
                payload=json.loads(self.rfile.read(size))
                if self.path=='/api/label':return self.reply(session.label(payload))
                if self.path=='/api/expand':return self.reply(session.expand(payload.get('revision')))
                if self.path=='/api/export':return self.reply(session.export(payload.get('revision')))
                return self.reply({'error':'Not found'},404)
            except Conflict as error:self.reply({'error':str(error)},409)
            except Exception as error:self.reply({'error':str(error)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print('Labeling screen: http://127.0.0.1:%d/'%port,flush=True)
    print('Session:',session.folder,flush=True)
    server.serve_forever()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();serve(Session(args.session),args.port)
