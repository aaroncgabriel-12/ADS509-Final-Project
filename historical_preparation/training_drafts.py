"""AI draft decisions from this assistant's reading of TRAINING passages only.

These are suggestions, not accepted annotations. No keyword-rule classifier or
external API generated them. The server will require individual human action.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SESSION = Path(__file__).resolve().parents[1] / 'music_review_runs/2026-10-15_music_insight_assisted_v1'
DRAFTS = [
('0f77abd5b7541a9ffaeb1a61','A','The passage evaluates the solo as melodic and thoughtful, identifying a particular performance element.'),
('0cd841d26cefc074072692a7','A','It discusses the album concept, rock style, and a cover song. The opening is a fragment, so confirm that the remaining musical description is sufficient.'),
('7988fa423942a101d40f1427','A','It describes the singer’s expressive delivery and the fit between the songs and her voice.'),
('4a0f3870f6266ba564d369e1','A','It describes the kick drum, guitar textures, electronic sounds, and the way the arrangement changes.'),
('9d6563a4bfbc992ac2d3340c','B','It broadly praises the quality of singers in the scene without describing a particular vocal or musical feature.'),
('9676e8d9c762cfecd21b21e1','A','It connects the sound of the arrangement and quoted lyrics to the songs’ themes of guilt and money.'),
('343808bfcaaa8907143216c4',None,'“This” is not identified. The humor could concern lyrics, performance, or something else; the isolated passage lacks context.'),
('b0e2fe283892bcc78bcde882','A','It describes the tune’s arrangement, rhythmic style, melody, and horns.'),
('bcb36a0c143924a42b6ea959',None,'The fragment does not identify what has been hard or establish a music-related observation or opinion.'),
('2cecbb907ac557953275b005','B','The reviewer calls the collaboration wonderful and wants longer appearances, but gives no specific musical detail.'),
('afbcae439abce901ffa82c89',None,'The subject and meaning of “pain and misery” are missing; it could refer to lyrical themes or unrelated context.'),
('dc48c89c6588e8ba1a474f78','A','It evaluates how songwriting and production are brought together on the album.'),
('57acb096df602d07abd16490','C','This is a comment about terminology or classification, without a described musical feature or an overall quality judgment.'),
('6692a2d70810ee2420bcef08','C','The sentence primarily identifies a guest bassist. It does not describe his playing; check the guide’s personnel-list distinction.'),
('92c8d88f0c1c5a228cc8e592','C','It lists repertoire and composers as background rather than describing how the pieces sound or are performed.'),
('512c16bb50a3df28159f3eb8','A','It describes the marimba’s timbral range and how the instrumentation evokes seasonal images.'),
('5ca652f0d39193853830c831','A','The passage describes a bouncy musical character recurring in another track and its commercial style.'),
('191c1ab5f21e995091f17243','A','It identifies the rhythm section’s supporting role and contrasts the guitar contributions.'),
('e2e2dddd5f4e7598019bc059','A','It describes a blend of country and indie musical styles.'),
('bb0024bb94a8ce66ab74b0ac','A','It describes punk and radio-rock characteristics of particular tracks as part of its praise.'),
('804b64b69705151beacda9d9','B','This expresses disappointment about the release’s contents, without evaluating an identifiable musical element.'),
('ee2f61d6ddc5042ab381292c','B','It gives a broad judgment about the bands’ attitude rather than a specific observation about the music.'),
('71174d9f58587badd506d99c','A','“Dense, knotty works” describes compositional complexity; the biographical information does not remove that observation.'),
('6e03745157cbf21030c91864',None,'The “nagging feeling” and what the DVD changed are not explained in this passage.'),
('a4ed034a68255cadc2c59762','B','“Try These” is an overall recommendation of recordings; no specific musical element is described.'),
('a4819bbdad225c9de2587fa4','A','It describes algorithmic sequencing, improvisational soundscapes, and the addition of vibes.'),
('f93e5361902d3aa32f963f58','A','It gives detailed observations about piano accompaniment, improvisation, and the solo.'),
('610338bae8ebb74c2b6b47f4','B','This is an overall judgment that the album is more mainstream than desired. Borderline: decide whether it gives enough musical specificity under the guide.'),
('85a5679a4a315c2a89340453','A','The passage identifies a song and quotes its lyrical perspective. Confirm that this is enough lyrical content despite the shortened quotation.'),
('5500867a7848ef12bcb8476d',None,'It says the track is typical of the composer but does not explain the relevant musical characteristic.'),
('d13ef461fe74e3866acca6fc','A','It evaluates Rodgers’ portrayal of a specific operatic role and describes the changing emotional performance.'),
('bbae0e68467cd079bd8551bc','C','This describes the artist’s interest in Electronic Voice Phenomena as background, not a musical assessment.'),
('e78f4ea9fc367247876265b8','A','It contrasts repertoire choices and identifies a consistent peace-and-protest theme in the material.'),
('27f80e17c5655f752abf6bc5',None,'The geographic metaphor is too indirect to identify the intended musical observation confidently from this passage alone.'),
('d3320321460c24a8a36b714f','A','It contrasts the albums’ thematic character: seduction versus a bitter breakup.'),
('c92d8a207578c9b98728f4a3','B','It compares the overall quality of versions without describing the musical reason for the preference.'),
('38eb3b46f56735b3cd2a9efb','B','“Far from airy-fairy” is a broad evaluative reaction without an identifiable musical element.'),
('0a2290b3678071d80ffcafaf',None,'The fragment depends entirely on a missing preceding claim.'),
('f14911dc2b54d25440da16da','B','It gives a broad relaxed-and-warm impression of the session without identifying a particular musical element. Review this borderline wording.'),
('185d4de97830e84a21b7dfd8','B','This is general praise of the performer through a comparison with competitors.'),
('cfc26dda3fa319c6b41e99e0','B','The passage recommends listening to the recording and names possible audiences, without explaining a musical feature.'),
('963385384304871aab0ea8c7','A','It identifies the R&B ballad style and rich, multi-layered vocals.'),
('364fd564e973e8f167d395fb','B','It expresses a hypothetical, underwhelmed reaction to a release plan; the songwriting credit and chart history are background.'),
('e6f605bc601bfe3946006acf','A','It describes the rhythm section playing behind the beat and the saxophone’s supporting role.'),
('bd9939745325eb2cba1f3dc9','A','It describes acoustic/synthetic textures and the sparing, electronica-derived rhythm tracks.'),
('f8c551fc22a3302ebcd9644e','C','It mainly identifies works and dates from a career period, without describing their music.'),
('533761b90fb9e19f71f64e72',None,'“That” has no referent in this short fragment, so the passage cannot be reliably categorized on its own.'),
('7cfefb6deac4de442e85f6e0','A','It describes the lyrics’ romantic declarations and the performer’s lyrical persona.'),
('ec914d047fd6daeb48c114ec','A','It describes the sound through post-rock/postfusion, lo-fi, and Miles Davis stylistic comparisons.'),
('719b3e58a14d919204a8d311','A','It evaluates programmed rhythms and production in relation to the lyrical message, and comments on the lyrics’ content.'),
('6b0fffa22fafd7b4ba360ff7','A','It identifies the tempo, soft pads, chimes, finger snaps, and harmonies.'),
('a8107fa733cfa35c0f89c0f2','B','The album is called idiosyncratic and brilliant, but no particular musical element is described.'),
('a38f118ef729d3a52e79563a','B','The passage broadly rates songs as a stormer, painful, or weedy without explaining specific musical features.'),
('d30b629bb65433f79d3626f7','B','This is general praise of the debut and its future potential.'),
('5c9556d67d668f52dc76d7b3','C','It discusses a cultural tradition of early jazz education rather than a passage-level musical assessment.'),
('3088cf4affe23e5c0412cbd5','A','It describes style and instrumentation, specifically spoons and stride piano.'),
('0293721215ba6d3c657a4666','A','It comments on arrangement detail and the recording’s intimate character.'),
('03ceec486f93c20832374713','B','It endorses a highly positive judgment of the mass/performance without specifying the musical or performance features.'),
('1634a48c3a40390dd16dff4e','A','It describes the pedal steel and jaunty piano and their effect on the songs.'),
('b5b440b71876ce6c8fa03471','A','It describes chill beats, multi-layered musicianship, and how the elements work together.'),
('a3b8726df37c07787bc829e6','A','It identifies the closing track’s lyrical message and its role as a mantra or epilogue.'),
('dc661d9acf0b6b6ddb2086b2','A','It describes a recording style that retains breath and incidental guitar noises.'),
('91b0ba9a5378ec06cdeb3ac0','B','Despite the stray markup, it clearly asserts the series’ importance without explaining a musical feature.'),
('5b500acb1fec05415c10eb3d','C','It offers religious and geographic context for the artist; the actual musical methods and sounds are not described.'),
('44f95a2e4bee35dcb3ddcf25','B','It gives general praise of some tracks with no specific musical detail.'),
('396a6de435b55cfd5aff77af','A','It explicitly compares the sound of parts of the album to another band’s sound.'),
('4110b255bb4e7ced6a61dbef','A','It describes trombone timbre, precision, and timbral range on a specific piece.'),
('c6282309d4edf0d8c77988e6','B','The reviewer gives an overall middle-of-the-road judgment and audience preference without naming a specific musical mechanism.'),
('e125aa65322f3e975665c9c7','C','This is a chronological description of career milestones and personal history.'),
('1028c23e76e535f8b53d4f58','A','It describes guitar textures, an organ drone, and electronic/noise effects in the arrangement.'),
('30f50c4e9190a00e794c9943',None,'The referent of “these” and the intended archetypes are absent.'),
('4fcf38d7171fa5f081a4a8c0',None,'The arguments and the musical qualities that answer them are missing from this isolated passage.'),
('8b8880ffea36b0612775f91a','A','It describes a change from balladry to swing and the relationship between altered chords and melody.'),
('cddf846758c43101b7b9f122','A','It identifies a repeated chorus and its accompanying role relative to the instruments.'),
('6c97fc3256d078c221565351',None,'The sentence gives artist names but does not identify what “it” refers to or explain the comparison.'),
('5cb68273066b2d6a986dc52f','A','“Smoky voice” identifies a vocal timbre within otherwise general commentary.'),
('a2815828f4aa6b649bdabf95','C','It describes the formation of Column One as background information.'),
]

def write_drafts(session=SESSION):
    session=Path(session)
    manifest=json.loads((session/'session_manifest.json').read_text())
    reference={r['passage_id']:r for r in json.loads((session/'reference.json').read_text())}
    states=json.loads((session/'state.json').read_text())
    active=set(manifest['group_order'][:states['active_group_count']])
    labels={'A':'specific_musical_observation','B':'general_opinion','C':'background_other',None:None}
    drafts={}
    created=datetime.now(timezone.utc).isoformat()
    for pid,code,reason in DRAFTS:
        assignment=manifest['membership'][pid]
        assert assignment['split']=='train' and assignment['candidate'] and assignment['group_id'] in active
        row=reference[pid]
        drafts[pid]={'suggestion_id':'codex_training_draft_v1_'+pid,'suggested_label':labels[code],
            'reason':reason,'text_sha256':hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
            'created_at':created,'provider':'Codex assistant judgment in this conversation',
            'method':'Direct model reading of frozen training passages only; not keyword rules',
            'human_review_required':True}
    target=session/'training_suggestions.json'
    assert json.loads(target.read_text())=={},'Existing drafts present; do not replace.'
    target.write_text(json.dumps(drafts,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Saved',len(drafts),'training-only AI drafts. No accepted labels or statuses changed.')

if __name__=='__main__': write_drafts()
