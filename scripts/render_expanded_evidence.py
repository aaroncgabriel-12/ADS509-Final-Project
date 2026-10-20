from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,collections,copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter,MultipleLocator

import argparse
parser=argparse.ArgumentParser(description="Render figures from saved expanded-evaluation results; no model execution.")
parser.add_argument("--output",type=Path,default=Path("outputs/expanded_presentation_evidence"))
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]/"expanded_evaluation"
RESULTS=ROOT
RUN=ROOT/"results"
OUT=args.output.resolve()
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.mkdir(exist_ok=False)
comparison_path=RUN/'one_time_test_comparison/comparison.json';summary_path=RESULTS/'VERIFIED_RESULTS_SUMMARY.json'
comparison=json.loads(comparison_path.read_text());summary=json.loads(summary_path.read_text())
labels=['specific_musical_observation','general_opinion','background_other']
names=['Specific musical observation','General opinion','Background/other']
wrap=['Specific musical\nobservation','General\nopinion','Background /\nother']
assert comparison['status']=='completed_one_time_test_comparison' and comparison['test_groups']==99
assert comparison['selected_checkpoint']=='adapter_epoch_3' and comparison['test_used_for_selection'] is False
sources={'comparison_json':comparison_path,'verified_summary_json':summary_path,
 'adapted_predictions_csv':RUN/'one_time_test_comparison/adapted_test_predictions.csv',
 'unadapted_predictions_csv':RUN/'one_time_test_comparison/unadapted_test_predictions.csv',
 'frozen_test_csv':ROOT/'frozen_data/test.csv',
 'selection_receipt_json':RUN/'selection_receipt.json'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
for model in ['unadapted','adapted']:
 m=comparison[model]['metrics'];pred=csv_rows(sources[model+'_predictions_csv']);assert len(pred)==m['n']==200
 matrix=np.zeros((3,3),dtype=int)
 for row in pred:matrix[labels.index(row['label']),labels.index(row['predicted_label'])]+=1
 assert matrix.tolist()==m['confusion_matrix']
 assert matrix.sum(axis=0).tolist()==[m['prediction_counts'][l] for l in labels]
 assert matrix.sum(axis=1).tolist()==[m['per_class'][l]['support'] for l in labels]
 assert np.trace(matrix)/200==m['accuracy']

BG='#F6F6F1';INK='#20313C';TEAL='#147D82';MUTED='#697D89';GRID='#D6DED9';BASE='#91A3AC';SOFT='#E5F1EE'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':INK,'axes.labelcolor':INK,
 'axes.edgecolor':GRID,'xtick.color':INK,'ytick.color':MUTED,'figure.facecolor':BG,'axes.facecolor':BG,
 'savefig.facecolor':BG,'svg.fonttype':'path','pdf.fonttype':42})
FOOT='n = 200 test passages from 99 review/release groups'
METHOD='AI-assisted labels with recorded human review · Checkpoint selected on validation before test'
def footer(fig,extra=None):
 fig.text(.055,.065,FOOT,fontsize=13,color=INK)
 fig.text(.055,.035,METHOD,fontsize=11.5,color=MUTED)
 if extra:fig.text(.945,.065,extra,fontsize=11.5,color=MUTED,ha='right')
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=300)
 fig.savefig(OUT/(name+'.svg'))
 plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13.333333,7.5))
fig.subplots_adjust(left=.075,right=.95,bottom=.22,top=.74,wspace=.30)
fig.text(.055,.91,'Expanded test: unadapted and adapted performance',fontsize=25,weight='bold')
fig.text(.055,.855,'Same test set · Pinned Qwen base vs validation-selected LoRA checkpoint (epoch 3)',fontsize=14,color=MUTED)
for ax,key,title in zip(axes,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
 values=[comparison[m]['metrics'][key] for m in ['unadapted','adapted']]
 bars=ax.bar([0,1],values,width=.55,color=[BASE,TEAL],zorder=3)
 ax.set_xticks([0,1],['Unadapted','Adapted']);ax.set_ylim(0,1);ax.set_xlim(-.65,1.65)
 ax.set_title(title,fontsize=19,loc='left',pad=14,weight='bold')
 ax.yaxis.set_major_locator(MultipleLocator(.2));ax.grid(axis='y',color=GRID,linewidth=.8,zorder=0)
 for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
 ax.tick_params(axis='both',length=0,pad=9,labelsize=14)
 if key=='accuracy':ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
 for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.035,f'{v:.1%}' if key=='accuracy' else f'{v:.3f}',ha='center',fontsize=20,weight='bold',color=INK)
 delta=values[1]-values[0]
 ax.text(.5,-.25,f'+{delta*100:.1f} percentage points' if key=='accuracy' else f'+{delta:.3f} macro-F1',transform=ax.transAxes,ha='center',fontsize=16,weight='bold',color=TEAL)
footer(fig)
save(fig,'01_same_test_accuracy_macro_f1')

matrix=np.array(comparison['adapted']['metrics']['confusion_matrix'])
fig=plt.figure(figsize=(13.333333,7.5));ax=fig.add_axes([.30,.23,.46,.57]);cax=fig.add_axes([.79,.25,.018,.53])
fig.text(.055,.92,'Adapted model: test confusion matrix',fontsize=26,weight='bold')
fig.text(.055,.86,'Counts · Epoch 3 · 142 of 200 passages classified correctly (71.0%)',fontsize=15,color=MUTED)
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('soft_teal',['#EFF4F1',TEAL])
im=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=90,aspect='auto')
ax.set_xticks(range(3),wrap,fontsize=14);ax.set_yticks(range(3),wrap,fontsize=14)
ax.set_xlabel('Predicted category',fontsize=16,labelpad=12);ax.set_ylabel('Actual category',fontsize=16,labelpad=20)
ax.tick_params(axis='both',length=0,pad=10)
ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
ax.grid(which='minor',color=BG,linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
for i in range(3):
 for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=31,weight='bold',color='white' if matrix[i,j]>=45 else INK)
cb=fig.colorbar(im,cax=cax);cb.set_label('Passages',fontsize=13,labelpad=10);cb.ax.tick_params(labelsize=12,length=0);cb.outline.set_visible(False)
footer(fig,'Rows = actual · Columns = predicted')
save(fig,'02_adapted_confusion_matrix_counts')

fig=plt.figure(figsize=(13.333333,7.5));ax=fig.add_axes([.045,.24,.91,.48]);ax.axis('off')
fig.text(.055,.91,'Class performance and prediction counts',fontsize=26,weight='bold')
fig.text(.055,.85,'Precision, recall and F1 describe the adapted model; prediction counts show both models.',fontsize=13.3,color=MUTED)
headers=['Category','Precision','Recall','F1','True\nsupport','Predicted\nunadapted','Predicted\nadapted']
rows=[]
for label,name in zip(labels,['Specific musical\nobservation','General opinion','Background/other']):
 m=comparison['adapted']['metrics']['per_class'][label]
 rows.append([name,f"{m['precision']:.3f}",f"{m['recall']:.3f}",f"{m['f1']:.3f}",str(m['support']),str(comparison['unadapted']['metrics']['prediction_counts'][label]),str(comparison['adapted']['metrics']['prediction_counts'][label])])
table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',colLoc='center',colWidths=[.27,.105,.105,.095,.10,.115,.11],bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(14)
for (i,j),cell in table.get_celld().items():
 cell.set_edgecolor(GRID);cell.set_linewidth(.7);cell.PAD=.08
 if i==0:cell.set_facecolor(INK);cell.get_text().set_color('white');cell.get_text().set_weight('bold');cell.get_text().set_fontsize(12.8)
 else:
  cell.set_facecolor('white' if i%2 else '#EDF1EC')
  if j==0:cell.get_text().set_ha('left');cell.get_text().set_weight('bold')
  if j==6:cell.set_facecolor(SOFT);cell.get_text().set_color(TEAL);cell.get_text().set_weight('bold')
fig.text(.055,.16,'General-opinion recall remained 0.500: 27 of 54 were identified.',fontsize=15,color=TEAL,weight='bold')
footer(fig,'Scores: 0–1 · Counts: passages')
save(fig,'03_adapted_class_metrics_prediction_counts')

with (OUT/'same_test_comparison.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=['model','n_passages','n_groups','accuracy','macro_f1','selected_epoch']);writer.writeheader()
 for model in ['unadapted','adapted']:writer.writerow({'model':model,'n_passages':200,'n_groups':99,'accuracy':comparison[model]['metrics']['accuracy'],'macro_f1':comparison[model]['metrics']['macro_f1'],'selected_epoch':3 if model=='adapted' else ''})
with (OUT/'per_class_metrics_and_prediction_counts.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['model','category','category_display','precision','recall','f1','true_support','prediction_count','n_passages','n_groups'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for model in ['unadapted','adapted']:
  for label,name in zip(labels,names):
   m=comparison[model]['metrics']['per_class'][label];writer.writerow({'model':model,'category':label,'category_display':name,'precision':m['precision'],'recall':m['recall'],'f1':m['f1'],'true_support':m['support'],'prediction_count':comparison[model]['metrics']['prediction_counts'][label],'n_passages':200,'n_groups':99})
for model in ['unadapted','adapted']:
 with (OUT/(model+'_confusion_matrix_counts.csv')).open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['actual_category',*labels])
  for label,row in zip(labels,comparison[model]['metrics']['confusion_matrix']):writer.writerow([label,*row])
source_audit={name:{'path':str(path),'sha256':sha(path)} for name,path in sources.items()}
metrics={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_files':source_audit,'comparison':comparison,
 'label_order':labels,'display_names':names,'n_passages':200,'n_groups':99,
 'metric_verification':'Both confusion matrices, supports and prediction counts recomputed from saved prediction CSVs; all match comparison.json.',
 'no_model_rerun':True,'historical_charts_changed':False,'figure_format':'PNG 300 dpi and SVG; 16:9; counts are unnormalized; scores rounded only for display.'}
(OUT/'presentation_metrics_and_provenance.json').write_text(json.dumps(metrics,indent=2)+'\n')
(OUT/'PROVENANCE_AND_CAPTIONS.txt').write_text('''EXPANDED TEST PRESENTATION EVIDENCE — verified saved outputs only

All figures describe the same 200 test passages from 99 review/release groups.
Labels were AI-assisted with recorded Aaron review, not independent human gold.
Epoch 3 was selected using validation, before test scoring. No test-driven tuning.
The baseline uses the same fresh pinned base with adapters disabled; development
preflight verified its predictions against the saved unadapted validation reference.
No model was rerun and no labels or historical charts were changed to build evidence.

01_same_test_accuracy_macro_f1.png/.svg
Caption: On the same 200-passage test, adaptation increased accuracy from 31.5% to 71.0%
and macro-F1 from 0.238963 to 0.662959. Macro-F1 difference +0.423996; accuracy difference
+39.5 percentage points. This is a paired dataset comparison, not a business impact test.

02_adapted_confusion_matrix_counts.png/.svg
Caption: Adapted epoch 3 confusion matrix. Rows are actual categories; columns are
predicted categories, both ordered specific musical observation / general opinion /
background-other. Counts are unnormalized. Matrix [[90,12,11],[16,27,11],[4,4,25]].
Supports 113/54/33; adapted prediction counts 110/43/47; correct 142 of 200.

03_adapted_class_metrics_prediction_counts.png/.svg
Caption: Adapted precision/recall/F1 with true support and both prediction counts.
Unadapted predictions 83/1/116; adapted 110/43/47. General-opinion recall remained 0.500.
Scores in images are rounded for readability; CSV/JSON retain full saved precision.

The JSON and CSV files are the editable data behind these figures. Use all 3 figures
with their sample/provenance footer visible. Passages are clustered by source group;
label subjectivity and possible pretraining overlap limit generalization. These
results do not establish business time savings or independent-human evaluation.

SOURCE FILES / SHA-256
'''+''.join(name+'\n'+value['path']+'\nSHA-256 '+value['sha256']+'\n\n' for name,value in source_audit.items()))
(OUT/'build_expansion_presentation_evidence.py').write_bytes(Path(__file__).read_bytes())
hashes={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file()};(OUT/'EVIDENCE_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'staging':str(OUT),'files':len(hashes)+1,'adapted_confusion_matrix':matrix.tolist(),'supports':matrix.sum(axis=1).tolist(),'adapted_prediction_counts':matrix.sum(axis=0).tolist(),'unadapted_prediction_counts':[comparison['unadapted']['metrics']['prediction_counts'][l] for l in labels]},indent=2))
