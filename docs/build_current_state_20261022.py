"""Build an evidence-backed PDF snapshot without changing models or datasets."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import csv,json,sys,hashlib,re
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));
import torch
from ml.model import load_checkpoint
from ml.sequence_model import load_sequence_checkpoint
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf
from PIL import Image,ImageDraw
OUT=ROOT/'output/pdf/ASignX-Current-System-State-2026-10-22.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
current=read('output/current-model-comparison/comparison.json')
correction=read('output/data-repair/latest-correction.json')
cleanup=read('output/system-cleanup/cleanup-record.json')
before=read('output/predeletion-training/20261021T162040Z/comparison.json')
with (ROOT/'ml/data/landmarks.csv').open(newline='') as f:
 reader=csv.DictReader(f);schema=reader.fieldnames;rows=list(reader)
datahash=sha(ROOT/'ml/data/landmarks.csv');assert datahash==current['evaluation_dataset_sha256']==correction['reviewed_dataset_sha256']
counts=Counter(r['label'] for r in rows);hands=Counter(r['handedness'] for r in rows);sessions={r['session_id'] for r in rows}
sm,sl,sc=load_checkpoint(ROOT/'ml/models/sign_classifier_canonical.pth')
dm,dl,dc=load_sequence_checkpoint(ROOT/'ml/models/sequence_classifier.pth')
assert sha(ROOT/'ml/models/sign_classifier_canonical.pth')=='98559a10c553c23d2c6aad227f52416b59ad92483bf4f079cfca5eb512bf5b66'
metric=current['models']['existing']['splits'];test=metric['test'];val=metric['validation']
testlog=(ROOT/'output/system-cleanup/current-report-tests.log').read_text(encoding='utf-16');testmatch=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',testlog);assert testmatch
modelpaths=list((ROOT/'ml/models').glob('*.pth'));assert len(modelpaths)==2
references=['main.py','asignx/detector.py','asignx/models.py','asignx/camera.py','asignx/visualizer.py','ml/preprocessing.py','ml/model.py','ml/sequence_model.py','ml/dataset.py','ml/data_io.py','ml/sequence_io.py','ml/train.py','ml/evaluate.py','ml/predict.py','ml/predict_sequence.py','ml/extract_dataset.py','ml/extractor_gui.py','ASignX_TestRecognition.bat','ASignX_DataExtractor.bat','output/current-model-comparison/comparison.json','output/data-repair/latest-correction.json','output/archive4-M-pose-review/mapping-summary.json','output/archive4-M-pose-review/final-verification.json','output/system-cleanup/cleanup-record.json','output/system-cleanup/package-verification.json','output/system-cleanup/current-report-tests.log','output/predeletion-training/20261021T162040Z/comparison.json']
evidence={'report_date_local':'2026-10-22','timezone':'Asia/Singapore','generated_at_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':datahash,'rows':len(rows),'labels':dict(sorted(counts.items())),'sessions':len(sessions),'handedness':dict(hands),'schema_columns':len(schema),'static_parameters':sum(p.numel() for p in sm.parameters()),'dynamic_parameters':sum(p.numel() for p in dm.parameters()),'static_config':sm.config,'dynamic_config':dm.config,'sequence_length':dc['sequence_length'],'tests':testmatch.group(0),'metrics_source':'Saved current comparison; recognition was not rerun to author this report. Software tests were rerun.','sources_sha256':{p:sha(ROOT/p) for p in references},'retained_models_sha256':{str(p.relative_to(ROOT)):sha(p) for p in modelpaths}}
OUT.with_suffix('.evidence.json').write_text(json.dumps(evidence,indent=2))
FONT=Path('C:/Windows/Fonts')
for name,file in [('Body','arial.ttf'),('Bold','arialbd.ttf'),('Mono','consola.ttf')]:pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
NAVY=colors.HexColor('#17324d');TEAL=colors.HexColor('#007e87');GRAY=colors.HexColor('#526577');PALE=colors.HexColor('#eef5f7');W=A4[0]-100
styles={
 'title':ParagraphStyle('title',fontName='Bold',fontSize=27,leading=33,textColor=NAVY,spaceAfter=16),
 'h1':ParagraphStyle('h1',fontName='Bold',fontSize=21,leading=27,textColor=NAVY,spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='Bold',fontSize=11.2,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Body',fontSize=9.4,leading=13.6,textColor=NAVY,spaceAfter=9,splitLongWords=True),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.3,leading=11.2,textColor=NAVY,splitLongWords=True),
 'th':ParagraphStyle('th',fontName='Bold',fontSize=8.3,leading=11.2,textColor=colors.white),
 'source':ParagraphStyle('source',fontName='Body',fontSize=8,leading=11,textColor=GRAY,spaceBefore=8,spaceAfter=7),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=8.2,leading=12,textColor=NAVY,spaceAfter=8)}
pages=[];page=None
def section(title):
 global page
 page={'title':title,'items':[]};pages.append(page)
def p(text,kind='body'):page['items'].append(('p',text,kind))
def h(text):p(text,'h2')
def table(rows,widths=None):page['items'].append(('table',rows,widths))
def pct(x):return f'{x:.2%}'
def src(text):p('Evidence: '+text,'source')

section('ASignX\nCurrent system state')
p('22 October 2026 | Asia/Singapore | Repository and saved-evaluation review','h2')
p('ASignX recognizes static alphabet hand poses and the dynamic signs J and Z using hand landmarks. The working system retains the older static checkpoint because it scored higher than the recent retraining runs on the same reviewed test data.')
table([['Current item','Verified state'],['Static dataset',f'{len(rows):,} rows; 24 classes; {len(sessions)} session IDs'],['Retained static model','63,681 parameters; cleaned-test macro F1 63.07%'],['Retained dynamic model','TCN-attention; 296,641 parameters; J/Z'],['Dataset corrections','2,253 M rows removed; 207 source rows unresolved'],['Software checks',testmatch.group(0)],['Workspace cleanup','10 experimental checkpoints removed; 2.37 GiB build/package cleanup']], [.34,.66])
h('How to read this report')
p('Cleaned-test scores and original-test scores describe different evaluation populations. The report labels both explicitly. A higher score after removing wrong labels is not evidence that the saved model learned anything new. Neither offline landmark accuracy nor passing software tests establishes live recognition reliability.')
p('Contents: architecture; data contracts; model architecture; data inventory; source corrections; performance; per-class results; training history; validation; operations and cleanup; priorities; evidence and hashes.')
src('Current model comparison, retained checkpoints, correction records and cleanup logs. Full machine-readable hashes accompany this PDF.')

section('1. Current architecture')
p('The desktop pipeline separates camera capture, hand detection, feature preparation, classification, and display. Offline tools manage collection, import, training, evaluation, and source review.')
page['items'].append(('diagram',None,None))
table([['Layer','Implementation and responsibility'],['Capture and detection','asignx/camera.py and detector.py: OpenCV capture, MediaPipe Tasks hand landmarks, duplicate-hand suppression, handedness stabilization.'],['Recognition','ml/predict.py and predict_sequence.py: preprocessing contracts, confidence thresholds, frame stability, sequence buffer and motion-based routing.'],['Presentation','main.py and asignx/visualizer.py: annotated camera output and recognition state.'],['Dataset tools','Extractor GUI, CSV merger, manual static collection, video/sequence preparation and Dynamic Studio.'],['Training and evaluation','PyTorch trainers, versioned manifests, class metrics, source diagnostics, calibration and review artifacts.'],['Browser integration','web_client/asignx_client_sdk.ts and ONNX export tooling exist; no browser-deployment or mobile performance claim is made here.']], [.25,.75])
p('Default combined recognition uses confidence 0.8 for both predictors, 5 stable frames and 10 release frames. The CLI motion threshold defaults to 0.045. These are runtime settings, distinct from the unthresholded offline scores reported later.')
src('main.py; asignx/*; ml/predict*.py; web_client/asignx_client_sdk.ts.')

section('2. Data models and preprocessing')
table([['Record','Contract'],['Landmark / hand / detection','Structured objects describe 21 named points, handedness, detections and hand collections. Detector outputs feed both visualization and feature preparation.'],['Static Sample / CSV',f'{len(schema)} columns: label, session_id, handedness, preprocessing, then 63 landmark coordinates in the code-defined feature order.'],['SequenceSample / NPZ','label, session_id, handedness, preprocessing version, frame-by-feature array and optional JSON metadata. Missing or incompatible contracts are rejected.'],['Split manifest','Dataset SHA-256, labels, seed, preprocessing contract and index lists. Every row must belong to exactly one partition.'],['Checkpoint','Weights, model configuration, label order, preprocessing, training metadata, dataset hash and split assignments.']], [.26,.74])
h('Static transformation')
p('For each hand, subtract the wrist from all 21 xyz points and divide by the largest wrist-to-point Euclidean distance. Flatten to 63 float32 values. Reject nonfinite or degenerate geometry; validate a zero wrist and unit maximum radius. The CSV keeps its original hand convention.')
p('The retained classifier expects left_x_reflection_v1: reflect only x for a Left hand. Keep y and z. This standardizes handedness; it is not palm rotation alignment. Extraction and live detection must use consistent handedness and mirror conventions.')
h('Dynamic transformation')
p('sequence_anchor_trajectory_v2 uses a common clip origin at the earliest visible wrist and a median visible-hand radius scale. It retains whole-hand motion and two-hand relative placement. Missing-hand slots remain distinguishable; fixed-length interpolation prepares model input. The retained model expects 126 values per frame, not a single static 63-value vector.')
src('asignx/models.py; ml/data_io.py; sequence_io.py; preprocessing.py; dataset.py.')

section('3. Retained model architecture')
h('Static: kinematic residual MLP')
p('63 coordinates -> 91 features -> 128-wide stage -> 64-wide stage -> cosine-prototype classifier -> 24 class logits. The 28 added features are 15 joint-flexion cosines, 10 fingertip distances and a 3D palm normal. Each stage uses Linear, LayerNorm, GELU and dropout 0.2, followed by a residual MLP block.')
p('The cosine head normalizes the embedding and class vectors, then applies a learned clamped scale. The saved checkpoint has 63,681 parameters. Its labels are A-I and K-Y; J and Z use the sequence branch.')
h('Dynamic: temporal convolution plus attention')
p(f'{dc["sequence_length"]} frames x 126 coordinates -> 182 kinematic features per frame -> projection to width 128 -> two temporal residual blocks with dilations 1 and 2 -> four-head self-attention with residual normalization -> mean temporal pooling -> 64-dimensional embedding -> cosine-prototype head for J/Z.')
p('The retained sequence model has 296,641 parameters. It is loaded through the compatibility class SignLSTM, but its checkpoint configuration selects backbone=tcn_attention; it is not using an LSTM recurrent backbone. Alternate implementations remain in source code, but their experimental weight files were removed.')
h('Training algorithms')
p('The retained static checkpoint records 50 epochs, batch size 64, AdamW learning rate 0.002, weight decay 0.0001, label smoothing 0.05, dropout 0.2, seed 42, CPU training and augmentation; its selected epoch is 25. Current trainers support validation-macro-F1 selection with validation loss as a tie-break, cosine learning-rate decay, class weighting and 3D rotation/jitter augmentation. The old checkpoint does not record every newer option, so newer defaults must not be inferred as its historical configuration.')
src('Retained checkpoint metadata; ml/model.py; sequence_model.py; train.py; train_sequence.py.')

section('4. Current dataset inventory')
table([['Measure','Current state'],['Static rows',f'{len(rows):,}'],['Distinct static labels','24'],['Session IDs',str(len(sessions))],['Handedness',', '.join(f'{k}: {v:,}' for k,v in sorted(hands.items()))],['Training / validation / test','109,223 / 13,791 / 34,720'],['Original pre-deletion snapshot','159,987 rows; original test 36,973'],['Recovered sequence benchmark','11 clips: training 7, validation 2, test 2']], [.45,.55])
h('Current class counts')
classrows=[['Class','Rows','Class','Rows','Class','Rows']]
labels=sorted(counts)
for i in range(8):classrows.append(sum(([labels[i+j*8],f'{counts[labels[i+j*8]]:,}'] for j in range(3)),[]))
table(classrows,[.12,.21,.12,.21,.12,.22])
h('Interpretation')
p('A session ID is a recording/import group, not verified signer identity. Session-separated partitions prevent exact group reuse, but do not establish independence of people, source archives or transformed images. Class imbalance and source-specific appearance can still dominate results.')
p('The weak-class Parquet preparation and its combined experiment CSV are retained as historical data artifacts. They are separate from the current main CSV and must not be silently treated as the active training set. The two historical dataset backups preserve original row identities and reproducibility.')
src('Current ml/data/landmarks.csv; reviewed.splits.json in the latest correction directory; sequence_classifier.splits.json.')

section('5. Source audit and M corrections')
p('The archive (4) M folder contained 7,900 image files, representing 4,876 unique image contents. Source recovery identified the archive-4 M session as ext_c72cf97fc7fa5995335a73a1. Stored rows do not inherently include their original image path, so direct geometric correspondence was required.')
table([['Action / evidence','Result'],['Initial reference correction','Original CSV row 110182, linked to M (100).jpg: one M row removed at the owner\'s instruction.'],['Full pose-family review','The owner-defined downward three-extended-fingers pose was reviewed across 18 candidate contact sheets; five other-family sheets were also reviewed.'],['Additional confirmed exclusions','2,252 M rows removed. Matching required all 63 canonical float32 coordinates within absolute 1e-6, with no relative tolerance.'],['Combined removal','2,253 M landmark rows total. 4,083 source filenames link to those rows because identical images appear under multiple names.'],['Remaining archive-4 M rows','1,062: 855 matched retained images; 207 could not be linked reliably and were preserved.'],['Scope of supported deletion','The recorded exclusion was M-specific. These records do not substantiate a separate N-class deletion.']], [.31,.69])
p('The source ZIP and retained main-CSV rows were preserved. A historical retry without padding-coordinate restoration tested 208 unique images and found no additional exact matches. No deletion was based solely on a classifier prediction or a nearest-pose guess.')
p('The filenames-only list is output/archive4-M-pose-review/removed-image-filenames.txt. Contact sheets, image decisions, unresolved rows, removed-row evidence, full CSV backups and row remaps remain available. Original CSV row numbers and current row numbers must not be interchanged.')
src('output/archive4-M-pose-review/*; output/data-repair/latest-correction.json; archive/data-corrections/*.')

section('6. Recognition performance')
p('The table below concerns the same retained static weights evaluated on two populations. Scores are unthresholded, offline landmark classification, not live camera measurements.')
orig=before['existing']
table([['Metric','Current cleaned test','Original historical test'],['Test sample count','34,720','36,973'],['Accuracy',pct(test['accuracy']),pct(orig['test']['accuracy'])],['Macro precision',pct(test['macro_precision']),pct(orig['test']['macro_precision'])],['Macro recall',pct(test['macro_recall']),pct(orig['test']['macro_recall'])],['Macro F1',pct(test['macro_f1']),pct(orig['test']['macro_f1'])],['Weighted F1',pct(test['weighted_f1']),pct(orig['test']['weighted_f1'])],['M F1',pct(test['per_class']['M']['f1']),pct(orig['test']['per_class']['M']['f1'])],['N F1',pct(test['per_class']['N']['f1']),pct(orig['test']['per_class']['N']['f1'])]], [.38,.31,.31])
h('Validation and the gap to test')
p(f'Validation contains 13,791 samples. Retained validation macro F1 is {pct(val["macro_f1"])} and accuracy is {pct(val["accuracy"])}. The large gap to test is consistent with source/domain mismatch and difficult or disputed examples; it is not proof of one particular preprocessing fault.')
h('What macro F1 measures')
p('For each class, precision measures how many predictions of that class are correct; recall measures how many true examples are recovered. F1 = 2 x precision x recall / (precision + recall). Macro F1 averages the 24 class F1 values equally; weighted F1 weights them by support. Accuracy counts all correct predictions over all rows.')
p('Removing the 2,253 owner-reviewed M test rows changed the test composition. The 63.07% score does not mean the model improved from training. Because the test has informed development, an untouched benchmark with verified identities is still needed.')
src('output/current-model-comparison/comparison.json; original evaluation in output/predeletion-training/.../comparison.json.')

section('7. Per-class cleaned-test results')
pc=[['Class','Support','Precision','Recall','F1']]
for label in test['labels']:
 m=test['per_class'][label];pc.append([label,str(m['support']),pct(m['precision']),pct(m['recall']),pct(m['f1'])])
table(pc,[.12,.19,.23,.23,.23])
weak=sorted(test['per_class'],key=lambda k:test['per_class'][k]['f1'])[:5]
p('Lowest class F1 values: '+', '.join(f'{k} ({pct(test["per_class"][k]["f1"])})' for k in weak)+'. Prioritize original-image review and targeted collection for these classes; do not remove hard examples solely to raise the score.','source')
src('Existing-model metrics on the reviewed 34,720-sample test set. Supports and confusion matrix are preserved in current-model-comparison/comparison.json.')

section('8. Retraining and model selection history')
table([['Run / evaluation','Validation macro F1','Test macro F1','Test accuracy'],['Retained old model / cleaned test','97.33%','63.07%','56.62%'],['Recent retraining / cleaned test','97.72%','57.77%','49.57%'],['Retained old model / original test','97.33%','59.49%','53.19%'],['Recent retraining / original test','97.72%','54.13%','46.56%']], [.43,.19,.19,.19])
p('Two 50-epoch retraining runs used the reviewed and pre-deletion snapshots with preserved partitions. Both selected epoch 43 by validation macro F1 and produced identical learned weights; this equality was checked before the experimental checkpoints were removed. Deletion affected only test rows, so the training and validation inputs were the same.')
p('The newer model improved validation slightly but underperformed the older model on both versions of the test. The older canonical checkpoint remains the selected application model. Its original training dataset hash and split manifest were not rewritten to match the cleaned CSV.')
h('Artifact disposition')
p('Ten unused experimental checkpoints were deleted at the owner\'s request: five static F1 variants, two dynamic variants, the weak-class candidate, and the two recent retraining checkpoints. Their metrics, predictions, manifests and cleanup hashes remain. Reports describing those models are historical evidence, not promises that the checkpoint files are still runnable.')
h('Dynamic results require a separate qualification')
p('The retained J/Z model scored 100% test macro F1 on only two clips, one per class, from the recovered 11-clip benchmark. That is insufficient for a reliability claim. Existing temporal diagnostics also flag possible shortcuts: tiny test performance staying unchanged after motion removal needs a much broader sequence benchmark.')
src('Retained model metadata; cleaned/pre-deletion comparisons; output/model-cleanup; recognition-reevaluation/summary.json.')

section('9. Validation and current checks')
table([['Check','Current evidence / limit'],['Fresh software suite',testmatch.group(0)+'. Warnings concern the existing tracing/export paths; see the saved log.'],['Static data contract','63 finite float32 features; wrist-centering and unit-radius checks; required preprocessing and source groups.'],['Partition integrity','Every row assigned once; each split contains all labels; session leakage, exact canonical feature leakage and conflicting exact labels rejected.'],['Import protections','Schema checks, archive path checks, byte preservation, duplicate/conflict detection, output locks, source grouping and audit records.'],['Source-image exclusion','Image-led decisions plus coordinate correspondence; retained rows byte-identical, only M rows excluded, source ZIP hash unchanged.'],['Retained extractor package','2,016 package files matched the ZIP by hash. Packaged smoke import completed, skipping two label conflicts and a blank no-hand image.']], [.28,.72])
h('What these checks do not establish')
p('Passing integrity checks cannot verify every linguistic label, signer identity or subtle near-duplicate. The package smoke test checks startup and a limited import path; it does not quantify landmark quality on representative images. Offline predictions omit detector failures, unknown-sign rejection, transitions, repeated gestures and live end-to-end latency.')
p('The packaged extractor is the newest retained packaged release, not a statement that its embedded code was rebuilt after every later source-code repair. The smoke audit records its preprocessing identifier. Preserve the versioned contracts and re-extract into a separate file when comparing package and current-source behavior.')
src('output/system-cleanup/current-report-tests.log; package-verification.json; smoke extraction audit; ml/dataset.py; tests/test_extractor_integrity.py.')

section('10. Operations and workspace cleanup')
table([['Retained component','Location / use'],['Static checkpoint','ml/models/sign_classifier_canonical.pth'],['Dynamic checkpoint','ml/models/sequence_classifier.pth'],['Current extractor package','output/extractor-reviewed/ASignX_DataExtractor'],['Matching distribution ZIP','output/extractor-reviewed/ASignX_DataExtractor.zip'],['Dynamic Studio','dist/ASignX_DynamicStudio.exe'],['Current comparison','output/current-model-comparison/report.md']], [.32,.68])
h('Launch commands from the project folder')
p('.\\ASignX_TestRecognition.bat','code')
p('.\\ASignX_DataExtractor.bat','code')
p('.\\ASignX_DynamicStudio.bat','code')
p('The recognition launcher selects the retained static and sequence checkpoints. The extractor launcher selects the reviewed package and falls back to Python source if it is missing. The Dynamic Studio launcher keeps its executable-first behavior.')
h('Removed and preserved')
p(f'Build/package cleanup removed {cleanup["bytes_removed"]/1024**3:.2f} GiB across 14 directories: three older extractor packages, build intermediates, five temporary build folders and caches. The launcher and guide were updated; build outputs and .venv-py314 were added to ignore rules. Git\'s index was not modified.')
p('Both active model hashes, the main dataset, historical backups, reviewed extractor, its ZIP and Dynamic Studio executable were checked unchanged. Audit/recovery scripts and temporary dependency installations were kept. Reports and manifests for deleted experiments remain intentionally historical.')
src('ASignX_*.bat; .gitignore; docs/extractor-guide.md; output/system-cleanup/cleanup-record.json; output/model-cleanup/*.')

section('11. Remaining issues and next steps')
table([['Priority','Action','Evidence of completion'],['1 - Label evidence','Review weak-class images by original source; resolve the 207 unmatched archive-4 M rows without deleting by prediction alone.','Recorded image identity, reviewer decision, reason and traceable row correction.'],['2 - Independent benchmark','Collect new signers and recording sessions across cameras, lighting, backgrounds, distances and hand orientations. Reserve an untouched global identity split.','All static labels and J/Z covered; signer/source groups stay in one split across labels.'],['3 - Targeted training data','Add verified examples for weak classes and confusing pairs. Preserve pose diversity and audit near-duplicates across partitions.','Source-balanced learning curves and per-class precision/recall on validation; one final held-out test.'],['4 - Dynamic evidence','Expand J/Z clips, wrong attempts, idle frames and transitions; assess gesture events and motion-removal diagnostics.','Per-class support large enough to report uncertainty; demonstrated temporal benefit.'],['5 - Pipeline parity','Compare current-source and packaged extraction on the same verified images; track mirror/handedness, padding and preprocessing versions.','Same feature contract or documented migration into a separate dataset.'],['6 - Runtime benchmark','Measure live detector failures, end-to-end latency, false triggers, coverage and calibrated unknown rejection.','Reproducible live sessions with device and environment metadata.']], [.15,.46,.39])
p('Keep the current model until a candidate passes a predefined validation gate and then improves on an independent benchmark. Do not shrink the test set, change labels from predictions, or repeatedly tune against the historical test to create a higher headline score.')
src('Current per-class results, unresolved-row audit, historical source diagnostics and dynamic benchmark support.')

section('12. Evidence, identity and reproducibility')
p('This report captures the repository state on 22 October 2026 in Asia/Singapore. Recognition scores were read from saved evaluations tied to explicit dataset/model identities. Software tests were rerun for this report. Models and datasets were not changed to produce the document.')
for title,value in [('Current cleaned static CSV',datahash),('Original static dataset',sc['dataset_sha256']),('Retained static checkpoint',sha(ROOT/'ml/models/sign_classifier_canonical.pth')),('Retained sequence checkpoint',sha(ROOT/'ml/models/sequence_classifier.pth'))]:
 h(title);p(value[:32]+'<br/>'+value[32:],'code')
h('Primary local evidence index')
for label,path in [('Current evaluation','output/current-model-comparison/comparison.json'),('Image decisions and filenames','output/archive4-M-pose-review/'),('Correction chain and backups','archive/data-corrections/'),('Latest correction summary','output/data-repair/latest-correction.json'),('Original-test historical scores','output/predeletion-training/20261021T162040Z/comparison.json'),('Removed model inventory','output/model-cleanup/'),('Package/build cleanup and fresh tests','output/system-cleanup/')]:p(label+': '+path,'source')
p('The companion .evidence.json records source hashes, model configurations, current counts and the fresh test result. The editable Markdown and builder are under docs/. Retained historical model manifests are bound to their original data hash: using them directly with the cleaned CSV should fail strict identity checks, not be bypassed by overwriting metadata.')

def safe(text):return escape(str(text)).replace('&lt;br/&gt;','<br/>')
def make_table(rows,fracs):
 n=len(rows[0]);fracs=fracs or [1/n]*n
 cells=[[Paragraph(safe(x),styles['th' if i==0 else 'cell']) for x in row] for i,row in enumerate(rows)]
 t=Table(cells,colWidths=[W*f for f in fracs],repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[PALE,colors.HexColor('#f9fbfc')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#ccdde5'))]))
 return t
def diagram():
 d=Drawing(W,123)
 blocks=[('Camera','OpenCV'),('Hand points','MediaPipe Tasks'),('Features','Versioned contracts'),('Predictors','Static + sequence')]
 bw=(W-36)/4
 for i,(top,bottom) in enumerate(blocks):
  x=i*(bw+12);d.add(Rect(x,61,bw,49,rx=5,fillColor=PALE,strokeColor=TEAL,strokeWidth=.8));d.add(String(x+bw/2,90,top,fontName='Bold',fontSize=8.8,textAnchor='middle',fillColor=NAVY));d.add(String(x+bw/2,75,bottom,fontName='Body',fontSize=7.7,textAnchor='middle',fillColor=NAVY))
  if i<3:d.add(Line(x+bw+2,85,x+bw+10,85,strokeColor=TEAL))
 d.add(Rect(0,5,W,35,rx=4,fillColor=NAVY,strokeColor=None));d.add(String(W/2,18,'Confidence + stability + motion routing -> display / recognition events',fontName='Body',fontSize=9,textAnchor='middle',fillColor=colors.white));d.add(Line(W-bw/2,60,W-bw/2,41,strokeColor=TEAL))
 return d
story=[];md=[]
for i,pg in enumerate(pages):
 if i:story.append(PageBreak());md.append('\n---\n')
 story.append(Paragraph(safe(pg['title']).replace('\n','<br/>'),styles['title' if i==0 else 'h1']));md+=['# '+pg['title'].replace('\n',' - '),'']
 for kind,value,extra in pg['items']:
  if kind=='p':story.append(Paragraph(safe(value),styles[extra]));md.extend([('## ' if extra=='h2' else '')+value.replace('<br/>','\n'),''])
  elif kind=='table':
   story.extend([make_table(value,extra),Spacer(1,10)]);md+=['| '+' | '.join(r)+' |' for r in value]+['']
  else:story.extend([diagram(),Spacer(1,6)]);md+=['Camera -> MediaPipe hand points -> versioned features -> static/sequence predictors -> runtime gating -> display','']
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#ccdde5'));c.line(50,A4[1]-37,A4[0]-50,A4[1]-37);c.setFont('Bold',8);c.setFillColor(TEAL);c.drawString(50,A4[1]-27,'ASIGNX / CURRENT SYSTEM STATE');c.setFont('Body',8);c.setFillColor(GRAY);c.drawString(50,28,'22 October 2026 | Evidence-backed repository review');c.drawRightString(A4[0]-50,28,str(doc.page))
class Doc(SimpleDocTemplate):
 def afterFlowable(self,item):
  if isinstance(item,Paragraph) and item.style.name in ('h1','title'):
   key=f'section-{self.page}';self.canv.bookmarkPage(key);self.canv.addOutlineEntry(item.getPlainText(),key,0)
Doc(str(OUT),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=56,bottomMargin=49,title='ASignX - Current System State - 22 October 2026',author='ASignX project review').build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'docs/current-system-state-2026-10-22.md').write_text('\n'.join(md),encoding='utf-8')
qa=ROOT/'tmp/pdfs/current-state-20261022';qa.mkdir(parents=True,exist_ok=True);pdf=pymupdf.open(OUT)
assert len(pdf)==len(pages),(len(pdf),len(pages))
thumbs=[];text=[]
for i,pg in enumerate(pdf):
 text.append(pg.get_text());pix=pg.get_pixmap(matrix=pymupdf.Matrix(1.35,1.35),alpha=False);path=qa/f'page-{i+1:02}.png';pix.save(path)
 im=Image.open(path).convert('RGB');im.thumbnail((298,422));thumbs.append(im)
 # All text should stay in the page and outside the footer rule.
 for block in pg.get_text('blocks'):
  assert block[0]>=45 and block[2]<=A4[0]-44 and block[1]>=12 and block[3]<=A4[1]-12,(i,block[:4])
sheet=Image.new('RGB',(4*318,4*454),'#dce5eb');draw=ImageDraw.Draw(sheet)
for i,im in enumerate(thumbs):x=(i%4)*318+10;y=(i//4)*454+10;sheet.paste(im,(x,y));draw.text((x,y+425),f'Page {i+1}',fill='#17324d')
sheet.save(qa/'contact-sheet.jpg',quality=92);(qa/'extracted-text.txt').write_text('\n'.join(text),encoding='utf-8')
joined='\n'.join(text)
assert all(s in joined for s in ['63.07%','59.49%','157,734','207','114 passed','2,253','296,641'])
assert sha(ROOT/'ml/data/landmarks.csv')==datahash
print(json.dumps({'pdf':str(OUT),'pages':len(pdf),'contact_sheet':str(qa/'contact-sheet.jpg'),'tests':testmatch.group(0)},indent=2))
