"""Build and render the detailed system report from its editable Markdown."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
import pymupdf
from PIL import Image, ImageDraw

EVIDENCE = ROOT/'output/system-report-2026-10-22/evidence.json'
e = json.loads(EVIDENCE.read_text(encoding='utf-8'))
source = ROOT/'docs/detailed-system-report-2026-10-22.md'
OUT = ROOT/'output/pdf/ASignX-Detailed-System-Report-2026-10-22.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
qa = ROOT/'tmp/pdfs/detailed-system-report-20261022'
qa.mkdir(parents=True, exist_ok=True)

def pct(value): return f'{value:.2%}'
def mdtable(rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in [rows[0], ['---']*len(rows[0]), *rows[1:]])

manual = next(k for k in e['paired_evaluation'] if 'manual_runs' in k)
canonical = 'ml/models/sign_classifier_canonical.pth'
model_names = {canonical:'Canonical default', manual:'Manual run'}
classes = list(e['dataset']['class_counts'].items())
class_rows = [['Class','Rows','Class','Rows','Class','Rows']]
for i in range(8):
    class_rows.append([v for j in range(3) for v in (classes[i+j*8][0], f'{classes[i+j*8][1]:,}')])
paired = [['Model / partition','Samples','Accuracy','Macro F1','Weighted F1']]
for key in [canonical, manual]:
    for part, scores in e['paired_evaluation'][key].items():
        paired.append([model_names[key]+' / '+part, f"{e['current_split_counts'][part]:,}",pct(scores['accuracy']),pct(scores['macro_f1']),pct(scores['weighted_f1'])])
overlap = [['New partition','From old train sessions','From old validation sessions','From old test sessions']]
for part, counts in e['cross_benchmark_overlap'].items():
    overlap.append([part, *[f'{counts[k]:,}' for k in ['train','validation','test']]])
metrics = e['paired_evaluation'][manual]['test']
per_class = [['Class','Support','Precision','Recall','F1']]
for label, m in metrics['per_class'].items():
    per_class.append([label,f"{m['support']:,}",pct(m['precision']),pct(m['recall']),pct(m['f1'])])
weak = sorted(metrics['per_class'].items(), key=lambda item:item[1]['f1'])[:5]
weak_text = 'The current manual-test lowest F1 classes are '+', '.join(f"{k} ({pct(v['f1'])})" for k,v in weak)+'. '
matrix = metrics['confusion_matrix']; labels = metrics['labels']
errors = sorted([(matrix[i][j],labels[i],labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j], reverse=True)[:5]
weak_text += 'The largest current confusion counts are '+', '.join(f'{a} predicted as {b}: {n:,}' for n,a,b in errors)+'.'
identities = [
    'Current main CSV SHA-256: '+e['dataset']['sha256'],
    'Historical cleaned CSV SHA-256: 4a160bec73556408c12f9457577c765c89e9f8cf09eae3d658fe39d3064f58cd',
    'Historical original CSV SHA-256: '+e['models'][canonical]['dataset_sha256'],
]
for key in [canonical,manual,'ml/models/sequence_classifier.pth']:
    identities.append(key+' SHA-256: '+e['models'][key]['sha256'])
replacements = {'CLASS_COUNTS':mdtable(class_rows),'PAIRED_RESULTS':mdtable(paired),
                'OVERLAP':mdtable(overlap),'PER_CLASS':mdtable(per_class),
                'WEAK_CLASSES':weak_text,'IDENTITIES':'\n\n'.join(identities)}
text = source.read_text(encoding='utf-8')
for key, value in replacements.items():
    text = text.replace('{{'+key+'}}',value)
assert '{{' not in text
source.write_text(text,encoding='utf-8')

font_dir = Path('C:/Windows/Fonts')
for name, filename in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold')
NAVY=colors.HexColor('#17324d'); TEAL=colors.HexColor('#007e87'); GRAY=colors.HexColor('#526577')
PAGE_W,PAGE_H=A4; WIDTH=PAGE_W-104
styles = {
    'title':ParagraphStyle('title',fontName='BodyBold',fontSize=29,leading=35,textColor=NAVY,spaceAfter=20),
    'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=18,leading=23,textColor=NAVY,spaceBefore=17,spaceAfter=12,keepWithNext=True),
    'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=12,leading=17,textColor=TEAL,spaceBefore=10,spaceAfter=8,keepWithNext=True),
    'body':ParagraphStyle('body',fontName='Body',fontSize=10,leading=14.5,textColor=NAVY,spaceAfter=9,splitLongWords=True,allowWidows=0,allowOrphans=0),
    'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.6,leading=12,textColor=NAVY,splitLongWords=True),
    'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.5,leading=11.7,textColor=colors.white),
    'toc':ParagraphStyle('toc',fontName='Body',fontSize=9,leading=12,textColor=NAVY,spaceBefore=0,spaceAfter=0,leftIndent=0,firstLineIndent=0),
}

def markup(value):
    value = escape(value)
    return re.sub(r'https://[^\s]+',lambda m: '<link href="'+m.group(0)+'" color="#007e87">'+m.group(0)+'</link>',value)

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d6e2e9'))
    canvas.line(52,PAGE_H-38,PAGE_W-52,PAGE_H-38)
    canvas.setFont('BodyBold',8);canvas.setFillColor(TEAL)
    canvas.drawString(52,PAGE_H-28,'ASIGNX / DETAILED SYSTEM REPORT')
    canvas.setFont('Body',8);canvas.setFillColor(GRAY)
    canvas.drawString(52,29,'22 October 2026 | Repository evidence and explained machine learning')
    canvas.drawRightString(PAGE_W-52,29,str(doc.page))
    canvas.restoreState()

def table(rows):
    size=len(rows[0])
    if size==2: ratios=[.30,.70]
    elif size==3:ratios=[.24,.31,.45]
    elif size==4:ratios=[.34,.22,.22,.22]
    elif size==5:ratios=[.34,.14,.17,.18,.17] if rows[0][0].startswith('Model') else [.12,.20,.23,.23,.22]
    elif size==6:ratios=[.10,.23,.10,.23,.10,.24]
    else:ratios=[1/size]*size
    if rows[0][0]=='Priority':ratios=[.10,.31,.59]
    if rows[0][0]=='New partition':ratios=[.19,.27,.27,.27]
    cells=[[Paragraph(markup(value),styles['th' if i==0 else 'cell']) for value in row] for i,row in enumerate(rows)]
    tab=Table(cells,colWidths=[WIDTH*x for x in ratios],repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf5f7'),colors.HexColor('#f7fafc')]),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#d6e2e9')),
    ]))
    return tab

class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='h1':
            title=flowable.getPlainText();key='section-'+re.match(r'\d+',title).group(0)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0)
            self.notify('TOCEntry',(0,title,self.page,key))

story=[];lines=text.splitlines();i=0; toc_added=False
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line=='---PAGE---':
        story.append(PageBreak())
        story.append(Paragraph('Contents',styles['h2']))
        toc=TableOfContents();toc.levelStyles=[styles['toc']];toc.dotsMinLevel=0
        story.extend([toc,PageBreak()]);toc_added=True;i+=1;continue
    if line.startswith('# '):
        style='h1' if re.match(r'# \d+\.',line) else 'title'
        story.append(Paragraph(markup(line[2:]),styles[style]));i+=1;continue
    if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['h2']));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',cell) for cell in cells):rows.append(cells)
            i+=1
        story.extend([table(rows),Spacer(1,12)]);continue
    para=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','---PAGE---')):
        para.append(lines[i].strip());i+=1
    story.append(Paragraph(markup(' '.join(para)),styles['body']))

doc=ReportDoc(str(OUT),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=57,bottomMargin=51,
              title='ASignX - Detailed System Report - 22 October 2026',
              author='ASignX project review',subject='Architecture, explained ML, data, results and browser development')
doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
pdf=pymupdf.open(OUT);thumbs=[];page_counts=[];bounds=[]
for index,page in enumerate(pdf):
    page_counts.append(len(page.get_text().split()))
    pix=page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False)
    path=qa/f'page-{index+1:02}.png';pix.save(path)
    im=Image.open(path).convert('RGB');im.thumbnail((298,422))
    thumb=Image.new('RGB',(318,450),'#dce5eb');thumb.paste(im,((318-im.width)//2,5))
    ImageDraw.Draw(thumb).text((12,430),f'Page {index+1}',fill='#17324d');thumbs.append(thumb)
    for block in page.get_text('blocks'):
        if block[0]<50 or block[2]>PAGE_W-49 or block[1]<12 or block[3]>PAGE_H-12:bounds.append([index+1,list(block[:4])])
for start in range(0,len(thumbs),9):
    sheet=Image.new('RGB',(954,1350),'white')
    for j,im in enumerate(thumbs[start:start+9]):sheet.paste(im,((j%3)*318,(j//3)*450))
    sheet.save(qa/f'contact-{start//9+1}.jpg',quality=93)
extracted='\n\f\n'.join(page.get_text() for page in pdf)
(qa/'extracted.txt').write_text(extracted,encoding='utf-8')
assert not bounds,bounds
for value in ['159,534','74.23%','74.84%','947','114 tests passed','1,800','296,641','63,681']:
    assert value in extracted,value
assert '{{' not in extracted
assert all(n>30 for n in page_counts),page_counts
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT/'ml/data/landmarks.csv')==e['dataset']['sha256']
for key,m in e['models'].items():assert sha(ROOT/key)==m['sha256']
qa_evidence={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'pdf':str(OUT.relative_to(ROOT)),
    'pdf_sha256':sha(OUT),'source_sha256':sha(source),'evidence_sha256':sha(EVIDENCE),
    'pages':len(pdf),'word_count':len(text.split()),'page_word_counts':page_counts,
    'text_bounds_issues':bounds,'datasets_and_models_unchanged':True,
    'render_directory':str(qa.relative_to(ROOT)),'visual_review':'Rendered; review contact sheets before delivery.'}
(OUT.with_suffix('.qa.json')).write_text(json.dumps(qa_evidence,indent=2),encoding='utf-8')
print(json.dumps(qa_evidence,indent=2))
