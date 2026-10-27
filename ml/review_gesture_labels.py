"""Review captured motions as HELLO/BACKGROUND/UNSURE without altering takes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(0,17),(17,18),(18,19),(19,20)]


def project_frame(features):
    """Reconstruct image-plane hand positions from the 180-feature contract.

    Hand-local coordinates lost their scale, but the stored wrist/index-tip
    offsets and head width recover it. No arbitrary hand-size estimate is used.
    This reconstructs tracked geometry, not original pixels or physical contact.
    """
    row=np.asarray(features,dtype=np.float32)
    if row.shape!=(180,) or not np.isfinite(row).all():
        raise ValueError('Review requires finite 180-feature body-context takes')
    context=row[126:150].reshape(8,3)
    body=np.where(context[:,2:3]==1,context[:,:2],np.nan)
    anchors=row[150:156].reshape(2,3)
    temples=np.where(anchors[:,2:3]==1,anchors[:,:2],np.nan)
    offsets=row[156:].reshape(2,2,2,3)
    hands=np.full((2,21,2),np.nan,dtype=np.float32)
    if np.isfinite(temples).all():
        head_width=float(np.linalg.norm(temples[0]-temples[1]))
        for slot in range(2):
            local=row[slot*63:(slot+1)*63].reshape(21,3)[:,:2]
            wrist=body[6+slot]
            if not np.isfinite(wrist).all() or not np.any(local):continue
            anchor=next((a for a in range(2) if offsets[slot,a,1,2]==1),None)
            if anchor is None:continue
            tip=temples[anchor]+offsets[slot,anchor,1,:2]*head_width
            vector=local[8]
            denominator=float(vector@vector)
            if denominator<1e-8:continue
            scale=float((tip-wrist)@vector)/denominator
            if scale<=0:continue
            hands[slot]=wrist+local*scale
    return body,temples,hands


def make_review_take(path,features,stamps,metadata):
    projections=[project_frame(row) for row in features]
    all_points=np.concatenate([np.concatenate((body,temples,hands.reshape(-1,2)))
                               for body,temples,hands in projections])
    finite=all_points[np.isfinite(all_points).all(axis=1)]
    if not len(finite):raise ValueError(f'No reviewable geometry in {path.name}')
    return {'path':path,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'metadata':metadata,'times':stamps-stamps[0],'poses':projections,
            'low':finite.min(axis=0)-.15,'high':finite.max(axis=0)+.15}


def load_review_takes(run, target='REST', include_unaccepted=False, selected_files=None):
    takes=[]
    for path in sorted((Path(run)/'gesture_inputs').glob('*.npz')):
        if selected_files is not None and path.name not in selected_files:continue
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            if metadata.get('target')!=target:continue
            if not include_unaccepted and not metadata.get('emitted'):continue
            if include_unaccepted:metadata['hide_model']=True
            features=data['frames'].copy(); stamps=data['timestamps'].copy()
            if 'present' in data:
                features[~data['present']]=0
        if features.ndim!=2 or features.shape[1]!=180:continue
        takes.append(make_review_take(path,features,stamps,metadata))
    if not takes:raise ValueError(f'No supported {target} captures found in this run')
    return takes


def load_saved_backgrounds(directory):
    from ml.sequence_io import load_sequence_file
    takes=[]
    for path in sorted(Path(directory).rglob('*.npz')):
        sample=load_sequence_file(path)
        if sample.label!='BACKGROUND':continue
        if sample.features.shape[1]!=180:
            raise ValueError('Saved-example review currently requires 180-feature recordings')
        duration=float(sample.metadata.get('duration_seconds',(len(sample.features)-1)/15))
        if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid saved recording duration')
        metadata={'target':sample.label,'emitted':None,'source_type':'saved_dataset',
                  'playback_timing':'Uniform gesture progress; original frame timestamps unavailable',
                  'hide_model':True}
        stamps=np.linspace(0,duration,len(sample.features))
        takes.append(make_review_take(path,sample.features,stamps,metadata))
    if not takes:raise ValueError('No saved BACKGROUND examples in the selected folder')
    return takes


def save_decision(path, take, label):
    if label not in ('J','Z','HELLO','BACKGROUND','UNSURE'):raise ValueError('Invalid review label')
    path=Path(path)
    payload=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'purpose':'Human label review; original inputs unchanged; not applied to training',
        'accepted_hello_forms':['forehead-salute','ordinary greeting wave'],'decisions':{}}
    key=take['path'].name
    previous=payload['decisions'].get(key)
    if previous and previous['sha256']!=take['sha256']:
        raise ValueError('A reviewed source file changed; preserve the previous review')
    payload['decisions'][key]={'source':str(take['path'].resolve()),'sha256':take['sha256'],
                               'review_label':label,'original_target':take['metadata']['target'],
                               'model_output':take['metadata']['emitted'],
                               'source_type':take['metadata'].get('source_type','diagnostic_capture')}
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8');os.replace(temporary,path)


class LabelReviewer:
    def __init__(self,root,run,takes=None,pilot_vocabulary=False):
        self.root=root;self.run=Path(run);self.takes=load_review_takes(run) if takes is None else takes
        self.run.mkdir(parents=True,exist_ok=True)
        self.output=self.run/'label_review.json';self.index=0;self.playing=True;self.start=time.monotonic()
        self.updating_slider=False
        root.title('ASignX - Review HELLO examples');root.geometry('1000x780')
        outer=ttk.Frame(root,padding=14);outer.pack(fill='both',expand=True)
        question='Which sign is actually shown?' if pilot_vocabulary else 'Is this a HELLO greeting or an unrelated movement?'
        guidance=('Choose J, Z, or HELLO for a recognizable sign. BACKGROUND means ordinary non-sign movement. Choose Unsure if unclear.'
                  if pilot_vocabulary else 'HELLO includes a salute and a greeting wave. Head contact is not required. Choose Unsure if the replay is unclear.')
        ttk.Label(outer,text=question,font=('Segoe UI',17,'bold')).pack(anchor='w')
        ttk.Label(outer,text=guidance,wraplength=960).pack(anchor='w',pady=7)
        self.title=tk.StringVar();ttk.Label(outer,textvariable=self.title,wraplength=960).pack(anchor='w')
        self.canvas=tk.Canvas(outer,width=940,height=470,bg='#101827',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True,pady=8)
        self.step=tk.IntVar(value=0)
        self.slider=ttk.Scale(outer,from_=0,to=1,command=self.seek);self.slider.pack(fill='x')
        controls=ttk.Frame(outer);controls.pack(fill='x',pady=8)
        for title,callback in [('Previous',lambda:self.change(-1)),('Play / pause',self.toggle),('Next',lambda:self.change(1))]:
            ttk.Button(controls,text=title,command=callback).pack(side='left',padx=4)
        choices=[('Valid HELLO (H)','HELLO'),('Unrelated movement (B)','BACKGROUND'),('Unsure (U)','UNSURE')]
        if pilot_vocabulary:choices=[('Valid J (J)','J'),('Valid Z (Z)','Z')]+choices
        for title,label in choices:
            ttk.Button(controls,text=title,command=lambda label=label:self.mark(label)).pack(side='left',padx=4)
        self.status=tk.StringVar(value='Slow replay at half speed. Slider pauses playback. Decisions save immediately.')
        ttk.Label(outer,textvariable=self.status,wraplength=960).pack(anchor='w')
        ttk.Label(outer,text='Tracked stick figure only: cannot show original appearance, palm texture, facial expression or physical contact.').pack(anchor='w',pady=5)
        for key,label in [(label[0].lower() if label!='BACKGROUND' else 'b',label) for _,label in choices]:
            root.bind(key,lambda event,label=label:self.mark(label))
        root.bind('<space>',lambda event:self.toggle())
        self.show_take();root.after(35,self.tick)

    def show_take(self):
        take=self.takes[self.index];self.start=time.monotonic();self.playing=True;self.step.set(0)
        self.slider.configure(to=max(1,len(take['poses'])-1));self.slider.set(0)
        self.playing=True
        if take['metadata'].get('hide_model'):
            self.title.set(f"Example {self.index+1}/{len(self.takes)} | Judge the movement; no model scores shown.")
        else:
            self.title.set(f"Example {self.index+1}/{len(self.takes)} | {take['path'].name} | Recorded in {take['metadata']['target']}; model output: {take['metadata']['emitted']}")

    def seek(self,value):
        if self.updating_slider:return
        self.step.set(int(float(value)));self.playing=False;self.draw()

    def change(self,offset):
        self.index=(self.index+offset)%len(self.takes);self.show_take()

    def toggle(self):
        self.playing=not self.playing
        if self.playing:
            self.start=time.monotonic()-float(self.takes[self.index]['times'][self.step.get()])*2

    def mark(self,label):
        try:save_decision(self.output,self.takes[self.index],label)
        except (OSError,ValueError) as error:
            messagebox.showerror('Review could not save',str(error),parent=self.root);return
        payload=json.loads(self.output.read_text(encoding='utf-8'))
        completed=sum(t['path'].name in payload['decisions'] for t in self.takes)
        self.status.set(f'Saved {label}. Reviewed {completed}/{len(self.takes)}. Close the window when finished.')
        if completed==len(self.takes):
            self.playing=False;return
        self.change(1)

    def draw(self):
        take=self.takes[self.index];i=min(self.step.get(),len(take['poses'])-1)
        body,temples,hands=take['poses'][i];self.canvas.delete('all')
        w=max(100,self.canvas.winfo_width());h=max(100,self.canvas.winfo_height())
        span=np.maximum(take['high']-take['low'],.1);scale=min((w-60)/span[0],(h-60)/span[1])
        origin=np.array([w/2,h/2])-(take['high']+take['low'])/2*scale
        def xy(p):return tuple(origin+p*scale)
        def line(a,b,color,width=2):
            if np.isfinite([a,b]).all():self.canvas.create_line(*xy(a),*xy(b),fill=color,width=width)
        for a,b in [(2,3),(2,4),(3,5),(4,6),(5,7)]:line(body[a],body[b],'#8695ad',3)
        line(temples[0],temples[1],'#cbd5e1',4)
        line(body[0],body[1],'#cbd5e1',3)
        for slot,color in enumerate(['#5eead4','#fbbf24']):
            for a,b in HAND_EDGES:line(hands[slot,a],hands[slot,b],color,3)
            for p in hands[slot]:
                if np.isfinite(p).all():
                    x,y=xy(p);self.canvas.create_oval(x-3,y-3,x+3,y+3,fill=color,outline='')
        for p in np.concatenate((body,temples)):
            if np.isfinite(p).all():
                x,y=xy(p);self.canvas.create_oval(x-4,y-4,x+4,y+4,fill='#cbd5e1',outline='')
        self.canvas.create_text(12,12,text=f'Observation {i+1}/{len(take["poses"])} | Left hand: teal; right: gold | 0.5x playback',anchor='nw',fill='#e2e8f0')

    def tick(self):
        if self.playing:
            take=self.takes[self.index];duration=float(take['times'][-1])
            elapsed=((time.monotonic()-self.start)*.5)%(duration+.5)
            self.step.set(min(int(np.searchsorted(take['times'],elapsed)),len(take['poses'])-1))
            self.updating_slider=True;self.slider.set(self.step.get());self.updating_slider=False
        self.draw();self.root.after(35,self.tick)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run');source.add_argument('--sequences')
    parser.add_argument('--output-dir')
    parser.add_argument('--target',choices=['REST','HELLO','UNMARKED','J','Z'],default='REST')
    parser.add_argument('--selection',type=Path,help='JSON list of captured take filenames to review')
    parser.add_argument('--pilot-vocabulary',action='store_true',help='Allow J/Z/HELLO/BACKGROUND/UNSURE decisions')
    parser.add_argument('--include-unaccepted',action='store_true',
                        help='Include rejected and uncertain captures; hide model outputs during review')
    args=parser.parse_args();root=tk.Tk()
    try:
        if args.sequences:
            LabelReviewer(root,args.output_dir or 'output/label-reviews/saved-background',
                          takes=load_saved_backgrounds(args.sequences),pilot_vocabulary=args.pilot_vocabulary)
        else:
            selected=None
            if args.selection:
                names=json.loads(args.selection.read_text(encoding='utf-8'))
                if not isinstance(names,list) or not all(isinstance(n,str) and Path(n).name==n for n in names):
                    raise ValueError('Selection must be a list of take filenames')
                selected=set(names)
            takes=load_review_takes(args.run,target=args.target,
                                   include_unaccepted=args.include_unaccepted,selected_files=selected)
            if selected is not None and {t['path'].name for t in takes}!=selected:
                raise ValueError('Some selected captures are missing or incompatible')
            LabelReviewer(root,args.output_dir or args.run,takes=takes,pilot_vocabulary=args.pilot_vocabulary)
    except (ValueError,OSError) as error:
        root.withdraw();messagebox.showerror('Cannot open review',str(error));root.destroy();return
    root.mainloop()


if __name__=='__main__':main()
