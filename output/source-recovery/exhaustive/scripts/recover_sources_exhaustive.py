"""Resumable image-to-current-row provenance recovery for the two legacy archives."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import argparse, hashlib, json, sqlite3, sys, time, zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from asignx.detector import HandDetector
from ml.data_io import read_samples
from ml.data_quality import file_digest
from ml.extract_dataset import _detect_hand_with_retries
from ml.preprocessing import normalize_landmarks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',choices=['archive3','archive5'],required=True)
    parser.add_argument('--subject-folder',choices=list('ABCDE'),help='Parallel archive3 partition; separate resumable cache')
    args=parser.parse_args()
    if args.subject_folder and args.archive!='archive3':parser.error('Subject partitions apply only to archive3')
    cv2.setNumThreads(1)
    out=ROOT/'output/source-recovery/exhaustive';out.mkdir(parents=True,exist_ok=True)
    archive=Path('C:/Users/jibbi/Downloads')/('archive (3).zip' if args.archive=='archive3' else 'archive (5).zip')
    data=ROOT/'ml/data/landmarks.csv'
    print('Loading dataset and verifying recovery inputs...',flush=True)
    metadata={'dataset_sha256':file_digest(data),'archive_sha256':file_digest(archive),
              'detector_sha256':file_digest(ROOT/'asignx/assets/hand_landmarker.task'),
              'extractor_sha256':file_digest(ROOT/'ml/extract_dataset.py'),
              'tolerance':1e-6,'handedness_policy':'must_match','opencv':cv2.__version__}
    if args.subject_folder:metadata['subject_folder_partition']=args.subject_folder
    samples=read_samples(data)
    labels={s.label for s in samples};features=np.stack([s.features for s in samples])
    buckets=defaultdict(list);dims=[3,12,27];width=.02;tol=1e-6
    for i,s in enumerate(samples):
        cell=tuple(np.floor(s.features[dims]/width).astype(int))
        buckets[(s.label,s.handedness,*cell)].append(i)
    cache_name=args.archive+('-'+args.subject_folder if args.subject_folder else '')
    db=sqlite3.connect(out/f'{cache_name}.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
    prior=db.execute("SELECT value FROM meta WHERE key='inputs'").fetchone()
    encoded=json.dumps(metadata,sort_keys=True)
    if prior and prior[0]!=encoded:raise ValueError('Recovery input/config changed; use a new cache')
    db.execute("INSERT OR IGNORE INTO meta VALUES('inputs',?)",(encoded,))
    db.execute('CREATE TABLE IF NOT EXISTS images(member TEXT PRIMARY KEY,label TEXT,status TEXT,image_sha256 TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS matches(member TEXT,csv_row INTEGER,max_delta REAL,handedness TEXT,subject_folder TEXT,PRIMARY KEY(member,csv_row))')
    db.commit();done={r[0] for r in db.execute('SELECT member FROM images')}
    started=time.monotonic();processed=0
    with zipfile.ZipFile(archive) as z,HandDetector(static_image_mode=True,max_num_hands=2,min_detection_confidence=.5) as detector:
        members=[]
        for m in z.infolist():
            path=Path(m.filename)
            if path.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
            if args.archive=='archive3' and not path.name.startswith('color_'):continue
            if args.subject_folder and path.parts[1]!=args.subject_folder:continue
            classes=[p.upper() for p in path.parts[:-1] if len(p)==1 and p.isalpha()]
            if not classes or classes[-1] not in labels:continue
            if m.file_size>64*1024*1024:continue
            members.append((m,classes[-1]))
        print(f'{args.archive}: {len(members)} eligible color images; {len(done)} cached',flush=True)
        def progress(complete=False):
            db.commit()
            unique=db.execute('SELECT count(DISTINCT csv_row) FROM matches').fetchone()[0]
            count=db.execute('SELECT count(*) FROM images').fetchone()[0]
            elapsed=time.monotonic()-started
            state={'archive':str(archive),'dataset_sha256':metadata['dataset_sha256'],'total_images':len(members),
                   'processed_images':count,'unique_current_rows_matched':unique,'complete':complete,
                   'elapsed_seconds_this_process':round(elapsed,2),'images_per_second':round(processed/max(elapsed,.01),2),
                   'scope':'Image/recording-folder evidence only. No signer identity invented.'}
            (out/f'{cache_name}-progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
            print(json.dumps(state),flush=True)
        progress()
        for member,label in members:
            if member.filename in done:continue
            raw=z.read(member);image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            status='unreadable';match_rows=[]
            if image is not None:
                result=_detect_hand_with_retries(detector,image)
                status='no_single_hand'
                if len(result.all_hands)==1:
                    hand=result.all_hands[0]
                    try:f=normalize_landmarks(hand)
                    except ValueError:f=None
                    status='no_geometry_match'
                    if f is not None:
                        cells=[range(int(np.floor((float(f[d])-tol)/width)),int(np.floor((float(f[d])+tol)/width))+1) for d in dims]
                        candidates=[]
                        for cell in product(*cells):candidates.extend(buckets.get((label,hand.handedness,*cell),[]))
                        if candidates:
                            delta=np.max(np.abs(features[candidates]-f),axis=1)
                            subject=Path(member.filename).parts[1] if args.archive=='archive3' else ''
                            for pos in np.flatnonzero(delta<=tol):
                                i=candidates[pos]
                                match_rows.append((member.filename,i+2,float(delta[pos]),hand.handedness,subject))
                        if match_rows:status='geometry_match'
            image_hash=hashlib.sha256(raw).hexdigest()
            db.execute('INSERT INTO images VALUES(?,?,?,?)',(member.filename,label,status,image_hash))
            db.executemany('INSERT OR IGNORE INTO matches VALUES(?,?,?,?,?)',match_rows)
            processed+=1
            if processed%200==0:db.commit()
            if processed%1000==0:progress()
        progress(complete=True)
    assert file_digest(data)==metadata['dataset_sha256'],'Source dataset changed during recovery'
    db.close()

if __name__=='__main__':main()
