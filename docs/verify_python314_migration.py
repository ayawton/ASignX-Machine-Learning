"""Capture cross-version model/detector outputs and optionally validate ONNX parity."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from asignx.detector import HandDetector
from ml.model import load_checkpoint
from ml.sequence_model import load_sequence_checkpoint
from ml.preprocessing import canonicalize_static_features
from ml.sequence_io import load_sequence_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onnx', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result = {'python':sys.version,'packages':{p:metadata.version(p) for p in ['torch','numpy','mediapipe','opencv-contrib-python']},'models':{}}
    with (ROOT/'ml/data/landmarks.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        rows=[next(reader) for _ in range(128)]
    sequence = next((ROOT/'ml/data/recovered_sequence_baseline').rglob('*.npz'))
    inputs={'static':np.stack([canonicalize_static_features(np.array([float(row[f'{axis}{i}']) for i in range(21) for axis in 'xyz'],dtype=np.float32),row['handedness'],'left_x_reflection_v1') for row in rows]),
            'sequence':load_sequence_file(sequence).features[np.newaxis].astype(np.float32)}
    for path in sorted((ROOT/'ml/models').rglob('*.pth')):
        kind='sequence' if path.name=='sequence_classifier.pth' else 'static'
        loader=load_sequence_checkpoint if kind=='sequence' else load_checkpoint
        model,labels,_=loader(path);model.eval();x=inputs[kind]
        with torch.inference_mode(): logits=model(torch.from_numpy(x)).numpy()
        item={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'labels':labels,'logits':logits.tolist(),'predictions':logits.argmax(axis=1).tolist()}
        if args.onnx:
            import onnxruntime as ort
            from ml.export_onnx import export_static_checkpoint,export_sequence_checkpoint
            target=args.output.parent/'onnx'/path.parent.name/(path.stem+'.onnx')
            exporter=export_sequence_checkpoint if kind=='sequence' else export_static_checkpoint
            artifacts=exporter(path,target,export_weights_json=False)
            assert 'onnx' in artifacts, f'ONNX export fell back for {path}'
            session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
            y=session.run(None,{session.get_inputs()[0].name:x})[0]
            np.testing.assert_allclose(y,logits,atol=1e-4,rtol=1e-4)
            item['onnx_max_absolute_error']=float(np.max(np.abs(y-logits)))
        result['models'][str(path.relative_to(ROOT)).replace('\\','/')]=item
    image=cv2.imread(str(ROOT/'output/data-repair/row-110182-original.jpg'))
    assert image is not None
    from ml.extract_dataset import _detect_hand_with_retries
    with HandDetector(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as detector:
        detected=_detect_hand_with_retries(detector,image)
    result['image_detection']=[{'handedness':hand.handedness,'landmarks':hand.to_numpy().tolist()} for hand in detected.all_hands]
    assert result['image_detection'], 'Reference image should have a detectable hand'
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verified',len(result['models']),'models; output:',args.output)

if __name__=='__main__':main()
