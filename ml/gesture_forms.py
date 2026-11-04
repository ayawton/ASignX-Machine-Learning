"""Necessary form evidence for the J/Z/forehead-salute pilot.

Engineering guardrails, not an ASL correctness classifier. The temporal model
still determines the movement class. Geometry cannot prove physical contact.
"""
from dataclasses import dataclass
import numpy as np


def extended_fingers(points):
    """Four long-finger extension flags, independent of camera orientation."""
    points=np.asarray(points,dtype=np.float32)
    if points.shape!=(21,3) or not np.isfinite(points).all() or not np.any(points):
        return np.zeros(4,dtype=bool)
    a,b,c,d=(points[indices] for indices in ([5,9,13,17],[6,10,14,18],
                                            [7,11,15,19],[8,12,16,20]))
    u,v=a-b,c-b
    denominator=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    cosine=np.divide((u*v).sum(1),denominator,out=np.ones(4),where=denominator>1e-8)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    length=np.linalg.norm(b-a,axis=1)+np.linalg.norm(c-b,axis=1)+np.linalg.norm(d-c,axis=1)
    reach=np.divide(np.linalg.norm(d-a,axis=1),length,out=np.zeros(4),where=length>1e-8)
    return (angle>=145.) & (reach>=.8)


def sustained(stamps,mask,minimum):
    longest=current=0.
    for i,dt in enumerate(np.diff(stamps)):
        current=current+dt if mask[i] and mask[i+1] and 0<dt<=.2 else 0.
        longest=max(longest,current)
    return longest+1e-8>=minimum


@dataclass(frozen=True)
class GestureFormAcceptance:
    min_handshape_fraction: float=.35
    min_handshape_seconds: float=.10
    head_start_radius: float=.45  # index tip to a temple proxy, in head widths
    head_start_seconds: float=.30
    policy_version='j_z_salute_form_v1'

    def __post_init__(self):
        if not 0<self.min_handshape_fraction<=1:
            raise ValueError('Handshape fraction must be in (0, 1]')
        if any(not np.isfinite(v) or v<=0 for v in (self.min_handshape_seconds,
               self.head_start_radius,self.head_start_seconds)):
            raise ValueError('Form thresholds must be positive and finite')

    def validate(self,label,frames,*,motion_frames=None):
        if label not in ('J','Z','HELLO'):return
        observations=list(frames)
        if len(observations)<2:raise ValueError(f'{label}: too little form evidence')
        stamps=np.array([t for t,_ in observations],dtype=np.float64)
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
            raise ValueError('Form evidence timestamps must increase')
        shape=next((np.asarray(f).shape for _,f in observations if f is not None),None)
        if shape is None:raise ValueError(f'{label}: no visible handshape evidence')
        rows=np.stack([np.asarray(f,dtype=np.float32).reshape(-1) if f is not None
                       else np.zeros(int(np.prod(shape)),dtype=np.float32) for _,f in observations])
        if rows.shape[1] not in (63,126,150,180) or not np.isfinite(rows).all():
            raise ValueError('Unsupported form evidence features')
        slots=1 if rows.shape[1]==63 else 2
        poses=[rows[:,slot*63:(slot+1)*63].reshape(-1,21,3) for slot in range(slots)]
        visible=[np.any(points!=0,axis=(1,2)) for points in poses]
        extensions=[np.stack([extended_fingers(p) for p in points]) for points in poses]
        eligible=list(range(slots))
        # Do not borrow an idle second hand's correct shape for a moving fist.
        if motion_frames is not None and slots==2:
            excursions=[]
            for name in ('Left','Right'):
                values=[np.asarray(hands[name],dtype=np.float32) for _,hands in motion_frames if name in hands]
                if len(values)<3:excursions.append(0.);continue
                values=np.stack(values)[:,:,:2]
                scale=np.median(np.linalg.norm(values[:,5]-values[:,17],axis=1))
                track=values[:,[0,4,8,12,16,20]]
                extent=np.linalg.norm(np.quantile(track,.95,axis=0)-np.quantile(track,.05,axis=0),axis=1).max()
                excursions.append(float(extent/max(float(scale),1e-5)))
            if max(excursions)>0:
                eligible=[i for i,v in enumerate(excursions) if v>=max(excursions)*.5]
        matched=[]
        for slot in eligible:
            ext=extensions[slot]
            mask=(ext[:,3] & ((~ext[:,:3]).sum(1)>=2) if label=='J' else
                  ext[:,0] & ((~ext[:,1:]).sum(1)>=2) if label=='Z' else ext.sum(1)>=3)
            mask &= visible[slot]
            # Time weighting avoids treating camera bursts as sustained evidence.
            dt=np.diff(stamps)
            supported=float(dt[mask[:-1]&mask[1:]&(dt<=.2)].sum())
            duration=float(dt.sum())
            if supported/max(duration,1e-8)>=self.min_handshape_fraction and sustained(stamps,mask,self.min_handshape_seconds):
                matched.append((slot,mask))
        if not matched:
            requirement={'J':'extended pinky with the other fingers folded',
                         'Z':'extended index finger with the other fingers folded',
                         'HELLO':'an open hand'}[label]
            raise ValueError(f'{label} requires sustained {requirement}')
        if label!='HELLO':return
        if rows.shape[1]!=180:
            raise ValueError('Salute HELLO requires face-reference features')
        # Require open-hand/forehead proximity together at the start, not a
        # late incidental pass near the head. This is proximity, not contact.
        end=stamps[0]+min(self.head_start_seconds,(stamps[-1]-stamps[0])*.4)
        prefix=stamps<=end+1e-8
        offsets=rows[:,156:].reshape(len(rows),2,2,2,3)
        anchors=rows[:,150:156].reshape(len(rows),2,3)
        for slot,open_hand in matched:
            tip=offsets[:,slot,:,1,:]
            valid=(tip[:,:,2]==1)&(anchors[:,:,2]==1)
            near=(np.linalg.norm(tip[:,:,:2],axis=2)<=self.head_start_radius)&valid
            mask=near.any(axis=1)&open_hand&prefix
            # No artificial contact hold: a fast salute may leave the region
            # immediately. Two consecutive observations reject isolated spikes.
            if np.any(mask[:-1]&mask[1:]&(np.diff(stamps)<=.2)):return
        raise ValueError('Salute HELLO must start with an open hand near the forehead')
