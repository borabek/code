"""Goreli threshold: NESTED-CV dogrulamasi (ratio train-fold'da secilir, test-fold'da uygulanir)."""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
r=np.load('results/rich_feats.npz',allow_pickle=True)
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts']); META=lock['parts']
X13,XR,Y,G,MF=r['X13'],r['XR'],r['y'],r['groups'],r['mfg']
RICH=np.hstack([X13,XR[:,0:33]])
rp=[str(v) for v in r['part_ids']]; gseen=np.array([int(r['seen'][int(g)]) for g in G])
gpid=np.array([rp[int(g)] for g in G]); ngt=dict(zip(r['grp_ids'].tolist(), r['ngt'].tolist()))
WORK=(gseen==0)&np.array([p not in LOCK for p in gpid])
fam=np.array([META.get(p,{}).get('family',f'nr:{p}') for p in gpid])
FG=np.array([{v:i for i,v in enumerate(sorted(set(fam)))}[v] for v in fam])
o=np.zeros(len(Y)); idx=np.where(WORK)[0]
for tr,te in GroupKFold(5).split(RICH[idx],Y[idx],FG[idx]):
    c=RandomForestClassifier(400,min_samples_leaf=3,n_jobs=-1,random_state=0).fit(RICH[idx][tr],Y[idx][tr])
    o[idx[te]]=c.predict_proba(RICH[idx][te])[:,1]

def counts(parts, rule):
    TP=NK=GT=0; zero=0
    for g in parts:
        m=np.where(WORK&(G==g))[0]
        if not len(m): continue
        sel=rule(m)
        if len(sel)==0: zero+=1
        TP+=int(Y[sel].sum()); NK+=len(sel); GT+=ngt[int(g)]
    return TP,NK,GT,zero
def f1(TP,NK,GT):
    p=TP/max(NK,1); rr=TP/max(GT,1); return 2*p*rr/max(p+rr,1e-9)

GLOB=[np.round(x,2) for x in np.arange(0.10,0.71,0.02)]
FRAC=[np.round(x,2) for x in np.arange(0.30,0.91,0.05)]
parts=np.unique(G[WORK]); rs=np.random.RandomState(0); parts=parts[rs.permutation(len(parts))]
folds=np.array_split(parts,5)
for name in ('GLOBAL threshold','GORELI threshold (part-ici)'):
    TP=NK=GT=Z=0; picks=[]
    for f_ in folds:
        te=set(f_.tolist()); trp=[g for g in parts if g not in te]; tep=list(f_)
        best=None
        grid = GLOB if name.startswith('GLOBAL') else FRAC
        for v in grid:
            rule=(lambda m,v=v: m[o[m]>=v]) if name.startswith('GLOBAL') else (lambda m,v=v: m[o[m]>=v*o[m].max()])
            s=f1(*counts(trp,rule)[:3])
            if best is None or s>best[0]: best=(s,v)
        v=best[1]; picks.append(v)
        rule=(lambda m,v=v: m[o[m]>=v]) if name.startswith('GLOBAL') else (lambda m,v=v: m[o[m]>=v*o[m].max()])
        a,b,c_,z=counts(tep,rule); TP+=a; NK+=b; GT+=c_; Z+=z
    p=TP/max(NK,1); rr=TP/max(GT,1)
    print(f'{name:26s} F1 {f1(TP,NK,GT):.4f} | P {p:.4f} R {rr:.4f} | bos {Z:2d} | selected {picks}', flush=True)
