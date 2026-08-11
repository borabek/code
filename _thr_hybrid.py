"""GORELI + MUTLAK TABAN: parca-ici goreli esik ama cok dusuk skorlu adayi yine de tutma."""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
r=np.load('results/rich_feats.npz',allow_pickle=True)
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts']); META=lock['parts']
X13,XR,Y,G=r['X13'],r['XR'],r['y'],r['groups']
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
def ev(fn,label):
    TP=NK=GT=0; zero=0
    for g in np.unique(G[WORK]):
        m=np.where(WORK&(G==g))[0]; sel=fn(m)
        if len(sel)==0: zero+=1
        TP+=int(Y[sel].sum()); NK+=len(sel); GT+=ngt[int(g)]
    p=TP/max(NK,1); rr=TP/max(GT,1); f1=2*p*rr/max(p+rr,1e-9)
    print(f'{label:52s} F1 {f1:.4f} | P {p:.4f} R {rr:.4f} | bos {zero:2d}', flush=True)
ev(lambda m: m[o[m]>=0.34], 'mevcut urun: global 0.34')
ev(lambda m: m[o[m]>=0.41], 'global 0.41 (nested-CV secimi)')
ev(lambda m: m[o[m]>=0.55*o[m].max()], 'goreli 0.55 (nested-CV secimi)')
for fl in (0.15,0.20,0.25,0.30):
    ev(lambda m,fl=fl: m[(o[m]>=0.55*o[m].max())&(o[m]>=fl)], f'goreli 0.55 + mutlak taban {fl}')
for fl in (0.20,0.25):
    ev(lambda m,fl=fl: m[(o[m]>=0.41)|((o[m]>=0.55*o[m].max())&(o[m]>=fl))], f'global 0.41 VEYA (goreli 0.55 + taban {fl})')
