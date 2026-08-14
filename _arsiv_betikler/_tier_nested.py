"""AUTO esigi (%98 precision hedefi) KARARLI mi? Nested: esik train-fold'da secilir, test-fold'da olculur."""
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
kept=WORK&(o>=0.34)
parts=np.unique(G[WORK]); rs=np.random.RandomState(0); parts=parts[rs.permutation(len(parts))]
folds=np.array_split(parts,5)
for target in (0.95,0.98):
    TPa=NAa=0; GTt=0; picks=[]
    for f_ in folds:
        te=set(f_.tolist())
        trm=kept&np.array([g not in te for g in G]); tem=kept&np.array([g in te for g in G])
        # train-fold'da hedefi saglayan EN DUSUK esik
        thr=0.99
        for t in np.arange(0.34,0.99,0.01):
            a=trm&(o>=t); n=int(a.sum())
            if n>=20 and int(Y[a].sum())/n>=target: thr=round(float(t),2); break
        picks.append(thr)
        a=tem&(o>=thr); TPa+=int(Y[a].sum()); NAa+=int(a.sum())
        GTt+=int(sum(ngt[int(g)] for g in np.unique(G[tem])))
    pa=TPa/max(NAa,1)
    print(f'hedef %{int(target*100)}: TEST-fold AUTO precision {pa:.4f} | AUTO {NAa} aday | CP kapsama %{100*TPa/max(GTt,1):.0f} | secilen esikler {picks}', flush=True)
