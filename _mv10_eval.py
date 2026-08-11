# -*- coding: utf-8 -*-
"""YENI POST-ISLEME (min_v10) CP F1'i artiriyor mu?
Kiyas ayni protokol: WORK 541 parca, family-out, nested-CV esik, zengin feature (46d), RF.
Soru: aday 1.5x arttiginda gate gurultuyu eleyebiliyor mu."""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts']); META=lock['parts']
THRS=np.round(np.arange(0.10,0.71,0.02),3)

def run(path,label):
    r=np.load(path,allow_pickle=True)
    X13,XR,Y,G,MF=r['X13'],r['XR'],r['y'],r['groups'],r['mfg']
    RICH=np.hstack([X13,XR[:,0:33]])
    rp=[str(v) for v in r['part_ids']]
    gpid=np.array([rp[int(g)] for g in G]); gseen=np.array([int(r['seen'][int(g)]) for g in G])
    ngt=dict(zip(r['grp_ids'].tolist(), r['ngt'].tolist()))
    M=(gseen==0)&np.array([p not in LOCK for p in gpid])
    fam=np.array([META.get(p,{}).get('family',f'nr:{p}') for p in gpid])
    FG=np.array([{v:i for i,v in enumerate(sorted(set(fam)))}[v] for v in fam])
    o=np.zeros(len(Y)); idx=np.where(M)[0]
    for tr,te in GroupKFold(5).split(RICH[idx],Y[idx],FG[idx]):
        c=RandomForestClassifier(400,min_samples_leaf=3,n_jobs=-1,random_state=0).fit(RICH[idx][tr],Y[idx][tr])
        o[idx[te]]=c.predict_proba(RICH[idx][te])[:,1]
    def prf(tp,nk,gt):
        p=tp/max(nk,1); rr=tp/max(gt,1); return 2*p*rr/max(p+rr,1e-9)
    def gt_of(m): return int(sum(ngt[int(g)] for g in np.unique(G[m])))
    def nested(m):
        gp=np.unique(FG[np.where(m)[0]]); rs=np.random.RandomState(0); gp=gp[rs.permutation(len(gp))]
        TP=NK=GT=0
        for f in np.array_split(gp,5):
            tg=set(f.tolist())
            trm=m&np.array([g not in tg for g in FG]); tem=m&np.array([g in tg for g in FG])
            if not trm.any() or not tem.any(): continue
            bt,bf=0.35,-1
            for t in THRS:
                k=trm&(o>=t); ff=prf(int(Y[k].sum()),int(k.sum()),gt_of(trm))
                if ff>bf: bf,bt=ff,t
            k=tem&(o>=bt); TP+=int(Y[k].sum()); NK+=int(k.sum()); GT+=gt_of(tem)
        return prf(TP,NK,GT)
    gt=gt_of(M); ceil=int(Y[M].sum())/max(gt,1)
    print("%-26s aday %5d (%.2fx GT) | tavan %.3f | AUC %.4f | ALL %.4f | WEI %.4f | PXC %.4f" % (
        label, int(M.sum()), int(M.sum())/max(gt,1), ceil, roc_auc_score(Y[M],o[M]),
        nested(M), nested(M&(MF==1)), nested(M&(MF==0))), flush=True)

print("POST-ISLEME KIYASI (WORK 541 parca, family-out, nested-CV esik)")
run('results/rich_feats.npz','MEVCUT min_v30')
run('results/rich_mv10.npz','YENI min_v10')
