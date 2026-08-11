"""REVIEW/AUTO KALIBRASYONU.
Mevcut kusur: tier, SEGMENTASYON guvenine bakiyor (conf>=0.5) -> her sey AUTO oluyor (holdout'ta REVIEW BOS).
Dogrusu: AUTO/REVIEW karari 'bu gercekten tel girisi mi' sorusudur -> GATE skoru (wire_score).
Kalibrasyon: AUTO precision hedefini saglayan gate esigi. Robot fiziksel eylem yaptigi icin
AUTO'nun precision'i yuksek olmali; gerisi insana (REVIEW) gitmeli."""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
r=np.load('results/rich_feats.npz',allow_pickle=True)
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts']); META=lock['parts']
X13,XR,Y,G,MF=r['X13'],r['XR'],r['y'],r['groups'],r['mfg']
RICH=np.hstack([X13,XR[:,0:33]])
CONF=X13[:,12]                      # mevcut tier'in baktigi sinyal
rp=[str(v) for v in r['part_ids']]; gseen=np.array([int(r['seen'][int(g)]) for g in G])
gpid=np.array([rp[int(g)] for g in G]); ngt=dict(zip(r['grp_ids'].tolist(), r['ngt'].tolist()))
WORK=(gseen==0)&np.array([p not in LOCK for p in gpid])
fam=np.array([META.get(p,{}).get('family',f'nr:{p}') for p in gpid])
FG=np.array([{v:i for i,v in enumerate(sorted(set(fam)))}[v] for v in fam])
o=np.zeros(len(Y)); idx=np.where(WORK)[0]
for tr,te in GroupKFold(5).split(RICH[idx],Y[idx],FG[idx]):
    c=RandomForestClassifier(400,min_samples_leaf=3,n_jobs=-1,random_state=0).fit(RICH[idx][tr],Y[idx][tr])
    o[idx[te]]=c.predict_proba(RICH[idx][te])[:,1]

kept = WORK & (o>=0.34)                       # urunun tuttugu adaylar
gt_all = int(sum(ngt[int(g)] for g in np.unique(G[WORK])))
print(f'urun ciktisi: {int(kept.sum())} aday | gercek CP {gt_all}')
print()
print('=== MEVCUT TIER (segmentasyon guveni >= 0.5) ===')
auto = kept & (CONF>=0.5); rev = kept & (CONF<0.5)
pa = int(Y[auto].sum())/max(int(auto.sum()),1)
print(f'  AUTO  : {int(auto.sum()):4d} aday | precision {pa:.4f}')
print(f'  REVIEW: {int(rev.sum()):4d} aday   <-- kusur: neredeyse bos')
print()
print('=== ONERILEN TIER (gate skoru = wire_score) ===')
print("%11s %7s %10s %12s %9s" % ("AUTO esigi","AUTO n","AUTO prec","AUTO recall","REVIEW n"))
best=None
for t in np.arange(0.34,0.96,0.02):
    a = kept & (o>=t); rv = kept & (o<t)
    na=int(a.sum()); pa=int(Y[a].sum())/max(na,1); ra=int(Y[a].sum())/max(gt_all,1)
    if na<10: break
    mark=''
    if best is None and pa>=0.95: best=(t,na,pa,ra,int(rv.sum())); mark=' <-- prec>=0.95'
    if abs(t-0.34)<1e-9 or abs(t%0.10)<1e-9 or mark:
        print(f'{t:11.2f} {na:7d} {pa:10.4f} {ra:12.4f} {int(rv.sum()):9d}{mark}')
print()
if best:
    t,na,pa,ra,nr=best
    print(f'>>> AUTO precision >=0.95 icin esik {t:.2f}')
    print(f'    AUTO {na} aday (precision {pa:.3f}, gercek CPlerin %{100*ra:.0f}i) | REVIEW {nr} aday')
    print(f'    yani robot CPlerin %{100*ra:.0f}ini tek basina yapar, gerisi insana gider')
