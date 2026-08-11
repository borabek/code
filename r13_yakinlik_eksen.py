# -*- coding: utf-8 -*-
"""R13: YAKINLIK-ONCELIKLI B-REP EKSENI GT'yi daha iyi buluyor mu? (kod DEGISMEDEN)

R12 bulgusu: adayin %74'unde 5mm icinde bir silindir VAR, ama `axis_at`'in ACI KAPISI
(max_turn_deg=60) onlarin %40'ini reddediyor -- cunku tohum yon, egimi goremeyen
channel_axis+snap'ten geliyor. Yanlis tohum -> dogru silindir "90 derece sapik" gorunuyor -> red.

BU BETIK URUNU DEGISTIRMEZ. Yalnizca olcer: silindiri YAKINLIKLA secip aci kapisini
kaldirirsak, eksen GT'ye daha mi yakin olur? Kazanmazsa kod DEGISMEZ.
"""
import sys, os, json, io, pickle, collections
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,'.')
import numpy as np, wire_gate
import brep_axes as BX
from f2_12_veri_kolu import egit
from big_arbiter import eligible
import olcum_kumesi as OK

G6={r['pid']:r for r in pickle.load(open('results/_der_yeni_g6.pkl','rb'))}
yol={p:s for m,p,jf,s in eligible()}
m,_=egit('results/zengin_parite_v3.npz')

def kos(DER, ad, sinir=70, maxoff=5.0):
    A_mev=[]; A_yak=[]; kur=0; boz=0; yok=0
    n=0
    for r in DER:
        if n>=sinir: break
        if r['X'] is None or r.get('XR') is None: continue
        M=np.hstack([r['X'],r['XR']]).astype(float)
        if M.shape[1]*2!=m['n_feat']: continue
        k=wire_gate.karar_maskesi(wire_gate.karar_skoru(m,M))
        P=np.asarray(r['P'],float)[k]; Pd=np.asarray(r['Pd'],float)[k]
        G=np.asarray(r['G'],float); Gd=np.asarray(r['Gd'],float)
        if not len(P) or not len(G): continue
        try: cyl=BX.cylinders(yol[r['pid']])
        except Exception: cyl=None
        n+=1
        C=np.asarray(cyl[0],float) if cyl is not None and len(cyl[0]) else None
        AX=np.asarray(cyl[1],float) if C is not None else None
        d=P[:,None,:]-G[None,:,:]; al=(d*Gd[None,:,:]).sum(-1)
        pe=np.linalg.norm(d-al[...,None]*Gd[None,:,:],axis=-1)
        pe2=np.where(np.abs(al)<=40,pe,np.inf)
        nP=Pd/(np.linalg.norm(Pd,axis=1,keepdims=True)+1e-9)
        tol=max(3.0,0.06*float(r['diag'])); up,ug=set(),set()
        for dd,a_,b_ in sorted((pe2[a,b],a,b) for a in range(len(P)) for b in range(len(G))):
            if dd>tol or a_ in up or b_ in ug: continue
            up.add(a_); ug.add(b_)
            g=Gd[b_]
            am=np.degrees(np.arccos(min(1,abs(float(nP[a_]@g))))); A_mev.append(am)
            if C is None: A_yak.append(am); yok+=1; continue
            rel=P[a_]-C; t=(rel*AX).sum(1)
            dik=np.linalg.norm(rel-t[:,None]*AX,axis=1)
            j=int(np.argmin(dik))
            if dik[j]>maxoff:
                A_yak.append(am); yok+=1; continue
            ay=np.degrees(np.arccos(min(1,abs(float(AX[j]@g))))); A_yak.append(ay)
            if am>80 and ay<=10: kur+=1
            if am<=10 and ay>80: boz+=1
    Am=np.array(A_mev); Ay=np.array(A_yak)
    print(f'=== {ad} (n={len(Am)}, silindirsiz/uzak {yok}) ===')
    print(f'  aci<=10  MEVCUT %{100*(Am<=10).mean():.0f}  ->  YAKINLIK %{100*(Ay<=10).mean():.0f}')
    print(f'  DIK>80   MEVCUT %{100*(Am>80).mean():.0f}  ->  YAKINLIK %{100*(Ay>80).mean():.0f}')
    print(f'  DIK KURTARILAN {kur} | IYIYI BOZAN {boz}   NET {kur-boz:+d}')

D194_e,_=OK.kume('results/_der_tam.pkl')
D194=[G6[p] for p in sorted({r['pid'] for r in D194_e}) if p in G6]
sv=json.load(io.open('results/d5_4_sinav_kumesi.json',encoding='utf-8'))
SIN=[G6[p] for p in sorted(set(sv['pidler'])) if p in G6]
kos(D194,'194luk TANIDIK'); kos(SIN,'GORULMEMIS')
