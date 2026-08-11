# -*- coding: utf-8 -*-
"""8 uyeli havuzda min_votes taramasi -- DURUST secimle.

GEREKCE: min_votes=1 (birlesim) karari 4 UYE icin verilmisti; gerekcesi "birlesimin yuksek
recall'i hayatta kalir cunku gate kesinligi geri getirir". 8 uyede aday sayisi GT basina
2.4'ten 4.6'ya cikti ve gate yuku kaldiramadi (tespit -0.031). Artik "3 uye anlasti" gibi
anlamli bir sinyal var; birlesim zorunlu degil.

DURUST SECIM: esik BIR YARIDA secilir, DIGER yaride olculur. Ayni kumede tarayip en iyiyi
almak sisirmedir (bu gece G5'te ayni tuzagi olcmustum).
"""
import io, json, os, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,".")
import olcum_kumesi, wire_gate
from sina_kume import esle, f1w
from sklearn.ensemble import RandomForestClassifier

d=np.load("results/gate_8uye.npz",allow_pickle=True)
Xtr=np.hstack([np.asarray(d["X22"],float),np.asarray(d["XR"],float)])
ytr=np.asarray(d["y"]); ptr=np.array([str(x) for x in d["pids"]]); vtr=np.asarray(d["votes"])
DER,rap=olcum_kumesi.kume("results/_der_8uye.pkl"); olcum_kumesi.rapor_bas(rap)
AD=None
try:
    import gate_tezgah as T; AD=T.yukle()["ad"]
except Exception: pass
i_vot = AD.index("votes") if AD and "votes" in AD else 11

Ztr=np.zeros((len(Xtr),Xtr.shape[1]*2))
for u in np.unique(ptr):
    i=np.where(ptr==u)[0]; Ztr[i]=wire_gate.parca_ici(Xtr[i],"zskor")

def gate_ile(mv):
    m=vtr>=mv
    if m.sum()<500 or len(np.unique(ytr[m]))<2: return None
    return {"clf":RandomForestClassifier(n_estimators=400,min_samples_leaf=3,n_jobs=-1,
            random_state=0).fit(Ztr[m],ytr[m]),"n_feat":Ztr.shape[1],"donusum":"zskor"}

def puanla(mv,alt):
    g=gate_ile(mv)
    if g is None: return None,None,None
    det,rob,gg=[],[],[]
    for r in alt:
        P=np.zeros((0,3)); Pd=np.zeros((0,3))
        if r["X"] is not None and r.get("XR") is not None:
            X=np.hstack([r["X"],r["XR"]])
            vv=X[:,i_vot]
            sec=vv>=mv
            if sec.any():
                Xs=X[sec]
                k=wire_gate.karar_maskesi(wire_gate.karar_skoru(g,Xs))
                if k.any():
                    idx=np.where(sec)[0][k]
                    P=np.asarray(r["P"],float)[idx].copy(); Pd=np.asarray(r["Pd"],float)[idx].copy()
                    c=[{"point":P[i],"direction":Pd[i]} for i in range(len(P))]
                    c=wire_gate.pose_duzelt(X[idx],c); c=wire_gate.aci_duzelt(X[idx],c)
                    if r.get("UYE"): c=wire_gate.uye_yonu_sec(X[idx],c,r["UYE"])
                    P=np.array([x["point"] for x in c],float); Pd=np.array([x["direction"] for x in c],float)
        rj="cok" if r["n"]>=8 else "dusuk"
        G=np.asarray(r["G"],float); Gd=np.asarray(r["Gd"],float)
        det.append((rj,)+esle(P,Pd,G,Gd,r["diag"],0.0,180.0,True))
        rob.append((rj,)+esle(P,Pd,G,Gd,r["diag"],2.0,10.0,False))
        gg.append(r["geo"])
    return det,rob,gg

MV=(1,2,3,4)
print(f"\n{'min_votes':<11}{'egitim aday':>13}{'tespit':>9}{'robot':>9}")
TAM={}
for mv in MV:
    det,rob,gg=puanla(mv,DER)
    if det is None: continue
    TAM[mv]=(f1w(det),f1w(rob))
    print(f"{mv:<11}{int((vtr>=mv).sum()):>13}{f1w(det):>9.4f}{f1w(rob):>9.4f}",flush=True)

# --- DURUST CAPRAZ SECIM
gruplar=sorted({r["geo"] for r in DER}); rng=np.random.default_rng(0)
kar=list(gruplar); rng.shuffle(kar); A=set(kar[:len(kar)//2])
altA=[r for r in DER if r["geo"] in A]; altB=[r for r in DER if r["geo"] not in A]
dc,gg2=[],[]
for sec,olc in ((altA,altB),(altB,altA)):
    en,ea=-1,1
    for mv in MV:
        dd,_,_=puanla(mv,sec)
        if dd and f1w(dd)>en: en,ea=f1w(dd),mv
    d1,_,g1=puanla(ea,olc)
    dc+=d1; gg2+=g1
    print(f"  yarida secilen min_votes {ea} -> diger yaride olculdu ({len(olc)} parca)")
print(f"\nDURUST (capraz secim) TESPIT: {f1w(dc):.4f}   [taban 0.7584]")
json.dump({"tam":{str(k):{"tespit":v[0],"robot":v[1]} for k,v in TAM.items()},
           "durust_tespit":f1w(dc)}, io.open("results/g4_minvotes.json","w"),indent=1)
print("makbuz -> results/g4_minvotes.json")
