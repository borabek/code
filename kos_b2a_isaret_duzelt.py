# -*- coding: utf-8 -*-
"""B2a: ISARET DUZELTME -- ogrenmeden, saf fizikle.

DAYANAK (`results/b2_yon_isaret.json`): YON_YOK kovasinin **%44.3'u (312 GT)**
duz 180 derece ISARET hatasi. Kova aritmetigi: 312 kurtarilirsa F1 0.2773 ->
**0.376**.

B1 (mesh normalini secenek yapmak) yalniz +0.0023 verdi: aday basina onlarca
normal secenegi var ve secici dogrusunu bulamiyor (pozitif oran %6.4). Isaret
karari ise TEK IKILI karar -- cok daha kolay.

FIZIKSEL KURAL (ogrenme YOK): tel DISARIDAN girer. Adayin yonu `d` ise,
  `erisim` = +d yonunde serbest yol (disari)
  `girme`  = -d yonunde serbest yol (iceri, govdeye dogru)
Yon dogruysa disarisi ACIK, icerisi kanal olmali. `erisim < girme` ise yon
TERS cevrilir. Iki olcu de `agiz_tanimlayici` sutunlarindan gelir (3=girme,
5=erisim) ve ZATEN onbellekte -- ek hesap yok.

Ayrica ogrenilmis bir isaret siniflandiricisi de kiyas icin kosulur.
TEZE SADIK: konum ve aday sayisi degismez, yalnizca yonun ISARETI.
"""
import collections, json, os, pickle, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, kanonik_d7 as K, urun_zinciri, wire_gate
from sina_kume import esle_macar

OZ="results/_tam_oz"; TAN="results/_tan_hizali"
OB={"d7":"results/_p1_olasilik_d7","tam":"results/_p1_olasilik_brepegit"}
KAYNAKLAR=(0,1); ESIK=0.05; YANAL,ACI,EKSENEL=2.0,10.0,40.0
GIRME, ERISIM = 3, 5           # agiz_tanimlayici.AD indeksleri
gate = pickle.load(open("results/kazanan_hgb_derin.pkl","rb"))["HGB-derin"]
S = K.step_haritasi()
_D6 = {str(p): r for p, r in d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])).items()}
Rk = K.yukle(None)
kayit = lambda p: Rk.get(p) or _D6.get(p)   # noqa: E731


def yukle(on, pid):
    z=np.load(f"{OZ}/{on}_{pid}.npz")
    m=np.isin(np.asarray(z["kaynak"],int),KAYNAKLAR)
    T=np.asarray(np.load(f"{TAN}/{on}_{pid}.npz")["T"],float)
    X=np.hstack([np.asarray(z["X"],float),T])[m]
    if len(X)<2: return None
    return {"X":X,"T":T[m],"P":np.asarray(z["P"],float)[m],
            "D":np.asarray(z["D"],float)[m]}


def gate_sec(d):
    s=np.asarray(gate.predict_proba(wire_gate.parca_ici(d["X"],"zskor"))[:,1],float)
    k=s>=ESIK
    if not k.any(): return d["P"][:0],d["D"][:0],d["T"][:0],s[:0]
    P,D,T,sk=d["P"][k],d["D"][k],d["T"][k],s[k]
    if len(P)>1:
        nm=wire_gate.kalabalik_maskesi(P,sk); P,D,T,sk=P[nm],D[nm],T[nm],sk[nm]
    return P,D,T,sk


def fiziksel_ters(T):
    """erisim < girme  ->  yon TERS (disarisi kapali, icerisi acik)."""
    return T[:, ERISIM] < T[:, GIRME]


def olc(kol, clf=None):
    rob=collections.defaultdict(lambda:[0,0,0]); tes=[]
    for pid in [f[3:-4] for f in sorted(os.listdir(OZ)) if f.startswith("d7_")]:
        r=kayit(pid)
        if r is None or not len(r.get("G",[])): continue
        d=yukle("d7",pid)
        if d is None: continue
        P,D,T,sk=gate_sec(d)
        if len(P):
            if kol=="fizik":
                D=np.where(fiziksel_ters(T)[:,None], -D, D)
            elif kol=="ogrenilmis" and clf is not None:
                F=np.hstack([T, sk[:,None], np.full((len(T),1),len(P))])
                D=np.where((clf.predict_proba(F)[:,1]>=0.5)[:,None], -D, D)
            f=f"{OB['d7']}/{pid}.npz"
            if os.path.exists(f):
                z=np.load(f)
                P,D=urun_zinciri.tam_poz(np.ascontiguousarray(z["V"],np.float64),
                    np.ascontiguousarray(z["F"],np.int64),
                    np.asarray(z["pbs"],float).mean(0),P,D,step_path=S.get(pid))
        G=np.asarray(r["G"],float); Gd=np.asarray(r["Gd"],float); dg=float(r["diag"])
        tp,fp,fn=esle_macar(P,D,G,Gd,dg,YANAL,ACI,False,isaretli=True)[:3]
        a=rob[r["mfg"]]; a[0]+=tp; a[1]+=fp; a[2]+=fn
        tes.append((len(G),)+esle_macar(P,D,G,Gd,dg,max(3.0,0.06*dg),180.0,True)[:3])
    pm={m:2*v[0]/max(2*v[0]+v[1]+v[2],1) for m,v in rob.items()}
    mi=float(2*sum(v[0] for v in rob.values())/max(sum(2*v[0]+v[1]+v[2] for v in rob.values()),1))
    return {"robot":mi,"tespit":K.mikro(tes),"makro":float(np.mean(list(pm.values()))),
            "en_kotu":float(min(pm.values())),"marka":pm}


# --- ogrenilmis isaret siniflandiricisi (korpusta egitilir)
X,Y=[],[]
for pid in [f[4:-4] for f in sorted(os.listdir(OZ)) if f.startswith("tam_")]:
    r=kayit(pid)
    if r is None or not len(r.get("G",[])): continue
    d=yukle("tam",pid)
    if d is None: continue
    P,D,T,sk=gate_sec(d)
    if not len(P): continue
    G=np.asarray(r["G"],float); Gd=np.asarray(r["Gd"],float)
    for a in range(len(P)):
        for t in range(len(G)):
            n=np.linalg.norm(Gd[t])
            if n<1e-9: continue
            u=Gd[t]/n; w=P[a]-G[t]; e=float(w@u)
            if np.linalg.norm(w-e*u)>YANAL or abs(e)>EKSENEL: continue
            duz=float(np.degrees(np.arccos(np.clip(float(D[a]@u),-1,1))))
            ters=float(np.degrees(np.arccos(np.clip(float(-D[a]@u),-1,1))))
            if min(duz,ters)>ACI: break
            X.append(np.concatenate([T[a],[sk[a],len(P)]]))
            Y.append(int(ters<duz)); break
X=np.asarray(X,float); Y=np.asarray(Y,int)
print(f"isaret egitimi {X.shape} | TERS olmasi gereken {Y.mean():.4f}", flush=True)
clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,
                                   max_leaf_nodes=31,random_state=0).fit(X,Y)
out={}
for ad,kol in (("TABAN (isaret sabit)",None),("FIZIKSEL KURAL","fizik"),
               ("OGRENILMIS ISARET","ogrenilmis")):
    out[ad]=olc(kol,clf)
    c=out[ad]
    print(f"{ad:<22} robot {c['robot']:.4f} | tespit {c['tespit']:.4f} | "
          f"makro {c['makro']:.4f} | en kotu {c['en_kotu']:.4f}", flush=True)
t=out["TABAN (isaret sabit)"]["robot"]
for ad in out:
    if ad!="TABAN (isaret sabit)":
        print(f"  {ad:<22} {out[ad]['robot']-t:+.4f}")
print("KAPI: >= +0.02 | kova aritmetigi beklentisi 0.376")
json.dump({"damga":makbuz_hash.damga(),"sonuc":out,
           "not":"Isaret duzeltme. Konum ve aday sayisi degismez. D7 marka-disi, TAM ZINCIR."},
          open("results/b2a_isaret.json","w"), indent=1)
print("makbuz -> results/b2a_isaret.json")
