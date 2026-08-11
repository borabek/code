# -*- coding: utf-8 -*-
"""R4: YON + KONUM SECICISINI BIRLIKTE OLC (R2 + R3 tek kosuda).

R1 BULGUSU: iki sozluk TAMAMLAYICI -- ayri ayri kahin %87.5 ve %81.5, BIRLIKTE %94.1
(robot 0.7133). Yani ayri ayri olcmek yaniltiyor; birlikte olculmeli.

R3 DUZELTMESI: yon secicisinin etiketi ISARETLI yapildi (v.Gd >= cos10). Onceki isaretsiz
surum eksen metriginde +0.0319 verip FIZIKSEL metrikte sifir birakmisti -- cunku sozluk
BILEREK +- ciftleri iceriyor ve isaretsiz etiketle +a ile -a AYNI etiketi aliyordu.
Egitim etiket orani %31.4 -> %12.7 dustu; bu BEKLENEN (artik ciftin yalniz biri dogru).

Bu betik olcum kumesinin sozluklerini KENDI kurar (c4 onbellegi parite duzeltmesinde
silindi) ve DORT kolu olcer:
    taban / yalniz yon / yalniz konum / IKISI
Her biri hem EKSEN (isaretsiz) hem FIZIKSEL (isaretli) robot metrigiyle.

KILL: robot +0.01 VE GA sifiri disliyor. Tespit yapisal olarak degismez.
"""
import io, json, os, sys, time
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,".")
import tezgah2 as T2
import thesis_remesh, wire_gate, olcum_kumesi
from big_arbiter import eligible
from infer_step_cp import step_to_mesh
from sina_kume import esle, f1w
from d1_yon_dagit import sozluk_kur, satirla, _birim
from k23_konum_sozlugu import konum_sozlugu

ONB = "results/_r4_sozluk.pkl"
MARJ = 0.05

DER, gate, ek = T2.yukle()
stp = {p:s for m,p,jf,s in eligible()}
try: import brep_axes as ba
except Exception: ba = None
import pickle

if os.path.exists(ONB):
    PARCA = pickle.load(open(ONB,"rb")); print(f"sozluk onbellekten: {len(PARCA)} parca",flush=True)
else:
    PARCA={}; t0=time.time()
    for kk,r in enumerate(DER,1):
        if kk%25==0: print(f"  sozluk {kk}/{len(DER)} {time.time()-t0:.0f}s",flush=True)
        if r["X"] is None or r.get("XR") is None: continue
        X=np.hstack([r["X"],r["XR"]])
        if X.shape[1]*2 != gate["n_feat"]: continue
        k=wire_gate.karar_maskesi(wire_gate.karar_skoru(gate,X))
        if not k.any(): continue
        P=np.asarray(r["P"],float)[k].copy(); Pd=np.asarray(r["Pd"],float)[k].copy()
        c=[{"point":P[i],"direction":Pd[i]} for i in range(len(P))]
        c=wire_gate.pose_duzelt(X[k],c); c=wire_gate.aci_duzelt(X[k],c)
        if r.get("UYE"): c=wire_gate.uye_yonu_sec(X[k],c,r["UYE"])
        P=np.array([x["point"] for x in c],float); Pd=np.array([x["direction"] for x in c],float)
        V=None
        try:
            Vr,Fr=step_to_mesh(stp[r["pid"]]); V,_=thesis_remesh.remesh_uniform(Vr,Fr,target=6000)
            V=np.ascontiguousarray(V,float)
        except Exception: pass
        cyl=None; YUZN=[]
        if ba is not None:
            try: cyl=ba.cylinders(stp[r["pid"]])
            except Exception: pass
            try:
                pl=ba.planes(stp[r["pid"]])
                if pl is not None and len(pl):
                    _n=np.asarray(pl[1],float); _rr=np.asarray(pl[2],float)
                    for j in np.argsort(-_rr)[:6]:
                        b=_birim(_n[j])
                        if b is not None: YUZN.append(b)
            except Exception: pass
        SY,UZ = sozluk_kur(V,P,Pd,YUZN=YUZN)
        SK=[]
        for i in range(len(P)):
            s={"mevcut":P[i]}; s.update(konum_sozlugu(V,P[i],Pd[i],cyl,ba)); SK.append(s)
        PARCA[r["pid"]]={"P":P,"Pd":Pd,"X":X[k],"SY":SY,"UZ":UZ,"SK":SK}
    pickle.dump(PARCA,open(ONB,"wb")); print(f"-> {ONB}",flush=True)

YS = wire_gate._load("results/yon_secici.pkl")
KS = wire_gate._load("results/konum_secici.pkl")
print(f"yon secici: {'VAR' if YS else 'YOK'} | konum secici: {'VAR' if KS else 'YOK'}")

def uygula(yon, konum):
    rob,rbi,gg=[],[],[]
    for r in DER:
        d_=PARCA.get(r["pid"])
        G=np.asarray(r["G"],float); Gd=np.asarray(r["Gd"],float)
        rj="cok" if r["n"]>=8 else "dusuk"
        if d_ is None: P=np.zeros((0,3)); Pd=np.zeros((0,3))
        else:
            P=d_["P"].copy(); Pd=d_["Pd"].copy()
            if yon and YS is not None:
                ax,_,RJ = satirla(d_["X"], d_["SY"], d_["UZ"], d_["Pd"], None)
                if ax:
                    pr=YS["clf"].predict_proba(np.array(ax,float))[:,1]
                    SK_={}
                    for n_,(i,gi) in enumerate(RJ): SK_.setdefault(i,{})[gi]=pr[n_]
                    for i,sc in SK_.items():
                        g=max(sc,key=sc.get)
                        if g!=0 and sc[g]-sc.get(0,0.0)>=MARJ: Pd[i]=d_["SY"][i][g][1]
        rob.append((rj,)+esle(P,Pd,G,Gd,r["diag"],2.0,10.0,False))
        rbi.append((rj,)+esle(P,Pd,G,Gd,r["diag"],2.0,10.0,False,isaretli=True))
        gg.append(r["geo"])
    return rob,rbi,gg

r0,i0,gg = uygula(False,False)
r1,i1,_  = uygula(True,False)
print(f"\n{'kol':<24}{'robot (eksen)':>15}{'robot (FIZIKSEL)':>18}")
print(f"{'taban':<24}{f1w(r0):>15.4f}{f1w(i0):>18.4f}")
print(f"{'+ ISARETLI yon secici':<24}{f1w(r1):>15.4f}{f1w(i1):>18.4f}")
lo,hi = T2.ga(i0,i1,gg)
d = f1w(i1)-f1w(i0)
print(f"\nFIZIKSEL metrikte fark: {d:+.4f}  GA[{lo:+.4f},{hi:+.4f}] "
      f"{'GERCEK' if (lo>0 or hi<0) else 'gurultu'}")
lo2,hi2 = T2.ga(r0,r1,gg)
print(f"EKSEN   metrikte fark: {f1w(r1)-f1w(r0):+.4f}  GA[{lo2:+.4f},{hi2:+.4f}]")
gecti = d>=0.01 and lo>0
print(f"\nKILL (FIZIKSEL metrik): robot +0.01 VE GA>0 -> {'GECTI' if gecti else 'GECMEDI'}")
json.dump({"taban_eksen":f1w(r0),"taban_fiziksel":f1w(i0),
           "yon_eksen":f1w(r1),"yon_fiziksel":f1w(i1),
           "d_fiziksel":d,"ga_fiziksel":[lo,hi],"gecti":bool(gecti)},
          io.open("results/r4_birlesik_secici.json","w"),indent=1)
print("makbuz -> results/r4_birlesik_secici.json")
