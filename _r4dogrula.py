import io,json,os,pickle,sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,".")
import tezgah2 as T2, wire_gate, olcum_kumesi
from sina_kume import esle,f1w
from d1_yon_dagit import satirla
from sklearn.ensemble import RandomForestClassifier
DER,gate,ek=T2.yukle()
PARCA=pickle.load(open("results/_r4_sozluk.pkl","rb"))
YS=wire_gate._load("results/yon_secici.pkl"); MARJ=0.05
gk=json.load(io.open("results/_strict_geometry_keys.json",encoding="utf-8"))
mfg_of={}
from big_arbiter import eligible
for m,p,jf,s in eligible(): mfg_of[p]=m

def kol(alt, uygula, g=None):
    G_=g or gate
    det,rob,rbi,gg=[],[],[],[]
    for r in alt:
        d_=PARCA.get(r["pid"]); G=np.asarray(r["G"],float); Gd=np.asarray(r["Gd"],float)
        rj="cok" if r["n"]>=8 else "dusuk"
        P=np.zeros((0,3)); Pd=np.zeros((0,3))
        if g is None and d_ is not None:
            P=d_["P"].copy(); Pd=d_["Pd"].copy()
            if uygula and YS is not None:
                ax,_,RJ=satirla(d_["X"],d_["SY"],d_["UZ"],d_["Pd"],None)
                if ax:
                    pr=YS["clf"].predict_proba(np.array(ax,float))[:,1]
                    S={}
                    for n_,(i,gi) in enumerate(RJ): S.setdefault(i,{})[gi]=pr[n_]
                    for i,sc in S.items():
                        q=max(sc,key=sc.get)
                        if q!=0 and sc[q]-sc.get(0,0.0)>=MARJ: Pd[i]=d_["SY"][i][q][1]
        elif g is not None and r["X"] is not None and r.get("XR") is not None:
            X=np.hstack([r["X"],r["XR"]])
            if X.shape[1]*2==G_["n_feat"]:
                k=wire_gate.karar_maskesi(wire_gate.karar_skoru(G_,X))
                if k.any():
                    P=np.asarray(r["P"],float)[k].copy(); Pd=np.asarray(r["Pd"],float)[k].copy()
                    c=[{"point":P[i],"direction":Pd[i]} for i in range(len(P))]
                    c=wire_gate.pose_duzelt(X[k],c); c=wire_gate.aci_duzelt(X[k],c)
                    if r.get("UYE"): c=wire_gate.uye_yonu_sec(X[k],c,r["UYE"])
                    P=np.array([x["point"] for x in c],float); Pd=np.array([x["direction"] for x in c],float)
                    if uygula and YS is not None and d_ is not None and len(P)==len(d_["P"]):
                        ax,_,RJ=satirla(X[k],d_["SY"],d_["UZ"],Pd,None)
                        if ax:
                            pr=YS["clf"].predict_proba(np.array(ax,float))[:,1]
                            S={}
                            for n_,(i,gi) in enumerate(RJ): S.setdefault(i,{})[gi]=pr[n_]
                            for i,sc in S.items():
                                q=max(sc,key=sc.get)
                                if q!=0 and sc[q]-sc.get(0,0.0)>=MARJ: Pd[i]=d_["SY"][i][q][1]
        det.append((rj,)+esle(P,Pd,G,Gd,r["diag"],0.0,180.0,True))
        rob.append((rj,)+esle(P,Pd,G,Gd,r["diag"],2.0,10.0,False))
        rbi.append((rj,)+esle(P,Pd,G,Gd,r["diag"],2.0,10.0,False,isaretli=True))
        gg.append(r["geo"])
    return det,rob,rbi,gg

d0,r0,i0,gg=kol(DER,False); d1,r1,i1,_=kol(DER,True)
print(f"\n{'':<26}{'tespit':>9}{'robot':>9}{'fiziksel':>10}")
print(f"{'taban':<26}{f1w(d0):>9.4f}{f1w(r0):>9.4f}{f1w(i0):>10.4f}")
print(f"{'+ isaretli yon secici':<26}{f1w(d1):>9.4f}{f1w(r1):>9.4f}{f1w(i1):>10.4f}")
print(f"  TESPIT degisimi: {f1w(d1)-f1w(d0):+.6f}  (yapisal olarak 0 olmali)")
# URETICI-DISI
X=ek["Z"]; y=ek["y"]; pid=ek["pid"]; mfg=ek["mfg"]; keep=ek["keep"]
print(f"\n{'uretici-disi':<26}{'tespit':>9}{'robot':>9}{'fiziksel':>10}")
UD={}
for k in np.unique(mfg):
    m=keep&(mfg!=k)
    if m.sum()<500: continue
    g={"clf":RandomForestClassifier(n_estimators=400,min_samples_leaf=3,n_jobs=-1,random_state=0).fit(X[m],y[m]),
       "n_feat":X.shape[1],"donusum":"zskor"}
    da,ra,ia,_=kol(DER,False,g); db,rb,ib,_=kol(DER,True,g)
    UD[str(k)]={"taban":{"tespit":f1w(da),"robot":f1w(ra),"fiziksel":f1w(ia)},
                "yeni":{"tespit":f1w(db),"robot":f1w(rb),"fiziksel":f1w(ib)}}
    print(f"{'  '+str(k)+' disarida (taban)':<26}{f1w(da):>9.4f}{f1w(ra):>9.4f}{f1w(ia):>10.4f}")
    print(f"{'  '+str(k)+' disarida (+secici)':<26}{f1w(db):>9.4f}{f1w(rb):>9.4f}{f1w(ib):>10.4f}")
dus=[k for k,v in UD.items() if v["yeni"]["fiziksel"] < v["taban"]["fiziksel"]-0.002]
print(f"\nFIZIKSEL metrikte DUSEN uretici: {dus if dus else 'YOK'}")
print("KARAR:", "DAGITILABILIR" if not dus and abs(f1w(d1)-f1w(d0))<1e-6 else "DIKKAT")
json.dump({"havuz":{"taban":{"tespit":f1w(d0),"robot":f1w(r0),"fiziksel":f1w(i0)},
                    "yeni":{"tespit":f1w(d1),"robot":f1w(r1),"fiziksel":f1w(i1)}},
           "uretici_disi":UD,"dusen":dus},io.open("results/r4_dogrulama.json","w"),indent=1)
print("makbuz -> results/r4_dogrulama.json")
