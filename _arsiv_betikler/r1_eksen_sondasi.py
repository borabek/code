# -*- coding: utf-8 -*-
"""R1 SONDASI: DIK eksen vakalarinin kaci KURTARILABILIR?

Gorulmemis ureticide eslesen ciftlerin %36'sinda aday ekseni GT eksenine DIK (>80 deg).
Bu bir secim hatasi (ortanca hata 0.0 deg -- yani yari zaten mukemmel, kalan yanlis eksen).
SORU: dik vakalarin adaylari uzerinde `channel_axis_robust` (mesh kanal ekseni) kosulursa
GT eksenini buluyor mu? Buluyorsa R1 gercek bir kol; bulmuyorsa dik vakalar kare/yay
girisler demektir ve baska mekanizma gerekir.
"""
import sys, os, json, io, pickle, collections
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,'.')
import numpy as np, trimesh
import wire_gate, thesis_remesh
from f2_12_veri_kolu import egit
from big_arbiter import eligible
from infer_step_cp import step_to_mesh
from cp_geometry import channel_axis_robust, channel_axis

sv=json.load(io.open('results/d5_4_sinav_kumesi.json',encoding='utf-8'))
G6={r['pid']:r for r in pickle.load(open('results/_der_yeni_g6.pkl','rb'))}
D=[G6[p] for p in sorted(set(sv['pidler'])) if p in G6]
yol={p:s for m,p,jf,s in eligible()}
m,_=egit('results/zengin_parite_v3.npz')

dik=[]  # (pid, aday_nokta, aday_yon, gt_yon)
for r in D:
    if r['X'] is None or r.get('XR') is None: continue
    M=np.hstack([r['X'],r['XR']]).astype(float)
    if M.shape[1]*2!=m['n_feat']: continue
    k=wire_gate.karar_maskesi(wire_gate.karar_skoru(m,M))
    P=np.asarray(r['P'],float)[k]; Pd=np.asarray(r['Pd'],float)[k]
    G=np.asarray(r['G'],float); Gd=np.asarray(r['Gd'],float)
    if not len(P) or not len(G): continue
    d=P[:,None,:]-G[None,:,:]; al=(d*Gd[None,:,:]).sum(-1)
    pe=np.linalg.norm(d-al[...,None]*Gd[None,:,:],axis=-1)
    pe2=np.where(np.abs(al)<=40,pe,np.inf)
    tol=max(3.0,0.06*float(r['diag'])); up,ug=set(),set()
    for dd,a_,b_ in sorted((pe2[a,b],a,b) for a in range(len(P)) for b in range(len(G))):
        if dd>tol or a_ in up or b_ in ug: continue
        up.add(a_); ug.add(b_)
        nP=Pd[a_]/(np.linalg.norm(Pd[a_])+1e-9)
        c=abs(float(nP@Gd[b_]))
        if c<0.342:                      # >70 derece = dik vaka
            dik.append((r['pid'],P[a_],Pd[a_],Gd[b_]))
print(f"dik vaka: {len(dik)} cift")
rng=np.random.RandomState(0); sec=[dik[i] for i in rng.permutation(len(dik))[:60]]
kur=0; olc=0; mesh_cache={}
for pid,p,pd,gd in sec:
    try:
        if pid not in mesh_cache:
            V,F=step_to_mesh(yol[pid]); V,F=thesis_remesh.remesh_uniform(V,F,target=6000)
            mesh_cache[pid]=trimesh.Trimesh(vertices=np.ascontiguousarray(V,np.float64),
                                            faces=np.ascontiguousarray(F,np.int64),process=False)
        ax=channel_axis_robust(mesh_cache[pid], np.asarray(p,float), np.asarray(pd,float))
        if ax is None:
            ax=channel_axis(mesh_cache[pid], np.asarray(p,float), fallback=np.asarray(pd,float))
        if ax is None: continue
        a=np.asarray(ax[0] if isinstance(ax,tuple) else ax,float).reshape(3)
        a=a/(np.linalg.norm(a)+1e-9)
        olc+=1
        if abs(float(a@gd))>=np.cos(np.deg2rad(10)): kur+=1
    except Exception: pass
print(f"olculen {olc} | kanal ekseni GT'yi (<=10deg) buluyor: {kur} (%{100*kur/max(olc,1):.0f})")
print("-> %50+ ise R1 GERCEK KOL; %20 alti ise dik vakalar kare/yay giris, baska mekanizma")
