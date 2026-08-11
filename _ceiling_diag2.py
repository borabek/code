"""ADAY-TAVANI TESHISI: kacirilan GT CP'lerinde segmentasyon ATESLEDI mi?
  atesledi ama aday olmadi  -> POST-ISLEME sorunu (min_v/dedupe) = UCUZ duzeltme
  hic ateslemedi            -> SEGMENTASYON kor = YENIDEN EGITIM gerek
Bu, 'segmentasyonu yeniden egitmek tavani yukseltir mi' sorusunun dogrudan cevabi."""
import os, sys, json
import numpy as np, torch, trimesh
sys.path.insert(0,'.'); os.environ['BA_ALLOW_SEEN']='1'
import diffusionnet as D, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2
from cp_geometry import seat_to_mouth   # ORTAK: GT yuvada, agiz baska yerde

r=np.load('results/rich_feats.npz',allow_pickle=True)
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts'])
Y,G=r['y'],r['groups']; rp=[str(v) for v in r['part_ids']]
gseen=np.array([int(r['seen'][int(g)]) for g in G]); gpid=np.array([rp[int(g)] for g in G])
ngt=dict(zip(r['grp_ids'].tolist(), r['ngt'].tolist()))
WORK=(gseen==0)&np.array([p not in LOCK for p in gpid])
# en cok CP kaciran parcalar
miss=[]
for g in np.unique(G[WORK]):
    m=WORK&(G==g); n_miss=ngt[int(g)]-int(Y[m].sum())
    if n_miss>0: miss.append((n_miss, rp[int(g)]))
miss.sort(reverse=True)
sample=[p for _,p in miss[:30]]
print(f'kacirilan CP olan {len(miss)} parca | ilk 30 inceleniyor', flush=True)
paths={p:(jf,s) for m_,p,jf,s in eligible()}
cfg=json.load(open('cp_config.json')); dev='cuda' if torch.cuda.is_available() else 'cpu'
models=[load_any(c,dev=dev)[:2] for c in cfg['current_product']['checkpoints']]
FIRED=0; BLIND=0; det=[]
for pid in sample:
    if pid not in paths: continue
    try:
        jf,stp=paths[pid]
        j=json.load(open(jf,encoding='utf-8-sig'))
        Vj=np.array([[q['X'],q['Y'],q['Z']] for q in j['Graphic3d']['Points']],float)
        Gp=np.array([[c['Point']['X'],c['Point']['Y'],c['Point']['Z']] for c in j['ConnectionPoints']],float)
        Gd=np.array([[c['InsertDirection']['X'],c['InsertDirection']['Y'],c['InsertDirection']['Z']] for c in j['ConnectionPoints']],float)
        Gd=Gd/(np.linalg.norm(Gd,axis=1,keepdims=True)+1e-9)
        Vr,Fr=step_to_mesh(stp); V,F=thesis_remesh.remesh_uniform(Vr,Fr,target=6000)
        V=np.ascontiguousarray(V,np.float64); F=np.ascontiguousarray(F,np.int64)
        acc=None; per=[]
        for m_,meta in models:
            _,pb=D.predict(m_,meta,V,F,device=dev,op_cache_dir=f"{OP}_k{int(meta.get('k_eig',64))}",return_probs=True)
            pb=np.asarray(pb,float); acc=pb if acc is None else acc+pb
            per.append(cp_openings.connection_points(V,F,pb.argmax(-1),min_v=30,classes=(CE,CT),dedupe_mm=10.0,probs=pb,vertex_conf=0.5,ct_depth_min_mm=1.0,cluster_mm=5.0))
        probs=acc/len(models); lab=probs.argmax(-1)
        cps=_vote2(per,min_votes=1)
        body=trimesh.Trimesh(V,F,process=False)
        R,t,_=align_frames(Vr,Vj); Gm=(Gp-t)@R; Gdm=Gd@R
        P=np.array([c['point'] for c in cps],float) if cps else np.zeros((0,3))
        diag=float(np.linalg.norm(V.max(0)-V.min(0))); tol=max(3.0,0.06*diag)
        for g_,gd in zip(Gm,Gdm):
            hit=False
            if len(P):
                d=P-g_; al=d@gd; pp=np.linalg.norm(d-al[:,None]*gd,axis=1)
                hit=bool(((np.abs(al)<=40.0)&(pp<=tol)).any())
            if hit: continue                       # bu CP zaten bulunmus
            # DUZELTME: GT yuvada durur; once GERCEK AGZI bul (iki yone de bak), sonra oraya bak
            mouth,_=seat_to_mouth(body, g_, gd)
            rel=V-mouth; alv=rel@gd; perp=np.linalg.norm(rel-alv[:,None]*gd,axis=1)
            zone=(np.abs(alv)<=8.0)&(perp<=8.0)
            if not zone.any(): BLIND+=1; continue
            nce=int((lab[zone]==CE).sum()); nct=int((lab[zone]==CT).sum())
            if nce+nct>0: FIRED+=1; det.append((pid,nce+nct,int(zone.sum())))
            else: BLIND+=1
    except Exception:
        continue
tot=FIRED+BLIND
print()
print(f'KACIRILAN {tot} CP incelendi:')
print(f'  segmentasyon ATESLEDI ama aday olmadi : {FIRED:3d} (%{100*FIRED/max(tot,1):.0f})  -> POST-ISLEME sorunu (ucuz)')
print(f'  segmentasyon HIC ateslemedi           : {BLIND:3d} (%{100*BLIND/max(tot,1):.0f})  -> SEGMENTASYON kor (yeniden egitim)')
if det:
    print(f'  atesleyen ornekler (parca, CE+CT vertex, bolge vertex): {det[:6]}')
