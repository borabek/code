"""POST-ISLEME SWEEP: aday-tavanini (kapsama) yukselten parametreleri bul.
Mevcut degerler (min_v 30 / vc 0.5 / cluster 5 / dedupe 10) F1 icin ayarlanmisti.
YENI HEDEF = KAPSAMA -> bu parametreler yeniden ayarlanmali.
Verimli: her parca icin segmentasyon BIR KEZ kosar, sonra parametreler ucuz taranir."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0,'.'); os.environ['BA_ALLOW_SEEN']='1'
import diffusionnet as D, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2

N=int(sys.argv[1]) if len(sys.argv)>1 else 60
r=np.load('results/rich_feats.npz',allow_pickle=True)
lock=json.load(open('results/split_lock.json')); LOCK=set(lock['locked_parts'])
Y,G=r['y'],r['groups']; rp=[str(v) for v in r['part_ids']]
gseen=np.array([int(r['seen'][int(g)]) for g in G]); gpid=np.array([rp[int(g)] for g in G])
ngt=dict(zip(r['grp_ids'].tolist(), r['ngt'].tolist()))
WORK=(gseen==0)&np.array([p not in LOCK for p in gpid])
parts=[rp[int(g)] for g in np.unique(G[WORK])]
rs=np.random.RandomState(0); rs.shuffle(parts); parts=parts[:N]
paths={p:(jf,s) for m_,p,jf,s in eligible()}
cfg=json.load(open('cp_config.json')); dev='cuda' if torch.cuda.is_available() else 'cpu'
models=[load_any(c,dev=dev)[:2] for c in cfg['current_product']['checkpoints']]

cache=[]; t0=time.time()
for i,pid in enumerate(parts,1):
    if pid not in paths: continue
    try:
        jf,stp=paths[pid]
        j=json.load(open(jf,encoding='utf-8-sig'))
        Vj=np.array([[q['X'],q['Y'],q['Z']] for q in j['Graphic3d']['Points']],float)
        Gp=np.array([[c['Point']['X'],c['Point']['Y'],c['Point']['Z']] for c in j['ConnectionPoints']],float)
        Gd=np.array([[c['InsertDirection']['X'],c['InsertDirection']['Y'],c['InsertDirection']['Z']] for c in j['ConnectionPoints']],float)
        if not len(Gp): continue
        Gd=Gd/(np.linalg.norm(Gd,axis=1,keepdims=True)+1e-9)
        Vr,Fr=step_to_mesh(stp); V,F=thesis_remesh.remesh_uniform(Vr,Fr,target=6000)
        V=np.ascontiguousarray(V,np.float64); F=np.ascontiguousarray(F,np.int64)
        pbs=[]
        for m_,meta in models:
            _,pb=D.predict(m_,meta,V,F,device=dev,op_cache_dir=f"{OP}_k{int(meta.get('k_eig',64))}",return_probs=True)
            pbs.append(np.asarray(pb,float))
        R,t,_=align_frames(Vr,Vj)
        cache.append((V,F,pbs,(Gp-t)@R,Gd@R,float(np.linalg.norm(V.max(0)-V.min(0)))))
    except Exception: continue
    if i%20==0: print(f'  {i}/{len(parts)} cikarildi {time.time()-t0:.0f}s',flush=True)
print(f'{len(cache)} parca hazir ({time.time()-t0:.0f}s). Sweep basliyor...\n',flush=True)

def evaluate(mv,vc,cl,dd):
    hit=tot=ncand=0
    for V,F,pbs,Gm,Gdm,diag in cache:
        per=[cp_openings.connection_points(V,F,pb.argmax(-1),min_v=mv,classes=(CE,CT),dedupe_mm=dd,
             probs=pb,vertex_conf=vc,ct_depth_min_mm=1.0,cluster_mm=cl) for pb in pbs]
        cps=_vote2([c for c in per if c],min_votes=1); ncand+=len(cps)
        P=np.array([c['point'] for c in cps],float) if cps else np.zeros((0,3))
        tol=max(3.0,0.06*diag)
        for g_,gd in zip(Gm,Gdm):
            tot+=1
            if len(P):
                d=P-g_; al=d@gd; pp=np.linalg.norm(d-al[:,None]*gd,axis=1)
                hit+= int(bool(((np.abs(al)<=40.0)&(pp<=tol)).any()))
    return hit/max(tot,1), ncand/max(tot,1)

print("%-42s %10s %10s" % ("parametreler","TAVAN(kapsama)","aday/GT"))
base=evaluate(30,0.5,5.0,10.0)
print("%-42s %9.3f %10.2f  <-- MEVCUT URUN" % ("min_v30 vc0.50 cluster5 dedupe10", base[0], base[1]))
for mv in (20,10,5):
    c,n=evaluate(mv,0.5,5.0,10.0); print("%-42s %9.3f %10.2f" % (f"min_v{mv} vc0.50 cluster5 dedupe10",c,n))
for vc in (0.40,0.30,0.20):
    c,n=evaluate(30,vc,5.0,10.0); print("%-42s %9.3f %10.2f" % (f"min_v30 vc{vc:.2f} cluster5 dedupe10",c,n))
for dd in (6.0,4.0,2.0):
    c,n=evaluate(30,0.5,5.0,dd); print("%-42s %9.3f %10.2f" % (f"min_v30 vc0.50 cluster5 dedupe{dd:.0f}",c,n))
for cl in (3.0,2.0):
    c,n=evaluate(30,0.5,cl,10.0); print("%-42s %9.3f %10.2f" % (f"min_v30 vc0.50 cluster{cl:.0f} dedupe10",c,n))
for mv,vc,cl,dd in ((10,0.30,3.0,6.0),(5,0.20,2.0,4.0),(10,0.35,3.0,5.0)):
    c,n=evaluate(mv,vc,cl,dd); print("%-42s %9.3f %10.2f  <-- kombinasyon" % (f"min_v{mv} vc{vc:.2f} cluster{cl:.0f} dedupe{dd:.0f}",c,n))
