import json, glob, os, time, numpy as np, torch
import diffusionnet as D, connector3d, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
CE=int(connector3d.CABLE_ENTRY); CT=int(connector3d.CONTACT); OP='results/step_infer/ops'
DS='C:/Users/DE0002~1/AppData/Local/Temp/dsz/DataSet'
dev='cuda' if torch.cuda.is_available() else 'cpu'
model,meta,_=load_any('results/seg_extra/human77c_s0.pt',dev=dev)
def gm(P,G,tol):
    if not len(P) or not len(G): return 0,len(P),len(G)
    dm=np.linalg.norm(P[:,None,:]-G[None,:,:],axis=2); pr=sorted((dm[i,j],i,j) for i in range(len(P)) for j in range(len(G)))
    up,ug,tp=set(),set(),0
    for d,i,j in pr:
        if d>tol: break
        if i in up or j in ug: continue
        up.add(i); ug.add(j); tp+=1
    return tp,len(P)-tp,len(G)-tp
pids=['2861344','2861289','2985631','2985688','2703994','2861250','2904622']  # kucukten buyuge
T=Fp=Fn=0
for pid in pids:
    t0=time.time()
    stps=sorted(glob.glob(f'C:/Users/DE00024082/Downloads/wscaduniverse_{pid}_*.stp'))
    jf=glob.glob(f'{DS}/PXC.{pid}_*.json')
    if not stps or not jf: print(f'{pid}: file none',flush=True); continue
    j=json.load(open(jf[0],encoding='utf-8-sig'))
    Vj=np.array([[p['X'],p['Y'],p['Z']] for p in j['Graphic3d']['Points']],float)
    G=np.array([[c['Point']['X'],c['Point']['Y'],c['Point']['Z']] for c in j['ConnectionPoints']],float)
    try:
        Vr,Fr=step_to_mesh(stps[-1]); tstep=time.time()-t0
        V,Fm=thesis_remesh.remesh_uniform(Vr,Fr,target=6000); tremesh=time.time()-t0-tstep
        V=np.ascontiguousarray(V,np.float64); Fm=np.ascontiguousarray(Fm,np.int64)
        plab,probs=D.predict(model,meta,V,Fm,device=dev,op_cache_dir=OP,return_probs=True)
    except Exception as e:
        print(f'{pid}: ERR {str(e)[:60]}',flush=True); continue
    cps=cp_openings.connection_points(V,Fm,np.asarray(plab),min_v=60,classes=(CE,CT),dedupe_mm=10.0,probs=np.asarray(probs),vertex_conf=0.7,ct_depth_min_mm=1.0,cluster_mm=10.0)
    R,t,ares=align_frames(Vr,Vj); P=(np.array([np.asarray(c['point']) for c in cps],float)@R.T+t) if cps else np.zeros((0,3))
    tol=max(3.0,0.06*float(np.linalg.norm(Vj.max(0)-Vj.min(0))))
    tp,fp,fn=gm(P,G,tol); T+=tp; Fp+=fp; Fn+=fn
    print(f'{pid}: mfgCP{len(G)} pred{len(P)} align{ares:.1f}mm -> {tp}/{fp}/{fn}  [{time.time()-t0:.0f}s step{tstep:.0f}/remesh{tremesh:.0f}]',flush=True)
pr=T/max(T+Fp,1); rc=T/max(T+Fn,1); f1=2*pr*rc/max(pr+rc,1e-9)
print(f'\n=== 7 YENI PXC: TP{T} FP{Fp} FN{Fn} | P={pr:.3f} R={rc:.3f} F1={f1:.3f} ({T+Fn} CP) ===',flush=True)
