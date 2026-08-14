# -*- coding: utf-8 -*-
"""WIRE-PILOT: 'Insan, geometriden ureticinin CP listesini uretebilir mi?' -- HIC SORULMAMIS soru.

WHY IMPORTANT (2026-07-28 bulgulari):
  * Ogrenme egrisi duzlesmiyor -> VERI acligi ana teshis; 2838 STEP'imiz present but CP'si absent.
  * O 2838'i INSAN etiketiyle acabilir miyiz? Ancak insan ureticinin tanimini yeniden
    uretebiliyorsa. Eski adjudication 'real opening mi?' diye sordu ('here TEL present mi?' not)
    and that yuzden precision %98'e sisti -- i.e. this soru correct bicimde HIC sorulmadi.
DECISION KAPISI:
  uyum >= %90  -> insan etiketi valid ikame; 2838 part acilabilir (data acligina ilac)
  uyum <  %75  -> ureticinin listesinde geometride OLMAYAN katalog bilgisi present (R4 kesinlesir);
                  insan etiketi GURULTU adds -> path kapatilir, 100 saat bosa gitmez
Tasarim: GT'si BILINEN parts is used but kullaniciya GOSTERILMEZ (kor test).
Cikti: results/wire_pilot/wire_pilot.html (single file, three.js, localStorage, JSON export)
       + results/wire_pilot/truth.json (cevap anahtari -- ACMA, only skorlayici reads)
Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe build_wire_pilot.py [--n 30]
"""
import os ,sys ,json ,base64 ,argparse 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

OUT_DIR ="results/wire_pilot"


def pack (a ,dt ):
    return base64 .b64encode (np .ascontiguousarray (a ,dt ).tobytes ()).decode ()


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--n",type =int ,default =30 ,help ="kac part (15-20 dk for ~30 eligible)")
    ap .add_argument ("--seed",type =int ,default =0 )
    a =ap .parse_args ()

    r =np .load ("results/rich_feats.npz",allow_pickle =True )
    lock =json .load (open ("results/split_lock.json"))
    LOCKED =set (lock ["locked_parts"])
    Y ,G ,MF =r ["y"],r ["groups"],r ["mfg"]
    pids =[str (x )for x in r ["part_ids"]]
    seen =r ["seen"]
    ngt =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))

    # candidate count 4-10 arasi, KILITLI OLMAYAN, temiz parts (kisa and ogretici)
    cand_g =[int (g )for g in np .unique (G )
    if 4 <=int ((G ==g ).sum ())<=10 and seen [int (g )]==0 and pids [int (g )]not in LOCKED ]
    rs =np .random .RandomState (a .seed );rs .shuffle (cand_g )
    chosen =cand_g [:a .n ]
    print (f"{len (chosen )} part secildi (candidate 4-10, kilitli not)",flush =True )

    import thesis_remesh 
    from infer_step_cp import step_to_mesh 
    from big_arbiter import eligible 
    os .environ ["BA_ALLOW_SEEN"]="1"
    paths ={p :s for m ,p ,jf ,s in eligible ()}

    parts ,truth =[],{}
    for gi in chosen :
        pid =pids [gi ]
        if pid not in paths :continue 
        try :
            Vr ,Fr =step_to_mesh (paths [pid ])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        except Exception :
            continue 
        idx =np .where (G ==gi )[0 ]
        P =r ["pos"][idx ]# JSON frame konumlari
        # mesh'i same frame'e tasi: pos JSON frame'de, mesh STEP frame'de -> align_frames with
        from cad_eval import align_frames 
        import json as _j 
        jf =[x for m ,p ,x ,s in eligible ()if p ==pid ]
        if not jf :continue 
        jj =_j .load (open (jf [0 ],encoding ="utf-8-sig"))
        Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in jj ["Graphic3d"]["Points"]],float )
        R ,t ,_ =align_frames (Vr ,Vj )
        Vm =V @R .T +t # mesh -> JSON frame (adaylarla same)
        c =Vm .mean (0 )
        parts .append ({"pid":pid ,"v":pack (Vm -c ,np .float32 ),"f":pack (F ,np .uint32 ),
        "pts":[(P [k ]-c ).round (3 ).tolist ()for k in range (len (idx ))]})
        truth [pid ]=[int (Y [i ])for i in idx ]
        print (f"  {pid }: {len (V )}v, {len (idx )} candidate, {ngt [gi ]} manufacturer-CP",flush =True )

    os .makedirs (OUT_DIR ,exist_ok =True )
    json .dump (truth ,open (os .path .join (OUT_DIR ,"truth.json"),"w"))
    data =json .dumps (parts ,separators =(",",":"))
    doc ="""<!doctype html><meta charset=utf-8><title>WIRE PILOT -- tel mi alet mi?</title>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
body{margin:0;font:14px system-ui;background:#111;color:#eee;display:flex;height:100vh}
#v{flex:1}#s{width:290px;padding:14px;overflow:auto;background:#1b1b1b;border-left:1px solid #333}
h3{margin:0 0 8px}.q{margin:10px 0;padding:8px;background:#232323;border-radius:6px}
button{margin:2px;padding:6px 10px;border:0;border-radius:5px;cursor:pointer;font-weight:600}
.w{background:#1a7f37;color:#fff}.t{background:#a11;color:#fff}.u{background:#555;color:#fff}
.sel{outline:3px solid #ffd400}
#pg{font-size:12px;color:#9a9}#exp{background:#0a5ad6;color:#fff;width:100%;padding:10px;margin-top:12px}
small{color:#999;display:block;margin-top:6px;line-height:1.45}
</style>
<div id=v></div><div id=s>
<h3>Bu acikliga TEL girer mi?</h3>
<small>Sari kure = degerlendirilecek opening. Surukle=dondur, tekerlek=zoom.
<b>TEL</b>=kablo girisi &nbsp; <b>ALET</b>=tornavida/test/vida/montaj &nbsp; <b>?</b>=emin degilim.
Cevap anahtarini GORMUYORSUN -- this kor test.</small>
<div id=pg></div><div id=q></div>
<button id=exp>CEVAPLARI INDIR (bitince)</button></div>
<script>
const PARTS=DATA_HERE, KEY='wire_pilot_v1';
let ans=JSON.parse(localStorage.getItem(KEY)||'{}'), pi=0, sel=0;
const b64=(s,T)=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)};
const sc=new THREE.Scene(), cam=new THREE.PerspectiveCamera(45,1,.1,5000), rd=new THREE.WebGLRenderer({antialias:true});
document.getElementById('v').appendChild(rd.domElement);
const ctr=new THREE.OrbitControls(cam,rd.domElement);
sc.add(new THREE.AmbientLight(0xffffff,.75));const dl=new THREE.DirectionalLight(0xffffff,.7);dl.position.set(1,1,1);sc.add(dl);
let grp=new THREE.Group();sc.add(grp);
function rs(){const w=document.getElementById('v').clientWidth,h=window.innerHeight;cam.aspect=w/h;cam.updateProjectionMatrix();rd.setSize(w,h)}
window.onresize=rs;
function show(){
  grp.clear();const P=PARTS[pi];
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(b64(P.v,Float32Array),3));
  g.setIndex(new THREE.BufferAttribute(b64(P.f,Uint32Array),1));g.computeVertexNormals();
  grp.add(new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:0xb9b9b9,side:THREE.DoubleSide})));
  const bb=new THREE.Box3().setFromObject(grp),sz=bb.getSize(new THREE.Vector3()).length();
  P.pts.forEach((p,i)=>{
    const m=new THREE.Mesh(new THREE.SphereGeometry(sz*0.022,16,16),
      new THREE.MeshBasicMaterial({color:i===sel?0xffd400:0x888888}));
    m.position.set(p[0],p[1],p[2]);grp.add(m);
  });
  cam.position.set(sz,sz*.8,sz);ctr.target.set(0,0,0);ctr.update();
  const done=Object.keys(ans).length, tot=PARTS.reduce((s,p)=>s+p.pts.length,0);
  document.getElementById('pg').textContent=`part ${pi+1}/${PARTS.length} -- cevaplanan ${done}/${tot}`;
  let h='';
  P.pts.forEach((p,i)=>{
    const k=P.pid+'#'+i, v=ans[k]||'';
    h+=`<div class="q ${i===sel?'sel':''}"><b>Aciklik ${i+1}</b><br>
      <button class=w onclick="A('${k}','wire',${i})">TEL</button>
      <button class=t onclick="A('${k}','tool',${i})">ALET</button>
      <button class=u onclick="A('${k}','unsure',${i})">?</button>
      <span style="margin-left:6px">${v?'&#10003; '+v:''}</span></div>`;
  });
  h+=`<div style="margin-top:10px">
    <button onclick="N(-1)">&larr; onceki</button><button onclick="N(1)">sonraki &rarr;</button></div>`;
  document.getElementById('q').innerHTML=h;
}
function A(k,v,i){ans[k]=v;localStorage.setItem(KEY,JSON.stringify(ans));sel=Math.min(i+1,PARTS[pi].pts.length-1);show();
  const P=PARTS[pi];if(P.pts.every((_,j)=>ans[P.pid+'#'+j]))setTimeout(()=>N(1),250);}
function N(d){pi=(pi+d+PARTS.length)%PARTS.length;sel=0;show()}
document.getElementById('exp').onclick=()=>{
  const b=new Blob([JSON.stringify(ans,null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='wire_pilot_answers.json';a.click();};
rs();show();(function loop(){requestAnimationFrame(loop);ctr.update();rd.render(sc,cam)})();
</script>"""
    doc =doc .replace ("DATA_HERE",data )
    p =os .path .join (OUT_DIR ,"wire_pilot.html")
    open (p ,"w",encoding ="utf-8").write (doc )
    n =sum (len (x ["pts"])for x in parts )
    print (f"\n-> {p }  ({len (parts )} part, {n } opening, {os .path .getsize (p )/1e6 :.1f} MB)")
    print ("   Tarayicida ac, each acikliga TEL/ALET/? de, bitince 'CEVAPLARI INDIR'.")
    print ("   Sonra: python score_wire_pilot.py <indirilen json>")


if __name__ =="__main__":
    main ()
