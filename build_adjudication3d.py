# -*- coding: utf-8 -*-
"""Interactive 3D adjudication page for the model-vs-human disagreements.

Two static renders are not enough to judge whether a red marker sits ten a real wire opening or ten a
screw head -- you have to turn the part. This builds ONE self-contained page (three.js from CDN, same
engine and controls as label_tool.html, which the annotator already knows: drag = rotate, wheel =
zoom) holding every part's mesh, with the annotator's own marks in green and each disputed model CP
as a numbered red marker. Answers live in localStorage and export as the same adjudication.json that
apply_adjudication.py consumes.

Meshes are packed as base64 Float32/Uint32 so the page stays a few MB rather than tens.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe build_adjudication3d.py
"""
import os ,sys ,json ,base64 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from region_label_helper import load_obj 
import connector3d 

CE =int (connector3d .CABLE_ENTRY )
OUT ="results/adjudicate/adjudicate3d.html"


def pack (a ,dtype ):
    return base64 .b64encode (np .ascontiguousarray (a ,dtype ).tobytes ()).decode ("ascii")


def main ():
    import argparse 
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--input",default ="results/disagreements.json")
    ap .add_argument ("--out",default =OUT )
    ap .add_argument ("--key",default ="cp_adjudication_v1",
    help ="localStorage key -- give each round its OWN key so a later round never "
    "shows (or overwrites) the previous round's answers")
    a =ap .parse_args ()
    out_path =a .out 
    dis =json .load (open (a .input ))
    by_part ={}
    for it in dis ["items"]:
        by_part .setdefault ((it ["part_id"],it ["dir"]),[]).append (it )

    parts =[]
    for (pid ,root ),items in sorted (by_part .items ()):
        of =os .path .join (root ,pid ,f"{pid }.obj")
        lf =os .path .join (root ,pid ,f"{pid }.labels.txt")
        if not (os .path .exists (of )and os .path .exists (lf )):
            continue 
        V ,F =load_obj (of )
        L =np .array ([int (x )for x in open (lf ).read ().split ()],np .int64 )
        if len (L )!=len (V ):
            continue 
        c =V .mean (0 )
        parts .append ({
        "pid":pid ,
        "v":pack (V -c ,np .float32 ),# centre it so the camera framing is trivial
        "f":pack (F ,np .uint32 ),
        "m":pack ((L ==CE ).astype (np .uint8 ),np .uint8 ),# the annotator's own marks
        "pts":[{"p":(np .asarray (it ["point"],float )-c ).round (3 ).tolist (),
        "src":it ["source"],"nv":it ["n_verts"],
        "near":it ["nearest_human_mm"]}for it in items ],
        })
        print (f"  {pid }: {len (V )}v, {len (items )} anlasmazlik",flush =True )

    n_items =sum (len (p ["pts"])for p in parts )
    data =json .dumps (parts ,separators =(",",":"))
    doc ="""<!doctype html><meta charset=utf-8><title>CP hakemleme 3D</title>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
*{box-sizing:border-box} body{margin:0;font:14px system-ui;background:#14161b;color:#e8e8e8;display:flex;height:100vh}
#left{flex:1;position:relative} canvas{width:100%;height:100%;display:block}
#side{width:340px;background:#191c22;border-left:1px solid #2a2f3a;padding:16px;overflow:auto}
h2{margin:0 0 2px;font-size:16px} .sub{color:#9aa3b2;font-size:12.5px;margin-bottom:12px}
.nav{display:flex;gap:8px;margin-bottom:14px}
button{background:#262b35;color:#e8e8e8;border:0;padding:8px 12px;border-radius:7px;cursor:pointer;font-size:13px}
button:hover{background:#333a47} button.primary{background:#3b82f6}
.q{background:#1e222a;border-radius:9px;padding:11px;margin-bottom:9px;border:1px solid #2a2f3a}
.q.done{border-color:#3b7a4a}
.qh{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.dot{width:19px;height:19px;border-radius:50%;background:#e0342c;color:#fff;font-size:12px;
     display:flex;align-items:center;justify-content:center;font-weight:700}
.meta{color:#93a0b3;font-size:11.5px}
.opts{display:flex;flex-direction:column;gap:5px}
label{padding:6px 9px;border-radius:6px;background:#232833;cursor:pointer;font-size:13px}
label:hover{background:#2c3341} input{margin-right:7px}
#prog{color:#9aa3b2;font-size:12.5px;margin:10px 0}
#hint{position:absolute;left:14px;bottom:12px;color:#7e8899;font-size:12px;
      background:#14161bcc;padding:6px 10px;border-radius:6px}
</style>
<div id=left><canvas id=cv></canvas><div id=hint>surukle = dondur &middot; tekerlek = yakinlas</div></div>
<div id=side>
  <h2 id=title>-</h2>
  <div class=sub id=sub>-</div>
  <div class=nav>
    <button onclick=go(-1)>&larr; onceki</button>
    <button onclick=go(1)>sonraki &rarr;</button>
    <button class=primary onclick=dl()>JSON indir</button>
  </div>
  <div id=prog></div>
  <div id=qs></div>
</div>
<script>
const PARTS = __DATA__;
const KEY = "__KEY__";
let ans = JSON.parse(localStorage.getItem(KEY) || "{}");
let cur = 0, mesh=null, markers=[];

const cv=document.getElementById("cv");
const renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x14161b);
const camera=new THREE.PerspectiveCamera(45,1,0.1,5000);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.8));
const d1=new THREE.DirectionalLight(0xffffff,0.55); d1.position.set(1,1,2); scene.add(d1);
const d2=new THREE.DirectionalLight(0xffffff,0.35); d2.position.set(-1,-1,-1); scene.add(d2);
function resize(){const w=cv.clientWidth,h=cv.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
addEventListener("resize",resize);
(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();

function b64(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}

function show(i){
  cur=(i+PARTS.length)%PARTS.length;
  const P=PARTS[cur];
  if(mesh){scene.remove(mesh);mesh.geometry.dispose();}
  markers.forEach(m=>scene.remove(m)); markers=[];
  const V=b64(P.v,Float32Array), F=b64(P.f,Uint32Array), M=b64(P.m,Uint8Array);
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.BufferAttribute(V,3));
  const col=new Float32Array(V.length);
  for(let k=0;k<M.length;k++){
    if(M[k]){col[k*3]=0.10;col[k*3+1]=0.82;col[k*3+2]=0.26;}      // annotator's mark
    else    {col[k*3]=0.66;col[k*3+1]=0.67;col[k*3+2]=0.70;}
  }
  g.setAttribute("color",new THREE.BufferAttribute(col,3));
  g.setIndex(new THREE.BufferAttribute(F,1));
  g.computeVertexNormals(); g.computeBoundingSphere();
  mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.92,metalness:0.0}));
  scene.add(mesh);
  const r=g.boundingSphere.radius;
  P.pts.forEach((q,k)=>{
    const s=new THREE.Mesh(new THREE.SphereGeometry(r*0.045,20,14),
                           new THREE.MeshStandardMaterial({color:0xe0342c,roughness:0.5}));
    s.position.set(q.p[0],q.p[1],q.p[2]); scene.add(s); markers.push(s);
    // NUMBER the marker: with several disputed points on one part the side-panel list is otherwise
    // impossible to map onto the 3D view (the annotator hit exactly this).
    const cn=document.createElement("canvas"); cn.width=cn.height=64;
    const cx=cn.getContext("2d");
    cx.fillStyle="#fff"; cx.font="bold 46px system-ui"; cx.textAlign="center"; cx.textBaseline="middle";
    cx.strokeStyle="#000"; cx.lineWidth=6; cx.strokeText(k+1,32,34); cx.fillText(k+1,32,34);
    const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cn),depthTest:false}));
    sp.position.set(q.p[0],q.p[1],q.p[2]); sp.scale.set(r*0.13,r*0.13,1);
    scene.add(sp); markers.push(sp);
  });
  controls.target.set(0,0,0);
  camera.position.set(0,0,r*2.7); controls.update(); resize();
  document.getElementById("title").textContent=`${cur+1}/${PARTS.length}  ${P.pid}`;
  document.getElementById("sub").textContent=`${P.pts.length} anlasmazlik | yesil = senin isaretin, kirmizi = modelin fazladan verdigi`;
  const qs=document.getElementById("qs"); qs.innerHTML="";
  P.pts.forEach((q,k)=>{
    const id=`${P.pid}_${k}`, v=ans[id]||"";
    const el=document.createElement("div"); el.className="q"+(v?" done":"");
    el.innerHTML=`<div class=qh><div class=dot>${k+1}</div>
      <div class=meta>${q.src}, ${q.nv} vertex, en yakin isaretin ${q.near}mm</div></div>
      <div class=opts>
      <label><input type=radio name="${id}" value=opening ${v==="opening"?"checked":""}>gercek opening (atlamisim)</label>
      <label><input type=radio name="${id}" value=not_cp ${v==="not_cp"?"checked":""}>CP not (vida/yuva)</label>
      <label><input type=radio name="${id}" value=unsure ${v==="unsure"?"checked":""}>emin degilim</label></div>`;
    el.onmouseenter=()=>highlight(k); el.onclick=()=>highlight(k);
    qs.appendChild(el);
  });
  qs.oninput=e=>{ans[e.target.name]=e.target.value;localStorage.setItem(KEY,JSON.stringify(ans));
                 e.target.closest(".q").classList.add("done");prog();};
  prog();
}
function highlight(k){
  // markers[] holds [sphere, label] per point, in panel order
  for(let i=0;i*2<markers.length;i++){
    const sph=markers[i*2], on=(i===k);
    if(!sph) continue;
    sph.material.color.setHex(on?0xffd21e:0xe0342c);
    const sc=on?1.55:1.0; sph.scale.set(sc,sc,sc);
  }
}
function prog(){
  const tot=__NITEMS__, n=Object.values(ans).filter(Boolean).length;
  document.getElementById("prog").textContent=`${n} / ${tot} cevaplandi`;
}
function go(d){show(cur+d);}
function dl(){
  const b=new Blob([JSON.stringify(ans,null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="adjudication.json";a.click();
}
addEventListener("keydown",e=>{if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1);});
show(0);
</script>"""
    doc =doc .replace ("__DATA__",data ).replace ("__NITEMS__",str (n_items )).replace ("__KEY__",a .key )
    os .makedirs (os .path .dirname (OUT ),exist_ok =True )
    open (OUT ,"w",encoding ="utf-8").write (doc )
    mb =os .path .getsize (OUT )/1e6 
    print (f"\n-> {OUT }  ({len (parts )} part, {n_items } anlasmazlik, {mb :.1f} MB)")


if __name__ =="__main__":
    main ()
