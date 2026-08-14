# -*- coding: utf-8 -*-
"""Build the RECALL painting tool: openings the model MISSES, located by the manufacturer.

WHY: two adjudication rounds proved the model's precision is really ~98% -- 183 of 189 "false
positives" were real openings our labels lacked. So the remaining error is RECALL: openings the model
never proposes, which adjudication cannot reach (there is nothing to review). The manufacturer data
can reach them: every unmatched ConnectionPoint is a KNOWN opening the model missed.

That splits the work the right way:
    manufacturer -> WHERE the opening is (no searching)
    human        -> WHAT SHAPE it is (a few brush strokes)
Exactly the split that beat four failed attempts to generate regions automatically (best IoU 0.18).

The camera opens already aimed at the missed CP, marked with a red sphere, so each item is a few
seconds of painting rather than a hunt.

Also the first human supervision WEI will ever have -- and WEI is where the model collapses
(F1 0.157 vs PXC 0.620 ten the manufacturer arbiter).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe build_recall_tool.py \
         [--input results/misses_wei.json] [--out results/recall/paint.html]
"""
import os ,sys ,json ,base64 ,argparse 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import thesis_remesh 
from infer_step_cp import step_to_mesh 


def pack (a ,dtype ):
    return base64 .b64encode (np .ascontiguousarray (a ,dtype ).tobytes ()).decode ("ascii")


HTML ="""<!doctype html><meta charset=utf-8><title>Recall boyama</title>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
*{box-sizing:border-box} body{margin:0;font:14px system-ui;background:#14161b;color:#e8e8e8;display:flex;height:100vh}
#left{flex:1;position:relative} canvas{width:100%;height:100%;display:block}
#side{width:330px;background:#191c22;border-left:1px solid #2a2f3a;padding:16px;overflow:auto}
h2{margin:0 0 2px;font-size:16px} .sub{color:#9aa3b2;font-size:12.5px;margin-bottom:12px;line-height:1.45}
button{background:#262b35;color:#e8e8e8;border:0;padding:8px 12px;border-radius:7px;cursor:pointer;font-size:13px;margin:0 6px 6px 0}
button:hover{background:#333a47} button.primary{background:#3b82f6} button.warn{background:#8a3b3b}
#prog{color:#9aa3b2;font-size:12.5px;margin:10px 0}
#modebtn.on{background:#2f7d4a}
#stat{position:absolute;right:14px;top:12px;font-size:12px;background:#14161bcc;padding:6px 10px;border-radius:6px;color:#9aa3b2}
.row{margin:12px 0;padding:11px;background:#1e222a;border-radius:9px}
label{display:block;color:#93a0b3;font-size:12px;margin-bottom:5px}
input[type=range]{width:100%}
#hint{position:absolute;left:14px;bottom:12px;color:#7e8899;font-size:12px;background:#14161bcc;padding:7px 11px;border-radius:6px;line-height:1.5}
.ok{color:#5fbf72}
</style>
<div id=left><canvas id=cv></canvas><div id=stat>boya modu KAPALI</div><div id=hint>
<b>sol tik + surukle</b> = boya &nbsp;|&nbsp; <b>sag tik surukle</b> = dondur &nbsp;|&nbsp; tekerlek = yakinlas<br>
<b>R</b> = delige bak &nbsp;|&nbsp; <b>B</b> = boya modu ac/kapa &nbsp;|&nbsp; <b>Z</b> = geri al &nbsp;|&nbsp; <b>Space</b> = sonraki &nbsp;|&nbsp; kirmizi kure = manufacturer CP-si (<b>parcanin ICINDE</b>, temas noktasi) &nbsp;|&nbsp; <b>sari ok</b> = disari, boyanacak yuzeye dogru
</div></div>
<div id=side>
  <h2 id=title>-</h2>
  <div class=sub id=sub>-</div>
  <div>
    <button onclick=go(-1)>&larr; onceki</button>
    <button onclick=go(1)>sonraki &rarr;</button>
    <button id=modebtn onclick=toggleMode()>BOYA MODU: KAPALI</button>
    <button onclick=resetView()>gorunumu sifirla (R)</button>
    <button class=warn onclick=clearPaint()>temizle</button>
    <button class=primary onclick=dl()>JSON indir</button>
  </div>
  <div class=row>
    <label>Firca yaricapi: <span id=bs>4.0</span> mm</label>
    <input type=range id=brush min=1 max=12 step=0.5 value=4 oninput="BR=+this.value;document.getElementById('bs').textContent=BR.toFixed(1)">
  </div>
  <div id=prog></div>
  <div class=sub style="margin-top:14px">
    Sari okun <b>gosterdigi yondeki yuzey acikligini</b> boya. Kirmizi kure manufacturer CP-si ve parcanin icinde durur (temas noktasi) &mdash; boyanacak yer okun cikis yaptigi <b>dis yuzey</b>. Model burayi kaciriyor; senin boyadigin sekil
    egitime <b>pozitif</b> olarak girecek.<br><br>
    Aciklik gorunmuyorsa ya da emin degilsen <b>bos birak</b> ve gec &mdash; bos olanlar kullanilmaz.
  </div>
</div>
<script>
const ITEMS = __DATA__;
const KEY = "__KEY__";
let store = JSON.parse(localStorage.getItem(KEY) || "{}");
let cur = 0, BR = 4.0, mesh=null, geom=null, colAttr=null, V=null, F=null, painted=null, undo=[], marker=null;

const cv=document.getElementById("cv");
const renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x14161b);
const camera=new THREE.PerspectiveCamera(45,1,0.1,5000);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
// PAINT MODE rather than fighting OrbitControls for the left button: with LEFT:null the control
// still swallowed the drag and nothing painted at all (the annotator hit exactly this). In paint
// mode the controls are switched OFF entirely, so a left drag can only mean paint.
let PAINT=false;
function toggleMode(){
  PAINT=!PAINT; controls.enabled=!PAINT;
  const b=document.getElementById("modebtn");
  b.textContent = PAINT ? "BOYA MODU: ACIK" : "BOYA MODU: KAPALI";
  b.classList.toggle("on", PAINT);
  document.getElementById("stat").textContent =
    PAINT ? "BOYA MODU ACIK -- sol tik boyar" : "boya modu KAPALI -- sol tik dondurur";
}
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl1=new THREE.DirectionalLight(0xffffff,0.5); dl1.position.set(1,1,2); scene.add(dl1);
const dl2=new THREE.DirectionalLight(0xffffff,0.35); dl2.position.set(-1,-1,-1); scene.add(dl2);
const ray=new THREE.Raycaster(), ptr=new THREE.Vector2();
function resize(){const w=cv.clientWidth,h=cv.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
addEventListener("resize",resize);
(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
function b64(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}

function show(i){
  cur=(i+ITEMS.length)%ITEMS.length;
  const it=ITEMS[cur];
  if(mesh){scene.remove(mesh);geom.dispose();}
  if(marker){scene.remove(marker);}
  V=b64(it.v,Float32Array); F=b64(it.f,Uint32Array);
  const nV=V.length/3;
  painted=new Uint8Array(nV);
  const saved=store[it.key];
  if(saved) for(const k of saved) if(k<nV) painted[k]=1;
  undo=[];
  geom=new THREE.BufferGeometry();
  geom.setAttribute("position",new THREE.BufferAttribute(V,3));
  colAttr=new THREE.BufferAttribute(new Float32Array(nV*3),3);
  geom.setAttribute("color",colAttr);
  geom.setIndex(new THREE.BufferAttribute(F,1));
  geom.computeVertexNormals(); geom.computeBoundingSphere();
  mesh=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.9,
    transparent:true, opacity:0.93}));
  scene.add(mesh); recolor();
  const r=geom.boundingSphere.radius;
  // THE MANUFACTURER'S CP SITS ~15mm INSIDE THE PART (it is the contact, not the mouth -- measured
  // today: tangential 1.4mm, axial +14.8mm). A plain sphere there is BURIED and invisible, which is
  // exactly what the annotator hit. So: draw it depth-test-free so it always shows through, and add
  // an arrow along the insertion axis pointing OUT to the face whose opening must be painted.
  marker=new THREE.Group();
  const sph=new THREE.Mesh(new THREE.SphereGeometry(r*0.030,20,14),
      new THREE.MeshBasicMaterial({color:0xe0342c, depthTest:false, transparent:true, opacity:0.95}));
  sph.renderOrder=999; marker.add(sph);
  const dirv=new THREE.Vector3(it.dir[0],it.dir[1],it.dir[2]).normalize();
  const arrow=new THREE.ArrowHelper(dirv, new THREE.Vector3(0,0,0), r*0.9, 0xffcc22, r*0.16, r*0.09);
  arrow.line.material.depthTest=false; arrow.cone.material.depthTest=false;
  arrow.line.renderOrder=999; arrow.cone.renderOrder=999;
  arrow.line.material.linewidth=3;
  marker.add(arrow);
  marker.position.set(it.cp[0],it.cp[1],it.cp[2]); scene.add(marker);
  RAD=r; resetView();
  resize();
  document.getElementById("title").textContent=`${cur+1}/${ITEMS.length}  ${it.pid} (${it.mfg})`;
  document.getElementById("sub").textContent=
    `manufacturer bu parcada ${it.n_mfg} CP tanimliyor, model ${it.n_pred} buldu -- bu kacan biri`;
  prog();
}
let RAD=1;
function resetView(){
  // Look straight INTO the opening: camera on the +InsertDirection side, aimed at the CP. Rotating
  // loses this framing and the part's orientation stops being obvious (the annotator read the arrow
  // as pointing "down" when it actually runs along the 56mm DEPTH axis, where the wire really enters),
  // so R always brings the honest view back.
  const it=ITEMS[cur];
  const c=new THREE.Vector3(it.cp[0],it.cp[1],it.cp[2]);
  const d=new THREE.Vector3(it.dir[0],it.dir[1],it.dir[2]).normalize();
  controls.target.copy(c);
  camera.position.copy(c).addScaledVector(d, RAD*1.6);
  camera.up.set(0,0,1);
  controls.update();
}
function recolor(){
  const a=colAttr.array;
  for(let i=0;i<painted.length;i++){
    if(painted[i]){a[i*3]=0.10;a[i*3+1]=0.82;a[i*3+2]=0.26;}
    else{a[i*3]=0.66;a[i*3+1]=0.67;a[i*3+2]=0.70;}
  }
  colAttr.needsUpdate=true;
}
function paintAt(e){
  const r=cv.getBoundingClientRect();
  ptr.x=((e.clientX-r.left)/r.width)*2-1; ptr.y=-((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(ptr,camera);
  const hit=ray.intersectObject(mesh,false)[0];
  const st=document.getElementById("stat");
  if(!hit){ st.textContent="parcaya denk gelmedi -- mesh uzerine tikla"; return; }
  const p=hit.point; const changed=[];
  for(let i=0;i<painted.length;i++){
    if(painted[i]) continue;
    const dx=V[i*3]-p.x, dy=V[i*3+1]-p.y, dz=V[i*3+2]-p.z;
    if(dx*dx+dy*dy+dz*dz<=BR*BR){ painted[i]=1; changed.push(i); }
  }
  if(changed.length){ undo.push(changed); recolor(); save();
    st.textContent = "boyandi: " + changed.length + " vertex (firca " + BR.toFixed(1) + "mm)"; }
  else st.textContent = "bu noktada zaten boyali";
}
let painting=false;
cv.addEventListener("pointerdown",e=>{
  if(!PAINT||e.button!==0) return;
  e.preventDefault(); cv.setPointerCapture(e.pointerId); painting=true; paintAt(e);
});
cv.addEventListener("pointermove",e=>{ if(painting) paintAt(e); });
addEventListener("pointerup",()=>painting=false);
cv.addEventListener("contextmenu",e=>e.preventDefault());
addEventListener("keydown",e=>{
  if(e.key==="z"||e.key==="Z"){const c=undo.pop(); if(c){for(const i of c)painted[i]=0; recolor(); save();}}
  if(e.key==="b"||e.key==="B"){toggleMode();}
  if(e.key==="r"||e.key==="R"){resetView();}
  if(e.code==="Space"){e.preventDefault(); go(1);}
  if(e.key==="ArrowRight")go(1); if(e.key==="ArrowLeft")go(-1);
});
function clearPaint(){undo.push([...painted.keys()].filter(i=>painted[i])); painted.fill(0); recolor(); save();}
function save(){
  const idx=[]; for(let i=0;i<painted.length;i++) if(painted[i]) idx.push(i);
  if(idx.length) store[ITEMS[cur].key]=idx; else delete store[ITEMS[cur].key];
  localStorage.setItem(KEY,JSON.stringify(store)); prog();
}
function prog(){
  const n=Object.keys(store).length;
  const here=(store[ITEMS[cur].key]||[]).length;
  document.getElementById("prog").innerHTML=
    `<b>${n}</b> / ${ITEMS.length} boyandi` + (here?` &nbsp;<span class=ok>&#10003; bu parcada ${here} vertex</span>`:"");
}
function go(d){show(cur+d);}
function dl(){
  const b=new Blob([JSON.stringify(store)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="recall_paint.json";a.click();
}
show(0);
</script>"""


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--input",default ="results/misses_wei.json")
    ap .add_argument ("--out",default ="results/recall/paint.html")
    ap .add_argument ("--key",default ="cp_recall_v1")
    ap .add_argument ("--limit",type =int ,default =0 )
    a =ap .parse_args ()

    d =json .load (open (a .input ))
    items =d ["items"][:a .limit ]if a .limit else d ["items"]
    cache ={}
    out =[]
    for k ,it in enumerate (items ):
        pid =it ["part_id"]
        if pid not in cache :
            Vr ,Fr =step_to_mesh (it ["step"])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            cache [pid ]=(np .ascontiguousarray (V ,float ),np .ascontiguousarray (F ,np .int64 ))
            print (f"  {pid }: {len (V )}v",flush =True )
        V ,F =cache [pid ]
        c =V .mean (0 )
        out .append ({"key":f"{pid }__{k }","pid":pid ,"mfg":it ["mfg"],
        "v":pack (V -c ,np .float32 ),"f":pack (F ,np .uint32 ),
        "cp":(np .asarray (it ["cp"])-c ).round (3 ).tolist (),
        "dir":np .asarray (it ["dir"]).round (4 ).tolist (),
        "n_mfg":it ["n_mfg_cps"],"n_pred":it ["n_pred"]})
    doc =HTML .replace ("__DATA__",json .dumps (out ,separators =(",",":"))).replace ("__KEY__",a .key )
    os .makedirs (os .path .dirname (a .out ),exist_ok =True )
    open (a .out ,"w",encoding ="utf-8").write (doc )
    print (f"\n-> {a .out }  ({len (out )} kacan CP, {len (cache )} part, "
    f"{os .path .getsize (a .out )/1e6 :.1f} MB)")


if __name__ =="__main__":
    main ()
