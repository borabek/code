# -*- coding: utf-8 -*-
"""Render the model-vs-human disagreements and build a one-click adjudication sheet.

Each disagreement is a place the PRODUCT emits a CP that the annotator did not mark. Adjudicating it
is useful whichever way it goes:
  "real opening"  -> the annotator missed it -> POSITIVE label correction (recall)
  "CP not"        -> a genuine distractor    -> NEGATIVE example (precision -- the signal this
                       project has never given; see mine_disagreements.py for why that matters)

Output: results/adjudicate/index.html (open in a browser, answer, press "JSON indir") plus one PNG
per part. The saved JSON feeds apply_adjudication.py.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe build_adjudication.py
"""
import os ,sys ,json ,html 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import matplotlib ;matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
from region_label_helper import load_obj 
import connector3d 

CE =int (connector3d .CABLE_ENTRY )
OUT ="results/adjudicate"


def render_part (pid ,root ,items ,path ):
    V ,F =load_obj (f"{root }/{pid }/{pid }.obj")
    L =np .array ([int (x )for x in open (f"{root }/{pid }/{pid }.labels.txt").read ().split ()],np .int64 )
    tri =V [F ]
    n =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    n =n /(np .linalg .norm (n ,axis =1 ,keepdims =True )+1e-9 )
    lt =np .array ([0.4 ,0.3 ,1.0 ]);lt =lt /np .linalg .norm (lt )
    base =np .stack ([0.42 +0.5 *np .abs (n @lt )]*3 ,1 )
    base [(L ==CE )[F ].any (1 )]=[0.15 ,0.8 ,0.25 ]# what the annotator already marked
    ext =V .max (0 )-V .min (0 );ctr =V .mean (0 );rr =ext .max ()*0.55 
    thin =int (np .argmin (ext ))
    views ={0 :[(0 ,0 ),(90 ,-90 )],1 :[(0 ,-90 ),(90 ,-90 )],2 :[(90 ,-90 ),(0 ,0 )]}[thin ]
    fig =plt .figure (figsize =(11 ,5.0 ))
    for k ,(ev ,az )in enumerate (views ):
        ax =fig .add_subplot (1 ,2 ,k +1 ,projection ="3d")
        ax .add_collection3d (Poly3DCollection (tri ,facecolors =np .clip (base ,0 ,1 ),edgecolors ="none"))
        for i ,it in enumerate (items ):
            p =np .asarray (it ["point"],float )
            ax .scatter (*p ,s =190 ,c ="red",marker ="o",depthshade =False ,edgecolors ="black",linewidths =1.4 )
            ax .text (p [0 ],p [1 ],p [2 ],f"  {i +1 }",color ="red",fontsize =13 ,weight ="bold")
        ax .set_xlim (ctr [0 ]-rr ,ctr [0 ]+rr );ax .set_ylim (ctr [1 ]-rr ,ctr [1 ]+rr );ax .set_zlim (ctr [2 ]-rr ,ctr [2 ]+rr )
        try :ax .set_box_aspect (ext )
        except Exception :pass 
        ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
    fig .suptitle (f"{pid }   yesil = senin isaretin   |   kirmizi = modelin fazladan verdigi CP",fontsize =11 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.94 ]);plt .savefig (path ,dpi =88 ,bbox_inches ="tight");plt .close (fig )


def main ():
    d =json .load (open ("results/disagreements.json"))
    by_part ={}
    for it in d ["items"]:
        by_part .setdefault ((it ["part_id"],it ["dir"]),[]).append (it )
    os .makedirs (OUT ,exist_ok =True )
    cards =[]
    for k ,((pid ,root ),items )in enumerate (sorted (by_part .items ()),1 ):
        png =f"{pid }.png"
        render_part (pid ,root ,items ,os .path .join (OUT ,png ))
        rows ="".join (
        f'<div class=q><b>#{i +1 }</b> <span class=meta>{html .escape (it ["source"])}, '
        f'{it ["n_verts"]}v, en yakin isaretin {it ["nearest_human_mm"]}mm</span>'
        f'<label><input type=radio name="{pid }_{i }" value="opening"> gercek opening (atlamisim)</label>'
        f'<label><input type=radio name="{pid }_{i }" value="not_cp"> CP not (vida/yuva/vs)</label>'
        f'<label><input type=radio name="{pid }_{i }" value="unsure"> emin degilim</label></div>'
        for i ,it in enumerate (items ))
        cards .append (f'<section><h3>{k }/{len (by_part )} &nbsp; {pid } '
        f'<small>({len (items )} anlasmazlik)</small></h3>'
        f'<img src="{png }" loading="lazy"><div class=qs>{rows }</div></section>')
    doc =f"""<meta charset=utf-8><title>CP hakemleme</title>
<style>
body{{font:15px system-ui;margin:0;background:#0f1115;color:#e6e6e6}}
header{{position:sticky;top:0;background:#171a21;padding:14px 20px;border-bottom:1px solid #2a2f3a;z-index:9}}
h1{{font-size:17px;margin:0 0 4px}} .sub{{color:#9aa3b2;font-size:13px}}
section{{padding:18px 20px;border-bottom:1px solid #222733}}
h3{{margin:0 0 8px;font-size:15px}} small{{color:#9aa3b2;font-weight:400}}
img{{max-width:100%;border-radius:8px;background:#fff}}
.q{{margin:9px 0;padding:9px 11px;background:#171a21;border-radius:8px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
.meta{{color:#9aa3b2;font-size:12.5px;flex:1}}
label{{cursor:pointer;padding:4px 9px;border-radius:6px;background:#20242d}}
label:hover{{background:#2b3140}}
button{{background:#3b82f6;color:#fff;border:0;padding:9px 16px;border-radius:8px;font-size:14px;cursor:pointer}}
#c{{color:#9aa3b2;font-size:13px;margin-left:12px}}
</style>
<header><h1>CP hakemleme &mdash; {d ['n_disagreements']} bolge / {len (by_part )} part</h1>
<div class=sub>Kirmizi nokta: modelin CP dedigi but senin isaretlemedigin yer. Her biri for: gercekten
a kablo/klemens agzi mi, yoksa not mi? &nbsp;<b>Emin degilsen "emin degilim" birak</b> &mdash;
tahmin, wrong etiketten iyidir not.</div>
<div style="margin-top:10px"><button onclick=dl()>JSON indir</button><span id=c></span></div></header>
{''.join (cards )}
<script>
const T={d ['n_disagreements']};
function upd(){{document.getElementById('c').textContent=
 document.querySelectorAll('input:checked').length+' / '+T+' cevaplandi';}}
document.addEventListener('change',upd); upd();
function dl(){{
 const o={{}};document.querySelectorAll('input:checked').forEach(i=>o[i.name]=i.value);
 const b=new Blob([JSON.stringify(o,null,1)],{{type:'application/json'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='adjudication.json';a.click();}}
</script>"""
    open (os .path .join (OUT ,"index.html"),"w",encoding ="utf-8").write (doc )
    print (f"-> {OUT }/index.html  ({d ['n_disagreements']} bolge, {len (by_part )} part)")


if __name__ =="__main__":
    main ()
