# -*- coding: utf-8 -*-
"""WSCAD partType YOKLAMASI -- amac: CP (ConnectionPoint) iceren a indirme formati present mi?

WHY: elimizde 2838 STEP present but CP'si absent; CP'li JSON'u which is 1926 parcanin cogunun STEP'i absent.
Kesisim only ~1146 -> ogrenme egrisinin ana kisiti this. Eger portal, STEP outside CP tasiyan a
format (WSCAD/EPLAN makro, urun-verisi JSON, vb.) sunuyorsa, 2838 parcanin ANAHTARI oradan gelebilir
= corpus 3x = measured_path egriye according to single basina ~+0.07 F1.

YONTEM: STEP indirmesi partType=2 with yapiliyor. Diger partType degerlerini TEK parcada dener,
donen icerigin first baytlarina/uzunluguna bakar. Minimal yuk: 1 part x birkac value, aralarda gecikme.
Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe wscad_probe_types.py
"""
import os ,sys ,time ,json 

PROFILE =os .path .abspath ("_wsprofile")
HOME ="https://www.wscaduniverse.com/most/"
MID ,PN =63 ,"3002162"# PXC, STEP'i basariyla indigi known part

JS ="""async ({mid, pn, pt}) => {
  let bearer = null;
  for (const k of Object.keys(localStorage)) {
    const v = localStorage.getItem(k) || "";
    const m = v.match(/eyJ[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+/);
    if (m) { bearer = m[0]; break; }
  }
  if (!bearer) return {err: "oturum tokeni absent"};
  const res = await fetch("https://bff.wscaduniverse.com/api/download/part", {
    method: "POST",
    headers: {"content-type": "application/json", "accept": "application/json",
              "authorization": "Bearer " + bearer},
    body: JSON.stringify({manufacturerId: mid, partNumber: pn, partType: pt, norm: 0, language: "most"})
  });
  const txt = await res.text();
  return {status: res.status, len: txt.length, head: txt.slice(0, 160)};
}"""


def main ():
    from playwright .sync_api import sync_playwright 
    hits ={}
    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (PROFILE ,channel ="msedge",headless =True ,
        accept_downloads =True )
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .goto (HOME ,wait_until ="domcontentloaded",timeout =60000 )
        print (f"part {MID }/{PN } uzerinde partType yoklamasi (STEP = 2)\n",flush =True )
        for pt in (0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ):
            try :
                r =page .evaluate (JS ,{"mid":MID ,"pn":PN ,"pt":pt })
            except Exception as e :
                print (f"  partType {pt }: HATA {str (e )[:60 ]}",flush =True );continue 
            if r .get ("err"):
                print ("  "+r ["err"]);break 
            st ,ln ,hd =r .get ("status"),r .get ("len",0 ),(r .get ("head")or "").replace ("\n"," ")[:90 ]
            kind =("STEP"if hd .startswith ("ISO-10303")else 
            "JSON"if hd .lstrip ().startswith (("{","["))else 
            "XML"if hd .lstrip ().startswith ("<")else 
            "ZIP/ikili"if "PK"in hd [:4 ]else "?")
            flag =""
            if st ==200 and ln >500 :
                low =hd .lower ()
                if any (w in low for w in ("connection","anschluss","terminal","klemme","point")):
                    flag ="  <<< CP IZI VAR"
                hits [pt ]=(ln ,kind ,hd )
            print (f"  partType {pt }: status {st } | {ln :9d} B | {kind :9s} | {hd [:70 ]}{flag }",flush =True )
            time .sleep (3 )
        ctx .close ()
    json .dump ({str (k ):{"len":v [0 ],"kind":v [1 ],"head":v [2 ]}for k ,v in hits .items ()},
    open ("results/wscad_parttypes.json","w"),indent =1 )
    print (f"\n-> results/wscad_parttypes.json  ({len (hits )} running format)")
    print ("   STEP outside 200 donen a format varsa, icerigini acip CP tasiyor mu bakariz.")


if __name__ =="__main__":
    main ()
