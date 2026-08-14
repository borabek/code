# -*- coding: utf-8 -*-
"""Fetch the missing STEP files for parts that already have manufacturer ConnectionPoints.

WHY: DataSet 1 holds 1101 JSONs with manufacturer CPs, but only 400 of those parts have a STEP in
all_wscad_stp -- so 696 parts / 4033 CPs of free, complete, multi-manufacturer ground truth are
unusable. WEI first: the product scores 0.780 ten PXC but 0.233 ten Weidmueller, and manufacturer
labels for WEI already lifted it to 0.493, so WEI geometry is the measured lever.

HOW (learned by capturing one manual download, results/wscad_net.json):
    POST https://bff.wscaduniverse.com/api/download/part
    {"manufacturerId":88,"partNumber":"1010100000","partType":2,"norm":0,"language":"most"}
    Authorization: Bearer <session JWT>     -> responds with the STEP text itself (verified: 200,
    800083 bytes, starts "ISO-10303-21;", byte-identical in size to the manual download).
The call is issued from INSIDE the logged-in page (playwright + Edge + the _wsprofile session), so
the browser supplies the session and no credentials are ever handled here. Clicking through the SPA
is not needed -- an earlier selector-based attempt scored 0/3.

manufacturerId: 63 = Phoenix (PXC), 88 = Weidmueller (WEI) -- from the project's own _kdl_* lists.

POLITE BY DESIGN: sequential, a real delay between parts, skips what is already ten disk (so it
resumes), and stops after repeated failures instead of hammering the service.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe wscad_fetch.py \
         [--mfg WEI] [--limit N] [--delay 3]
"""
import os ,sys ,csv ,glob ,time ,json ,argparse 
from datetime import datetime 

PROFILE =os .path .abspath ("_wsprofile")
OUT ="all_wscad_stp"
LIST ="results/eksik_step_listesi.csv"
HOME ="https://www.wscaduniverse.com/most/"
MFG_ID ={"PXC":63 ,"WEI":88 ,"ABB":1 ,"A-B":5 ,"SIE":77 }
# ids confirmed by querying /api/search and reading manufacturerName back -- NEVER guess one: part
# numbers collide across manufacturers (searching a KLM number returned "Gira"), and a wrong id would
# silently fetch a DIFFERENT part. verify_downloads.py re-checks every file against its JSON mesh.
# A-B was 4470 here and that was WRONG -- /api/search returns 5 for Allen Bradley. Prefer
# --from-census, which carries the id the service itself reported for each part.
#
# AVAILABILITY IS TWO SEPARATE QUESTIONS, learned the hard way 2026-07-30:
#   searchable  -- /api/search finds the exact part number (what wscad_census.py measures)
#   downloadable-- a STEP actually exists for it (partType 2 returns 200)
# They are NOT the same. All 186 Allen Bradley 1492 terminals are searchable and none has a STEP:
# the part's own metadata (partType 0 -> Data/part.json) says formats:[0,1] = WSCAD data + EPLAN
# .edz only. A 404 here therefore means "not published as STEP", not "something is broken", and it
# must not trip the consecutive-failure guard.

JS ="""async ({mid, pn}) => {
  let bearer = null;
  for (const k of Object.keys(localStorage)) {
    const v = localStorage.getItem(k) || "";
    const m = v.match(/eyJ[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+/);
    if (m) { bearer = m[0]; break; }
  }
  if (!bearer) return {err: "oturum tokeni bulunamadi -- tekrar giris gerekebilir"};
  const res = await fetch("https://bff.wscaduniverse.com/api/download/part", {
    method: "POST",
    headers: {"content-type": "application/json", "accept": "application/json",
              "authorization": "Bearer " + bearer},
    body: JSON.stringify({manufacturerId: mid, partNumber: pn, partType: 2, norm: 0, language: "most"})
  });
  const txt = await res.text();
  return {status: res.status, len: txt.length, body: txt};
}"""


CENSUS ="results/wscad_availability.csv"


def todo (mfg ,limit ,list_path =LIST ):
    have ={os .path .basename (s ).split ("_")[1 ]for s in glob .glob (f"{OUT }/*.stp")}
    rows =[]
    with open (list_path ,encoding ="utf-8-sig")as fh :
        for r in csv .DictReader (fh ):
            if r ["parca_no"]in have or (mfg and r ["manufacturer"]!=mfg ):
                continue 
            if r ["manufacturer"]not in MFG_ID :
                continue 
            r ["mid"]=MFG_ID [r ["manufacturer"]]
            rows .append (r )
    return rows [:limit ]if limit else rows 


    # our JSON prefix -> the manufacturer name WSCAD reports. An exact part-number match is only the
    # SAME part when the maker also matches: part numbers collide hard across catalogues. The census
    # found our "KLM" numbers matching Legrand, Pilz, Gira, EATON, Mitsubishi and MERTEN exactly, and
    # a "PXC" number matching Lapp. Downloading those would put geometry into the corpus that does not
    # belong to the JSON whose ConnectionPoints are the ground truth -- silent GT poisoning.
MFG_NAME ={"PXC":"phoenix","WEI":"weidmueller","ABB":"abb","A-B":"allen bradley",
"SIE":"siemens","WAGO":"wago"}


def same_maker (our_mfg ,wscad_name ):
    """True only when WSCAD's manufacturer is the one our prefix means."""
    want =MFG_NAME .get (our_mfg )
    if want is None :
        return False # unknown prefix -> identity cannot be proven, so refuse
    return want in (wscad_name or "").strip ().lower ()


def todo_census (mfg ,limit ,census_path =CENSUS ):
    """Drive the download from the availability census instead of the guessed MFG_ID table.

    The census called /api/search per part and wrote back the manufacturerId the service itself
    returned for an EXACT part-number match. That fixes two things at before: parts whose maker is
    not in MFG_ID become downloadable, and parts that simply do not exist there (the SIE 8WH
    block: 10/10 404) are never attempted.

    Accepted only when part number AND manufacturer both match -- see MFG_NAME. Every accepted
    download is then checked geometrically by verify_downloads.py, which aligns the STEP mesh to
    the JSON point cloud and rejects files whose residual says "different part".
    """
    have ={os .path .basename (s ).split ("_")[1 ]for s in glob .glob (f"{OUT }/*.stp")}
    rows ,rejected =[],[]
    with open (census_path ,encoding ="utf-8-sig")as fh :
        for r in csv .DictReader (fh ):
            if r ["found_exact"]!="1"or r ["part"]in have :
                continue 
            if mfg and r ["our_mfg"]!=mfg :
                continue 
            if not same_maker (r ["our_mfg"],r ["wscad_mfg_name"]):
                rejected .append ((r ["our_mfg"],r ["wscad_mfg_name"]))
                continue 
            rows .append ({"parca_no":r ["part"],"manufacturer":r ["our_mfg"],
            "mid":int (r ["wscad_mfg_id"]),"wscad_mfg":r ["wscad_mfg_name"]})
    if rejected :
        import collections 
        c =collections .Counter (rejected )
        print (f"  [manufacturer uyusmuyor] {len (rejected )} part ATLANDI (numara ayni, brand baska): "
        +", ".join (f"{a }->{b } x{n }"for (a ,b ),n in c .most_common (6 )),flush =True )
    return rows [:limit ]if limit else rows 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--mfg",default ="WEI")
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--delay",type =float ,default =3.0 ,help ="seconds between parts -- do not lower")
    ap .add_argument ("--max-fail",type =int ,default =10 )
    ap .add_argument ("--list",default =LIST ,help ="hangi CSV listesinden indirilecek")
    ap .add_argument ("--from-census",action ="store_true",
    help =f"part+manufacturer-id'sini {CENSUS } makbuzundan al (API'nin dogruladigi id; "
    "orada olmayan parts never denenmez)")
    a =ap .parse_args ()
    from playwright .sync_api import sync_playwright 

    rows =todo_census (a .mfg ,a .limit )if a .from_census else todo (a .mfg ,a .limit ,a .list )
    if not rows :
        print ("indirilecek part absent (all of them diskte may be)");return 
    src ="census (API-dogrulanmis id)"if a .from_census else "missing listesi (MFG_ID tablosu)"
    print (f"{len (rows )} part indirilecek ({a .mfg or 'all of them'}) | kaynak: {src } | "
    f"zaten diskte olanlar atlaniyor",flush =True )
    os .makedirs (OUT ,exist_ok =True )
    ok =fail =nostep =0 ;streak =0 ;t0 =time .time ()

    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (PROFILE ,channel ="msedge",headless =True ,
        accept_downloads =True )
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .goto (HOME ,wait_until ="domcontentloaded",timeout =60000 )
        page .wait_for_timeout (5000 )

        for i ,r in enumerate (rows ,1 ):
            pid =r ["parca_no"];mid =int (r ["mid"])
            try :
                res =page .evaluate (JS ,{"mid":mid ,"pn":pid })
                if res .get ("err"):
                    print (f"  {res ['err']}");break 
                body =res .get ("body")or ""
                if res .get ("status")==200 and body .startswith ("ISO-10303-21"):
                    ts =datetime .now ().strftime ("%Y-%m-%d-%H-%M-%S")
                    dest =os .path .join (OUT ,f"wscaduniverse_{pid }_{ts }.stp")
                    open (dest ,"w",encoding ="utf-8",newline ="").write (body )
                    ok +=1 ;streak =0 
                    print (f"  [{i }/{len (rows )}] {pid } OK  {len (body )//1024 } KB",flush =True )
                elif res .get ("status")==404 :
                # 404 = "this parcanin STEP'i YOK", ariza DEGIL. Dogrulandi: body uretilecek
                # file adinin kendisi ("1492-H7_5_Step_IEC.stp"), and parcanin own
                # part.json'u formats:[0,1] diyor -- i.e. only WSCAD verisi + EPLAN .edz,
                # format 2 (STEP) never absent. Bunu ard-arda-error sayacina katmak kosuyu
                # more first manufacturer blogunda durduruyordu.
                    nostep +=1 ;streak =0 
                    print (f"  [{i }/{len (rows )}] {pid } STEP yok (404)",flush =True )
                else :
                    fail +=1 ;streak +=1 
                    print (f"  [{i }/{len (rows )}] {pid } basarisiz (status {res .get ('status')}, "
                    f"{res .get ('len')} byte)",flush =True )
            except Exception as e :
                fail +=1 ;streak +=1 
                print (f"  [{i }/{len (rows )}] {pid } HATA {str (e )[:70 ]}",flush =True )
            if streak >=a .max_fail :
                print (f"\n{streak } ard arda GERCEK error (404 disi) -> duruyorum",flush =True );break 
            time .sleep (a .delay )
        ctx .close ()
    print (f"\nbitti: {ok } indi, {fail } basarisiz, {time .time ()-t0 :.0f}s",flush =True )


if __name__ =="__main__":
    main ()
