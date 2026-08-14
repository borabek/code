# -*- coding: utf-8 -*-
"""D5-3 ONKOSULU: YASAK GRUPLARI YENI ANAHTAR UZAYINA TASI (capraz leakage kontrolu).

SORUN (2026-08-04'te difference edildi, silent leakage):
`protocol.egitim_maskesi` gruplari `_strict_geometry_keys.json`'dan reads. O dosyadaki
anahtarlar ESKI ureticiden (`[8.1, 50.4, 71.8]|v12|f13|c19|p161|h...`). Yeni korpusun
anahtarlari whereas `geometri_anahtar.anahtar()`'dan geliyor and bicimi TAMAMEN FARKLI
(`b16.101.144|d31|h5.10.20...`).

Iki anahtar uzayi **never string as eslesmez**. `geometri_anahtar.dogrula()` new
uretecin old GRUPLAMAYI yeniden urettigini kanitlasa bile this yetmez: gruplama denkligi
!= anahtar esitligi. Yani OLCUM ya da LOCKED parcasinin IKIZI which is new a part, two
ayri uzayda durdugu for `egitim_maskesi`'nden YAKALANMADAN gecer and egitime girer.
Sonuc: headline SESSIZCE SISER -- full da [[geometry-twin-leakage]]'in ogrettigi error.

COZUM (ucuz which is): tum korpusu yeniden anahtarlamak instead of YALNIZ YASAK gruplarin
uyelerini (610 part) new uretecle anahtarla. Yeni parcanin anahtari this kumeye
dusuyorsa training disi birakilir. Eski parcalarin mevcut mekanizmasi AYNEN korunur.

Cikti: results/_geo_yasak.json  ->  {pid: yeni_anahtar}  (+ anahtar kumesi D5-3'te is used)
"""
import io 
import json 
import os 
import sys 
import time 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/_geo_yasak.json"


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import measure_set as OK 
    from big_arbiter import eligible 
    from geometri_anahtar import key_ 
    from infer_step_cp import step_to_mesh 

    gk =OK .geo_anahtarlari ()
    yasak_grup =set (OK .locked_gruplari ())
    D ,_ =OK .cluster ("results/_der_tam.pkl")
    yasak_grup |={r ["geo"]for r in D }
    hedef_pid ={p for p ,g in gk .items ()if g in yasak_grup }
    path ={p :s for _ ,p ,_ ,s in eligible ()if p in hedef_pid }
    print (f"YASAK grup {len (yasak_grup )} | uye part {len (hedef_pid )} | "
    f"STEP'i found {len (path )}")

    OUT ={}
    if os .path .exists (CIKTI ):
        OUT =json .load (io .open (CIKTI ,encoding ="utf-8"))
        print (f"devam: {len (OUT )} already anahtarlanmis")
    remaining =sorted (p for p in path if p not in OUT )
    t0 =time .time ();error =0 
    for k ,p in enumerate (remaining ,1 ):
        if k %25 ==0 :
            hiz =(time .time ()-t0 )/k 
            print (f"  {k }/{len (remaining )}  ({hiz :.1f}s/part, remaining ~{hiz *(len (remaining )-k )/60 :.0f} dk, "
            f"error {error })",flush =True )
            with io .open (CIKTI ,"w",encoding ="utf-8")as f :
                json .dump (OUT ,f )
        try :
            V ,F =step_to_mesh (path [p ])
            OUT [p ]=key_ (V ,F )
        except Exception as e :
            error +=1 
            if error <=5 :
                print (f"    {p }: {type (e ).__name__ }: {str (e )[:60 ]}",flush =True )
    with io .open (CIKTI ,"w",encoding ="utf-8")as f :
        json .dump (OUT ,f )

        # SAGLAMLIK KONTROLU: new anahtar, ESKI yasak gruplamayi koruyor mu?
        # Ayni old grupta which is two part new anahtarda da same must be; degilse new
        # uretec this bolgede AYIRIYOR demektir and koruma missing kalir -- gorunur olsun.
    ters ={}
    for p ,a in OUT .items ():
        ters .setdefault (gk .get (p ,"?"),set ()).add (a )
    bolunen ={g :len (s )for g ,s in ters .items ()if len (s )>1 }
    print (f"\n{len (OUT )} part anahtarlandi | error {error } | {time .time ()-t0 :.0f}s")
    print (f"  ESKI grup -> YENI anahtar: {len (ters )} gruptan {len (bolunen )}'i BOLUNDU")
    if bolunen :
        print (f"    (bolunen gruplar korumayi zayiflatir; en very bolunen "
        f"{sorted (bolunen .values (),reverse =True )[:5 ]})")
    print (f"  TEKIL YASAK ANAHTAR: {len (set (OUT .values ()))}  -> {CIKTI }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
