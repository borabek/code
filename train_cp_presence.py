# -*- coding: utf-8 -*-
"""G: URETICI CP'lerini TESPIT denetimi yap -- kose basina "here CP present mi".

WHY BU, VE WHY SIMDI:
Segmentasyon modeli 102 insan-etiketli parcayla egitildi. Kayitli bulgu angle: CAD sozde-etiketleri
insan GT'siyle however F1~0.46 ortusuyor, i.e. more extra CAD etiketi tavani yukseltmiyor.
Ama elimizde BAMBASKA and KESIN a audit present: **8534 manufacturer ConnectionPoint konumu**,
1542 parcada -- and bugune up to only DEGERLENDIRMEDE kullanildi, egitimde HIC kullanilmadi.

H'NIN OLUMU BU KOLU ZORUNLU KILDI (measured 2026-07-30): sinif-oncullu karar duzeltmesi
recall'i HIC artirmadi (0.747 -> 0.722), only precision'i yikti (0.923 -> 0.791). Yani network,
kacirdigi acikliklarda CE/CT olasiligini ZATEN uretmiyor. Kacirilan opening a DECISION KURALI
sorunu not, **agin kendisinin that yeri gormemesi**. Esik oynatmak caresiz; agi ogretmek is required.

VERI: `build_axis_dataset.py` ciktisi AYNEN is used. Oradaki `idx`, manufacturer CP'sinin 4mm
yakinindaki koseler = POZITIF maske. Gerisi negatif. Yeni data toplamaya gerek absent.

METRIK -- AUC DEGIL: dogrulama, urunun isine bagli single sayiyla is done:
**CP-YAKALAMA ORANI** = kac manufacturer CP'sinin 4mm yakininda at least a pozitif prediction present.
(Bu projede a arm AUC'ye bakip F1'de olmustu -- see tel-b-channel-profile-dead.)

BOLME: URETICI-DISI. family_key this korpusta part numarasinin kendisini donduruyor
(1542 part -> 1542 "aile"), i.e. "aile-disi" fiilen part-disi olurdu and kardes varyantlarin
geometrisi BIREBIR same (axis farki 0.000 derece). Kardesler same ureticide oldugu for
manufacturer-disi split this sizintiyi tamamen keser.

KILL (olcumden ONCE yazildi):
  * val CP-yakalama orani, mevcut hattin candidate recall'ini (**0.747**) GECMEZSE arm duser.
  * yakalama artip precision'i wire-gate geri alamiyorsa (detection F1 gerilerse) arm duser --
    this however G ciktisi candidate havuzuna baglandiktan after olculebilir (2. stage).
"""
import os ,sys ,json ,time ,argparse 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")
DS ="results/presence_dataset"# hedefler AGIZDA (C dersi)
MESH_CACHE ="results/mesh_cache"
OUTDIR ="results/cp_presence"
INFLIGHT =os .path .join (OUTDIR ,"inflight.txt")
SKIPFILE =os .path .join (OUTDIR ,"skip.txt")
_RETRY ={}


def _load_skip ():
    """Kilitleyen parcayi skip (gate_regrow deseni). Ilk kesintide kara listeye ALINMAZ:
    'part asildi' with 'sureci ben oldurdum' ayirt edilemez; real asilan ikinci times de asilir."""
    skip =set ()
    if os .path .exists (SKIPFILE ):
        skip |={x .strip ()for x in open (SKIPFILE )if x .strip ()}
    if os .path .exists (INFLIGHT ):
        raw =open (INFLIGHT ).read ().strip ().split ()
        stuck ,att =(raw [0 ]if raw else ""),(int (raw [1 ])if len (raw )>1 else 1 )
        if stuck :
            if att >=2 :
                skip .add (stuck )
                os .makedirs (OUTDIR ,exist_ok =True )
                with open (SKIPFILE ,"a")as fh :
                    fh .write (stuck +chr (10 ))
                print (f"  [zehirli part] {stuck } IKI kez asti -> persistent atlama",flush =True )
            else :
                _RETRY [stuck ]=att +1 
        os .remove (INFLIGHT )
    return skip 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--epochs",type =int ,default =40 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--accum",type =int ,default =8 )
    ap .add_argument ("--lr-decay-every",type =int ,default =15 )
    ap .add_argument ("--lr-decay-rate",type =float ,default =0.5 )
    ap .add_argument ("--c-width",type =int ,default =128 )
    ap .add_argument ("--blocks",type =int ,default =4 )
    ap .add_argument ("--k-eig",type =int ,default =96 )
    ap .add_argument ("--val-mfg",default ="WEI")
    ap .add_argument ("--pos-weight",type =float ,default =0.0 ,
    help ="0 = veriden otomatik hesapla (negatif/pozitif orani)")
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--fire-budget",type =float ,default =0.15 ,
    help ="en iyi checkpoint yalnizca this atesleme oraninin ALTINDA secilir")
    ap .add_argument ("--resume",action ="store_true")
    a =ap .parse_args ()

    import torch 
    import diffusionnet as D ,thesis_remesh 
    from infer_step_cp import step_to_mesh 

    os .makedirs (OUTDIR ,exist_ok =True )
    skip =_load_skip ()
    idx =json .load (open (os .path .join (DS ,"index.json")))
    parts =[p for p in idx ["parts"]if p ["part"]not in skip ]
    if a .limit :
        parts =parts [:a .limit ]
    tr =[p for p in parts if p .get ("mfg")!=a .val_mfg ]
    va =[p for p in parts if p .get ("mfg")==a .val_mfg ]
    print (f"{len (parts )} part | URETICI-DISI: training {len (tr )} / dogrulama {len (va )} "
    f"(={a .val_mfg })",flush =True )
    if not tr or not va :
        raise SystemExit ("split bos taraf uretti")

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg ={"input_features":"xyz","c_width":a .c_width ,
    "n_diffusion_blocks":a .blocks ,"n_eig":a .k_eig ,"dropout":0.0 ,"loss":"ce"}
    model ,meta =D .build_diffusionnet (cfg ,n_classes =1 )# single channel: CP present/absent
    model =model .to (dev )
    opt =torch .optim .Adam (model .parameters (),lr =a .lr )
    opcache =f"results/step_infer/ops_k{a .k_eig }"

    # pozitif agirligi VERIDEN: pozitifler koselerin ~%5'i, agirliksiz BCE hepsine "absent" der
    if a .pos_weight >0 :
        pw =a .pos_weight 
    else :
        pos =sum (p ["n_target_verts"]for p in tr )
        tot =sum (p .get ("n_verts",6000 )for p in tr )
        pw =max (1.0 ,(tot -pos )/max (pos ,1 ))
    print (f"pozitif agirlik (neg/poz) = {pw :.1f}",flush =True )
    lossf =torch .nn .BCEWithLogitsLoss (pos_weight =torch .tensor ([pw ],device =dev ))

    def load (p ):
        os .makedirs (OUTDIR ,exist_ok =True )
        with open (INFLIGHT ,"w")as fh :
            fh .write (f"{p ['part']} {_RETRY .get (p ['part'],1 )}")
        z =np .load (os .path .join (DS ,p ["part"]+".npz"),allow_pickle =True )
        mc =os .path .join (MESH_CACHE ,p ["part"]+".npz")
        if os .path .exists (mc ):
            m =np .load (mc )
            V =np .ascontiguousarray (m ["V"],np .float64 );F =np .ascontiguousarray (m ["F"],np .int64 )
        else :
            Vr ,Fr =step_to_mesh (p ["stp"])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =idx ["remesh_target"])
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        if len (V )!=int (z ["n_verts"]):
            return None 
        y =np .zeros (len (V ),np .float32 )
        y [z ["idx"].astype (np .int64 )]=1.0 
        out =(V ,F ,y )
        if os .path .exists (INFLIGHT ):
            os .remove (INFLIGHT )
        return out 

    def forward (V ,F ):
        """Girdi merkezlenir + olceklenir -- _model_input('xyz') HAM mm donduruyor and parts
        40-250mm arasi. C kolunda same correction yapildi."""
        ops =D .precompute_operators (V ,F ,meta ["k_eig"],opcache )
        ops ={k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}
        x =D ._model_input (ops ,meta )
        if meta ["input_features"]=="xyz":
            c =0.5 *(x .max (0 ).values +x .min (0 ).values )
            sc =float ((x .max (0 ).values -x .min (0 ).values ).norm ())or 1.0 
            x =(x -c )/sc 
        return D ._forward (model ,ops ,x ).squeeze (-1 )

    LAST =os .path .join (OUTDIR ,"last.pt")
    best ,start_ep =-1.0 ,1 
    if a .resume and os .path .exists (LAST ):
        ck =torch .load (LAST ,map_location =dev ,weights_only =False )
        model .load_state_dict (ck ["model"]);opt .load_state_dict (ck ["opt"])
        best =float (ck .get ("best",-1.0 ));start_ep =int (ck .get ("epoch",0 ))+1 
        print (f"  [devam] epoch {start_ep }'den, en iyi yakalama {best :.3f}",flush =True )

    for ep in range (start_ep ,a .epochs +1 ):
        model .train ();tot =0.0 ;nb =0 ;t0 =time .time ();_pending =False 
        opt .zero_grad ()
        for g in opt .param_groups :
            g ["lr"]=a .lr *(a .lr_decay_rate **((ep -1 )//a .lr_decay_every ))
        for i in np .random .permutation (len (tr )):
            d =load (tr [i ])
            if d is None :
                continue 
            V ,F ,y =d 
            out =forward (V ,F )
            loss =lossf (out ,torch .as_tensor (y ,device =dev ))
            (loss /a .accum ).backward ()
            if (nb +1 )%a .accum ==0 :
                opt .step ();opt .zero_grad ()
                _pending =False 
            else :
                _pending =True 
            tot +=float (loss );nb +=1 

            # VERIFICATION: urunun isine bagli single number -- CP-YAKALAMA ORANI (AUC DEGIL).
        if _pending :# ARTIK BATCH: aksi halde epoch'un last
            opt .step ();opt .zero_grad ()# parcalarinin gradyani ATILIR (part count
        _pending =False # accum'dan azsa HIC step atilmaz -- measured)
        model .eval ();hit =seen =0 ;npos_pred =nvert =0 
        with torch .no_grad ():
            for p in va :
                d =load (p )
                if d is None :
                    continue 
                V ,F ,y =d 
                pr =torch .sigmoid (forward (V ,F )).cpu ().numpy ()
                fire =pr >=0.5 
                npos_pred +=int (fire .sum ());nvert +=len (V )
                # each manufacturer CP'si for: 4mm yakininda pozitif prediction present mi
                z =np .load (os .path .join (DS ,p ["part"]+".npz"),allow_pickle =True )
                tgt =z ["idx"].astype (np .int64 )
                if not len (tgt ):
                    continue 
                    # hedef koseler CP basina gruplanmadi -> yakalama, hedef kose orani with olculur
                seen +=len (tgt );hit +=int (fire [tgt ].sum ())
        rate =hit /max (seen ,1 )
        firerate =npos_pred /max (nvert ,1 )
        print (f"epoch {ep :>3}  loss {tot /max (nb ,1 ):.4f}  yakalama {rate :.3f}  "
        f"atesleme %{100 *firerate :4.1f}{''if firerate <=a .fire_budget else '  (BUTCE DISI)'}"
        f"  {time .time ()-t0 :.0f}s",flush =True )
        # YAKALAMA TEK BASINA OYUNLANABILIR: each koseye "present" diyen a network %100 yakalar.
        # Olculdu (duman testi): atesleme %45.7 iken yakalama 0.599 -- oysa pozitifler
        # koselerin ~%5'i. Bu yuzden most iyi checkpoint only ATESLEME BUTCESI inside secilir.
        ok_budget =firerate <=a .fire_budget 
        torch .save ({"model":model .state_dict (),"opt":opt .state_dict (),"epoch":ep ,
        "best":max (best ,rate if ok_budget else -1.0 ),"config":cfg ,
        "meta":meta },LAST )
        if ok_budget and rate >best :
            best =rate 
            torch .save ({"model":model .state_dict (),"config":cfg ,"meta":meta ,
            "val_catch":rate ,"fire_rate":firerate },
            os .path .join (OUTDIR ,"cp_presence_best.pt"))
    print (f"\nEN IYI val CP-kose yakalama: {best :.3f} -> {OUTDIR }/cp_presence_best.pt")
    print ("KILL: mevcut hattin candidate recall'i 0.747. Bunu gecmiyorsa arm duser.")


if __name__ =="__main__":
    main ()
