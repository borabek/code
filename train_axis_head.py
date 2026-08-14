# -*- coding: utf-8 -*-
"""C ADIMI, 2/2: EKSEN AGI -- kose basina insert yonunu OGREN.

WHY SEPARATE BIR AG, mevcut kafaya 3 channel EKLEMEK not: segmentasyon modelinin last katmani
tum cikislara log_softmax uyguluyor (diffusionnet.build_diffusionnet). C_out'u 5'ten 8'e
cikarmak sinif olasiliklarini direction kanallariyla same softmax'a sokar and segmentasyonu breaks.
Ayrica 5 cikisli checkpoint'ler last layer sekli uyusmadigi for yuklenemezdi. Ayri a network
mevcut urune HIC dokunmaz and single basina olculebilir -- kaybederse silinir, kazanirsa eklenir.

KAYIP: maskeli, ISARETSIZ kosinus. 1 - |cos(prediction, hedef)|
  * maskeli: only manufacturer CP'sinin yakinindaki koseler katkida bulunur (see
    build_axis_dataset.py). CAD sozde-etiketlerine never dokunulmaz -- onlar insan GT'siyle
    however F1~0.46 ortusuyor, direction for guvenilmez.
  * unsigned: manufacturer sign konvansiyonu parcadan parcaya degisiyor; isareti already
    geometrik as (outward) belirliyoruz.

BOLME: URETICI-DISI (varsayilan). json_dataset.family_key with "aile-disi" bolmeye kalkistim and
OLCTUM: 1542 part -> 1542 aile, i.e. that fonksiyon part numarasinin KENDISINI donduruyor and
no gruplama yapmiyor. O split gercekte PARCA-DISI olurdu and sizintili olurdu: bugun
measured ki kardes varyantlarin geometrisi BIREBIR same (axis farki 0.000 derece, hizalama
residual 0.3mm) -- val'deki parcanin ikizi egitimde olabilirdi. Kardesler same ureticide
oldugu for URETICI-DISI split this sizintiyi tamamen keser and more hard a sinavdir.

TABAN CIZGILERI (853 part / 272291 hedef kose on MEASURED, egitimden ONCE):
    sabit +Z (most aptal prediction)          medyan 7.00d   >15d %43.2
    korpusun most sik yonu (= +Z)         medyan 7.00d   >15d %43.2
    parcanin KENDI baskin yonu (KOPYA)  medyan 0.00d   >15d %10.2   <- ULASILABILIR TAVAN
    mevcut urun (B-rep + geometri)                     >15d %15.8

Iki sey birden correct:
  * Gorev sanildigi up to hard DEGIL: parcalarin %89'u TEK YONLU (part basina medyan 1 different
    direction), i.e. is large olcude "this part hangi yonu kullaniyor" sorusuna iniyor.
  * Ama onemsiz de not: sabit +Z %43.2 error veriyor. 11 parcalik duman testinde agin 0.96
    dereceye inmesi, "each yere +Z head" not "parcanin geometrisinden yonunu cikar" demek.

KILL (olcumden ONCE yazildi):
  * val >15d orani **%15.8'in altina** inmezse arm DUSER (mevcut urunu gecmiyor demektir).
  * val >15d orani **%43.2'ye yakinsa** network no sey ogrenmemis demektir -- arm DUSER.
  * TAVAN %10.2: part-yonu tahmini KUSURSUZ olsa bile very-yonlu parts this up to error
    birakiyor. Yani C'nin kazanabilecegi most extra sey ~5.6 score; B-rep'in single basina getirdigi
    9.1 puandan (24.9 -> 15.8) KUCUK. Bu, kolu kosmadan before bilinmesi gereken a boundary.
"""
import os ,sys ,json ,glob ,time ,argparse 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")
DS ="results/axis_dataset"
OUTDIR ="results/axis_net"
INFLIGHT ="results/axis_net/inflight.txt"# this an islenen part (zehirli-part korumasi)
SKIPFILE ="results/axis_net/skip.txt"# kalici atlama listesi


def _load_skip ():
    """Bizi kilitleyen parcalari atla -- gate_regrow'da kanitlanmis desen.

    2026-07-30: training single a parcada 22 DAKIKA asili kaldi (op onbellegine new file
    yazilmadi, GPU %12). Ayni imza corpus betiklerinde de yasandi (spektral ayristirmada
    47 dakika donen WEI parcasi). Parca ISLENMEDEN ONCE adi INFLIGHT'a yazilir; kosu yeniden
    baslatildiginda orada duran part "bizi olduren part"dir.

    ILK kesintide kara listeye ALINMAZ: mekanizma "part asildi" with "sureci ben oldurdum"u
    ayirt edemez. Gercekten asilan part IKINCI denemede de asilir.
    """
    skip =set ()
    if os .path .exists (SKIPFILE ):
        skip |={x .strip ()for x in open (SKIPFILE )if x .strip ()}
    if os .path .exists (INFLIGHT ):
        raw =open (INFLIGHT ).read ().strip ().split ()
        stuck ,att =(raw [0 ]if raw else ""),(int (raw [1 ])if len (raw )>1 else 1 )
        if stuck :
            if att >=2 :
                skip .add (stuck )
                os .makedirs (os .path .dirname (SKIPFILE ),exist_ok =True )
                with open (SKIPFILE ,"a")as fh :
                    fh .write (stuck +chr (10 ))
                print (f"  [zehirli part] {stuck } IKI kez asti -> kalici atlama",flush =True )
            else :
                _RETRY [stuck ]=att +1 
                print (f"  [yeniden dene] {stuck } bir kez yarim kaldi -> {att +1 }. deneme",
                flush =True )
        os .remove (INFLIGHT )
    return skip 


_RETRY ={}


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--epochs",type =int ,default =40 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--accum",type =int ,default =8 ,
    help ="gradyan biriktirme (etkili batch). batch=1 salinimin caresi;",)
    ap .add_argument ("--lr-decay-every",type =int ,default =15 )
    ap .add_argument ("--lr-decay-rate",type =float ,default =0.5 )
    ap .add_argument ("--no-normalize",action ="store_true",
    help ="girdiyi merkezleme/olcekleme (tanida yararli)")
    ap .add_argument ("--c-width",type =int ,default =128 )
    ap .add_argument ("--blocks",type =int ,default =4 )
    ap .add_argument ("--k-eig",type =int ,default =96 )
    ap .add_argument ("--val-frac",type =float ,default =0.15 )
    ap .add_argument ("--split",choices =("mfg","part"),default ="mfg",
    help ="mfg = URETICI-disi (varsayilan, sizintisiz); part = part-disi (zayif)")
    ap .add_argument ("--val-mfg",default ="WEI",help ="mfg bolmesinde dogrulamaya ayrilan manufacturer")
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--seed",type =int ,default =20260730 )
    ap .add_argument ("--resume",action ="store_true",
    help ="last.pt varsa AYNI yerden devam et (model+optimizer+epoch)")
    a =ap .parse_args ()

    import torch 
    import diffusionnet as D ,thesis_remesh 
    from infer_step_cp import step_to_mesh 
    from json_dataset import family_key 

    os .makedirs (OUTDIR ,exist_ok =True )
    skip =_load_skip ()
    idx =json .load (open (os .path .join (DS ,"index.json")))
    parts =idx ["parts"]
    if skip :
        parts =[p for p in parts if p ["part"]not in skip ]
        print (f"  {len (skip )} zehirli part atlandi",flush =True )
    if a .limit :
        parts =parts [:a .limit ]
    rng =np .random .RandomState (a .seed )
    if a .split =="mfg":
        tr =[p for p in parts if p .get ("mfg")!=a .val_mfg ]
        va =[p for p in parts if p .get ("mfg")==a .val_mfg ]
        print (f"{len (parts )} part | URETICI-DISI: training {len (tr )} "
        f"(={sorted ({p .get ('mfg')for p in tr })}) / dogrulama {len (va )} (={a .val_mfg })",
        flush =True )
    else :
        keys =sorted ({family_key (p ["part"])for p in parts })
        rng .shuffle (keys )
        val_k =set (keys [:max (1 ,int (len (keys )*a .val_frac ))])
        tr =[p for p in parts if family_key (p ["part"])not in val_k ]
        va =[p for p in parts if family_key (p ["part"])in val_k ]
        print (f"{len (parts )} part | PARCA-DISI (ZAYIF, kardes sizintisi olabilir): "
        f"training {len (tr )} / dogrulama {len (va )}",flush =True )
    if not tr or not va :
        raise SystemExit ("split bos taraf uretti -- --val-mfg degerini kontrol et")

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg ={"input_features":"xyz","c_width":a .c_width ,
    "n_diffusion_blocks":a .blocks ,"n_eig":a .k_eig ,"dropout":0.0 ,"loss":"ce"}
    model ,meta =D .build_diffusionnet (cfg ,n_classes =3 )# 3 channel = direction vektoru
    model =model .to (dev )
    opt =torch .optim .Adam (model .parameters (),lr =a .lr )
    opcache =f"results/step_infer/ops_k{a .k_eig }"

    MESH_CACHE ="results/mesh_cache"
    os .makedirs (MESH_CACHE ,exist_ok =True )

    def load (p ):
        """Mesh DISKTEN onbelleklenir. Profillendi (6 part): STEP okuma gmsh with 0.638s =
        surenin %83'u, GPU isi whereas only 0.109s. Mesh epoch'lar arasi DEGISMIYOR, i.e. each
        epoch'ta yeniden uretmek saf israfti: 29 dk/epoch -> ~2 dk/epoch (15 fold).
        """
        os .makedirs (OUTDIR ,exist_ok =True )
        with open (INFLIGHT ,"w")as fh :# ISLENMEDEN ONCE isaretle
            fh .write (f"{p ['part']} {_RETRY .get (p ['part'],1 )}")
        z =np .load (os .path .join (DS ,p ["part"]+".npz"),allow_pickle =True )
        mc =os .path .join (MESH_CACHE ,p ["part"]+".npz")
        if os .path .exists (mc ):
            m =np .load (mc )
            V =np .ascontiguousarray (m ["V"],np .float64 )
            F =np .ascontiguousarray (m ["F"],np .int64 )
        else :
            Vr ,Fr =step_to_mesh (p ["stp"])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =idx ["remesh_target"])
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            try :
                np .savez (mc ,V =V .astype (np .float32 ),F =F .astype (np .int32 ))
            except Exception :
                pass 
        if len (V )!=int (z ["n_verts"]):
            return None # remesh degismis -> hedef indeksleri gecersiz
        out =(V ,F ,z ["idx"].astype (np .int64 ),z ["tgt"].astype (np .float32 ))
        if os .path .exists (INFLIGHT ):
            os .remove (INFLIGHT )# basariyla bitti -> isareti kaldir
        return out 

    def forward (V ,F ):
        """Girdi MERKEZLENIR and OLCEKLENIR.

        diffusionnet._model_input("xyz") HAM mm koordinati donduruyor -- merkezleme absent,
        olcekleme absent. Parcalar 40-250mm arasi and orijinleri keyfi. Yon whereas konumdan and
        olcekten BAGIMSIZ a buyukluk; ham koordinat vermek aga "mutlak konumu gormezden
        gel"i also ogretmek demek. Egitim kaybinin 0.10'da (mean ~26 derece)
        duzlesmesinin most olasi sebebi this.
        """
        """Turevlenebilir ileri gecis. D.predict() no_grad inside kosuyor and argmax donduruyor,
        i.e. training for kullanilamaz -- operatorleri same sekilde hazirlayip _forward'i
        dogrudan cagiriyoruz (same op onbellegi, same input kurulumu)."""
        ops =D .precompute_operators (V ,F ,meta ["k_eig"],opcache )
        ops ={k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}
        x =D ._model_input (ops ,meta )
        if (not a .no_normalize )and meta ["input_features"]=="xyz":
            c =0.5 *(x .max (0 ).values +x .min (0 ).values )
            sc =float ((x .max (0 ).values -x .min (0 ).values ).norm ())or 1.0 
            x =(x -c )/sc 
        return D ._forward (model ,ops ,x )

    def angle_deg (pred ,tgt ):
        pn =pred /(np .linalg .norm (pred ,axis =1 ,keepdims =True )+1e-9 )
        c =np .abs ((pn *tgt ).sum (1 )).clip (0 ,1 )
        return np .degrees (np .arccos (c ))

        # DEVAM ETME. Yalniz "most iyi" agirligi saklamak DEVAM for yetmez: Adam'in momentleri,
        # epoch sayaci and most iyi skor da is required. Onlarsiz yeniden baslatma "sicak restart" becomes --
        # first epochlar geriye gider and sifirlanan "best" kotu a epochun uzerine yazmasina izin
        # gives. Bugun process three times became; this yetenek olmadan each seferinde bastan baslanirdi.
    LAST =os .path .join (OUTDIR ,"last.pt")
    best ,start_ep =1e9 ,1 
    if a .resume and os .path .exists (LAST ):
        ck =torch .load (LAST ,map_location =dev ,weights_only =False )
        model .load_state_dict (ck ["model"]);opt .load_state_dict (ck ["opt"])
        best =float (ck .get ("best",1e9 ));start_ep =int (ck .get ("epoch",0 ))+1 
        print (f"  [devam] epoch {start_ep }'den, en iyi val medyan {best :.2f}d",flush =True )
    for ep in range (start_ep ,a .epochs +1 ):
        model .train ();tot =0.0 ;nb =0 ;t0 =time .time ()
        opt .zero_grad ()
        for g in opt .param_groups :# LR AZALTMA (seg egiticisindeki desen)
            g ["lr"]=a .lr *(a .lr_decay_rate **((ep -1 )//a .lr_decay_every ))
        order =np .random .permutation (len (tr ))
        for i in order :
            d =load (tr [i ])
            if d is None :
                continue 
            V ,F ,ii ,tt =d 
            out =forward (V ,F )
            pred =out [torch .as_tensor (ii ,device =dev )]
            tg =torch .as_tensor (tt ,device =dev )
            pn =pred /(pred .norm (dim =1 ,keepdim =True )+1e-9 )
            loss =(1.0 -(pn *tg ).sum (1 ).abs ()).mean ()# ISARETSIZ kosinus
            # GRADYAN BIRIKTIRME: batch=1 with val 10.7-23.7 derece arasi zipliyordu.
            # Segmentasyon egiticisi this depoda already accum kullaniyor (~row 530).
            (loss /a .accum ).backward ()
            if (nb +1 )%a .accum ==0 :
                opt .step ();opt .zero_grad ()
            tot +=float (loss );nb +=1 
        model .eval ();angs =[]
        with torch .no_grad ():
            for p in va :
                d =load (p )
                if d is None :
                    continue 
                V ,F ,ii ,tt =d 
                out =forward (V ,F ).detach ().cpu ().numpy ()
                angs .append (angle_deg (out [ii ],tt ))
        A =np .concatenate (angs )if angs else np .array ([90.0 ])
        med =float (np .median (A ));bad =float ((A >15 ).mean ())
        print (f"epoch {ep :>3}  loss {tot /max (nb ,1 ):.4f}  val medyan aci {med :5.2f}d  "
        f">15d %{bad *100 :5.1f}  {time .time ()-t0 :.0f}s",flush =True )
        torch .save ({"model":model .state_dict (),"opt":opt .state_dict (),"epoch":ep ,
        "best":min (best ,med ),"config":cfg ,"meta":meta },LAST )
        if med <best :
            best =med 
            torch .save ({"model":model .state_dict (),"config":cfg ,"meta":meta ,
            "val_median_deg":med ,"val_bad15":bad },
            os .path .join (OUTDIR ,"axis_net_best.pt"))
    print (f"\nEN IYI val medyan aci: {best :.2f}d -> {OUTDIR }/axis_net_best.pt")
    print ("KILL kontrolu: mevcut urun ayni olcutte >15d sapmayi %16.0'ya indirmisti; "
    "bu ag onu gecmiyorsa arm duser.")


if __name__ =="__main__":
    main ()
