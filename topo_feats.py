# -*- coding: utf-8 -*-
"""ICBUKEY KENAR TOPOLOJISI -- hole-tanima alaninin birinci sinyali.

FIKIR: a opening/cep, ICBUKEY kenarlarla cevrili face kumesidir; disa cikinti whereas DISBUKEY
kenarlarla. Gate'in 18 ozelliginin HICBIRI topolojik not (all of them probability istatistigi ya da
point-geometrisi). Bu, bugun +0.0704 getiren "gate'e sahip olmadigi bilgiyi ver" hamlesinin
acilmamis ikinci kanadi.

WHY MESH, WHY B-rep DEGIL: mesh each parcada present; B-rep renk/eslesme yollarinda gordugumuz
kapsama kaybi here YOK. Ikiyuzlu (dihedral) angle mesh'ten dogrudan is computed.

ICBUKEYLIK TESTI: kenari paylasan two ucgenden birinin duzlemi and OTEKI ucgenin karsi kosesi.
Karsi kose duzlemin ARKASINDA whereas disbukey, ONUNDE whereas icbukey (disa bakan normal kabulu).

DORT OZELLIK (all of them candidate noktasi p and axis d etrafinda, radius R):
  kon_oran  : R icindeki kenarlarin kaci icbukey
  kon_sayi  : R icindeki icbukey edge count (log scale)
  kon_aci   : icbukey kenarlarin mean bukulme acisi (derece)
  kon_cevre : ASIL SINYAL -- icbukey kenarlar axis d etrafinda TAM TUR sariyor mu?
              (12 acisal kovadan kaci full). Bir mouth "icbukey a cemberle" cevrilidir;
              duz a yuzeydeki noise ya da single a oluk cevrelemez.
"""
import numpy as np 

R_VARSAYILAN =6.0 
N_KOVA =12 


def _kenar_yuz (F ):
    """edge -> ona komsu face indeksleri (only TAM 2 komsulu kenarlar)."""
    import collections 
    m =collections .defaultdict (list )
    for fi ,(a ,b ,c )in enumerate (F ):
        for u ,v in ((a ,b ),(b ,c ),(c ,a )):
            m [(u ,v )if u <v else (v ,u )].append (fi )
    return {k :v for k ,v in m .items ()if len (v )==2 }


def icbukey_kenarlar (V ,F ):
    """Doner: (M,2) edge kose indeksleri, (M,) icbukey mi, (M,) bukulme acisi derece, (M,3) middle point."""
    V =np .asarray (V ,float );F =np .asarray (F ,int )
    n =np .cross (V [F [:,1 ]]-V [F [:,0 ]],V [F [:,2 ]]-V [F [:,0 ]])
    ln =np .linalg .norm (n ,axis =1 ,keepdims =True )
    n =n /np .maximum (ln ,1e-12 )
    km =_kenar_yuz (F )
    E =np .array (list (km .keys ()),int )
    if not len (E ):
        return E ,np .zeros (0 ,bool ),np .zeros (0 ),np .zeros ((0 ,3 ))
    FF =np .array ([km [tuple (e )]for e in E ],int )
    n1 ,n2 =n [FF [:,0 ]],n [FF [:,1 ]]
    mid =0.5 *(V [E [:,0 ]]+V [E [:,1 ]])
    # ikinci ucgenin KARSI kosesi (kenarda olmayan)
    karsi =np .empty (len (E ),int )
    for i ,(e ,ff )in enumerate (zip (E ,FF )):
        t =F [ff [1 ]]
        karsi [i ]=[x for x in t if x not in (e [0 ],e [1 ])][0 ]
        # disa bakan normal kabulu: karsi kose n1 duzleminin ONUNDE whereas ICBUKEY (vadi)
    d =((V [karsi ]-mid )*n1 ).sum (1 )
    ic =d >1e-9 
    cosang =np .clip ((n1 *n2 ).sum (1 ),-1.0 ,1.0 )
    aci =np .degrees (np .arccos (cosang ))
    return E ,ic ,aci ,mid 


def topo_ozellik (V ,F ,p ,dvec ,R =R_VARSAYILAN ,cache =None ):
    """Bir candidate for 4 topolojik feature. `cache` = icbukey_kenarlar ciktisi (part basina a times)."""
    E ,ic ,aci ,mid =cache if cache is not None else icbukey_kenarlar (V ,F )
    out =np .zeros (4 ,float )
    if not len (E ):
        return out 
    p =np .asarray (p ,float )
    d =np .asarray (dvec ,float )
    nd =np .linalg .norm (d )
    d =d /nd if nd >1e-9 else np .array ([0.0 ,0.0 ,1.0 ])
    yakin =np .linalg .norm (mid -p ,axis =1 )<=R 
    if not yakin .any ():
        return out 
    icy =ic &yakin 
    out [0 ]=float (icy .sum ())/float (max (yakin .sum (),1 ))# kon_oran
    out [1 ]=float (np .log1p (icy .sum ()))# kon_sayi (log)
    out [2 ]=float (aci [icy ].mean ())if icy .any ()else 0.0 # kon_aci
    if icy .any ():
        rel =mid [icy ]-p 
        al =rel @d 
        perp =rel -al [:,None ]*d 
        # eksene dik duzlemde acisal kapsama
        e1 =np .array ([1.0 ,0.0 ,0.0 ])
        if abs (d @e1 )>0.9 :
            e1 =np .array ([0.0 ,1.0 ,0.0 ])
        e1 =e1 -(e1 @d )*d ;e1 /=np .linalg .norm (e1 )+1e-12 
        e2 =np .cross (d ,e1 )
        th =np .arctan2 (perp @e2 ,perp @e1 )
        bucket =((th +np .pi )/(2 *np .pi )*N_KOVA ).astype (int )%N_KOVA 
        out [3 ]=float (len (set (bucket .tolist ())))/N_KOVA # kon_cevre
    return out 


ISIMLER =["kon_oran","kon_sayi","kon_aci","kon_cevre"]
