# -*- coding: utf-8 -*-
"""gate_regrow devam mantigi: corpus BUYUDUGUNDE satirlar correct parcaya bagli kalmali.

WHY BU TEST VAR: devam kaydi eskiden DONGU INDEKSI with tutuluyordu. eligible() diske new
STEP dustukce buyur and each enumerate indeksi kayar; old indekslerle devam etmek wrong
parcalari skips and ngt'yi (part basina GT count) kaydirir. Bu SESSIZ a bozulmadir:
regime ayrimi (low/very-CP) and recall paydasi yanlislasir, i.e. metrigin kendisi bozulur,
but no error mesaji cikmaz.
"""
import os 
import numpy as np 
import pytest 

import gate_regrow 


def _parts (pids ):
    """eligible() bicimi: (mfg, pid, json_yolu, step_yolu)."""
    return [("PXC",p ,f"{p }.json",f"{p }.stp")for p in pids ]


def _write_partial (path ,rows ,ngt_of ):
    """rows: (pid, aday_sayisi) -- old kosuda islenmis parts, that sirayla indekslenmis."""
    pid_col ,grp_col =[],[]
    for k ,(pid ,n )in enumerate (rows ,1 ):
        pid_col +=[pid ]*n 
        grp_col +=[k ]*n 
    m =len (pid_col )
    gid =sorted ({g for g in grp_col })
    np .savez (path ,
    X =np .arange (m *2 ,dtype =float ).reshape (m ,2 ),
    votes =np .ones (m ),y =np .zeros (m ),
    groups =np .array (grp_col ),mfg =np .zeros (m ),fams =np .array (["f"]*m ),
    pts =np .zeros ((m ,3 )),dirs =np .zeros ((m ,3 )),pids =np .array (pid_col ),
    grp_ids =np .array (gid ),
    ngt =np .array ([ngt_of [rows [g -1 ][0 ]]for g in gid ]))


def test_resume_remaps_indices_when_corpus_grows (tmp_path ):
    """Eski run A,B,C gormus. Yeni korpusta basa X eklenmis -> indeksler kaymis."""
    part =str (tmp_path /"partial.npz")
    _write_partial (part ,[("A",2 ),("B",1 ),("C",3 )],{"A":5 ,"B":9 ,"C":2 })

    # new corpus: X basa input, D sona -> A/B/C residual 2/3/4. indeksinde
    parts =_parts (["X","A","B","C","D"])
    (Xs ,votes ,tp ,grp ,mfgs ,fams ,pts ,dirs ,pids ,ngt ,done ,gone )=gate_regrow .resume_partial (parts ,part )

    assert gone ==0 
    # each row still KENDI parcasina bagli
    assert [(g ,p )for g ,p in zip (grp ,pids )]==[
    (2 ,"A"),(2 ,"A"),(3 ,"B"),(4 ,"C"),(4 ,"C"),(4 ,"C")]
    # ngt new indekslere tasindi and DEGERLER karismadi
    assert ngt =={2 :5 ,3 :9 ,4 :2 }
    # atlanacaklar = new indeksler; X (1) and D (5) islenmemis kalmali
    assert done =={2 ,3 ,4 }
    assert 1 not in done and 5 not in done 


def test_resume_drops_parts_that_left_the_corpus (tmp_path ):
    """Korpustan produced parcanin satirlari atilmali, rest bozulmamali."""
    part =str (tmp_path /"partial.npz")
    _write_partial (part ,[("A",2 ),("B",3 ),("C",1 )],{"A":4 ,"B":7 ,"C":1 })

    parts =_parts (["A","C"])# B residual absent
    (Xs ,votes ,tp ,grp ,mfgs ,fams ,pts ,dirs ,pids ,ngt ,done ,gone )=gate_regrow .resume_partial (parts ,part )

    assert gone ==3 # B'nin 3 adayi atildi
    assert pids ==["A","A","C"]
    assert grp ==[1 ,1 ,2 ]
    assert ngt =={1 :4 ,2 :1 }
    assert Xs [0 ].shape [0 ]==3 and len (votes [0 ])==3 and len (tp [0 ])==3 
    assert pts [0 ].shape [0 ]==3 and dirs [0 ].shape [0 ]==3 


def test_resume_rejects_old_format_without_part_numbers (tmp_path ):
    """Parca numarasi olmayan old PARTIAL guvenle esleneMEZ -> sessizce devam etme, DUR."""
    part =str (tmp_path /"old.npz")
    np .savez (part ,X =np .zeros ((2 ,2 )),votes =np .ones (2 ),y =np .zeros (2 ),
    groups =np .array ([1 ,1 ]),mfg =np .zeros (2 ),fams =np .array (["f","f"]),
    grp_ids =np .array ([1 ]),ngt =np .array ([3 ]))
    with pytest .raises (SystemExit ):
        gate_regrow .resume_partial (_parts (["A"]),part )


def test_index_based_resume_would_have_been_wrong (tmp_path ):
    """Duzeltilen hatayi ACIKCA gosterir: old rule wrong parcalari atlardi.

    Eski kod done = set(groups) yapip loop indeksiyle karsilastiriyordu. Korpus basa a
    part alinca same indeksler ARTIK BASKA parcalari sign eder.
    """
    part =str (tmp_path /"partial.npz")
    _write_partial (part ,[("A",1 ),("B",1 ),("C",1 )],{"A":1 ,"B":1 ,"C":1 })
    parts =_parts (["X","A","B","C","D"])

    eski_done ={1 ,2 ,3 }# old rule: ham gruplar
    eski_atlanan ={p [1 ]for k ,p in enumerate (parts ,1 )if k in eski_done }
    assert eski_atlanan =={"X","A","B"}# X never islenmemisti -> ATLANIRDI
    assert "C"not in eski_atlanan # C islenmisti -> IKI KEZ islenirdi

    *_ ,done ,_ =gate_regrow .resume_partial (parts ,part )
    dogru_atlanan ={p [1 ]for k ,p in enumerate (parts ,1 )if k in done }
    assert dogru_atlanan =={"A","B","C"}
