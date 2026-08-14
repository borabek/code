# -*- coding: utf-8 -*-
"""FB-2b (2/2): yardimci KAFA + KAYIP.

Ana 5-sinif yolu does not change: `out = D._forward(...)` and `loss = D._compute_loss(...)`
aynen kalir. Yardimci kafa `last_lin` ONCESI hidden temsilden (128-d) beslenir --
body paylasilir, cikis ayridir.
"""
import io 

P ="train_seg_extra.py"
N ="\r\n"
s =io .open (P ,encoding ="utf-8",newline ="").read ()


def yama (old ,new ,s ):
    assert old in s ,"KALIP YOK: "+repr (old [:80 ])
    return s .replace (old ,new ,1 )


    # --- 4) modelden SONRA yardimci kafayi kur and optimize edilecekler listesine fold
o4 ="    opt = torch.optim.Adam(model.parameters(), lr=a.lr)"
n4 =(
'    # FB-2: YARDIMCI KAFA. Govde paylasilir (last_lin ONCESI 128-d hidden temsil),'+N 
+'    # cikis ayridir. Ana 5-sinif kafasi and kaybi HIC DEGISMEZ.'+N 
+'    aux_lin = None; _gizli = {}'+N 
+'    if a.aux_wire:'+N 
+'        _w = int(cfg.get("width", 64))'+N 
+'        aux_lin = torch.nn.Linear(_w, 1).to(dev)'+N 
+'        def _kanca(mod, giren, cikan):'+N 
+'            _gizli["z"] = giren[0]'+N 
+'        model.last_lin.register_forward_hook(_kanca)'+N 
+'        n_aux = sum(1 for _d in tr_d if _d.get("aux") is not None)'+N 
+'        print(f"  [aux-wire] {n_aux}/{len(tr_d)} parcada TEL/ALET etiketi present "'+N 
+'              f"| weight {a.aux_w} | pos_w {a.aux_pos_weight}", flush=True)'+N 
+'    _par = list(model.parameters()) + (list(aux_lin.parameters()) if aux_lin else [])'+N 
+"    opt = torch.optim.Adam(_par, lr=a.lr)")
s =yama (o4 ,n4 ,s )

# --- 5) kayba yardimci terimi EKLE (ana loss hesaplandiktan SONRA)
o5 ="            loss.backward(); opt.step(); tot += float(loss)"
n5 =(
'            # FB-2 YARDIMCI KAYIP: ana loss above hesaplandi and DEGISMEDI;'+N 
+'            # here only USTUNE ekleniyor. Maskeli: -1 which is vertices sinyal vermez.'+N 
+'            if aux_lin is not None and d.get("aux") is not None and "z" in _gizli:'+N 
+'                ya = d["aux"].to(dev)'+N 
+'                mk = ya >= 0'+N 
+'                if bool(mk.any()):'+N 
+'                    lg = aux_lin(_gizli["z"]).squeeze(-1)'+N 
+'                    pw = torch.tensor(float(a.aux_pos_weight), device=dev)'+N 
+'                    la = torch.nn.functional.binary_cross_entropy_with_logits('+N 
+'                        lg[mk], ya[mk].float(), pos_weight=pw)'+N 
+'                    loss = loss + float(a.aux_w) * la'+N 
+"            loss.backward(); opt.step(); tot += float(loss)")
s =yama (o5 ,n5 ,s )

# --- 6) checkpoint'e yardimci kafayi da yaz
o6 ='torch.save({"state": model.state_dict(), "cfg": cfg, "meta": meta,'
n6 ='torch.save({"state": model.state_dict(), "cfg": cfg, "meta": meta,\r\n                            "aux_state": (aux_lin.state_dict() if aux_lin is not None else None),'
s =yama (o6 ,n6 ,s )

io .open (P ,"w",encoding ="utf-8",newline ="").write (s )
print ("4-6 yamalandi: yardimci kafa + loss + checkpoint")
