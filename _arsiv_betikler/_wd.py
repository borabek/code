# SADECE saglik + disk-acil. EPOCH DURDURMA YOK. Liveness = log buyume (glitch-dayanikli).
import time, os, json, datetime, shutil
LOG="train_run_hp_v31_ftc6.log"
def now(): return datetime.datetime.now().strftime("%H:%M")
def wlog(m):
    with open("_health6h.log","a",encoding="utf-8") as f: f.write(f"{now()} {m}\n")
wlog("=== nobetci: LOG-ONLY + disk-acil, liveness=log-buyume (epoch durdurma YOK) ===")
stale=0
last=os.path.getsize(LOG) if os.path.exists(LOG) else 0
while True:
    time.sleep(300)
    cur=os.path.getsize(LOG) if os.path.exists(LOG) else 0
    grew = cur>last; last=cur
    stale = 0 if grew else stale+1
    d=shutil.disk_usage("c:/").free/1e9
    ep="?"
    try:
        h=json.load(open("checkpoints/cp_hp_v31_ftc6_history.json",encoding="utf-8"))
        r=[x for x in (h if isinstance(h,list) else h.get("history",[])) if isinstance(x,dict) and "val_micro_f1" in x]
        if r: ep=f"ep{r[-1]['epoch']} F1={r[-1]['val_micro_f1']:.4f}"
    except: pass
    wlog(f"disk={d:.1f}G  {ep}  {'(log akiyor)' if grew else f'(log durdu x{stale})'}")
    if stale>=3:      # 15 dk log yok -> egitim bitmis/olmus
        wlog("=== log 15dk durdu -> egitim yok, nobetci cikiyor ==="); break
    if d<1.2:
        wlog(f"=== ACIL disk {d:.1f}G -> cokme onleme ===")
        import subprocess; subprocess.run(["powershell","-NoProfile","-Command","Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_cp' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],timeout=30); break
