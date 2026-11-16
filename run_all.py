"""Execute all numbered notebooks in order with papermill (outputs saved in place).
Usage:  python run_all.py            (from the project root)"""
import os, sys, time
from pathlib import Path
import papermill as pm

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
order = sorted(p for p in NB.glob("0*.ipynb"))
for p in order:
    t0 = time.time()
    print(f"==> {p.name}", flush=True)
    pm.execute_notebook(str(p), str(p), cwd=str(NB), kernel_name=os.environ.get("NB_KERNEL", "mlproj2"), progress_bar=False, log_output=False)
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
print("all notebooks executed")
