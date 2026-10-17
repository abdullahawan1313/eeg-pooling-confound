# 00_check_env.py
"""
Step 00: checks the Python packages, PyTorch/CUDA and the data paths. No training (a few seconds).
Run: python 00_check_env.py
"""
import sys, platform, importlib, time

print("=" * 60)
print(" STEP 0: ENVIRONMENT CHECK")
print("=" * 60)
print("python   :", sys.version.split()[0], "|", platform.system(), platform.release())
if sys.version_info[:2] not in [(3, 11), (3, 12)]:
    print("  [WARN] tested with Python 3.11 and 3.12.")

ok = True
for mod in ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "mne",
            "mat73", "xlrd", "openpyxl", "fooof", "tqdm", "torch"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:<11} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  [MISS] {mod:<11} -> pip install {'scikit-learn' if mod=='sklearn' else mod}   ({e.__class__.__name__})")

try:
    import torch
    print("torch threads:", torch.get_num_threads(), "| CUDA:", torch.cuda.is_available())
    t = time.time()
    x = torch.randn(256, 10, 512)
    conv = torch.nn.Conv1d(10, 64, 7, padding=3)
    for _ in range(20):
        conv(x).sum().backward()
    print(f"CPU speed test: {time.time()-t:.2f} s  (reference: < 3 s is fine)")
except Exception:
    pass

# ---------- paths ----------
from rev.paths import DATA_ROOT, OLD_PROJECT, DATASET_DIRS, META
print("\nDATA_ROOT:", DATA_ROOT)
checks = {f"data:{k}": v for k, v in DATASET_DIRS.items()}
checks.update({f"meta:{k}": v for k, v in META.items()})
for k, v in checks.items():
    flag = "OK  " if v.exists() else "MISSING"
    if not v.exists():
        ok = False
    print(f"  [{flag}] {k:<22} {v}")
print(f"  [{'OK  ' if OLD_PROJECT.exists() else 'n/a '}] {'earlier analysis':<22} {OLD_PROJECT}"
      "   (optional, only for step 01)")

print("\n" + ("All checks passed." if ok else "Something is MISSING (see above); fix it and run again."))
