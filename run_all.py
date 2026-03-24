"""
Run all experiments — LOSO then CV5.
8 methods: CSP, ACMCSP, FBCSP, RCSP, MDM, TS-FBCSP, FB-ACMCSP, FB-RCSP-RA.
Configuration: EA in loader + bandpass 8-30Hz + SVM RBF C=4.
"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    # ── LOSO (primary results) ──────────────────────────
    "experiments/run_csp_loso.py",
    "experiments/run_acmcsp_loso.py",
    "experiments/run_rcsp_loso.py",
    "experiments/run_mdm_loso.py",
    "experiments/run_fbrcspra_loso.py",
    # ── CV5 (secondary results) ─────────────────────────
    "experiments/run_csp_all.py",
    "experiments/run_acmcsp_all.py",
    "experiments/run_rcsp_all.py",
    "experiments/run_mdm_all.py",
    "experiments/run_fbrcspra_all.py",
]

total = len(scripts)
for i, script in enumerate(scripts, 1):
    print(f"\n{'='*55}")
    print(f"[{i}/{total}] {script}")
    print('='*55)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERROR in {script} — stopping.")
        sys.exit(1)
    print(f"Done: {script}")

print("\n" + "="*55)
print("ALL EXPERIMENTS FINISHED — Check results/ folder")
print("="*55)
