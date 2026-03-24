"""CV5 - Riemann MDM. EA in loader + bandpass + z-score + pure MDM."""
import sys, os, csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from mne.filter import filter_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import BCICIV2aLoader
from features.riemann import RiemannMDMClassifier

def save(subject, method, accuracies):
    os.makedirs("results", exist_ok=True)
    path = f"results/{method}_cv5_metrics.csv"
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["Subject","Fold_Accuracies","Mean","Std"])
        w.writerow([subject,"|".join([f"{x:.6f}" for x in accuracies]),
                    f"{np.mean(accuracies):.6f}",f"{np.std(accuracies):.6f}"])

subjects = [f"A0{i}" for i in range(1,10)]
for subj in subjects:
    print(f"\n==== {subj} ====")
    X,y = BCICIV2aLoader(f"data/bbci2a/{subj}T.gdf").extract_trials()
    X = np.array([filter_data(t,250,8,30,verbose=False) for t in X])
    X = (X-X.mean(axis=2,keepdims=True))/(X.std(axis=2,keepdims=True)+1e-6)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for fold,(tr,te) in enumerate(skf.split(X,y)):
        mdm=RiemannMDMClassifier()
        mdm.fit(X[tr],y[tr])
        acc=accuracy_score(y[te],mdm.predict(X[te]))
        print(f"  Fold {fold+1}:{acc:.4f}"); accs.append(acc)
    save(subj,"mdm",accs)
    print(f"  Mean:{np.mean(accs):.4f} Std:{np.std(accs):.4f}")
