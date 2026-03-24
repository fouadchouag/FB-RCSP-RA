"""CV5 - RCSP Ledoit-Wolf. EA in loader + bandpass + z-score + SVM C=4."""
import sys, os, csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mne.filter import filter_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import BCICIV2aLoader
from features.rcsp import RCSPFeatureExtractor

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
        rcsp=RCSPFeatureExtractor(n_components=6,reg='ledoit_wolf')
        Ftr=rcsp.fit_transform(X[tr],y[tr]); Fte=rcsp.transform(X[te])
        sc=StandardScaler(); Ftr=sc.fit_transform(Ftr); Fte=sc.transform(Fte)
        clf=SVC(kernel="rbf",C=4,gamma="scale",random_state=42)
        clf.fit(Ftr,y[tr])
        acc=accuracy_score(y[te],clf.predict(Fte))
        print(f"  Fold {fold+1}:{acc:.4f}"); accs.append(acc)
    save(subj,"rcsp",accs)
    print(f"  Mean:{np.mean(accs):.4f} Std:{np.std(accs):.4f}")
