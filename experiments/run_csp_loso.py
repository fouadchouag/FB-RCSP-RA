"""LOSO - CSP. EA in loader + bandpass 8-30Hz + SVM RBF C=4."""
import sys, os, csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mne.filter import filter_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import BCICIV2aLoader
from features.csp import CSPFeatureExtractor

def save(subjects, accs, name):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_loso.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Subject","Accuracy"])
        for s,a in zip(subjects,accs): w.writerow([s,f"{a:.6f}"])
        w.writerow([]); w.writerow(["Mean",f"{np.mean(accs):.6f}"]); w.writerow(["Std",f"{np.std(accs):.6f}"])

subjects = [f"A0{i}" for i in range(1,10)]
X_all,y_all,ids = [],[],[]
print("Loading...")
for i,s in enumerate(subjects):
    X,y = BCICIV2aLoader(f"data/bbci2a/{s}T.gdf").extract_trials()
    X = np.array([filter_data(t,250,8,30,verbose=False) for t in X])
    X_all.append(X); y_all.append(y); ids.extend([i]*len(y))
X_all=np.concatenate(X_all); y_all=np.concatenate(y_all); ids=np.array(ids)
accs=[]
for ts in range(len(subjects)):
    print(f"\nTesting {subjects[ts]}")
    tr=ids!=ts; te=ids==ts
    csp=CSPFeatureExtractor(n_components=6)
    Ftr=csp.fit_transform(X_all[tr],y_all[tr]); Fte=csp.transform(X_all[te])
    sc=StandardScaler(); Ftr=sc.fit_transform(Ftr); Fte=sc.transform(Fte)
    clf=SVC(kernel="rbf",C=4,gamma="scale",random_state=42)
    clf.fit(Ftr,y_all[tr])
    acc=accuracy_score(y_all[te],clf.predict(Fte)); print(f"  Acc:{acc:.4f}"); accs.append(acc)
save(subjects,accs,"csp")
print(f"\nMean:{np.mean(accs):.4f} Std:{np.std(accs):.4f}")
