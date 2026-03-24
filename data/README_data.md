# BCI Competition IV Dataset 2a — Download Instructions

The BCI Competition IV Dataset 2a is publicly available but must be
downloaded separately from this repository due to licensing restrictions.

---

## Download Steps

**1.** Visit the official competition website:
```
https://www.bbci.de/competition/iv/
```

**2.** Navigate to **"Data Sets"** → **"Dataset 2a"**

**3.** Download all **training session files** (9 subjects):
```
A01T.gdf    A02T.gdf    A03T.gdf
A04T.gdf    A05T.gdf    A06T.gdf
A07T.gdf    A08T.gdf    A09T.gdf
```

**4.** Place all `.gdf` files in the `data/bbci2a/` folder:

```
FB-RCSP-RA/
└── data/
    └── bbci2a/
        ├── A01T.gdf
        ├── A02T.gdf
        ├── A03T.gdf
        ├── A04T.gdf
        ├── A05T.gdf
        ├── A06T.gdf
        ├── A07T.gdf
        ├── A08T.gdf
        └── A09T.gdf
```

> **Note:** Only training session files (`A0xT.gdf`) are used.
> Evaluation files (`A0xE.gdf`) do not contain class labels and are not needed.

---

## Dataset Description

| Parameter | Value |
|-----------|-------|
| Subjects | 9 healthy subjects (A01–A09) |
| MI classes | 4: left hand (769), right hand (770), feet (771), tongue (772) |
| EEG channels | 22 Ag/AgCl electrodes (10-20 system) + 3 EOG channels |
| Sampling rate | 250 Hz |
| Hardware filter | 0.5–100 Hz bandpass + 50 Hz notch |
| Trials per subject | 288 (6 runs × 48 trials, 12 per class) |
| Epoch extraction | tmin = 0.5 s to tmax = 4.5 s relative to cue onset |
| Epoch length | 4.0 s (1001 samples at 250 Hz) |

---

## Trial Timing Structure

```
t = 0.0 s  — Fixation cross appears + acoustic beep
t = 2.0 s  — Visual cue appears (event marker: 769/770/771/772)
t = 2.0 s  — Motor imagery execution begins
t = 2.5 s  — ★ Epoch start  (tmin = +0.5 s relative to cue)
t = 3.25 s — Visual cue disappears (shown for 1.25 s)
t = 6.0 s  — Motor imagery execution ends
t = 6.5 s  — ★ Epoch end    (tmax = +4.5 s relative to cue)
t = 7.5 s  — End of trial, short break
```

The epoch window `[tmin=0.5, tmax=4.5]` captures the full 4-second MI
execution while avoiding the early visual-evoked potential (VEP) at cue onset.

---

## Preprocessing Applied in This Study

The `loader.py` script applies the following preprocessing steps:

```python
# Step 1 — Channel selection (exclude EOG)
raw.pick(list(range(22)))           # keep exactly 22 EEG channels by index

# Step 2 — Unit conversion
raw.apply_function(lambda x: x * 1e6)   # Volts → microvolts

# Step 3 — Euclidean Alignment (EA) per subject
# R_mean = mean covariance across all trials
# W = R_mean^(-1/2)  (whitening matrix)
# X_aligned = W @ X  for each trial
```

EA is applied **before** any frequency-band filtering and uses **no
information from the test subject** — it is computed independently
for each subject from their own trials only.

---

## Verification

After placing the files correctly, verify the setup:

```bash
python data/loader.py --subject A01 --verify
```

Expected output:
```
Subject A01: 288 trials, 22 channels, 1001 samples
Classes: {769: 72, 770: 72, 771: 72, 772: 72}
EA applied: covariance trace ratio before/after = 1.00 (normalized)
OK
```

---

## Reference

If you use this dataset, please cite the original source:

```
Brunner, C., Leeb, R., Müller-Putz, G.R., Schlögl, A., & Pfurtscheller, G. (2008).
BCI Competition 2008 — Graz data set A.
Institute for Knowledge Discovery, Graz University of Technology.
https://www.bbci.de/competition/iv/
```

---

## License

The BCI Competition IV Dataset 2a is provided by Graz University of Technology
under the terms specified at https://www.bbci.de/competition/iv/.
Please read and comply with their data usage terms before downloading.
