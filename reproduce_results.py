#!/usr/bin/env python3
"""Recompute the reported results of the paper from the stored predictions.

    python3 reproduce_results.py

Reads predictions/combined_model_predictions.csv, which holds the per-image
predictions of the three backbones and of the weighted soft-voting ensemble on the
20,118-image test set, together with the ground truth and the laboratory/field tag.

Needs only numpy and pandas; every metric and test is implemented here so that the
numbers do not depend on a particular scikit-learn or scipy version.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, BOOT = 42, 1000
MODELS = [("ResNet50", "resnet_pred"), ("InceptionV3", "inception_pred"),
          ("EfficientNetB0", "efficient_pred"), ("Weighted ensemble", "voting_pred")]


# ---------- metrics (macro-averaged, multi-class MCC) ------------------------
def confusion(y, p, k):
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def metrics(y, p, k=120):
    """Macro-averaged precision/recall/F1 and the multi-class MCC.

    The macro average is taken over the classes that actually occur in this subset
    in the ground truth, i.e. sklearn's f1_score(..., labels=np.unique(y_true)). Averaging over all 120 classes instead would count absent
    classes as F1 = 0 and understate the field subset, where only 74 classes exist.
    """
    full = confusion(y, p, k)

    # Accuracy and MCC use every image and every class.
    n = full.sum()
    acc = np.diag(full).sum() / n
    predf, truef = full.sum(0).astype(float), full.sum(1).astype(float)
    num = np.diag(full).sum() * n - (predf * truef).sum()
    d1 = n * n - (predf * predf).sum()
    d2 = n * n - (truef * truef).sum()
    mcc = num / np.sqrt(d1 * d2) if d1 > 0 and d2 > 0 else 0.0

    # The macro averages are taken over the classes present in this subset's ground
    # truth, i.e. sklearn's f1_score(..., labels=np.unique(y_true)). On the field subset
    # only 74 of the 120 classes exist; averaging over all 120 would count the absent
    # ones as zero and understate the result by some 27 points.
    present = np.unique(y)
    m = full[np.ix_(present, present)]
    k = len(present)
    tp = np.diag(m).astype(float)
    pred, true = m.sum(0).astype(float), m.sum(1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros(k), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros(k), where=true > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros(k), where=den > 0)
    return dict(acc=100 * acc, precision=100 * prec.mean(),
                recall=100 * rec.mean(), f1=100 * f1.mean(), mcc=mcc)


def mcnemar(y, a, b):
    """Paired test with continuity correction: a wrong & b right vs the reverse."""
    n01 = int(((a != y) & (b == y)).sum())
    n10 = int(((a == y) & (b != y)).sum())
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    # survival function of chi-square with 1 df, via the error function
    from math import erfc, sqrt
    return chi2, erfc(sqrt(chi2 / 2))


def bootstrap(y, p, q=None, reps=BOOT, seed=SEED):
    """Resample images with replacement; returns mean/std of accuracy and, if a
    second prediction vector is given, the CI of the accuracy difference."""
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, diffs = np.empty(reps), np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        accs[i] = (p[idx] == y[idx]).mean()
        if q is not None:
            diffs[i] = (p[idx] == y[idx]).mean() - (q[idx] == y[idx]).mean()
    out = dict(mean=100 * accs.mean(), std=100 * accs.std(ddof=1))
    if q is not None:
        out["ci"] = (100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5))
    return out


# ---------- data -------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "predictions", "combined_model_predictions.csv"))
df = df[df["actual"].notna()].reset_index(drop=True)
y = df["actual"].astype(int).to_numpy()
P = {name: df[col].astype(int).to_numpy() for name, col in MODELS}
print(f"test images: {len(y)}   classes: {y.max() + 1}\n")

print("Table 2 - overall performance on the test set")
print(f"{'model':22s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'MCC':>8s}")
for name, _ in MODELS:
    m = metrics(y, P[name])
    print(f"{name:22s} {m['acc']:7.2f} {m['precision']:7.2f} {m['recall']:7.2f} "
          f"{m['f1']:7.2f} {m['mcc']:8.4f}")

if "field_image" in df.columns:
    tag = df["field_image"]
    fld = tag.astype(str).str.strip().str.upper().eq("Y")
    print(f"\nTable 4 - laboratory ({(~fld).sum()}) vs field ({fld.sum()}) subsets")
    print(f"{'model':22s} {'lab acc':>9s} {'lab F1':>8s} {'field acc':>11s} {'field F1':>9s}")
    for name, _ in MODELS:
        L, F = metrics(y[~fld], P[name][~fld]), metrics(y[fld], P[name][fld])
        print(f"{name:22s} {L['acc']:9.2f} {L['f1']:8.2f} {F['acc']:11.2f} {F['f1']:9.2f}")

print("\nTable 5 - ensemble vs each backbone (McNemar, and bootstrap CI of the gain)")
ens = P["Weighted ensemble"]
for name, _ in MODELS[:3]:
    chi2, p_val = mcnemar(y, P[name], ens)
    b = bootstrap(y, ens, P[name])
    lo, hi = b["ci"]
    print(f"  vs {name:16s} chi2={chi2:8.2f}  p={'<0.001' if p_val < 1e-3 else f'{p_val:.4f}'}"
          f"   gain={100 * ((ens == y).mean() - (P[name] == y).mean()):+.2f} pts  95% CI [{lo:.2f}, {hi:.2f}]")

b = bootstrap(y, ens)
print(f"\n  ensemble accuracy over {BOOT} bootstrap resamples: {b['mean']:.2f} +/- {b['std']:.2f}")

print("\nTable 7 - ten most frequent confusions of the ensemble")
m = confusion(y, ens, int(y.max()) + 1)
np.fill_diagonal(m, 0)
names = (df.drop_duplicates("actual").set_index(df.drop_duplicates("actual")["actual"].astype(int))
         ["actual_label"].to_dict()) if "actual_label" in df.columns else {}
order = np.dstack(np.unravel_index(np.argsort(m.ravel())[::-1], m.shape))[0][:10]
true_tot = confusion(y, ens, int(y.max()) + 1).sum(1)
for t, p in order:
    print(f"  {names.get(t, t)!s:34s} -> {names.get(p, p)!s:34s} {m[t, p]:4d}"
          f"  ({100 * m[t, p] / true_tot[t]:5.1f}% of that class)")
