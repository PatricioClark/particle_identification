"""Which Stokes number does the classifier see? Nominal label, real (tau / tau_eta) or large-scale (tau / T).

In 010, trained on the paired runs, the species of the other 512 runs came out with the right model but a
lower Stokes, in the direction of their lower real Stokes. Here the question is turned into a regression:
the same features (5000-particle samples of one 4 T window, normalised with the L/U scales of the window,
moments of 010 and 009) predict log10 of the Stokes number of the particles, with four definitions of it:

  nominal  : the label (tau = St * 0.09416), the same in every window;
  real     : tau / tau_eta of the window, tau_eta = sqrt(nu / eps);
  real_run : tau / tau_eta averaged over all the windows of the run (one value per run);
  T        : tau / T of the window, T = L / U (the time unit of the features).

For the ONLD the drag factor <c> (slip_onld.py) is not included: tau is the nominal one for both models.
One regressor per model (HPP, ONLD), as if the model were known (the classifier gets it right in most
cases): an ONLD responds like an HPP of lower Stokes, so a single regressor would drag its predictions.

  - Paired runs, leave one 4 T block out without its neighbours, test particles separate from training
    ones: error of the prediction (log10 predicted / true) per definition.
  - Trained on all the blocks of the paired runs, tested on the species with the same label of the
    bespanol runs and of the new run (seed 2207): median predicted / true per species and definition.
    The definition the features really encode is the one for which this ratio stays close to 1.

Usage
-----
python stokes_real.py [--moments DIR --moments-009 DIR --moments-nueva DIR]
"""

import argparse
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "../010")
from train_schemes import features, load  # noqa: E402

NU, TAU_ETA_LABEL = 7e-4, 0.09416
DEFS = ["nominal", "real", "real_run", "T"]
W = "4"


def st_label(lab):
    return float(lab.split("_")[1][2:])


def window_st(lab, scales):
    """Stokes of each window under each definition: {def: (n_windows,)}."""
    tau = st_label(lab) * TAU_ETA_LABEL
    teta = np.array([np.sqrt(NU / s["eps"]) for s in scales])
    T = np.array([s["T"] for s in scales])
    return {"nominal": np.full(len(scales), st_label(lab)), "real": tau / teta,
            "real_run": np.full(len(scales), tau / teta.mean()), "T": tau / T}


def samples(M, windows, per_window, batch, rng, n_lags):
    """per_window samples of batch particles from each window; returns features and the window of each."""
    out, wid = [], []
    for w in windows:
        for _ in range(per_window):
            sel = rng.choice(M.shape[1], size=batch, replace=False)
            out.append(M[w, sel].astype(np.float64).mean(0))
            wid.append(w)
    return features(np.array(out), n_lags), np.array(wid)


def regressor(seed):
    return Pipeline([("scaler", StandardScaler()),
                     ("rf", RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=seed,
                                                  n_jobs=-1))])


def model_of(lab):
    return lab.split("_")[0]


def fit_predict(Xtr, ytr, mtr, Xte, mte, seed):
    """One regressor per model; each test sample predicted with the regressor of its (true) model."""
    pred = np.empty(len(Xte))
    for m in set(mte):
        reg = regressor(seed).fit(Xtr[mtr == m], ytr[mtr == m])
        pred[mte == m] = reg.predict(Xte[mte == m])
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--moments", default="/share/data4/msvidal/pid010/moments_3926")
    p.add_argument("--moments-009", default="/share/data4/msvidal/pid009/moments")
    p.add_argument("--moments-nueva", default="/share/data4/msvidal/pid009/moments_nueva")
    p.add_argument("--per-window", type=int, default=20, help="Training samples per species and window")
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runs = load(args.moments)
    labels = {r["label"] for r in runs}
    others = [dict(r, run="bespanol") for r in load(args.moments_009) if r["label"] in labels] + \
             [dict(r, label=r["expected"], run="nueva") for r in load(args.moments_nueva) if r["expected"] in labels]
    n_lags = len(runs[0]["windows"][W]["names"]) // 4 - 1
    rng = np.random.default_rng(args.seed)

    fac = {d: np.mean([window_st(r["label"], r["windows"][W]["scales"])[d] / st_label(r["label"]) for r in runs])
           for d in DEFS}
    print("Stokes / etiqueta promedio en las pareadas: " + ", ".join(f"{d} {fac[d]:.2f}" for d in DEFS))
    for name in ("bespanol", "nueva"):
        rr = [r for r in others if r["run"] == name]
        print(f"Stokes / etiqueta promedio en {name}: " + ", ".join(
            f"{d} {np.mean([window_st(r['label'], r['windows'][W]['scales'])[d] / st_label(r['label']) for r in rr]):.2f}"
            for d in DEFS))
    print()

    # 1) Paired runs, leave one block out without neighbours
    blocks = sorted(set(runs[0]["windows"][W]["block"]))
    err = {d: [] for d in DEFS}
    for b in blocks:
        tr = [k for k in blocks if abs(k - b) > 1]
        Xtr, ytr, Xte, yte = [], {d: [] for d in DEFS}, [], {d: [] for d in DEFS}
        mtr, mte = [], []
        for r in runs:
            Wd = r["windows"][W]
            st = window_st(r["label"], Wd["scales"])
            X, wid = samples(Wd["M"], tr, args.per_window, args.batch, rng, n_lags)
            Xtr.append(X)
            mtr += [model_of(r["label"])] * len(X)
            X, wte = samples(Wd["M_test"], [b], args.per_window, args.batch, rng, n_lags)
            Xte.append(X)
            mte += [model_of(r["label"])] * len(X)
            for d in DEFS:
                ytr[d].append(np.log10(st[d][wid]))
                yte[d].append(np.log10(st[d][wte]))
        Xtr, Xte, mtr, mte = np.vstack(Xtr), np.vstack(Xte), np.array(mtr), np.array(mte)
        for d in DEFS:
            pred = fit_predict(Xtr, np.concatenate(ytr[d]), mtr, Xte, mte, args.seed)
            err[d].append(pred - np.concatenate(yte[d]))
    print("Pareadas, dejando un bloque afuera sin vecinos: error de log10(St predicho / St verdadero)")
    for d in DEFS:
        e = np.concatenate(err[d])
        print(f"   {d:8s} sesgo {e.mean():+.3f}  error típico {np.sqrt((e ** 2).mean()):.3f}  "
              f"(factor típico {10 ** np.sqrt((e ** 2).mean()):.2f})")
    print()

    # 2) Train on all the blocks, test on the other runs
    Xall, yall, mall = [], {d: [] for d in DEFS}, []
    for r in runs:
        Wd = r["windows"][W]
        st = window_st(r["label"], Wd["scales"])
        X, wid = samples(Wd["M"], blocks, args.per_window, args.batch, rng, n_lags)
        Xall.append(X)
        mall += [model_of(r["label"])] * len(X)
        for d in DEFS:
            yall[d].append(np.log10(st[d][wid]))
    Xall, mall = np.vstack(Xall), np.array(mall)
    models = {(d, m): regressor(args.seed).fit(Xall[mall == m], np.concatenate(yall[d])[mall == m])
              for d in DEFS for m in ("HPP", "ONLD")}

    print("Otras corridas (entrenado con todas las pareadas): mediana de St predicho / St verdadero")
    print(f"   {'corrida':9s} {'especie':10s} " + " ".join(f"{d:>9s}" for d in DEFS)
          + "    (St verdadero: " + ", ".join(DEFS) + ")")
    ratios = {(name, d): [] for name in ("bespanol", "nueva") for d in DEFS}
    for r in sorted(others, key=lambda r: (r["run"], r["label"].split("_")[0], st_label(r["label"]))):
        Wd = r["windows"][W]
        st = window_st(r["label"], Wd["scales"])
        X, wid = samples(Wd["M"], range(len(Wd["block"])), 3, args.batch, rng, n_lags)
        row, true = [], []
        for d in DEFS:
            ratio = 10 ** np.median(models[(d, model_of(r["label"]))].predict(X) - np.log10(st[d][wid]))
            if st_label(r["label"]) > 0.1:   # at St 0.1 the features barely change with St (near tracers)
                ratios[(r["run"], d)].append(np.log10(ratio))
            row.append(f"{ratio:9.2f}")
            true.append(f"{np.median(st[d]):.2f}")
        sp = r["label"].split("_")[0] + " " + r["label"].split("_")[1][2:]
        print(f"   {r['run']:9s} {sp:10s} " + " ".join(row) + "    (" + ", ".join(true) + ")")
    print("   promedio (geométrico) por corrida, sin las de Stokes 0.1:")
    for name in ("bespanol", "nueva"):
        print(f"   {name:9s} {'':10s} " + " ".join(f"{10 ** np.mean(ratios[(name, d)]):9.2f}" for d in DEFS))


if __name__ == "__main__":
    main()
