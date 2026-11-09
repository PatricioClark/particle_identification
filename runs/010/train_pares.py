"""Test on the paired RND512 runs: are HPP and ONLD with the same Stokes told apart?

18 classes (HPP and ONLD at St 0.1, 0.2, 0.35, 0.6, 1, 1.4, 2, 4, 20), all in the same flow. Samples are
time-coherent (5000 particles of one window, as in 009). For each block length (6 and 4 T), feature set
(all, or without the two acceleration features, which are normalised with large scales U/T although the
acceleration follows the Kolmogorov ones) and training scheme:

  - Leave one block out: train on the other blocks, test on samples of the held-out block, a moment
    of the flow the classifier never saw (with moments built with --disjoint, also particles it never
    saw: test samples come from the separate test pool).
  - Without neighbours: the same, but the blocks next to the test block are also left out of training,
    because the energy level of the flow takes ~10 T to decorrelate and neighbouring blocks look alike.

Reported: exact class, model right (HPP/ONLD, whatever the St), St right (whatever the model), per St
how often each model is recognised, and per test block the local tau_eta = sqrt(nu/eps) with the factor
real St / label St = 0.09416 / tau_eta (the labels use tau = St * 0.09416).

Another realisation of the flow: train on all the blocks and predict the species with the same label in
the original RND512 runs (bespanol) and in RND512_nuevas (seed 2207), windows of the same length from the
009 moments, with their own real St / label St factor. This is Patricio's question: does it recognise the
particle model or the simulation?

Usage
-----
python train_pares.py --moments DIR [--moments-009 DIR --moments-nueva DIR]
"""

import argparse
from collections import Counter

import numpy as np

from train_schemes import coherent_samples, load, pipe

BLOCKS = ["6", "4"]
FEATURE_SETS = ["todas", "sin aceleración"]   # acc_variance and acc_flatness are the last two features
SCHEMES = {"resto de los bloques": 0, "sin bloques vecinos": 1}   # blocks within this distance left out
NU = 7e-4                # kinematic viscosity of the three RND512 runs
TAU_ETA_LABEL = 0.09416  # tau_eta of the St labels of 512


def model_of(lab):
    return lab.split("_")[0]


def st_of(lab):
    return lab.split("_")[1][2:]


def st_factor(scales):
    """Real St / label St of each window: 0.09416 / tau_eta, with tau_eta = sqrt(nu / eps)."""
    return np.array([TAU_ETA_LABEL / np.sqrt(NU / s["eps"]) for s in scales])


def samples(M, windows, n, args, rng, n_lags, fs):
    X = coherent_samples(M, windows, n, args.batch, rng, n_lags)
    return X[:, :-2] if fs == "sin aceleración" else X


def leave_one_block_out(runs, w, fs, gap, args, n_lags):
    rng = np.random.default_rng(args.seed)
    y_true, y_pred, blk = [], [], []
    all_blocks = sorted(set(runs[0]["windows"][w]["block"]))
    for b in all_blocks:
        if not any(abs(k - b) > gap for k in all_blocks):
            continue  # no block left to train on
        Xtr, ytr, Xte, yte = [], [], [], []
        for r in runs:
            W = r["windows"][w]
            tr, te = np.where(np.abs(W["block"] - b) > gap)[0], np.where(W["block"] == b)[0]
            Xtr.append(samples(W["M"], tr, args.n_train, args, rng, n_lags, fs))
            ytr += [r["label"]] * len(Xtr[-1])
            Xte.append(samples(W.get("M_test", W["M"]), te, args.n_test * len(te), args, rng, n_lags, fs))
            yte += [r["label"]] * len(Xte[-1])
        y_pred += list(pipe(args.seed).fit(np.vstack(Xtr), ytr).predict(np.vstack(Xte)))
        y_true += yte
        blk += [b] * len(yte)
    return np.array(y_true), np.array(y_pred), np.array(blk)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--moments", required=True)
    p.add_argument("--moments-009", default="/share/data4/msvidal/pid009/moments")
    p.add_argument("--moments-nueva", default="/share/data4/msvidal/pid009/moments_nueva")
    p.add_argument("--n-train", type=int, default=100, help="Training samples per species")
    p.add_argument("--n-test", type=int, default=20, help="Test samples per species and test window")
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runs = load(args.moments)
    classes = sorted({r["label"] for r in runs}, key=lambda l: (float(st_of(l)), model_of(l)))
    others = [r for r in load(args.moments_009) if r["label"] in classes] + \
             [dict(r, label=r["expected"]) for r in load(args.moments_nueva) if r["expected"] in classes]
    pools = "grupos de partículas distintos para entrenar y probar" if runs[0].get("disjoint") else "mismas partículas"
    print(f"{len(runs)} especies de las corridas pareadas, salidas {runs[0]['files'][0]}-{runs[0]['files'][1]}, "
          f"pool {runs[0]['n_pool']} ({pools}), muestras de {args.batch} partículas\n")

    summary = []
    for w in BLOCKS:
        n_lags = len(runs[0]["windows"][w]["names"]) // 4 - 1
        W0 = runs[0]["windows"][w]
        fac = st_factor(W0["scales"])
        print(f"████ bloques de {w} T: {len(W0['block'])} bloques; St real / St etiqueta por bloque: "
              + ", ".join(f"{f:.2f}" for f in fac) + "\n")
        for fs in FEATURE_SETS:
            for scheme, gap in SCHEMES.items():
                y_true, y_pred, blk = leave_one_block_out(runs, w, fs, gap, args, n_lags)
                if len(y_true) == 0:
                    print(f"══ {w} T | features: {fs} | entrenamiento: {scheme}: no quedan bloques para entrenar ══\n")
                    continue
                ok = y_true == y_pred
                m_ok = np.array([model_of(a) == model_of(b) for a, b in zip(y_true, y_pred)])
                s_ok = np.array([st_of(a) == st_of(b) for a, b in zip(y_true, y_pred)])
                print(f"══ {w} T | features: {fs} | entrenamiento: {scheme} ══")
                print(f"   clase exacta {ok.mean():.3f} | modelo (HPP/ONLD) bien {m_ok.mean():.3f} | "
                      f"St bien {s_ok.mean():.3f}   (azar: clase 0.056, modelo 0.5)")
                print(f"   {'St':>5s} {'HPP: modelo bien':>17s} {'ONLD: modelo bien':>18s}   predicción más frecuente")
                for st in sorted({st_of(c) for c in classes}, key=float):
                    cells, tops = [], []
                    for m in ("HPP", "ONLD"):
                        sel = y_true == f"{m}_St{st}_RND512"
                        if sel.any():
                            cells.append(f"{m_ok[sel].mean():.2f}")
                            top = Counter(y_pred[sel]).most_common(2)
                            tops.append(f"{m} → " + ", ".join(f"{k} {v / sel.sum():.2f}" for k, v in top))
                        else:
                            cells.append("—")
                    print(f"   {st:>5s} {cells[0]:>17s} {cells[1]:>18s}   " + " | ".join(tops))
                print("   por bloque de prueba (St real / etiqueta, clase exacta, modelo bien): " + "; ".join(
                    f"{b}: {fac[b]:.2f}, {ok[blk == b].mean():.2f}, {m_ok[blk == b].mean():.2f}"
                    for b in sorted(set(blk))) + "\n")
                summary.append((w, fs, scheme, ok.mean(), m_ok.mean(), s_ok.mean()))

            rng = np.random.default_rng(args.seed)
            Xall, yall = [], []
            for r in runs:
                W = r["windows"][w]
                Xall.append(samples(W["M"], np.arange(len(W["block"])), args.n_train, args, rng, n_lags, fs))
                yall += [r["label"]] * len(Xall[-1])
            model = pipe(args.seed).fit(np.vstack(Xall), yall)
            print(f"   otra realización del flujo ({w} T, features: {fs}; entrenado con todos los bloques; "
                  f"St real / etiqueta en las pares: {fac.mean():.2f}):")
            for r in others:
                W = r["windows"][w]
                pn = model.predict(samples(W["M"], np.arange(len(W["block"])), 100, args, rng, n_lags, fs))
                top = ", ".join(f"{k} {v / len(pn):.2f}" for k, v in Counter(pn).most_common(2))
                m_frac = np.mean([model_of(x) == model_of(r["label"]) for x in pn])
                print(f"      {r['name']:15s} St real/etiqueta {st_factor(W['scales']).mean():.2f}  "
                      f"clase {np.mean(pn == r['label']):4.2f}  modelo {m_frac:4.2f}  → {top}")
            print()

    print("── Resumen (dejando un bloque afuera) ──")
    print(f"{'bloque':>6s}  {'features':16s} {'entrenamiento':22s} {'clase':>6s} {'modelo':>7s} {'St':>6s}")
    for w, fs, scheme, a, am, ast in summary:
        print(f"{w + ' T':>6s}  {fs:16s} {scheme:22s} {a:6.3f} {am:7.3f} {ast:6.3f}")


if __name__ == "__main__":
    main()
