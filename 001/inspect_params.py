"""Print dt, pstep, and xlg file count for each simulation in simuls.yaml."""

import argparse
import re
from pathlib import Path

import yaml

from dataset import _glob_sorted


def parse_parameter_inp(sim_path):
    """Return dict of scalar values from parameter.inp (Fortran namelist)."""
    p = Path(sim_path)
    candidates = [p / "parameter.inp", p / "outs" / "parameter.inp"]
    for cand in candidates:
        if cand.exists():
            path = cand
            break
    else:
        return None

    vals = {}
    for line in path.read_text().splitlines():
        line = line.split("!")[0].strip()
        m = re.match(r"(\w+)\s*=\s*([^\s,/]+)", line)
        if m:
            key, raw = m.group(1), m.group(2)
            try:
                vals[key] = float(raw)
            except ValueError:
                vals[key] = raw
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", default="simuls.yaml")
    args = p.parse_args()

    with open(args.yaml) as f:
        simuls = yaml.safe_load(f)

    fmt = "{:<25s}  {:>10s}  {:>8s}  {:>10s}"
    print(fmt.format("name", "dt", "pstep", "n_xlg"))
    print("-" * 60)

    for sim in simuls:
        name = sim.get("name", "?")
        path = sim.get("path", "")
        if not Path(path).exists():
            continue

        vals = parse_parameter_inp(path)
        if vals is None:
            dt_s, pstep_s = "no param.inp", "-"
        else:
            dt    = vals.get("dt", float("nan"))
            tstep = int(vals.get("tstep", 0))
            lgmult = int(vals.get("lgmult", 1))
            pstep = tstep // lgmult
            dt_s    = f"{dt:.3e}"
            pstep_s = str(pstep)

        n_xlg = len(_glob_sorted(path, "xlg"))
        print(fmt.format(name, dt_s, pstep_s, str(n_xlg)))


if __name__ == "__main__":
    main()
