"""Scan .lag files for size anomalies (empty or inconsistent across timesteps).

All .lag files for the same quantity in a simulation should be identical in
size. Deviations indicate empty or partially written files.

Usage
-----
python inspect/inspect_lag_files.py --yaml simuls.yaml
python inspect/inspect_lag_files.py --path /path/to/sim
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml


def scan_sim(sim_path):
    """Return list of issue strings for sim_path, or [] if clean."""
    p = Path(sim_path)
    search_dir = p / "outs" if (p / "outs").is_dir() else p
    files = sorted(search_dir.glob("*.lag"))
    if not files:
        return ["no .lag files found"]

    by_quantity = defaultdict(list)
    for f in files:
        by_quantity[f.name.split(".")[0]].append(f.stat().st_size)

    issues = []
    for qty, sizes in sorted(by_quantity.items()):
        zeros = sizes.count(0)
        if zeros:
            issues.append(f"{qty}: {zeros}/{len(sizes)} empty files")
        unique = set(sizes) - {0}
        if len(unique) > 1:
            issues.append(f"{qty}: inconsistent sizes {sorted(unique)}")

    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--yaml", help="simuls.yaml — scan all accessible sims")
    g.add_argument("--path", help="single simulation directory to scan")
    args = p.parse_args()

    if args.path:
        sims = [{"name": Path(args.path).name, "path": args.path}]
    else:
        with open(args.yaml) as fh:
            entries = yaml.safe_load(fh)
        sims = [{"name": s.get("name", s["path"]), "path": s["path"]} for s in entries]

    total_issues = skipped = 0
    for sim in sims:
        if not Path(sim["path"]).is_dir():
            skipped += 1
            continue
        issues = scan_sim(sim["path"])
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"  {sim['name']:<35s}  [{status}]")
        for msg in issues:
            print(f"      ! {msg}")
        total_issues += len(issues)

    print(f"\nTotal issues: {total_issues}"
          + (f"  ({skipped} sim(s) skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
