#!/usr/bin/env python3
"""Offline Platt fit (no network): CSV with columns forecast,outcome (header
optional) -> JSON {"a":..,"b":..} usable as METAC_CALIBRATION.

    python3 scripts/fit_calibration.py pairs.csv > calibration.json
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast_extras import fit_platt  # noqa: E402


def read_pairs(path):
    fs, os_ = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                fs.append(float(row[0]))
                os_.append(float(row[1]))
            except (ValueError, IndexError):
                continue  # header / blank
    return fs, os_


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    fs, os_ = read_pairs(argv[1])
    if len(fs) < 2 or len(set(os_)) < 2:
        print("need >=2 pairs with both outcomes 0 and 1", file=sys.stderr)
        return 1
    a, b = fit_platt(fs, os_)
    print(json.dumps({"a": a, "b": b}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
