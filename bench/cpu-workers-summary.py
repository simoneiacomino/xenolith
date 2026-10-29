#!/usr/bin/env python3
"""Summarize the selected historical CPU worker samples; no benchmark runs."""
import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def changes(pairs, field):
    values = [100 * (float(b[field]) / float(a[field]) - 1) for a, b in pairs]
    return statistics.median(values), min(values), max(values)


def describe(values):
    median, low, high = values
    return f"{median:+.3f}% [{low:+.3f}%, {high:+.3f}%]"


remote = read("cpu-workers-remote-20261026.csv")
print("Remote prototype: prompt, median TPS 6 -> 12, paired TPS, paired package J/token, paired CPU seconds")
for depth in (512, 8192):
    cases = {(r["profile"], int(r["rep"])): r for r in remote if int(r["depth"]) == depth}
    pairs = [(cases["baseline6", i], cases["cpu12", i]) for i in (1, 2, 3)]
    medians = [statistics.median(float(pair[j]["decode_tps"]) for pair in pairs) for j in (0, 1)]
    print(depth, f"{medians[0]:.3f} -> {medians[1]:.3f}",
          *(describe(changes(pairs, field)) for field in ("decode_tps", "package_j_per_token", "cpu_seconds")), sep=" | ")

integrated = read("cpu-workers-integration-20261027.csv")
print("Integration control: workers, prompt, paired TPS change")
for workers in (4, 12):
    for depth in (512, 8192):
        cases = {}
        for row in integrated:
            if int(row["workers"]) == workers and int(row["depth"]) == depth:
                match = re.fullmatch(r"perf-(original|integrated)\d+-p\d+-g128-r([12])", row["case"])
                if not match:
                    raise ValueError(row["case"])
                cases[match.group(1), int(match.group(2))] = row
        pairs = [(cases["original", i], cases["integrated", i]) for i in (1, 2)]
        print(workers, depth, describe(changes(pairs, "decode_tps")), sep=" | ")
