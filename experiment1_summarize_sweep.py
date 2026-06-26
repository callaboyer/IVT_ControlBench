#!/usr/bin/env python3
"""
Summarize PPO sweep results from experiment1_sweep.py.

Example:
    python summarize_sweep.py --root sweep --out sweep_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="sweep_v8")
    parser.add_argument("--out", type=str, default="sweep_v8_summary.csv")
    parser.add_argument("--sort-by", type=str, default="total_reward_mean")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for csv_path in root.glob("*/experiment1_results.csv"):
        df = pd.read_csv(csv_path)
        if "method" not in df.columns:
            continue

        ppo = df[df["method"] == "ppo"]
        if len(ppo) == 0:
            continue

        row = ppo.iloc[0].to_dict()
        row["run"] = csv_path.parent.name
        rows.append(row)

    if not rows:
        raise SystemExit(f"No PPO results found under: {root}")

    out = pd.DataFrame(rows)

    preferred_cols = [
        "run",
        "qa_yield_mean",
        "total_reward_mean",
        "rna_full_mean",
        "integrity_mean",
        "total_ntp_fed_mean",
        "total_mg_fed_mean",
        "ppi_mean",
        "mg_ppi_mean",
        "stop_time_mean",
    ]

    cols = [c for c in preferred_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in cols]
    out = out[cols + remaining]

    if args.sort_by in out.columns:
        out = out.sort_values(args.sort_by, ascending=False)

    out.to_csv(args.out, index=False)
    print(out.head(20).to_string(index=False))
    print(f"\nSaved summary to: {args.out}")


if __name__ == "__main__":
    main()
