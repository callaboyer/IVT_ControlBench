#!/usr/bin/env python3
"""
Summarize Experiment 2 PPO results by observation mode.

Example:
    python summarize_experiment2.py --root experiment2 --out experiment2_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="experiment2")
    parser.add_argument("--out", type=str, default="experiment2_summary.csv")
    parser.add_argument("--sort-by", type=str, default="total_reward_mean")
    args = parser.parse_args()

    rows = []
    for csv_path in Path(args.root).glob("*/experiment1_results.csv"):
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
        raise SystemExit(f"No PPO result CSVs found under {args.root!r}")

    out = pd.DataFrame(rows)

    preferred = [
        "run",
        "obs_mode",
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
    cols = [c for c in preferred if c in out.columns]
    out = out[cols + [c for c in out.columns if c not in cols]]

    if args.sort_by in out.columns:
        out = out.sort_values(["obs_mode", args.sort_by], ascending=[True, False])

    out.to_csv(args.out, index=False)

    print("\nPer-run results:")
    print(out.head(40).to_string(index=False))

    if "obs_mode" in out.columns:
        metrics = [
            "qa_yield_mean",
            "total_reward_mean",
            "rna_full_mean",
            "integrity_mean",
            "total_ntp_fed_mean",
            "total_mg_fed_mean",
            "mg_ppi_mean",
            "stop_time_mean",
        ]
        metrics = [m for m in metrics if m in out.columns]
        grouped = out.groupby("obs_mode")[metrics].agg(["mean", "std"])
        print("\nGrouped by observation mode:")
        print(grouped.to_string())

    print(f"\nSaved summary to: {args.out}")


if __name__ == "__main__":
    main()
