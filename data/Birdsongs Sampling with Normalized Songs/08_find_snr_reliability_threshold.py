from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT_CSV = ROOT / "Acoustic_Results" / "acoustic_by_recording.csv"
OUTPUT_DIR = ROOT / "SNR_Threshold_Analysis"

SNR_COLUMN = "environment_adjusted_snr_db"
DISTANCE_ORDER = ["1ft", "4ft", "8ft", "12ft", "24ft", "36ft", "48ft"]
CANDIDATE_THRESHOLDS_DB = [0.0, 3.0, 6.0, 10.0]
DPI = 300

PLOT_NOTES = {
    "01_overall_snr_distribution.png":
        "Shows the SNR distribution across every bird and distance. Vertical lines mark candidate reliability thresholds.",
    "02_snr_by_distance.png":
        "Shows how SNR changes across distance and how much birds vary at the same distance.",
    "03_fdi_vs_snr.png":
        "Shows whether FDI becomes larger or more scattered at low SNR, which can indicate noise-contaminated distortion.",
    "04_bandwidth_change_vs_snr.png":
        "Shows whether bandwidth measurements become unstable at low SNR.",
    "05_high_edge_change_vs_snr.png":
        "Shows whether the measured upper spectral edge becomes unstable at low SNR.",
    "06_low_edge_change_vs_snr.png":
        "Shows whether the measured lower spectral edge becomes unstable at low SNR.",
    "07_candidate_threshold_retention.png":
        "Shows how many recordings remain above each candidate threshold; higher thresholds are more conservative."
}


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    note = PLOT_NOTES.get(path.name, "")
    if note:
        fig = plt.gcf()
        fig.text(0.5, 0.012, "Note: " + note,
                 ha="center", va="bottom", fontsize=8, wrap=True)
        plt.tight_layout(rect=[0, 0.075, 1, 1])
    else:
        plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def distance_number(value: str) -> int:
    text = str(value).strip().lower()
    if not text.endswith("ft"):
        raise ValueError(f"Unexpected distance label: {value}")
    return int(text[:-2])


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    required = {
        "bird", "distance", "file", SNR_COLUMN,
        "frequency_distortion_index_db",
        "bandwidth_change_hz",
        "usable_low_edge_change_hz",
        "usable_high_edge_change_hz",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    numeric_cols = [
        SNR_COLUMN,
        "frequency_distortion_index_db",
        "bandwidth_change_hz",
        "usable_low_edge_change_hz",
        "usable_high_edge_change_hz",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["distance_ft"] = df["distance"].map(distance_number)
    df["finite_snr"] = np.isfinite(df[SNR_COLUMN])
    return df


def plot_overall_distribution(df: pd.DataFrame) -> None:
    s = df.loc[df["finite_snr"], SNR_COLUMN]
    plt.figure(figsize=(10, 6))
    bins = max(10, min(30, int(round(math.sqrt(len(s)) * 2))))
    plt.hist(s, bins=bins, edgecolor="black", alpha=0.75)
    for t in CANDIDATE_THRESHOLDS_DB:
        plt.axvline(t, linestyle="--", linewidth=1.5, label=f"{t:g} dB")
    plt.title("Overall Environment-Adjusted SNR Distribution")
    plt.xlabel("Environment-adjusted SNR (dB)")
    plt.ylabel("Number of bird × distance recordings")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(title="Candidate thresholds")
    save_plot(OUTPUT_DIR / "01_overall_snr_distribution.png")


def plot_snr_by_distance(df: pd.DataFrame) -> None:
    finite = df[df["finite_snr"]]
    groups, labels = [], []
    for d in DISTANCE_ORDER:
        vals = finite.loc[finite["distance"] == d, SNR_COLUMN].dropna()
        if len(vals):
            groups.append(vals.to_numpy())
            labels.append(d)

    plt.figure(figsize=(11, 6))
    pos = np.arange(1, len(groups) + 1)
    plt.boxplot(groups, positions=pos, widths=0.55, showfliers=False)

    rng = np.random.default_rng(12345)
    for x, vals in zip(pos, groups):
        jitter = rng.normal(0, 0.055, len(vals))
        plt.scatter(np.full(len(vals), x) + jitter, vals,
                    s=28, alpha=0.7, zorder=3)

    for t, ls in [(0, ":"), (3, "-."), (6, "--"), (10, "--")]:
        plt.axhline(t, linestyle=ls, linewidth=1.1, label=f"{t:g} dB")

    plt.xticks(pos, labels)
    plt.title("Environment-Adjusted SNR by Distance")
    plt.xlabel("Distance")
    plt.ylabel("Environment-adjusted SNR (dB)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    save_plot(OUTPUT_DIR / "02_snr_by_distance.png")


def plot_metric_vs_snr(df: pd.DataFrame, ycol: str, ylabel: str,
                       title: str, filename: str, zero_line=False) -> None:
    data = df[df["finite_snr"] & np.isfinite(df[ycol])].copy()
    if data.empty:
        return

    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for d, marker in zip(DISTANCE_ORDER, markers):
        part = data[data["distance"] == d]
        if part.empty:
            continue
        plt.scatter(part[SNR_COLUMN], part[ycol],
                    marker=marker, s=45, alpha=0.75, label=d)

    for t in CANDIDATE_THRESHOLDS_DB:
        plt.axvline(t, linestyle="--", linewidth=1, alpha=0.6)

    if zero_line:
        plt.axhline(0, linestyle=":", linewidth=1)

    plt.title(title)
    plt.xlabel("Environment-adjusted SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    save_plot(OUTPUT_DIR / filename)


def candidate_summary(df: pd.DataFrame) -> pd.DataFrame:
    finite = df[df["finite_snr"]].copy()
    rows = []

    metrics = {
        "frequency_distortion_index_db": "fdi",
        "bandwidth_change_hz": "bandwidth_change",
        "usable_low_edge_change_hz": "low_edge_change",
        "usable_high_edge_change_hz": "high_edge_change",
    }

    for t in CANDIDATE_THRESHOLDS_DB:
        above = finite[finite[SNR_COLUMN] >= t]
        below = finite[finite[SNR_COLUMN] < t]

        row = {
            "threshold_db": t,
            "total_finite_snr_rows": len(finite),
            "rows_at_or_above_threshold": len(above),
            "rows_below_threshold": len(below),
            "percent_retained": 100 * len(above) / len(finite),
        }

        for col, short in metrics.items():
            row[f"median_{short}_above"] = above[col].median()
            row[f"median_{short}_below"] = below[col].median()
            row[f"median_abs_{short}_above"] = above[col].abs().median()
            row[f"median_abs_{short}_below"] = below[col].abs().median()

        rows.append(row)

    return pd.DataFrame(rows)


def plot_candidate_retention(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 5.5))
    x = np.arange(len(summary))
    plt.bar(x, summary["percent_retained"])
    plt.xticks(x, [f"{v:g} dB" for v in summary["threshold_db"]])

    for i, row in summary.iterrows():
        plt.text(i, row["percent_retained"] + 1,
                 f"{row['percent_retained']:.1f}%",
                 ha="center", va="bottom", fontsize=9)

    plt.ylim(0, 105)
    plt.title("Recordings Retained by Candidate SNR Threshold")
    plt.xlabel("Candidate threshold")
    plt.ylabel("Finite-SNR recordings retained (%)")
    plt.grid(True, axis="y", alpha=0.25)
    save_plot(OUTPUT_DIR / "07_candidate_threshold_retention.png")


def snr_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    finite = df[df["finite_snr"]].copy()

    edges = [-np.inf, 0, 3, 6, 10, 15, 20, np.inf]
    labels = ["<0", "0–3", "3–6", "6–10", "10–15", "15–20", ">=20"]

    finite["snr_bin"] = pd.cut(
        finite[SNR_COLUMN], bins=edges, labels=labels, right=False
    )

    return (
        finite.groupby("snr_bin", observed=True)
        .agg(
            recording_count=("file", "count"),
            median_snr_db=(SNR_COLUMN, "median"),
            median_fdi_db=("frequency_distortion_index_db", "median"),
            median_abs_bandwidth_change_hz=(
                "bandwidth_change_hz", lambda x: x.abs().median()
            ),
            median_abs_low_edge_change_hz=(
                "usable_low_edge_change_hz", lambda x: x.abs().median()
            ),
            median_abs_high_edge_change_hz=(
                "usable_high_edge_change_hz", lambda x: x.abs().median()
            ),
        )
        .reset_index()
    )


def distance_summary(df: pd.DataFrame) -> pd.DataFrame:
    finite = df[df["finite_snr"]].copy()

    out = (
        finite.groupby("distance", observed=True)
        .agg(
            recording_count=("file", "count"),
            mean_snr_db=(SNR_COLUMN, "mean"),
            median_snr_db=(SNR_COLUMN, "median"),
            std_snr_db=(SNR_COLUMN, "std"),
            minimum_snr_db=(SNR_COLUMN, "min"),
            maximum_snr_db=(SNR_COLUMN, "max"),
        )
        .reset_index()
    )

    out["distance_ft"] = out["distance"].map(distance_number)
    return out.sort_values("distance_ft").drop(columns="distance_ft")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    finite = df[df["finite_snr"]]

    print("=" * 78)
    print("FIND SNR RELIABILITY THRESHOLD")
    print("=" * 78)
    print(f"Finite SNR rows: {len(finite)} / {len(df)}")
    print(f"Minimum SNR: {finite[SNR_COLUMN].min():.2f} dB")
    print(f"25th percentile: {finite[SNR_COLUMN].quantile(0.25):.2f} dB")
    print(f"Median: {finite[SNR_COLUMN].median():.2f} dB")
    print(f"75th percentile: {finite[SNR_COLUMN].quantile(0.75):.2f} dB")
    print(f"Maximum SNR: {finite[SNR_COLUMN].max():.2f} dB")
    print()

    plot_overall_distribution(df)
    plot_snr_by_distance(df)

    plot_metric_vs_snr(
        df, "frequency_distortion_index_db", "FDI (dB)",
        "FDI vs Environment-Adjusted SNR",
        "03_fdi_vs_snr.png"
    )

    plot_metric_vs_snr(
        df, "bandwidth_change_hz", "Bandwidth change from 1 ft (Hz)",
        "Bandwidth Change vs Environment-Adjusted SNR",
        "04_bandwidth_change_vs_snr.png", True
    )

    plot_metric_vs_snr(
        df, "usable_high_edge_change_hz", "High-edge change from 1 ft (Hz)",
        "High Spectral-Energy Edge Change vs Environment-Adjusted SNR",
        "05_high_edge_change_vs_snr.png", True
    )

    plot_metric_vs_snr(
        df, "usable_low_edge_change_hz", "Low-edge change from 1 ft (Hz)",
        "Low Spectral-Energy Edge Change vs Environment-Adjusted SNR",
        "06_low_edge_change_vs_snr.png", True
    )

    candidates = candidate_summary(df)
    candidates.to_csv(
        OUTPUT_DIR / "candidate_threshold_comparison.csv", index=False
    )
    plot_candidate_retention(candidates)

    bins = snr_bin_summary(df)
    bins.to_csv(OUTPUT_DIR / "snr_bin_diagnostics.csv", index=False)

    dsum = distance_summary(df)
    dsum.to_csv(OUTPUT_DIR / "snr_by_distance_summary.csv", index=False)

    review_cols = [
        "bird", "distance", "file", SNR_COLUMN,
        "frequency_distortion_index_db",
        "bandwidth_change_hz",
        "usable_low_edge_change_hz",
        "usable_high_edge_change_hz",
    ]
    df[review_cols].sort_values(
        SNR_COLUMN, na_position="last"
    ).to_csv(
        OUTPUT_DIR / "recordings_sorted_by_snr.csv", index=False
    )

    print("CANDIDATE THRESHOLDS")
    for _, row in candidates.iterrows():
        print(
            f"{row['threshold_db']:>4.0f} dB -> "
            f"{int(row['rows_at_or_above_threshold'])}/"
            f"{int(row['total_finite_snr_rows'])} retained "
            f"({row['percent_retained']:.1f}%)"
        )

    print()
    print("Use the plots to look for where FDI, bandwidth, and spectral-edge")
    print("measurements become much more scattered or extreme as SNR decreases.")
    print("Then inspect recordings just above and below that region using")
    print("recordings_sorted_by_snr.csv before selecting the final threshold.")
    print()
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
