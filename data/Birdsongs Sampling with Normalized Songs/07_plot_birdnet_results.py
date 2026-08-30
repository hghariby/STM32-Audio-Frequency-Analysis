from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================ USER SETTINGS ============================

ROOT = Path(".")

INPUT_CSV = (
    ROOT
    / "BirdNET_Results"
    / "02_evaluated"
    / "birdnet_by_recording.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "BirdNET_Results"
    / "04_plots"
)

DISTANCE_ORDER = [
    "1ft",
    "4ft",
    "8ft",
    "12ft",
    "24ft",
    "36ft",
    "48ft",
]

EXPERIMENT_ORDER = [
    "E1_no_metadata",
    "E2_week_only",
    "E3_location_only",
    "E4_week_and_location",
]

EXPERIMENT_LABELS = {
    "E1_no_metadata": "E1: No metadata",
    "E2_week_only": "E2: Week only",
    "E3_location_only": "E3: Location only",
    "E4_week_and_location": "E4: Week + location",
}

DPI = 300

# =====================================================================


REQUIRED_COLUMNS = {
    "experiment",
    "bird",
    "distance",
    "status",
    "mean_correct_top1",
    "max_correct_top1",
    "expected_mean_active_window_confidence",
    "active_window_detection_rate",
}


def safe_name(text: str) -> str:
    """Convert a bird name into a folder/file-safe name."""
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(text))
    value = re.sub(r"\s+", "_", value.strip())
    return value


def distance_number(value: str) -> int:
    """Convert '24ft' to 24 for sorting and plotting."""
    match = re.fullmatch(r"(\d+)ft", str(value).strip(), flags=re.I)
    if not match:
        raise ValueError(f"Invalid distance value: {value}")
    return int(match.group(1))


def prepare_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"BirdNET recording-level CSV not found:\n{INPUT_CSV.resolve()}"
        )

    df = pd.read_csv(INPUT_CSV)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    # Use only successfully evaluated recordings.
    df = df[df["status"] == "OK"].copy()

    if df.empty:
        raise ValueError("No rows with status == 'OK' were found.")

    # Normalize boolean columns in case CSV reading returns text.
    for column in ["mean_correct_top1", "max_correct_top1"]:
        if df[column].dtype == object:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

        df[column] = df[column].astype(float)

    for column in [
        "expected_mean_active_window_confidence",
        "active_window_detection_rate",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["distance_ft"] = df["distance"].map(distance_number)

    # Keep the expected experimental ordering.
    df["experiment"] = pd.Categorical(
        df["experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )

    return df


def format_percent_axis(ax) -> None:
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def format_distance_axis(ax) -> None:
    distances = [distance_number(x) for x in DISTANCE_ORDER]
    ax.set_xticks(distances)
    ax.set_xticklabels(DISTANCE_ORDER)
    ax.set_xlabel("Distance")


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_single_experiment_top1(
    bird_df: pd.DataFrame,
    bird: str,
    experiment: str,
    metric: str,
    method_label: str,
    output_path: Path,
) -> None:
    data = (
        bird_df[bird_df["experiment"] == experiment]
        .sort_values("distance_ft")
    )

    if data.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(
        data["distance_ft"],
        data[metric],
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{bird}\n{method_label} Top-1 vs Distance — "
        f"{EXPERIMENT_LABELS.get(experiment, experiment)}"
    )
    plt.ylabel("Top-1 correct")
    format_percent_axis(plt.gca())
    format_distance_axis(plt.gca())
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def plot_combined_experiments_top1(
    bird_df: pd.DataFrame,
    bird: str,
    metric: str,
    method_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))

    for experiment in EXPERIMENT_ORDER:
        data = (
            bird_df[bird_df["experiment"] == experiment]
            .sort_values("distance_ft")
        )

        if data.empty:
            continue

        plt.plot(
            data["distance_ft"],
            data[metric],
            marker="o",
            linewidth=2,
            label=EXPERIMENT_LABELS.get(experiment, experiment),
        )

    plt.title(
        f"{bird}\n{method_label} Top-1 vs Distance — All Experiments"
    )
    plt.ylabel("Top-1 correct")
    format_percent_axis(plt.gca())
    format_distance_axis(plt.gca())
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(output_path)


def plot_metric_all_experiments(
    bird_df: pd.DataFrame,
    bird: str,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))

    for experiment in EXPERIMENT_ORDER:
        data = (
            bird_df[bird_df["experiment"] == experiment]
            .sort_values("distance_ft")
        )

        if data.empty:
            continue

        plt.plot(
            data["distance_ft"],
            data[metric],
            marker="o",
            linewidth=2,
            label=EXPERIMENT_LABELS.get(experiment, experiment),
        )

    plt.title(f"{bird}\n{title}")
    plt.ylabel(ylabel)
    format_percent_axis(plt.gca())
    format_distance_axis(plt.gca())
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(output_path)


def plot_total_top1(
    df: pd.DataFrame,
    metric: str,
    method_label: str,
    output_path: Path,
) -> None:
    """
    Total plot:
    average Top-1 correctness across all birds AND all experiments
    at each distance.
    """
    total = (
        df.groupby("distance_ft", as_index=False, observed=True)
        .agg(top1_accuracy=(metric, "mean"))
        .sort_values("distance_ft")
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        total["distance_ft"],
        total["top1_accuracy"],
        marker="o",
        linewidth=2.5,
    )

    plt.title(
        f"Overall {method_label} Top-1 Accuracy vs Distance\n"
        "All Birds + All Experiments"
    )
    plt.ylabel("Top-1 accuracy")
    format_percent_axis(plt.gca())
    format_distance_axis(plt.gca())
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def main() -> None:
    df = prepare_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    birds = sorted(df["bird"].dropna().astype(str).unique())

    print(f"Input: {INPUT_CSV.resolve()}")
    print(f"Birds: {len(birds)}")
    print(f"Output: {OUTPUT_ROOT.resolve()}\n")

    for index, bird in enumerate(birds, start=1):
        print(f"[{index}/{len(birds)}] {bird}")

        bird_df = df[df["bird"] == bird].copy()
        bird_folder = OUTPUT_ROOT / safe_name(bird)

        mean_folder = bird_folder / "Mean_Top1"
        max_folder = bird_folder / "Max_Top1"

        # ----------------------------------------------------------
        # Mean Top-1: one plot per experiment
        # ----------------------------------------------------------
        for experiment in EXPERIMENT_ORDER:
            plot_single_experiment_top1(
                bird_df=bird_df,
                bird=bird,
                experiment=experiment,
                metric="mean_correct_top1",
                method_label="Mean",
                output_path=(
                    mean_folder
                    / f"{safe_name(experiment)}_mean_top1_vs_distance.png"
                ),
            )

        # Mean Top-1: all four experiments on one plot.
        plot_combined_experiments_top1(
            bird_df=bird_df,
            bird=bird,
            metric="mean_correct_top1",
            method_label="Mean",
            output_path=(
                mean_folder
                / "combined_experiments_mean_top1_vs_distance.png"
            ),
        )

        # ----------------------------------------------------------
        # Max Top-1: one plot per experiment
        # ----------------------------------------------------------
        for experiment in EXPERIMENT_ORDER:
            plot_single_experiment_top1(
                bird_df=bird_df,
                bird=bird,
                experiment=experiment,
                metric="max_correct_top1",
                method_label="Max",
                output_path=(
                    max_folder
                    / f"{safe_name(experiment)}_max_top1_vs_distance.png"
                ),
            )

        # Max Top-1: all four experiments on one plot.
        plot_combined_experiments_top1(
            bird_df=bird_df,
            bird=bird,
            metric="max_correct_top1",
            method_label="Max",
            output_path=(
                max_folder
                / "combined_experiments_max_top1_vs_distance.png"
            ),
        )

        # ----------------------------------------------------------
        # Expected-bird confidence vs distance
        # One line per experiment.
        # ----------------------------------------------------------
        plot_metric_all_experiments(
            bird_df=bird_df,
            bird=bird,
            metric="expected_mean_active_window_confidence",
            ylabel="Expected-bird confidence",
            title="Expected-Bird Confidence vs Distance",
            output_path=(
                bird_folder
                / "expected_bird_confidence_vs_distance.png"
            ),
        )

        # ----------------------------------------------------------
        # Active-window detection rate vs distance
        # One line per experiment.
        # ----------------------------------------------------------
        plot_metric_all_experiments(
            bird_df=bird_df,
            bird=bird,
            metric="active_window_detection_rate",
            ylabel="Active-window detection rate",
            title="Active-Window Detection Rate vs Distance",
            output_path=(
                bird_folder
                / "active_window_detection_rate_vs_distance.png"
            ),
        )

    # --------------------------------------------------------------
    # TOTAL plots at the same level as the bird folders.
    # --------------------------------------------------------------
    plot_total_top1(
        df=df,
        metric="mean_correct_top1",
        method_label="Mean",
        output_path=(
            OUTPUT_ROOT
            / "TOTAL_mean_top1_accuracy_vs_distance.png"
        ),
    )

    plot_total_top1(
        df=df,
        metric="max_correct_top1",
        method_label="Max",
        output_path=(
            OUTPUT_ROOT
            / "TOTAL_max_top1_accuracy_vs_distance.png"
        ),
    )

    print("\nPLOTTING COMPLETE")
    print(f"Plots saved under:\n{OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
