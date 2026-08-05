from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")

METADATA_CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

RESULTS_ROOT = (
    ROOT
    / "BirdNET_Harmonic_Comparison"
    / "Raw_Results"
)

OUTPUT_FOLDER = (
    ROOT
    / "BirdNET_Harmonic_Comparison"
    / "Comparison_Reports"
)

BRANCHES = [
    "without_harmonics",
    "with_harmonics",
]

EXPERIMENTS = [
    "E1_no_metadata",
    "E2_week_only",
    "E3_location_only",
    "E4_week_and_location",
]

# Project-defined practical threshold, not a universal BirdNET rule.
MEANINGFUL_CONFIDENCE_DIFFERENCE = 0.02


def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = text.replace(".BirdNET.results", "")
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def harmonic_test_required(row: pd.Series) -> bool:
    return (
        str(row.get("harmonic_test_required", "no"))
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def resolve_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "None of these columns were found: "
        + ", ".join(candidates)
    )


def clean_species_name(row: pd.Series) -> str:
    for column in (
        "Common name",
        "Common Name",
        "common_name",
    ):
        if (
            column in row.index
            and pd.notna(row[column])
        ):
            return str(row[column]).strip()

    for column in (
        "species_name",
        "Species",
        "Scientific name",
        "Scientific Name",
    ):
        if (
            column in row.index
            and pd.notna(row[column])
        ):
            text = str(row[column]).strip()

            if "_" in text:
                return text.split("_", 1)[1].strip()

            return text

    return ""


def find_combined_table(folder: Path) -> Path:
    preferred = folder / "BirdNET_CombinedTable.csv"

    if preferred.exists():
        return preferred

    candidates = sorted(folder.glob("*Combined*.csv"))

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Expected one combined BirdNET CSV in "
        f"{folder.resolve()}, found {len(candidates)}."
    )


def identify_expected_bird(
    file_value: object,
    paired_names: dict[str, str],
) -> str | None:
    """Return the paired bird name, or None for a non-paired bird.

    The without-harmonics BirdNET table contains all 20 birds, while
    only the harmonic-test birds have a matching with-harmonics file.
    Non-paired birds must therefore be skipped instead of treated as
    errors.
    """
    file_key = normalize_name(
        Path(str(file_value)).name
    )

    if file_key in paired_names:
        return paired_names[file_key]

    matches = [
        (len(key), display_name)
        for key, display_name in paired_names.items()
        if key in file_key
    ]

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]


def prepare_table(
    csv_path: Path,
    branch: str,
    experiment: str,
    paired_names: dict[str, str],
) -> tuple[pd.DataFrame, str, str]:
    dataframe = pd.read_csv(csv_path)

    file_column = resolve_column(
        dataframe,
        ["File", "Begin Path", "file"],
    )

    start_column = resolve_column(
        dataframe,
        ["Start (s)", "Begin Time (s)", "start"],
    )

    end_column = resolve_column(
        dataframe,
        ["End (s)", "End Time (s)", "end"],
    )

    confidence_column = resolve_column(
        dataframe,
        ["Confidence", "confidence"],
    )

    dataframe = dataframe.copy()

    dataframe["branch"] = branch
    dataframe["experiment"] = experiment
    dataframe["file"] = dataframe[file_column].apply(
        lambda value: Path(str(value)).name
    )

    dataframe["expected"] = dataframe[file_column].apply(
        lambda value: identify_expected_bird(
            value,
            paired_names,
        )
    )

    # The without-harmonics results contain all 20 birds. Keep only
    # the 16 birds that also exist in the with-harmonics branch so
    # the comparison remains paired.
    dataframe = dataframe[
        dataframe["expected"].notna()
    ].copy()

    if dataframe.empty:
        raise ValueError(
            f"No paired birds were found in {csv_path}."
        )

    dataframe["species"] = dataframe.apply(
        clean_species_name,
        axis=1,
    )

    dataframe["species_key"] = dataframe["species"].apply(
        normalize_name
    )

    dataframe["expected_key"] = dataframe["expected"].apply(
        normalize_name
    )

    dataframe["confidence"] = (
        pd.to_numeric(
            dataframe[confidence_column],
            errors="coerce",
        )
        .fillna(0.0)
    )

    dataframe["start_s"] = dataframe[start_column]
    dataframe["end_s"] = dataframe[end_column]

    return dataframe, start_column, end_column


def summarize_windows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    group_columns = [
        "branch",
        "experiment",
        "file",
        "expected",
        "start_s",
        "end_s",
    ]

    for group_key, group in dataframe.groupby(
        group_columns,
        dropna=False,
    ):
        (
            branch,
            experiment,
            file_name,
            expected,
            start_s,
            end_s,
        ) = group_key

        group = group.sort_values(
            "confidence",
            ascending=False,
        )

        expected_key = normalize_name(expected)

        species = group["species"].astype(str).tolist()
        species_keys = [
            normalize_name(value)
            for value in species
        ]
        confidences = (
            group["confidence"]
            .astype(float)
            .tolist()
        )

        target_positions = [
            index
            for index, species_key
            in enumerate(species_keys)
            if species_key == expected_key
        ]

        detected = bool(target_positions)

        target_rank = (
            target_positions[0] + 1
            if detected
            else np.nan
        )

        target_confidence = (
            confidences[target_positions[0]]
            if detected
            else 0.0
        )

        competitor_confidences = [
            confidence
            for species_key, confidence
            in zip(species_keys, confidences)
            if species_key != expected_key
        ]

        strongest_competitor = (
            max(competitor_confidences)
            if competitor_confidences
            else 0.0
        )

        rows.append({
            "branch": branch,
            "experiment": experiment,
            "file": file_name,
            "expected": expected,
            "start_s": start_s,
            "end_s": end_s,
            "target_detected": detected,
            "target_rank": target_rank,
            "target_confidence": target_confidence,
            "strongest_competitor_confidence":
                strongest_competitor,
            "target_margin":
                target_confidence
                - strongest_competitor,
            "top1_prediction":
                species[0] if species else "",
            "top1_confidence":
                confidences[0] if confidences else 0.0,
            "correct_top1":
                detected and target_rank == 1,
            "correct_top3":
                detected and target_rank <= 3,
            "correct_top5":
                detected and target_rank <= 5,
        })

    return pd.DataFrame(rows)


def summarize_recordings(
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group_key, group in window_summary.groupby(
        ["branch", "experiment", "file", "expected"],
        dropna=False,
    ):
        branch, experiment, file_name, expected = group_key

        detected = group[group["target_detected"]]
        ranks = (
            detected["target_rank"]
            .dropna()
            .astype(float)
        )

        rows.append({
            "branch": branch,
            "experiment": experiment,
            "file": file_name,
            "expected": expected,
            "analysis_windows": int(len(group)),
            "target_detected": not detected.empty,
            "target_detection_windows":
                int(group["target_detected"].sum()),
            "target_detection_rate":
                float(group["target_detected"].mean()),
            "target_max_confidence":
                float(group["target_confidence"].max()),
            "target_mean_confidence_all_windows":
                float(group["target_confidence"].mean()),
            "target_best_rank":
                float(ranks.min())
                if not ranks.empty
                else np.nan,
            "top1_window_rate":
                float(group["correct_top1"].mean()),
            "top3_window_rate":
                float(group["correct_top3"].mean()),
            "top5_window_rate":
                float(group["correct_top5"].mean()),
            "target_best_margin":
                float(group["target_margin"].max()),
            "target_mean_margin":
                float(group["target_margin"].mean()),
            "strongest_competitor_max_confidence":
                float(
                    group[
                        "strongest_competitor_confidence"
                    ].max()
                ),
        })

    return pd.DataFrame(rows)


def decide_pair(row: pd.Series) -> tuple[str, str]:
    comparisons = [
        (
            "top1_window_rate",
            "higher target top-1 window rate",
        ),
        (
            "top3_window_rate",
            "higher target top-3 window rate",
        ),
        (
            "target_detection_rate",
            "higher target detection rate",
        ),
    ]

    for metric, reason in comparisons:
        with_value = float(row[f"with_{metric}"])
        without_value = float(row[f"without_{metric}"])

        if with_value > without_value:
            return "with_harmonics", reason

        if without_value > with_value:
            return "without_harmonics", reason

    max_difference = float(
        row["target_max_confidence_difference"]
    )

    if (
        max_difference
        >= MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "with_harmonics",
            "meaningfully higher maximum target confidence",
        )

    if (
        max_difference
        <= -MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "without_harmonics",
            "meaningfully higher maximum target confidence",
        )

    mean_difference = float(
        row["target_mean_confidence_difference"]
    )

    if (
        mean_difference
        >= MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "with_harmonics",
            "meaningfully higher mean target confidence",
        )

    if (
        mean_difference
        <= -MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "without_harmonics",
            "meaningfully higher mean target confidence",
        )

    margin_difference = float(
        row["target_mean_margin_difference"]
    )

    if (
        margin_difference
        >= MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "with_harmonics",
            "better separation from competing species",
        )

    if (
        margin_difference
        <= -MEANINGFUL_CONFIDENCE_DIFFERENCE
    ):
        return (
            "without_harmonics",
            "better separation from competing species",
        )

    return (
        "no_meaningful_difference",
        "top-k rates and confidence differences are approximately equivalent",
    )


def create_paired_comparison(
    recording_summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["experiment", "expected"]

    without = (
        recording_summary[
            recording_summary["branch"]
            == "without_harmonics"
        ]
        .drop(columns=["branch"])
        .rename(
            columns={
                column: (
                    column
                    if column in keys
                    else f"without_{column}"
                )
                for column
                in recording_summary.columns
                if column != "branch"
            }
        )
    )

    with_harmonics = (
        recording_summary[
            recording_summary["branch"]
            == "with_harmonics"
        ]
        .drop(columns=["branch"])
        .rename(
            columns={
                column: (
                    column
                    if column in keys
                    else f"with_{column}"
                )
                for column
                in recording_summary.columns
                if column != "branch"
            }
        )
    )

    paired = without.merge(
        with_harmonics,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    paired["target_max_confidence_difference"] = (
        paired["with_target_max_confidence"]
        - paired["without_target_max_confidence"]
    )

    paired["target_mean_confidence_difference"] = (
        paired[
            "with_target_mean_confidence_all_windows"
        ]
        - paired[
            "without_target_mean_confidence_all_windows"
        ]
    )

    paired["target_detection_rate_difference"] = (
        paired["with_target_detection_rate"]
        - paired["without_target_detection_rate"]
    )

    paired["top1_window_rate_difference"] = (
        paired["with_top1_window_rate"]
        - paired["without_top1_window_rate"]
    )

    paired["top3_window_rate_difference"] = (
        paired["with_top3_window_rate"]
        - paired["without_top3_window_rate"]
    )

    paired["top5_window_rate_difference"] = (
        paired["with_top5_window_rate"]
        - paired["without_top5_window_rate"]
    )

    paired["target_mean_margin_difference"] = (
        paired["with_target_mean_margin"]
        - paired["without_target_mean_margin"]
    )

    decisions = paired.apply(
        decide_pair,
        axis=1,
        result_type="expand",
    )

    decisions.columns = [
        "preferred_branch",
        "decision_reason",
    ]

    return (
        pd.concat([paired, decisions], axis=1)
        .sort_values(["experiment", "expected"])
    )


def create_overall_decisions(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for bird_name, group in paired.groupby("expected"):
        with_votes = int(
            (
                group["preferred_branch"]
                == "with_harmonics"
            ).sum()
        )

        without_votes = int(
            (
                group["preferred_branch"]
                == "without_harmonics"
            ).sum()
        )

        tie_votes = int(
            (
                group["preferred_branch"]
                == "no_meaningful_difference"
            ).sum()
        )

        mean_top1_difference = float(
            group[
                "top1_window_rate_difference"
            ].mean()
        )

        mean_max_difference = float(
            group[
                "target_max_confidence_difference"
            ].mean()
        )

        mean_confidence_difference = float(
            group[
                "target_mean_confidence_difference"
            ].mean()
        )

        if with_votes > without_votes:
            preferred = "with_harmonics"
            reason = "won more metadata experiments"

        elif without_votes > with_votes:
            preferred = "without_harmonics"
            reason = "won more metadata experiments"

        elif mean_top1_difference > 0:
            preferred = "with_harmonics"
            reason = (
                "experiment votes tied; "
                "higher average top-1 window rate"
            )

        elif mean_top1_difference < 0:
            preferred = "without_harmonics"
            reason = (
                "experiment votes tied; "
                "higher average top-1 window rate"
            )

        elif (
            mean_max_difference
            >= MEANINGFUL_CONFIDENCE_DIFFERENCE
        ):
            preferred = "with_harmonics"
            reason = (
                "experiment votes tied; "
                "meaningfully higher average "
                "maximum confidence"
            )

        elif (
            mean_max_difference
            <= -MEANINGFUL_CONFIDENCE_DIFFERENCE
        ):
            preferred = "without_harmonics"
            reason = (
                "experiment votes tied; "
                "meaningfully higher average "
                "maximum confidence"
            )

        else:
            preferred = "no_meaningful_difference"
            reason = (
                "no consistent or meaningful "
                "advantage across experiments"
            )

        rows.append({
            "expected": bird_name,
            "with_harmonics_votes": with_votes,
            "without_harmonics_votes":
                without_votes,
            "no_difference_votes": tie_votes,
            "mean_top1_window_rate_difference":
                mean_top1_difference,
            "mean_target_max_confidence_difference":
                mean_max_difference,
            "mean_target_confidence_difference":
                mean_confidence_difference,
            "overall_preferred_branch": preferred,
            "overall_decision_reason": reason,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("expected")
    )


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: "
            f"{METADATA_CSV.resolve()}"
        )

    metadata = pd.read_csv(METADATA_CSV)

    required = {
        "name",
        "harmonic_test_required",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            "Metadata CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    paired_metadata = metadata[
        metadata.apply(
            harmonic_test_required,
            axis=1,
        )
    ].copy()

    paired_names = {
        normalize_name(row["name"]):
            str(row["name"]).strip()
        for _, row
        in paired_metadata.iterrows()
    }

    if len(paired_names) != 16:
        print(
            "WARNING: Expected 16 paired birds, "
            f"found {len(paired_names)}."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_tables: list[pd.DataFrame] = []
    window_tables: list[pd.DataFrame] = []

    print("=" * 80)
    print("EVALUATE BIRDNET HARMONIC COMPARISON")
    print("=" * 80)

    for branch in BRANCHES:
        for experiment in EXPERIMENTS:
            result_folder = (
                RESULTS_ROOT
                / branch
                / experiment
            )

            combined_csv = find_combined_table(
                result_folder
            )

            print(f"Reading: {combined_csv.resolve()}")

            prepared, _, _ = prepare_table(
                combined_csv,
                branch,
                experiment,
                paired_names,
            )

            prepared_tables.append(prepared)
            window_tables.append(
                summarize_windows(prepared)
            )

    raw_combined = pd.concat(
        prepared_tables,
        ignore_index=True,
    )

    window_summary = pd.concat(
        window_tables,
        ignore_index=True,
    )

    recording_summary = summarize_recordings(
        window_summary
    )

    paired_comparison = create_paired_comparison(
        recording_summary
    )

    overall_decisions = create_overall_decisions(
        paired_comparison
    )

    experiment_summary = (
        paired_comparison.groupby(
            "experiment",
            as_index=False,
        )
        .agg(
            paired_birds=("expected", "count"),
            with_harmonics_preferred=(
                "preferred_branch",
                lambda values: int(
                    (
                        values
                        == "with_harmonics"
                    ).sum()
                ),
            ),
            without_harmonics_preferred=(
                "preferred_branch",
                lambda values: int(
                    (
                        values
                        == "without_harmonics"
                    ).sum()
                ),
            ),
            no_meaningful_difference=(
                "preferred_branch",
                lambda values: int(
                    (
                        values
                        == "no_meaningful_difference"
                    ).sum()
                ),
            ),
            mean_target_max_confidence_difference=(
                "target_max_confidence_difference",
                "mean",
            ),
            mean_target_confidence_difference=(
                "target_mean_confidence_difference",
                "mean",
            ),
            mean_top1_window_rate_difference=(
                "top1_window_rate_difference",
                "mean",
            ),
        )
    )

    raw_combined.to_csv(
        OUTPUT_FOLDER
        / "birdnet_raw_combined_clean.csv",
        index=False,
    )

    window_summary.to_csv(
        OUTPUT_FOLDER
        / "birdnet_summary_by_window.csv",
        index=False,
    )

    recording_summary.to_csv(
        OUTPUT_FOLDER
        / "birdnet_summary_by_recording.csv",
        index=False,
    )

    paired_comparison.to_csv(
        OUTPUT_FOLDER
        / "paired_with_without_harmonics_comparison.csv",
        index=False,
    )

    overall_decisions.to_csv(
        OUTPUT_FOLDER
        / "preferred_branch_by_bird.csv",
        index=False,
    )

    experiment_summary.to_csv(
        OUTPUT_FOLDER
        / "harmonic_comparison_by_experiment.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"Reports: {OUTPUT_FOLDER.resolve()}")

    print("\nOverall preferred branch by bird:")
    print(
        overall_decisions[
            [
                "expected",
                "overall_preferred_branch",
                "overall_decision_reason",
            ]
        ].to_string(index=False)
    )

    print("\nSummary by experiment:")
    print(
        experiment_summary.to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
