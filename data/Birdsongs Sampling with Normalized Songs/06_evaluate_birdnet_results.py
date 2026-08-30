from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

# ============================ USER SETTINGS ============================
ROOT = Path(".")
RAW_RESULTS_ROOT = ROOT / "BirdNET_Results" / "01_raw"
ACTIVE_INTERVALS_CSV = ROOT / "bird_active_call_ranges.csv"
EVALUATED_DIR = ROOT / "BirdNET_Results" / "02_evaluated"
REPORT_DIR = ROOT / "BirdNET_Results" / "03_report_tables"

EXPERIMENTS = [
    "E1_no_metadata",
    "E2_week_only",
    "E3_location_only",
    "E4_week_and_location",
]
DISTANCE_ORDER = ["1ft", "4ft", "8ft", "12ft", "24ft", "36ft", "48ft"]
MIN_ACTIVE_OVERLAP_SECONDS = 0.25

ALIASES = {
    "Annas Hummingbird": "Anna's Hummingbird",
    "Bewicks Wren": "Bewick's Wren",
    "Dark eyed Junco": "Dark-eyed Junco",
}

BIRDNET_NAME_MAP = {
    # Dataset name: exact BirdNET common name

    "Northern House Wren": "House Wren",
    "Northern Yellow Warbler": "Yellow Warbler",

}
# =====================================================================


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).replace("’", "'").lower())

def expected_birdnet_name(dataset_name: str) -> str:
    return BIRDNET_NAME_MAP.get(dataset_name, dataset_name)

def parse_intervals(text: str) -> list[tuple[float, float]]:
    intervals = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", item)
        if not match:
            raise ValueError(f"Invalid active interval: {item}")
        start, end = float(match.group(1)), float(match.group(2))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid active interval: {item}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError("No active intervals found.")
    return intervals


def overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    total = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
    return min(total, max(0.0, end - start))


def choose_column(columns: pd.Index, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find {label} column. Available: {list(columns)}")


def distance_from_path(value: str) -> str:
    match = re.search(r"(?<!\d)(\d+)ft(?!\w)", str(value), flags=re.I)
    return f"{int(match.group(1))}ft" if match else ""


def expected_from_path(value: str) -> str:
    name = Path(str(value)).name
    name = re.sub(r"\.BirdNET\.results\.csv$", "", name, flags=re.I)
    name = re.sub(r"\.wav$", "", name, flags=re.I)
    name = re.sub(r"[_\-\s]*\d+ft.*$", "", name, flags=re.I)
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return ALIASES.get(name, name)


def species_name(row: pd.Series) -> str:
    for column in ["Common name", "Common Name", "common_name", "species_name"]:
        if column in row.index and pd.notna(row[column]):
            text = str(row[column]).strip()
            if column == "species_name" and "_" in text:
                return text.split("_", 1)[1]
            return text
    return ""


def load_active_lookup() -> dict[str, dict]:
    df = pd.read_csv(ACTIVE_INTERVALS_CSV)
    required = {"name", "active_intervals_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Active-call CSV missing: " + ", ".join(sorted(missing)))

    lookup = {}
    for _, row in df.iterrows():
        bird = str(row["name"]).strip()
        lookup[normalize(bird)] = {
            "bird": bird,
            "intervals": parse_intervals(row["active_intervals_seconds"]),
        }
    return lookup


def process_experiment(experiment: str, active_lookup: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    path = RAW_RESULTS_ROOT / experiment / "BirdNET_CombinedTable.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path.resolve()}")

    print(f"Reading {path}")
    df = pd.read_csv(path)
    file_col = choose_column(df.columns, ["File", "Begin Path", "file"], "file")
    start_col = choose_column(df.columns, ["Start (s)", "Begin Time (s)", "start_s"], "start")
    end_col = choose_column(df.columns, ["End (s)", "End Time (s)", "end_s"], "end")
    conf_col = choose_column(df.columns, ["Confidence", "confidence"], "confidence")

    df["confidence"] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0)
    df["start_s"] = pd.to_numeric(df[start_col], errors="coerce")
    df["end_s"] = pd.to_numeric(df[end_col], errors="coerce")
    df = df.dropna(subset=["start_s", "end_s"]).copy()
    df["species"] = df.apply(species_name, axis=1)
    df["source_path"] = df[file_col].astype(str)
    df["file"] = df["source_path"].map(lambda x: Path(x).name)
    df["expected"] = df["source_path"].map(expected_from_path)
    df["distance"] = df["source_path"].map(distance_from_path)

    window_rows = []
    group_cols = ["file", "source_path", "start_s", "end_s"]

    for (file_name, source_path, start_s, end_s), group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("confidence", ascending=False).reset_index(drop=True)
        expected_file = str(group["expected"].iloc[0]).strip()
        key = normalize(expected_file)
        if key not in active_lookup:
            raise KeyError(f"'{expected_file}' from {file_name} does not match the active-call CSV.")

        bird = active_lookup[key]["bird"]
        intervals = active_lookup[key]["intervals"]
        overlap = overlap_seconds(float(start_s), float(end_s), intervals)
        duration = max(0.0, float(end_s) - float(start_s))
        is_active = overlap >= MIN_ACTIVE_OVERLAP_SECONDS

        names = group["species"].astype(str).tolist()
        confidences = group["confidence"].astype(float).tolist()
        normalized_names = [normalize(x) for x in names]

        expected_species = expected_birdnet_name(bird)
        expected_norm = normalize(expected_species)

        expected_rank = math.nan
        expected_confidence = 0.0
        for rank, (candidate, confidence) in enumerate(zip(normalized_names, confidences), start=1):
            if candidate == expected_norm:
                expected_rank = rank
                expected_confidence = confidence
                break

        window_rows.append({
            "experiment": experiment,
            "file": file_name,
            "source_path": source_path,
            "bird": bird,
            "expected_birdnet_species": expected_species,
            "distance": group["distance"].iloc[0],
            "start_s": float(start_s),
            "end_s": float(end_s),
            "window_duration_s": duration,
            "active_overlap_seconds": overlap,
            "active_overlap_fraction": overlap / duration if duration > 0 else 0.0,
            "is_active_window": is_active,
            "top1_prediction": names[0] if names else "",
            "top1_confidence": confidences[0] if confidences else 0.0,
            "top2_prediction": names[1] if len(names) > 1 else "",
            "top2_confidence": confidences[1] if len(confidences) > 1 else 0.0,
            "top3_prediction": names[2] if len(names) > 2 else "",
            "top3_confidence": confidences[2] if len(confidences) > 2 else 0.0,
            "expected_detected": not math.isnan(expected_rank),
            "expected_rank": expected_rank,
            "expected_confidence": expected_confidence,
            "correct_top1": expected_norm in normalized_names[:1],
            "correct_top3": expected_norm in normalized_names[:3],
            "correct_top5": expected_norm in normalized_names[:5],
        })

    window_df = pd.DataFrame(window_rows)
    recording_rows = []

    for (file_name, bird, distance), windows in window_df.groupby(["file", "bird", "distance"], dropna=False):
        active = windows[windows["is_active_window"]].copy()
        if active.empty:
            recording_rows.append({
                "experiment": experiment,
                "file": file_name,
                "bird": bird,
                "distance": distance,
                "status": "ERROR",
                "error": "No BirdNET window met the active-overlap threshold.",
            })
            continue

        active_keys = set(zip(active["start_s"], active["end_s"]))
        candidates = df[
            (df["file"] == file_name)
            & df.apply(lambda row: (row["start_s"], row["end_s"]) in active_keys, axis=1)
        ].copy()

        expected_species = expected_birdnet_name(bird)
        expected_norm = normalize(expected_species)

        # MAX method: each species is represented by its strongest single
        # confidence from any active window.
        max_ranking = (
            candidates.groupby("species", as_index=False)
            .agg(confidence=("confidence", "max"))
            .sort_values(["confidence", "species"], ascending=[False, True])
            .reset_index(drop=True)
        )
        max_names = max_ranking["species"].astype(str).tolist()
        max_conf = max_ranking["confidence"].astype(float).tolist()
        max_norm = [normalize(x) for x in max_names]
        max_rank = next(
            (i for i, value in enumerate(max_norm, start=1) if value == expected_norm),
            math.nan,
        )

        # MEAN method: take the top 5 detections from each active window,
        # sum each species' confidence, then divide by ALL active windows.
        # Therefore, a species missing from a window contributes zero.
        top5_rows = []
        active_windows = active[["start_s", "end_s"]].drop_duplicates()
        for start, end in active_windows.itertuples(index=False, name=None):
            top5 = (
                candidates[(candidates["start_s"] == start) & (candidates["end_s"] == end)]
                .sort_values("confidence", ascending=False)
                .head(5)
            )
            top5_rows.extend(
                {"species": row["species"], "confidence": float(row["confidence"])}
                for _, row in top5.iterrows()
            )

        if top5_rows:
            mean_ranking = (
                pd.DataFrame(top5_rows)
                .groupby("species", as_index=False)
                .agg(confidence=("confidence", "sum"))
            )
            mean_ranking["confidence"] /= len(active_windows)
            mean_ranking = mean_ranking.sort_values(
                ["confidence", "species"], ascending=[False, True]
            ).reset_index(drop=True)
        else:
            mean_ranking = pd.DataFrame(columns=["species", "confidence"])

        mean_names = mean_ranking["species"].astype(str).tolist()
        mean_conf = mean_ranking["confidence"].astype(float).tolist()
        mean_norm = [normalize(x) for x in mean_names]
        mean_rank = next(
            (i for i, value in enumerate(mean_norm, start=1) if value == expected_norm),
            math.nan,
        )

        detected_count = int(active["expected_detected"].sum())
        active_count = len(active)
        recording_rows.append({
            "experiment": experiment,
            "file": file_name,
            "bird": bird,
            "expected_birdnet_species": expected_species,
            "distance": distance,
            "active_window_count": active_count,
            "expected_detected_active_windows": detected_count,
            "active_window_detection_rate": detected_count / active_count,

            # Mean-based whole-recording ranking: main consistency result.
            "mean_top1_prediction": mean_names[0] if mean_names else "",
            "mean_top1_confidence": mean_conf[0] if mean_conf else 0.0,
            "mean_top2_prediction": mean_names[1] if len(mean_names) > 1 else "",
            "mean_top2_confidence": mean_conf[1] if len(mean_conf) > 1 else 0.0,
            "mean_top3_prediction": mean_names[2] if len(mean_names) > 2 else "",
            "mean_top3_confidence": mean_conf[2] if len(mean_conf) > 2 else 0.0,
            "mean_correct_top1": expected_norm in mean_norm[:1],
            "mean_correct_top3": expected_norm in mean_norm[:3],
            "mean_correct_top5": expected_norm in mean_norm[:5],
            "mean_expected_rank": mean_rank,

            # Max-based whole-recording ranking: strongest single detection.
            "max_top1_prediction": max_names[0] if max_names else "",
            "max_top1_confidence": max_conf[0] if max_conf else 0.0,
            "max_top2_prediction": max_names[1] if len(max_names) > 1 else "",
            "max_top2_confidence": max_conf[1] if len(max_conf) > 1 else 0.0,
            "max_top3_prediction": max_names[2] if len(max_names) > 2 else "",
            "max_top3_confidence": max_conf[2] if len(max_conf) > 2 else 0.0,
            "max_correct_top1": expected_norm in max_norm[:1],
            "max_correct_top3": expected_norm in max_norm[:3],
            "max_correct_top5": expected_norm in max_norm[:5],
            "max_expected_rank": max_rank,

            "expected_detected": detected_count > 0,
            "expected_max_active_window_confidence": float(active["expected_confidence"].max()),
            # Missing expected species contributes zero to this mean.
            "expected_mean_active_window_confidence": float(active["expected_confidence"].mean()),
            "expected_median_active_window_confidence": float(active["expected_confidence"].median()),
            "window_top1_accuracy": float(active["correct_top1"].mean()),
            "window_top3_accuracy": float(active["correct_top3"].mean()),
            "window_top5_accuracy": float(active["correct_top5"].mean()),
            "status": "OK",
            "error": "",
        })

    return recording_rows, window_rows


def add_reference_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric = "expected_mean_active_window_confidence"
    refs = (
        df[df["distance"] == "1ft"][["experiment", "bird", metric]]
        .rename(columns={metric: "reference_1ft_confidence"})
    )
    if refs.duplicated(["experiment", "bird"], keep=False).any():
        raise ValueError("More than one 1-ft reference exists for an experiment/bird pair.")

    result = df.merge(refs, on=["experiment", "bird"], how="left", validate="many_to_one")
    result["confidence_change_from_1ft"] = result[metric] - result["reference_1ft_confidence"]
    result["confidence_retention_from_1ft"] = result.apply(
        lambda row: row[metric] / row["reference_1ft_confidence"]
        if pd.notna(row["reference_1ft_confidence"]) and row["reference_1ft_confidence"] > 0
        else math.nan,
        axis=1,
    )
    return result


def main() -> None:
    EVALUATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lookup = load_active_lookup()

    recording_rows, window_rows = [], []
    for experiment in EXPERIMENTS:
        recordings, windows = process_experiment(experiment, lookup)
        recording_rows.extend(recordings)
        window_rows.extend(windows)

    recording_df = add_reference_metrics(pd.DataFrame(recording_rows))
    window_df = pd.DataFrame(window_rows)
    for frame in [recording_df, window_df]:
        frame["distance"] = pd.Categorical(frame["distance"], categories=DISTANCE_ORDER, ordered=True)

    recording_df = recording_df.sort_values(["experiment", "distance", "bird"])
    window_df = window_df.sort_values(["experiment", "distance", "bird", "start_s"])
    recording_df.to_csv(EVALUATED_DIR / "birdnet_by_recording.csv", index=False)
    window_df.to_csv(EVALUATED_DIR / "birdnet_by_window.csv", index=False)

    valid = recording_df[recording_df["status"] == "OK"].copy()
    by_distance = (
        valid.groupby(["experiment", "distance"], observed=True)
        .agg(
            recording_count=("file", "count"),
            mean_top1_accuracy=("mean_correct_top1", "mean"),
            mean_top3_accuracy=("mean_correct_top3", "mean"),
            mean_top5_accuracy=("mean_correct_top5", "mean"),
            max_top1_accuracy=("max_correct_top1", "mean"),
            max_top3_accuracy=("max_correct_top3", "mean"),
            max_top5_accuracy=("max_correct_top5", "mean"),
            expected_detection_rate=("expected_detected", "mean"),
            mean_expected_confidence=("expected_mean_active_window_confidence", "mean"),
            median_expected_confidence=("expected_mean_active_window_confidence", "median"),
            mean_confidence_retention=("confidence_retention_from_1ft", "mean"),
            mean_active_window_detection_rate=("active_window_detection_rate", "mean"),
        ).reset_index()
    )
    by_species_distance = valid[[
        "experiment", "bird", "distance", "file",
        "mean_correct_top1", "mean_correct_top3", "mean_correct_top5", "mean_expected_rank",
        "max_correct_top1", "max_correct_top3", "max_correct_top5", "max_expected_rank",
        "expected_detected", "expected_max_active_window_confidence",
        "expected_mean_active_window_confidence", "active_window_detection_rate",
        "reference_1ft_confidence", "confidence_change_from_1ft", "confidence_retention_from_1ft",
    ]]
    metadata = (
        valid.groupby("experiment")
        .agg(
            recording_count=("file", "count"),
            mean_top1_accuracy=("mean_correct_top1", "mean"),
            mean_top3_accuracy=("mean_correct_top3", "mean"),
            mean_top5_accuracy=("mean_correct_top5", "mean"),
            max_top1_accuracy=("max_correct_top1", "mean"),
            max_top3_accuracy=("max_correct_top3", "mean"),
            max_top5_accuracy=("max_correct_top5", "mean"),
            expected_detection_rate=("expected_detected", "mean"),
            mean_expected_confidence=("expected_mean_active_window_confidence", "mean"),
            mean_confidence_retention=("confidence_retention_from_1ft", "mean"),
            mean_active_window_detection_rate=("active_window_detection_rate", "mean"),
        ).reset_index()
    )
    by_distance.to_csv(REPORT_DIR / "summary_by_distance.csv", index=False)
    by_species_distance.to_csv(REPORT_DIR / "summary_by_species_distance.csv", index=False)
    metadata.to_csv(REPORT_DIR / "metadata_comparison.csv", index=False)

    print("\nBIRDNET EVALUATION COMPLETE")
    print(f"Recording rows: {len(recording_df)}")
    print(f"Window rows: {len(window_df)}")
    print(f"Evaluated files: {EVALUATED_DIR.resolve()}")
    print(f"Report tables: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
