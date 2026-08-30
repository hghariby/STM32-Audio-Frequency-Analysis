"""
Create one 7-panel spectrogram comparison for each bird.

Output:
    20 PNG files (one per bird)
    Each PNG contains 7 spectrograms:
        1 ft, 4 ft, 8 ft, 12 ft, 24 ft, 36 ft, 48 ft

Active-call intervals are highlighted on every panel.

Required packages:
    pip install numpy pandas scipy matplotlib soundfile

Expected metadata CSV:
    bird_active_call_ranges.csv

Required columns:
    name
    active_intervals_seconds

Example:
    name,active_intervals_seconds
    American Robin,0.40-2.10;3.25-4.80;6.10-8.30

The script searches recursively under AUDIO_ROOT for WAV files whose
file name or parent folder contains both the bird name and distance.

IMPORTANT:
    Use correctly aligned and split 10-second recordings.
    Do not use the currently incorrect 48-ft split files until 48-ft
    alignment has been corrected.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


# ============================================================================
# USER SETTINGS
# ============================================================================
ROOT = Path(".")
AUDIO_ROOT = ROOT / 'Birdsongs Recordings'

ACTIVE_INTERVALS_CSV = ROOT / 'bird_active_call_ranges.csv'

OUTPUT_FOLDER = ROOT / 'Spectrogram_7_Distance_Comparison'


DISTANCES_FT: Sequence[int] = (1, 4, 8, 12, 24, 36)

MAX_FREQUENCY_HZ = 15000
NPERSEG = 1024
OVERLAP_FRACTION = 0.75
WINDOW = "hann"

FIGURE_WIDTH_INCHES = 16
PANEL_HEIGHT_INCHES = 2.15
IMAGE_DPI = 180
DYNAMIC_RANGE_DB = 80.0

ACTIVE_HIGHLIGHT_COLOR = "lime"
ACTIVE_HIGHLIGHT_ALPHA = 0.14
ACTIVE_BOUNDARY_ALPHA = 0.75


# ============================================================================
# HELPERS
# ============================================================================

def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def safe_filename(text: str) -> str:
    text = text.replace("’", "'")
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    return re.sub(r"\s+", "_", text.strip())


def merge_intervals(
    intervals: Iterable[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []

    merged: List[List[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def parse_active_intervals(value: object) -> List[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip().replace("–", "-").replace("—", "-")
    if not text:
        return []

    intervals: List[Tuple[float, float]] = []
    for piece in re.split(r"[;,]+", text):
        piece = piece.strip()
        if not piece:
            continue

        match = re.fullmatch(
            r"\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*",
            piece,
        )
        if match is None:
            raise ValueError(
                f"Could not parse active interval {piece!r} from {value!r}."
            )

        start = float(match.group(1))
        end = float(match.group(2))
        if end <= start:
            raise ValueError(f"Invalid interval {piece!r}: end must be after start.")

        intervals.append((start, end))

    return merge_intervals(intervals)


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1:
        raise ValueError(f"Unsupported audio shape {audio.shape} in {path}")
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    return audio, int(sample_rate)


def path_contains_distance(path: Path, distance_ft: int) -> bool:
    text = str(path).replace("_", " ")

    direct_pattern = re.compile(
        rf"(?<!\d){distance_ft}\s*(?:ft|feet)(?!\d)",
        flags=re.IGNORECASE,
    )
    if direct_pattern.search(text):
        return True

    sequence_index = DISTANCES_FT.index(distance_ft) + 1
    folder_pattern = re.compile(
        rf"(?<!\d)d0?{sequence_index}(?:\s*{distance_ft}\s*ft)?(?!\d)",
        flags=re.IGNORECASE,
    )
    return folder_pattern.search(text) is not None


def bird_aliases(name: str) -> List[str]:
    aliases = {
        normalize_text(name),
        normalize_text(name.replace("'s", "s")),
        normalize_text(name.replace("'s", "")),
        normalize_text(name.replace("-", " ")),
    }
    return sorted(alias for alias in aliases if alias)


def path_contains_bird(path: Path, bird_name: str) -> bool:
    normalized_path = normalize_text(str(path))
    return any(alias in normalized_path for alias in bird_aliases(bird_name))


def build_wav_index(audio_root: Path) -> List[Path]:
    return sorted(
        path for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def find_audio_file(
    wav_files: Sequence[Path],
    bird_name: str,
    distance_ft: int,
) -> Optional[Path]:
    matches = [
        path for path in wav_files
        if path_contains_bird(path, bird_name)
        and path_contains_distance(path, distance_ft)
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None

    direct_matches = [
        path for path in matches
        if path_contains_bird(Path(path.stem), bird_name)
        and path_contains_distance(Path(path.stem), distance_ft)
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]

    formatted = "\n".join(f"    {path}" for path in matches)
    raise RuntimeError(
        f"Multiple files matched {bird_name!r} at {distance_ft} ft:\n"
        f"{formatted}\n"
        "Rename the files or remove old outputs from AUDIO_ROOT."
    )


def compute_spectrogram_db(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(NPERSEG, len(audio))
    noverlap = min(int(round(nperseg * OVERLAP_FRACTION)), nperseg - 1)

    frequencies, times, magnitude = signal.spectrogram(
        audio,
        fs=sample_rate,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )

    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, np.finfo(np.float64).tiny)
    )

    frequency_mask = frequencies <= min(MAX_FREQUENCY_HZ, sample_rate / 2.0)
    return frequencies[frequency_mask], times, magnitude_db[frequency_mask, :]


def load_metadata(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    required_columns = {"name", "active_intervals_seconds"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()
    dataframe["name"] = dataframe["name"].astype(str).str.strip()

    if dataframe["name"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["name"].duplicated(keep=False), "name"
        ].tolist()
        raise ValueError(f"Duplicate bird names in metadata CSV: {duplicates}")

    return dataframe


# ============================================================================
# PLOTTING
# ============================================================================

def create_bird_comparison(
    bird_name: str,
    active_intervals: List[Tuple[float, float]],
    distance_files: Dict[int, Path],
    output_path: Path,
) -> Dict[str, object]:
    spectrograms = {}
    global_max_db = -np.inf
    maximum_duration = 0.0
    minimum_nyquist = float("inf")

    for distance_ft in DISTANCES_FT:
        audio_path = distance_files[distance_ft]
        audio, sample_rate = load_audio_mono(audio_path)
        frequencies, times, magnitude_db = compute_spectrogram_db(audio, sample_rate)

        duration_seconds = len(audio) / sample_rate
        maximum_duration = max(maximum_duration, duration_seconds)
        minimum_nyquist = min(minimum_nyquist, sample_rate / 2.0)
        global_max_db = max(global_max_db, float(np.max(magnitude_db)))

        spectrograms[distance_ft] = (
            frequencies,
            times,
            magnitude_db,
            duration_seconds,
            sample_rate,
            audio_path,
        )

    vmax = global_max_db
    vmin = vmax - DYNAMIC_RANGE_DB
    common_max_frequency = min(MAX_FREQUENCY_HZ, minimum_nyquist)

    figure_height = PANEL_HEIGHT_INCHES * len(DISTANCES_FT)
    figure, axes = plt.subplots(
        nrows=len(DISTANCES_FT),
        ncols=1,
        figsize=(FIGURE_WIDTH_INCHES, figure_height),
        sharex=True,
        sharey=True,
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.89,
        top=0.945,
        bottom=0.06,
        hspace=0.12,
    )

    image = None
    for axis, distance_ft in zip(axes, DISTANCES_FT):
        frequencies, times, magnitude_db, duration_seconds, sample_rate, _ = (
            spectrograms[distance_ft]
        )

        image = axis.pcolormesh(
            times,
            frequencies,
            magnitude_db,
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        for start, end in active_intervals:
            clipped_start = max(0.0, start)
            clipped_end = min(duration_seconds, end)
            if clipped_end <= clipped_start:
                continue

            axis.axvspan(
                clipped_start,
                clipped_end,
                facecolor=ACTIVE_HIGHLIGHT_COLOR,
                alpha=ACTIVE_HIGHLIGHT_ALPHA,
                linewidth=0,
            )
            axis.axvline(
                clipped_start,
                color=ACTIVE_HIGHLIGHT_COLOR,
                linewidth=0.8,
                alpha=ACTIVE_BOUNDARY_ALPHA,
            )
            axis.axvline(
                clipped_end,
                color=ACTIVE_HIGHLIGHT_COLOR,
                linewidth=0.8,
                alpha=ACTIVE_BOUNDARY_ALPHA,
            )

        axis.set_ylabel(f"{distance_ft} ft\nFrequency (Hz)", fontsize=8.5)
        axis.set_ylim(0, common_max_frequency)
        axis.grid(False)

        axis.text(
            0.995,
            0.93,
            f"{sample_rate:,} Hz",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="white",
            bbox={
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
                "pad": 2,
            },
        )

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xlim(0, maximum_duration)

    interval_text = (
        "; ".join(f"{start:.2f}–{end:.2f} s" for start, end in active_intervals)
        if active_intervals
        else "No active intervals found"
    )

    figure.suptitle(
        f"{bird_name}: spectrogram comparison across 7 distances\n"
        f"Active calls highlighted: {interval_text}",
        fontsize=14,
        fontweight="bold",
    )

    if image is not None:
        colorbar_axis = figure.add_axes([0.91, 0.12, 0.018, 0.75])
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Magnitude (dB, common scale for this bird)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(figure)

    return {
        "bird": bird_name,
        "output_file": str(output_path),
        "active_intervals": interval_text,
        "status": "OK",
        "error": "",
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("7-DISTANCE BIRD SPECTROGRAM COMPARISON")
    print("=" * 78)

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"AUDIO_ROOT does not exist:\n{AUDIO_ROOT}\n"
            "Update AUDIO_ROOT near the top of the script."
        )

    if not ACTIVE_INTERVALS_CSV.exists():
        raise FileNotFoundError(
            f"Active-interval CSV does not exist:\n{ACTIVE_INTERVALS_CSV}"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(ACTIVE_INTERVALS_CSV)
    wav_files = build_wav_index(AUDIO_ROOT)

    if not wav_files:
        raise FileNotFoundError(f"No WAV files were found under:\n{AUDIO_ROOT}")

    print(f"Metadata birds: {len(metadata)}")
    print(f"WAV files found: {len(wav_files)}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print()

    missing_rows: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    file_map: Dict[str, Dict[int, Path]] = {}

    for _, row in metadata.iterrows():
        bird_name = row["name"]
        file_map[bird_name] = {}

        for distance_ft in DISTANCES_FT:
            audio_path = find_audio_file(wav_files, bird_name, distance_ft)
            if audio_path is None:
                missing_rows.append({
                    "bird": bird_name,
                    "distance_ft": distance_ft,
                    "status": "MISSING",
                })
            else:
                file_map[bird_name][distance_ft] = audio_path

    missing_report_path = OUTPUT_FOLDER / "missing_audio_files.csv"
    pd.DataFrame(
        missing_rows,
        columns=["bird", "distance_ft", "status"],
    ).to_csv(missing_report_path, index=False)

    if missing_rows:
        print("Missing bird-distance recordings were found.")
        print(f"See: {missing_report_path}")
        print("Figures are generated only for birds that have all seven files.")
        print()

    for bird_number, (_, row) in enumerate(metadata.iterrows(), start=1):
        bird_name = row["name"]
        distance_files = file_map[bird_name]

        if len(distance_files) != len(DISTANCES_FT):
            missing_distances = [
                distance for distance in DISTANCES_FT
                if distance not in distance_files
            ]
            message = f"Missing distances: {missing_distances}"
            print(f"[SKIP] {bird_name}: {message}")
            results.append({
                "bird": bird_name,
                "output_file": "",
                "active_intervals": "",
                "status": "SKIPPED",
                "error": message,
            })
            continue

        try:
            intervals = parse_active_intervals(row["active_intervals_seconds"])
            output_path = (
                OUTPUT_FOLDER / f"{safe_filename(bird_name)}_7_distances.png"
            )

            print(f"[{bird_number:02d}/{len(metadata):02d}] Creating {bird_name}")

            result = create_bird_comparison(
                bird_name=bird_name,
                active_intervals=intervals,
                distance_files=distance_files,
                output_path=output_path,
            )
            results.append(result)

        except Exception as exc:
            print(f"[ERROR] {bird_name}: {exc}")
            results.append({
                "bird": bird_name,
                "output_file": "",
                "active_intervals": "",
                "status": "ERROR",
                "error": str(exc),
            })

    report_path = OUTPUT_FOLDER / "spectrogram_generation_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    successful = sum(result["status"] == "OK" for result in results)
    skipped = sum(result["status"] == "SKIPPED" for result in results)
    errors = sum(result["status"] == "ERROR" for result in results)

    print()
    print("=" * 78)
    print("COMPLETE")
    print("=" * 78)
    print(f"Figures created: {successful}")
    print(f"Birds skipped:   {skipped}")
    print(f"Errors:          {errors}")
    print(f"Report:          {report_path}")
    print(f"Figures folder:  {OUTPUT_FOLDER}")

    if successful != 20:
        print()
        print(
            "Expected 20 completed figures. Review missing_audio_files.csv "
            "and spectrogram_generation_report.csv."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
