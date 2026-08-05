from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")
AUDIO_FOLDER = ROOT / "Trimmed"
FILTER_RESULTS_CSV = (
    ROOT / "Frequency Range Results" / "bird_frequency_ranges.csv"
)

WITH_HARMONICS_FOLDER = (
    ROOT 
    / "Filtered_With_Harmonics" 
    / "Audio"
)
WITHOUT_HARMONICS_FOLDER = (
    ROOT 
    / "Filtered_Without_Harmonics" 
    / "Audio"
)

WITH_HARMONICS_REPORT = (
    ROOT
    / "Filtered_With_Harmonics"
    / "filtered_with_harmonics_rms_report.csv"
)
WITHOUT_HARMONICS_REPORT = (
    ROOT
    / "Filtered_Without_Harmonics"
    / "filtered_without_harmonics_rms_report.csv"
)


# ============================================================
# FILTER SETTINGS
# ============================================================

FILTER_ORDER = 6
PEAK_HEADROOM_DB = 1.0
OUTPUT_SUBTYPE = "PCM_16"

LOW_COLUMN = "final_filter_low_hz"
HIGH_WITH_COLUMN = "final_filter_high_with_harmonics_hz"
HIGH_WITHOUT_COLUMN = "final_filter_high_without_harmonics_hz"
HARMONIC_FLAG_COLUMN = "harmonic_test_required"


# ============================================================
# HELPERS
# ============================================================

def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def find_audio_file(folder: Path, bird_name: str) -> Path:
    target = normalize_name(bird_name)

    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and normalize_name(path.name) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No WAV file found for '{bird_name}' in {folder.resolve()}"
        )

    raise ValueError(
        f"More than one WAV file matched '{bird_name}': "
        + ", ".join(path.name for path in matches)
    )


def parse_intervals(text: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []

    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*",
            part,
        )

        if not match:
            raise ValueError(
                f"Invalid interval '{part}'. "
                "Expected start-end;start-end"
            )

        start = float(match.group(1))
        end = float(match.group(2))

        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval '{part}'.")

        intervals.append((start, end))

    if not intervals:
        raise ValueError("No valid active intervals were provided.")

    return intervals


def load_audio_preserve_channels(
    path: Path,
) -> tuple[np.ndarray, int]:
    """
    Load WAV as floating-point audio while preserving mono/stereo layout.

    Returned shape:
      mono   -> (samples,)
      stereo -> (samples, channels)
    """
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{path.name} contains invalid samples.")

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"{path.name} has unsupported audio shape {audio.shape}."
        )

    return audio, int(sample_rate)


def get_filter_value(
    row: pd.Series,
    column: str,
) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None

    value = float(row[column])
    return value if value > 0 else None


def harmonic_test_required(row: pd.Series) -> bool:
    return (
        str(row.get(HARMONIC_FLAG_COLUMN, "no"))
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def apply_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float | None,
    high_hz: float | None,
) -> tuple[np.ndarray, str]:
    nyquist = sample_rate / 2.0

    if low_hz is not None and low_hz >= nyquist:
        raise ValueError(
            f"Low cutoff {low_hz:.1f} Hz must be below "
            f"Nyquist {nyquist:.1f} Hz."
        )

    if high_hz is not None:
        high_hz = min(high_hz, nyquist * 0.999)

    if low_hz is not None and high_hz is not None:
        if high_hz <= low_hz:
            raise ValueError(
                f"High cutoff {high_hz:.1f} Hz must exceed "
                f"low cutoff {low_hz:.1f} Hz."
            )

        sos = signal.butter(
            FILTER_ORDER,
            [low_hz, high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        filter_type = "bandpass"

    elif low_hz is not None:
        sos = signal.butter(
            FILTER_ORDER,
            low_hz,
            btype="highpass",
            fs=sample_rate,
            output="sos",
        )
        filter_type = "highpass"

    elif high_hz is not None:
        sos = signal.butter(
            FILTER_ORDER,
            high_hz,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        filter_type = "lowpass"

    else:
        return audio.copy(), "none"

    # axis=0 filters along time and preserves all channels.
    filtered = signal.sosfiltfilt(
        sos,
        audio,
        axis=0,
    )

    return filtered, filter_type


def extract_active_samples(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    duration = len(audio) / sample_rate
    pieces: list[np.ndarray] = []

    for start, end in intervals:
        if start >= duration:
            raise ValueError(
                f"Interval starts at {start:.3f}s, "
                f"but file is only {duration:.3f}s."
            )

        end = min(end, duration)

        start_sample = int(round(start * sample_rate))
        end_sample = int(round(end * sample_rate))

        piece = audio[start_sample:end_sample]

        if piece.shape[0] >= 2:
            pieces.append(piece)

    if not pieces:
        raise ValueError("No active samples were extracted.")

    return np.concatenate(pieces, axis=0)


def rms_dbfs(audio: np.ndarray) -> tuple[float, float]:
    """
    Compute RMS across all samples and channels.

    The audio files themselves remain mono/stereo; this is only one
    summary measurement for the branch report.
    """
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if rms <= 0:
        return 0.0, -math.inf

    return rms, 20.0 * math.log10(rms)


def peak_dbfs(audio: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(audio)))

    if peak <= 0:
        return 0.0, -math.inf

    return peak, 20.0 * math.log10(peak)


def process_branch(
    *,
    row: pd.Series,
    input_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    low_hz: float | None,
    high_hz: float | None,
    output_folder: Path,
    branch_name: str,
) -> dict[str, object]:
    filtered, filter_type = apply_filter(
        audio,
        sample_rate,
        low_hz,
        high_hz,
    )

    active = extract_active_samples(
        filtered,
        sample_rate,
        intervals,
    )

    rms, rms_db = rms_dbfs(active)
    peak, peak_db = peak_dbfs(filtered)

    maximum_safe_target = (
        -PEAK_HEADROOM_DB
        - (peak_db - rms_db)
    )

    output_path = output_folder / input_path.name

    sf.write(
        output_path,
        filtered,
        sample_rate,
        subtype=OUTPUT_SUBTYPE,
    )

    channels = 1 if filtered.ndim == 1 else filtered.shape[1]

    return {
        "name": str(row["name"]).strip(),
        "id": row.get("id", ""),
        "branch": branch_name,
        "harmonic_test_required": (
            "yes" if harmonic_test_required(row) else "no"
        ),
        "input_filename": input_path.name,
        "filtered_filename": output_path.name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "filter_type": filter_type,
        "filter_low_hz": low_hz,
        "filter_high_hz": high_hz,
        "active_rms_linear_filtered": rms,
        "active_rms_dbfs_filtered": rms_db,
        "full_file_peak_linear_filtered": peak,
        "full_file_peak_dbfs_filtered": peak_db,
        "maximum_safe_target_rms_dbfs": maximum_safe_target,
        "status": "OK",
        "error": "",
    }


def write_report(
    rows: list[dict[str, object]],
    report_path: Path,
) -> None:
    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)

    good = report[report["status"] == "OK"]

    print(f"  Report: {report_path.resolve()}")

    if not good.empty:
        common = float(
            good["maximum_safe_target_rms_dbfs"].min()
        )
        print(
            "  Highest safe common RMS target with "
            f"{PEAK_HEADROOM_DB:.1f} dB headroom: "
            f"{common:.2f} dBFS"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    WITH_HARMONICS_FOLDER.mkdir(parents=True, exist_ok=True)
    WITHOUT_HARMONICS_FOLDER.mkdir(parents=True, exist_ok=True)
    WITH_HARMONICS_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not AUDIO_FOLDER.exists():
        raise FileNotFoundError(
            f"Audio folder not found: {AUDIO_FOLDER.resolve()}"
        )

    if not FILTER_RESULTS_CSV.exists():
        raise FileNotFoundError(
            "Frequency-range CSV not found: "
            f"{FILTER_RESULTS_CSV.resolve()}"
        )

    filters = pd.read_csv(FILTER_RESULTS_CSV)

    required_columns = {
        "name",
        "id",
        "active_intervals_seconds",
        LOW_COLUMN,
        HIGH_WITHOUT_COLUMN,
        HIGH_WITH_COLUMN,
        HARMONIC_FLAG_COLUMN,
    }

    missing = required_columns - set(filters.columns)

    if missing:
        raise ValueError(
            "Frequency-range CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    with_rows: list[dict[str, object]] = []
    without_rows: list[dict[str, object]] = []

    print("=" * 78)
    print("FILTERING: SELECTED WITH-HARMONICS + ALL WITHOUT-HARMONICS")
    print(f"CSV rows: {len(filters)}")
    print("=" * 78)

    for index, (_, row) in enumerate(filters.iterrows(), start=1):
        name = str(row["name"]).strip()
        include_harmonic_test = harmonic_test_required(row)

        print(f"\n[{index:02d}/{len(filters)}] {name}")

        try:
            input_path = find_audio_file(AUDIO_FOLDER, name)

            audio, sample_rate = load_audio_preserve_channels(
                input_path
            )

            intervals = parse_intervals(
                row["active_intervals_seconds"]
            )

            low_hz = get_filter_value(row, LOW_COLUMN)
            without_high_hz = get_filter_value(
                row,
                HIGH_WITHOUT_COLUMN,
            )

            # The without-harmonics branch contains all 20 birds.
            without_result = process_branch(
                row=row,
                input_path=input_path,
                audio=audio,
                sample_rate=sample_rate,
                intervals=intervals,
                low_hz=low_hz,
                high_hz=without_high_hz,
                output_folder=WITHOUT_HARMONICS_FOLDER,
                branch_name="without_harmonics",
            )
            without_rows.append(without_result)

            print(
                "  Without harmonics: "
                f"{low_hz}-{without_high_hz} Hz "
                f"| RMS {without_result['active_rms_dbfs_filtered']:.2f} dBFS"
            )

            # The with-harmonics branch contains only birds selected
            # for the harmonic comparison.
            if include_harmonic_test:
                with_high_hz = get_filter_value(
                    row,
                    HIGH_WITH_COLUMN,
                )

                if with_high_hz is None:
                    raise ValueError(
                        f"{HIGH_WITH_COLUMN} is blank for a bird "
                        "marked for harmonic testing."
                    )

                with_result = process_branch(
                    row=row,
                    input_path=input_path,
                    audio=audio,
                    sample_rate=sample_rate,
                    intervals=intervals,
                    low_hz=low_hz,
                    high_hz=with_high_hz,
                    output_folder=WITH_HARMONICS_FOLDER,
                    branch_name="with_harmonics",
                )
                with_rows.append(with_result)

                print(
                    "  With harmonics:    "
                    f"{low_hz}-{with_high_hz} Hz "
                    f"| RMS {with_result['active_rms_dbfs_filtered']:.2f} dBFS"
                )
            else:
                print(
                    "  With harmonics:    skipped "
                    "(harmonic_test_required = no)"
                )

        except Exception as error:
            print(f"  ERROR: {error}")

            error_row = {
                "name": name,
                "id": row.get("id", ""),
                "status": "ERROR",
                "error": str(error),
            }

            if include_harmonic_test:
                with_rows.append(
                    {
                        **error_row,
                        "branch": "with_harmonics",
                    }
                )

            without_rows.append(
                {
                    **error_row,
                    "branch": "without_harmonics",
                }
            )

    print("\n" + "=" * 78)
    print("WITH-HARMONICS BRANCH")
    print(f"  Folder: {WITH_HARMONICS_FOLDER.resolve()}")
    write_report(with_rows, WITH_HARMONICS_REPORT)

    print("\nWITHOUT-HARMONICS BRANCH")
    print(f"  Folder: {WITHOUT_HARMONICS_FOLDER.resolve()}")
    write_report(without_rows, WITHOUT_HARMONICS_REPORT)

    with_ok = sum(row.get("status") == "OK" for row in with_rows)
    without_ok = sum(
        row.get("status") == "OK"
        for row in without_rows
    )

    print("\n" + "=" * 78)
    print("FILTERING COMPLETE")
    selected_harmonic_count = int(
        filters[HARMONIC_FLAG_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"yes", "y", "true", "1"})
        .sum()
    )

    print(
        f"With harmonics:    {with_ok}/{selected_harmonic_count} files"
    )
    print(
        f"Without harmonics: {without_ok}/{len(filters)} files"
    )
    print("=" * 78)



if __name__ == "__main__":
    main()
