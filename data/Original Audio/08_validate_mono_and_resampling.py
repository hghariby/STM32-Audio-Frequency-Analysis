from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

METADATA_CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

CHANNEL_SELECTION_CSV = (
    ROOT
    / "Frequency Range Results"
    / "mono_channel_selection_report.csv"
)

OUTPUT_REPORT_CSV = (
    ROOT
    / "Frequency Range Results"
    / "mono_resampling_validation_report.csv"
)

BRANCHES = {
    "without_harmonics": {
        "before_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Audio"
        ),
        "after_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Mono_Audio"
        ),
    },
    "with_harmonics": {
        "before_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Audio"
        ),
        "after_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Mono_Audio"
        ),
    },
}


# ============================================================
# EXPECTED OUTPUT FORMAT
# ============================================================

EXPECTED_SAMPLE_RATE_HZ = 44100
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"
EXPECTED_FORMAT = "WAV"

RMS_CHANGE_TOLERANCE_DB = 0.10
DURATION_TOLERANCE_SECONDS = 0.001
INACTIVE_ZERO_TOLERANCE = 0.0


# ============================================================
# HELPERS
# ============================================================

def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def harmonic_test_required(row: pd.Series) -> bool:
    return (
        str(row.get("harmonic_test_required", "no"))
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def find_wav(folder: Path, bird_name: str) -> Path:
    target = normalize_name(bird_name)

    matches = [
        path
        for path in folder.glob("*.wav")
        if normalize_name(path.name) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No WAV found for '{bird_name}' in {folder.resolve()}"
        )

    raise ValueError(
        f"Multiple WAV files matched '{bird_name}' in {folder.resolve()}: "
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
            raise ValueError(f"Invalid active interval: {part}")

        start, end = map(float, match.groups())

        if start < 0 or end <= start:
            raise ValueError(f"Invalid active interval: {part}")

        intervals.append((start, end))

    if not intervals:
        raise ValueError("No active intervals were found.")

    return intervals


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        always_2d=False,
        dtype="float64",
    )

    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"{path.name} has unsupported audio shape {audio.shape}."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"{path.name} contains NaN or infinite samples."
        )

    return audio, int(sample_rate)


def channel_count(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def select_expected_mono(
    audio: np.ndarray,
    selected_channel: str,
) -> np.ndarray:
    """
    Reconstruct the mono signal that the conversion script intended to use
    before resampling.
    """
    if audio.ndim == 1:
        return audio.copy()

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            f"Only mono or stereo input is supported; got shape {audio.shape}."
        )

    if selected_channel == "left":
        return audio[:, 0].copy()

    if selected_channel == "right":
        return audio[:, 1].copy()

    if selected_channel == "middle":
        return np.mean(audio, axis=1)

    if selected_channel == "already_mono":
        raise ValueError(
            "Channel report says already_mono, but the silenced input is stereo."
        )

    raise ValueError(
        f"Unsupported selected_channel value: {selected_channel}"
    )


def make_time_mask(
    frame_count: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    duration = frame_count / sample_rate

    for start, end in intervals:
        if start >= duration:
            continue

        clipped_end = min(end, duration)

        start_sample = max(
            0,
            min(frame_count, int(round(start * sample_rate))),
        )

        end_sample = max(
            0,
            min(frame_count, int(round(clipped_end * sample_rate))),
        )

        mask[start_sample:end_sample] = True

    if not np.any(mask):
        raise ValueError(
            "No active intervals overlap the audio duration."
        )

    return mask


def active_rms_dbfs(
    mono_audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> float:
    mask = make_time_mask(
        len(mono_audio),
        sample_rate,
        intervals,
    )

    active = mono_audio[mask]
    rms = float(np.sqrt(np.mean(np.square(active))))

    if rms <= 0:
        return float("-inf")

    return float(20.0 * math.log10(rms))


def inactive_max_abs(
    mono_audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> float:
    mask = make_time_mask(
        len(mono_audio),
        sample_rate,
        intervals,
    )

    inactive = mono_audio[~mask]

    if inactive.size == 0:
        return 0.0

    return float(np.max(np.abs(inactive)))


def determine_status(
    *,
    output_sample_rate: int,
    output_channels: int,
    output_subtype: str,
    output_format: str,
    duration_difference_seconds: float,
    active_rms_change_db: float,
) -> tuple[str, str]:
    errors: list[str] = []

    if output_sample_rate != EXPECTED_SAMPLE_RATE_HZ:
        errors.append(
            f"output sample rate is {output_sample_rate}, "
            f"expected {EXPECTED_SAMPLE_RATE_HZ}"
        )

    if output_subtype != EXPECTED_SUBTYPE:
        errors.append(
            f"output subtype is {output_subtype}, "
            f"expected {EXPECTED_SUBTYPE}"
        )

    if output_format != EXPECTED_FORMAT:
        errors.append(
            f"output format is {output_format}, "
            f"expected {EXPECTED_FORMAT}"
        )

    if abs(duration_difference_seconds) > DURATION_TOLERANCE_SECONDS:
        errors.append(
            f"duration changed by {duration_difference_seconds:.9f} s"
        )

    if not np.isfinite(active_rms_change_db):
        errors.append("active RMS change is invalid")
    elif abs(active_rms_change_db) > RMS_CHANGE_TOLERANCE_DB:
        errors.append(
            f"active RMS changed by {active_rms_change_db:.4f} dB, "
            f"exceeding ±{RMS_CHANGE_TOLERANCE_DB:.2f} dB"
        )

    if errors:
        return "FAIL", "; ".join(errors)

    return "OK", ""


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(METADATA_CSV)

    if not CHANNEL_SELECTION_CSV.exists():
        raise FileNotFoundError(CHANNEL_SELECTION_CSV)

    metadata = pd.read_csv(METADATA_CSV)
    choices_report = pd.read_csv(CHANNEL_SELECTION_CSV)

    metadata_required = {
        "name",
        "active_intervals_seconds",
        "harmonic_test_required",
    }

    choices_required = {
        "name",
        "selected_channel",
        "status",
    }

    missing_metadata = metadata_required - set(metadata.columns)
    missing_choices = choices_required - set(choices_report.columns)

    if missing_metadata:
        raise ValueError(
            "Metadata CSV is missing columns: "
            + ", ".join(sorted(missing_metadata))
        )

    if missing_choices:
        raise ValueError(
            "Channel-selection CSV is missing columns: "
            + ", ".join(sorted(missing_choices))
        )

    valid_choices = choices_report[
        choices_report["status"].astype(str).str.upper() == "OK"
    ].copy()

    selected_channels = {
        normalize_name(row["name"]): str(row["selected_channel"]).strip()
        for _, row in valid_choices.iterrows()
    }

    for paths in BRANCHES.values():
        if not paths["before_folder"].exists():
            raise FileNotFoundError(paths["before_folder"])

        if not paths["after_folder"].exists():
            raise FileNotFoundError(paths["after_folder"])

    rows: list[dict[str, object]] = []

    print("=" * 80)
    print("MONO AND SAMPLE-RATE CONVERSION VALIDATION")
    print("=" * 80)

    for index, (_, metadata_row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):
        bird_name = str(metadata_row["name"]).strip()
        bird_key = normalize_name(bird_name)

        branches = ["without_harmonics"]

        if harmonic_test_required(metadata_row):
            branches.append("with_harmonics")

        print(f"\n[{index:02d}/{len(metadata)}] {bird_name}")

        for branch in branches:
            report_row: dict[str, object] = {
                "name": bird_name,
                "branch": branch,
            }

            try:
                if bird_key not in selected_channels:
                    raise ValueError(
                        "No valid selected_channel was found in "
                        "mono_channel_selection_report.csv."
                    )

                selected_channel = selected_channels[bird_key]

                before_path = find_wav(
                    BRANCHES[branch]["before_folder"],
                    bird_name,
                )

                after_path = find_wav(
                    BRANCHES[branch]["after_folder"],
                    bird_name,
                )

                before_audio, before_sr = load_audio(before_path)
                after_audio, after_sr = load_audio(after_path)

                after_info = sf.info(after_path)

                if after_audio.ndim != 1:
                    raise ValueError(
                        f"Output is not mono; shape is {after_audio.shape}."
                    )

                expected_before_mono = select_expected_mono(
                    before_audio,
                    selected_channel,
                )

                intervals = parse_intervals(
                    metadata_row["active_intervals_seconds"]
                )

                duration_before = len(expected_before_mono) / before_sr
                duration_after = len(after_audio) / after_sr
                duration_change = duration_after - duration_before

                rms_before = active_rms_dbfs(
                    expected_before_mono,
                    before_sr,
                    intervals,
                )

                rms_after = active_rms_dbfs(
                    after_audio,
                    after_sr,
                    intervals,
                )

                rms_change = rms_after - rms_before

                inactive_after = inactive_max_abs(
                    after_audio,
                    after_sr,
                    intervals,
                )

                status, error = determine_status(
                    output_sample_rate=after_sr,
                    output_channels=channel_count(after_audio),
                    output_subtype=after_info.subtype,
                    output_format=after_info.format,
                    duration_difference_seconds=duration_change,
                    active_rms_change_db=rms_change,
                )

                report_row.update({
                    "selected_channel": selected_channel,
                    "input_sample_rate_hz": before_sr,
                    "output_sample_rate_hz": after_sr,
                    "input_channels": channel_count(before_audio),
                    "output_channels": channel_count(after_audio),
                    "output_format": after_info.format,
                    "output_subtype": after_info.subtype,
                    "duration_before_seconds": duration_before,
                    "duration_after_seconds": duration_after,
                    "duration_change_seconds": duration_change,
                    "active_rms_before_dbfs": rms_before,
                    "active_rms_after_dbfs": rms_after,
                    "active_rms_change_db": rms_change,
                    "status": status,
                    "error": error,
                })

                print(
                    f"  {branch}: {status}, "
                    f"{before_sr} -> {after_sr} Hz, "
                    f"RMS change {rms_change:.4f} dB"
                )

            except Exception as exception:
                report_row.update({
                    "selected_channel": selected_channels.get(
                        bird_key,
                        "",
                    ),
                    "input_sample_rate_hz": np.nan,
                    "output_sample_rate_hz": np.nan,
                    "input_channels": np.nan,
                    "output_channels": np.nan,
                    "output_format": "",
                    "output_subtype": "",
                    "duration_before_seconds": np.nan,
                    "duration_after_seconds": np.nan,
                    "duration_change_seconds": np.nan,
                    "active_rms_before_dbfs": np.nan,
                    "active_rms_after_dbfs": np.nan,
                    "active_rms_change_db": np.nan,
                    "status": "ERROR",
                    "error": str(exception),
                })

                print(
                    f"  {branch}: ERROR — {exception}"
                )

            rows.append(report_row)

    report_columns = [
        "name",
        "branch",
        "selected_channel",
        "input_sample_rate_hz",
        "output_sample_rate_hz",
        "input_channels",
        "output_channels",
        "output_format",
        "output_subtype",
        "duration_before_seconds",
        "duration_after_seconds",
        "duration_change_seconds",
        "active_rms_before_dbfs",
        "active_rms_after_dbfs",
        "active_rms_change_db",
        "status",
        "error",
    ]

    report = pd.DataFrame(
        rows,
        columns=report_columns,
    )

    OUTPUT_REPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_REPORT_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print(f"Report: {OUTPUT_REPORT_CSV.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
