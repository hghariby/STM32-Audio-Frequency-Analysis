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

TRIMMED_FOLDER = ROOT / "Trimmed"

BRANCHES = {
    "without_harmonics": {
        "filtered_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Audio"
        ),
        "mono_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Mono_Audio"
        ),
        "silenced_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Silenced_Audio"
        ),
        "normalized_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Normalized_Audio"
        ),
    },
    "with_harmonics": {
        "filtered_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Audio"
        ),
        "mono_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Mono_Audio"
        ),
        "silenced_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Silenced_Audio"
        ),
        "normalized_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Normalized_Audio"
        ),
    },
}

OUTPUT_REPORT = (
    ROOT
    / "Frequency Range Results"
    / "all_stage_audio_comparison_report.csv"
)


# ============================================================
# EXPECTED FINAL CONDITIONS
# ============================================================

TARGET_SAMPLE_RATE_HZ = 44100
TARGET_NORMALIZED_RMS_DBFS = -25.0
MAX_NORMALIZED_PEAK_DBFS = -1.0

MONO_RMS_CHANGE_TOLERANCE_DB = 0.10
SILENCING_RMS_CHANGE_TOLERANCE_DB = 0.10
NORMALIZED_RMS_TOLERANCE_DB = 0.10
DURATION_TOLERANCE_SECONDS = 0.001
INACTIVE_ZERO_TOLERANCE = 0.0


# ============================================================
# GENERAL HELPERS
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
        f"Multiple WAV files matched '{bird_name}' in "
        f"{folder.resolve()}: "
        + ", ".join(path.name for path in matches)
    )


def parse_intervals(text: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []

    for part in str(text).split(";"):
        part = part.strip()

        if not part:
            continue

        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*-\s*"
            r"(\d+(?:\.\d+)?)\s*",
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


def load_channel_choices(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Channel-selection report not found: {path.resolve()}"
        )

    report = pd.read_csv(path)

    required = {"name", "selected_channel", "status"}
    missing = required - set(report.columns)

    if missing:
        raise ValueError(
            "Channel-selection report is missing columns: "
            + ", ".join(sorted(missing))
        )

    valid = report[
        report["status"].astype(str).str.upper() == "OK"
    ].copy()

    choices: dict[str, str] = {}

    for _, row in valid.iterrows():
        name = normalize_name(row["name"])
        choice = str(row["selected_channel"]).strip().lower()

        # The conversion script used "middle" for channel averaging
        # and "already_mono" for source files that were already mono.
        if choice == "average":
            choice = "middle"
        elif choice == "mono":
            choice = "already_mono"

        if choice not in {
            "left",
            "right",
            "middle",
            "already_mono",
        }:
            raise ValueError(
                f"Unsupported selected_channel '{choice}' for "
                f"{row['name']}."
            )

        choices[name] = choice

    return choices


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio(
    path: Path,
) -> tuple[np.ndarray, int, str, str]:
    info = sf.info(path)

    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"{path.name} has unsupported shape {audio.shape}."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"{path.name} contains NaN or infinite samples."
        )

    return (
        audio,
        int(sample_rate),
        str(info.format),
        str(info.subtype),
    )


def channel_count(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def select_expected_source(
    audio: np.ndarray,
    selected_channel: str,
) -> np.ndarray:
    """
    Return the exact signal definition used to create the mono file.

    For an already-mono input, the file itself is returned.

    For stereo:
      left   -> channel 0
      right  -> channel 1
      middle -> arithmetic average of left and right
    """
    choice = selected_channel.strip().lower()

    if audio.ndim == 1:
        if choice not in {
            "already_mono",
            "left",
            "right",
            "middle",
        }:
            raise ValueError(
                f"Unsupported choice '{selected_channel}' for mono audio."
            )

        return audio.copy()

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            "Channel selection supports only mono or stereo audio; "
            f"received shape {audio.shape}."
        )

    if choice == "left":
        return audio[:, 0].copy()

    if choice == "right":
        return audio[:, 1].copy()

    if choice == "middle":
        return np.mean(audio, axis=1)

    if choice == "already_mono":
        raise ValueError(
            "The source file is stereo, but the channel-selection report "
            "says already_mono."
        )

    raise ValueError(
        f"Unsupported selected_channel '{selected_channel}'."
    )


def active_sample_mask(
    frame_count: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(
        frame_count,
        dtype=bool,
    )

    duration = frame_count / sample_rate

    for start, end in intervals:
        if start >= duration:
            continue

        clipped_end = min(end, duration)

        start_sample = max(
            0,
            min(
                frame_count,
                int(round(start * sample_rate)),
            ),
        )

        end_sample = max(
            0,
            min(
                frame_count,
                int(round(clipped_end * sample_rate)),
            ),
        )

        if end_sample > start_sample:
            mask[start_sample:end_sample] = True

    if not np.any(mask):
        raise ValueError(
            "No active samples overlapped the audio."
        )

    return mask


def rms_dbfs(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")

    rms = float(
        np.sqrt(
            np.mean(
                np.square(values)
            )
        )
    )

    if rms <= 0:
        return float("-inf")

    return float(
        20.0 * math.log10(rms)
    )


def peak_dbfs(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")

    peak = float(
        np.max(
            np.abs(values)
        )
    )

    if peak <= 0:
        return float("-inf")

    return float(
        20.0 * math.log10(peak)
    )


def inactive_max_abs(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> float:
    active_mask = active_sample_mask(
        len(audio),
        sample_rate,
        intervals,
    )

    inactive_mask = ~active_mask

    if not np.any(inactive_mask):
        return 0.0

    inactive = audio[inactive_mask]

    if inactive.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(inactive)
        )
    )


def measure_stage(
    path: Path,
    intervals: list[tuple[float, float]],
    *,
    selected_channel: str | None,
    check_inactive: bool,
) -> dict[str, object]:
    """
    For Trimmed and Filtered stages, selected_channel is supplied and
    measurements are made from the exact selected source.

    For Mono, Silenced, and Normalized stages, selected_channel is None
    because those files are already mono.
    """
    audio, sample_rate, audio_format, subtype = load_audio(path)
    original_channels = channel_count(audio)

    if selected_channel is None:
        if audio.ndim != 1:
            raise ValueError(
                f"{path.name} should be mono but has shape {audio.shape}."
            )

        comparison_audio = audio

    else:
        comparison_audio = select_expected_source(
            audio,
            selected_channel,
        )

    active_mask = active_sample_mask(
        len(comparison_audio),
        sample_rate,
        intervals,
    )

    active_values = comparison_audio[active_mask]

    active_rms = rms_dbfs(active_values)
    full_peak = peak_dbfs(comparison_audio)

    duration = len(comparison_audio) / sample_rate
    active_frame_count = int(np.sum(active_mask))
    active_duration_seconds = (
        active_frame_count / sample_rate
    )
    active_fraction_percent = (
        100.0
        * active_frame_count
        / len(comparison_audio)
    )

    crest_factor = (
        full_peak - active_rms
        if (
            np.isfinite(full_peak)
            and np.isfinite(active_rms)
        )
        else np.nan
    )

    inactive_max = (
        inactive_max_abs(
            comparison_audio,
            sample_rate,
            intervals,
        )
        if check_inactive
        else np.nan
    )

    return {
        "path": str(path),
        "sample_rate_hz": sample_rate,
        "original_channels": original_channels,
        "comparison_channels": 1,
        "format": audio_format,
        "subtype": subtype,
        "duration_seconds": duration,
        "active_frame_count": active_frame_count,
        "active_duration_seconds": active_duration_seconds,
        "active_fraction_percent": active_fraction_percent,
        "active_rms_dbfs": active_rms,
        "full_file_peak_dbfs": full_peak,
        "crest_factor_db": crest_factor,
        "inactive_max_abs": inactive_max,
    }


# ============================================================
# COMPARISON HELPERS
# ============================================================

def level_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return np.nan

    return float(after - before)


def determine_status(
    *,
    mono: dict[str, object],
    silenced: dict[str, object],
    normalized: dict[str, object],
    durations: list[float],
    rms_mono_change_db: float,
    rms_silencing_change_db: float,
) -> tuple[str, str]:
    errors: list[str] = []

    if int(mono["sample_rate_hz"]) != TARGET_SAMPLE_RATE_HZ:
        errors.append(
            f"mono sample rate is not {TARGET_SAMPLE_RATE_HZ} Hz"
        )

    for stage_name, stage in (
        ("mono", mono),
        ("silenced", silenced),
        ("normalized", normalized),
    ):
        if int(stage["original_channels"]) != 1:
            errors.append(f"{stage_name} stage is not mono")

    if not np.isfinite(rms_mono_change_db):
        errors.append("mono/resampling RMS change is invalid")
    elif (
        abs(rms_mono_change_db)
        > MONO_RMS_CHANGE_TOLERANCE_DB
    ):
        errors.append(
            "mono/resampling RMS changed by "
            f"{rms_mono_change_db:.4f} dB"
        )

    if not np.isfinite(rms_silencing_change_db):
        errors.append("silencing RMS change is invalid")
    elif (
        abs(rms_silencing_change_db)
        > SILENCING_RMS_CHANGE_TOLERANCE_DB
    ):
        errors.append(
            "silencing RMS changed by "
            f"{rms_silencing_change_db:.4f} dB"
        )

    normalized_rms = float(
        normalized["active_rms_dbfs"]
    )

    if not np.isfinite(normalized_rms):
        errors.append("normalized RMS is invalid")
    elif (
        abs(
            normalized_rms
            - TARGET_NORMALIZED_RMS_DBFS
        )
        > NORMALIZED_RMS_TOLERANCE_DB
    ):
        errors.append(
            "normalized RMS is not within "
            f"±{NORMALIZED_RMS_TOLERANCE_DB:.2f} dB of "
            f"{TARGET_NORMALIZED_RMS_DBFS:.2f} dBFS"
        )

    normalized_peak = float(
        normalized["full_file_peak_dbfs"]
    )

    if (
        normalized_peak
        > MAX_NORMALIZED_PEAK_DBFS
        + 1e-9
    ):
        errors.append(
            "normalized peak exceeds "
            f"{MAX_NORMALIZED_PEAK_DBFS:.2f} dBFS"
        )

    if (
        float(silenced["inactive_max_abs"])
        > INACTIVE_ZERO_TOLERANCE
    ):
        errors.append(
            "silenced inactive samples are not exactly zero"
        )

    if (
        float(normalized["inactive_max_abs"])
        > INACTIVE_ZERO_TOLERANCE
    ):
        errors.append(
            "normalized inactive samples are not exactly zero"
        )

    if (
        max(durations) - min(durations)
        > DURATION_TOLERANCE_SECONDS
    ):
        errors.append(
            "duration changed beyond "
            f"{DURATION_TOLERANCE_SECONDS} seconds"
        )

    return (
        ("FAIL", "; ".join(errors))
        if errors
        else ("OK", "")
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {METADATA_CSV.resolve()}"
        )

    if not TRIMMED_FOLDER.exists():
        raise FileNotFoundError(TRIMMED_FOLDER)

    metadata = pd.read_csv(METADATA_CSV)

    required_metadata = {
        "name",
        "active_intervals_seconds",
        "harmonic_test_required",
    }

    missing_metadata = (
        required_metadata
        - set(metadata.columns)
    )

    if missing_metadata:
        raise ValueError(
            "Metadata CSV is missing columns: "
            + ", ".join(sorted(missing_metadata))
        )

    channel_choices = load_channel_choices(
        CHANNEL_SELECTION_CSV
    )

    for branch_paths in BRANCHES.values():
        for folder in branch_paths.values():
            if not folder.exists():
                raise FileNotFoundError(folder)

    rows: list[dict[str, object]] = []

    print("=" * 80)
    print("ALL-STAGE AUDIO COMPARISON")
    print("Selected channel is used for Trimmed and Filtered measurements.")
    print("=" * 80)

    for index, (_, metadata_row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):
        bird_name = str(
            metadata_row["name"]
        ).strip()

        bird_key = normalize_name(
            bird_name
        )

        intervals = parse_intervals(
            metadata_row[
                "active_intervals_seconds"
            ]
        )

        branches = ["without_harmonics"]

        if harmonic_test_required(
            metadata_row
        ):
            branches.append(
                "with_harmonics"
            )

        print(
            f"\n[{index:02d}/{len(metadata)}] "
            f"{bird_name}"
        )

        for branch in branches:
            report_row: dict[str, object] = {
                "name": bird_name,
                "branch": branch,
                "status": "ERROR",
                "error": "",
            }

            try:
                if bird_key not in channel_choices:
                    raise ValueError(
                        "No valid selected_channel was found in "
                        "mono_channel_selection_report.csv."
                    )

                selected_channel = channel_choices[
                    bird_key
                ]

                trimmed_path = find_wav(
                    TRIMMED_FOLDER,
                    bird_name,
                )

                filtered_path = find_wav(
                    BRANCHES[branch][
                        "filtered_folder"
                    ],
                    bird_name,
                )

                mono_path = find_wav(
                    BRANCHES[branch][
                        "mono_folder"
                    ],
                    bird_name,
                )

                silenced_path = find_wav(
                    BRANCHES[branch][
                        "silenced_folder"
                    ],
                    bird_name,
                )

                normalized_path = find_wav(
                    BRANCHES[branch][
                        "normalized_folder"
                    ],
                    bird_name,
                )

                # Apples-to-apples:
                # Trimmed and Filtered are measured using the exact
                # channel selection that created Mono_Audio.
                trimmed = measure_stage(
                    trimmed_path,
                    intervals,
                    selected_channel=
                        selected_channel,
                    check_inactive=False,
                )

                filtered = measure_stage(
                    filtered_path,
                    intervals,
                    selected_channel=
                        selected_channel,
                    check_inactive=False,
                )

                mono = measure_stage(
                    mono_path,
                    intervals,
                    selected_channel=None,
                    check_inactive=False,
                )

                silenced = measure_stage(
                    silenced_path,
                    intervals,
                    selected_channel=None,
                    check_inactive=True,
                )

                normalized = measure_stage(
                    normalized_path,
                    intervals,
                    selected_channel=None,
                    check_inactive=True,
                )

                rms_filter_change = level_change(
                    filtered["active_rms_dbfs"],
                    trimmed["active_rms_dbfs"],
                )

                rms_mono_change = level_change(
                    mono["active_rms_dbfs"],
                    filtered["active_rms_dbfs"],
                )

                rms_silencing_change = level_change(
                    silenced["active_rms_dbfs"],
                    mono["active_rms_dbfs"],
                )

                rms_normalization_change = level_change(
                    normalized["active_rms_dbfs"],
                    silenced["active_rms_dbfs"],
                )

                rms_overall_change = level_change(
                    normalized["active_rms_dbfs"],
                    trimmed["active_rms_dbfs"],
                )

                peak_filter_change = level_change(
                    filtered["full_file_peak_dbfs"],
                    trimmed["full_file_peak_dbfs"],
                )

                peak_mono_change = level_change(
                    mono["full_file_peak_dbfs"],
                    filtered["full_file_peak_dbfs"],
                )

                peak_silencing_change = level_change(
                    silenced["full_file_peak_dbfs"],
                    mono["full_file_peak_dbfs"],
                )

                peak_normalization_change = level_change(
                    normalized["full_file_peak_dbfs"],
                    silenced["full_file_peak_dbfs"],
                )

                peak_overall_change = level_change(
                    normalized["full_file_peak_dbfs"],
                    trimmed["full_file_peak_dbfs"],
                )

                durations = [
                    float(trimmed["duration_seconds"]),
                    float(filtered["duration_seconds"]),
                    float(mono["duration_seconds"]),
                    float(silenced["duration_seconds"]),
                    float(normalized["duration_seconds"]),
                ]

                status, error = determine_status(
                    mono=mono,
                    silenced=silenced,
                    normalized=normalized,
                    durations=durations,
                    rms_mono_change_db=
                        rms_mono_change,
                    rms_silencing_change_db=
                        rms_silencing_change,
                )

                report_row.update({
                    "selected_channel":
                        selected_channel,

                    # Active RMS levels
                    "active_rms_trimmed_selected_dbfs":
                        trimmed["active_rms_dbfs"],
                    "active_rms_filtered_selected_dbfs":
                        filtered["active_rms_dbfs"],
                    "active_rms_mono_dbfs":
                        mono["active_rms_dbfs"],
                    "active_rms_silenced_dbfs":
                        silenced["active_rms_dbfs"],
                    "active_rms_normalized_dbfs":
                        normalized["active_rms_dbfs"],

                    # Active RMS changes
                    "rms_filter_change_db":
                        rms_filter_change,
                    "rms_mono_resampling_change_db":
                        rms_mono_change,
                    "rms_silencing_change_db":
                        rms_silencing_change,
                    "rms_normalization_change_db":
                        rms_normalization_change,
                    "rms_overall_change_db":
                        rms_overall_change,

                    # Full-file peaks, measured from the same
                    # selected source before the mono stage
                    "peak_trimmed_selected_dbfs":
                        trimmed["full_file_peak_dbfs"],
                    "peak_filtered_selected_dbfs":
                        filtered["full_file_peak_dbfs"],
                    "peak_mono_dbfs":
                        mono["full_file_peak_dbfs"],
                    "peak_silenced_dbfs":
                        silenced["full_file_peak_dbfs"],
                    "peak_normalized_dbfs":
                        normalized["full_file_peak_dbfs"],

                    # Peak changes
                    "peak_filter_change_db":
                        peak_filter_change,
                    "peak_mono_resampling_change_db":
                        peak_mono_change,
                    "peak_silencing_change_db":
                        peak_silencing_change,
                    "peak_normalization_change_db":
                        peak_normalization_change,
                    "peak_overall_change_db":
                        peak_overall_change,

                    # Crest factor
                    "crest_factor_trimmed_selected_db":
                        trimmed["crest_factor_db"],
                    "crest_factor_filtered_selected_db":
                        filtered["crest_factor_db"],
                    "crest_factor_mono_db":
                        mono["crest_factor_db"],
                    "crest_factor_silenced_db":
                        silenced["crest_factor_db"],
                    "crest_factor_normalized_db":
                        normalized["crest_factor_db"],

                    # Original file channel counts
                    "trimmed_original_channels":
                        trimmed["original_channels"],
                    "filtered_original_channels":
                        filtered["original_channels"],
                    "mono_channels":
                        mono["original_channels"],
                    "silenced_channels":
                        silenced["original_channels"],
                    "normalized_channels":
                        normalized["original_channels"],

                    # Sample rates
                    "trimmed_sample_rate_hz":
                        trimmed["sample_rate_hz"],
                    "filtered_sample_rate_hz":
                        filtered["sample_rate_hz"],
                    "mono_sample_rate_hz":
                        mono["sample_rate_hz"],
                    "silenced_sample_rate_hz":
                        silenced["sample_rate_hz"],
                    "normalized_sample_rate_hz":
                        normalized["sample_rate_hz"],

                    # File subtypes
                    "trimmed_subtype":
                        trimmed["subtype"],
                    "filtered_subtype":
                        filtered["subtype"],
                    "mono_subtype":
                        mono["subtype"],
                    "silenced_subtype":
                        silenced["subtype"],
                    "normalized_subtype":
                        normalized["subtype"],

                    # Durations
                    "duration_trimmed_seconds":
                        trimmed["duration_seconds"],
                    "duration_filtered_seconds":
                        filtered["duration_seconds"],
                    "duration_mono_seconds":
                        mono["duration_seconds"],
                    "duration_silenced_seconds":
                        silenced["duration_seconds"],
                    "duration_normalized_seconds":
                        normalized["duration_seconds"],

                    # Active-interval information
                    "active_duration_seconds":
                        normalized["active_duration_seconds"],
                    "active_fraction_percent":
                        normalized["active_fraction_percent"],

                    # Exact-zero checks
                    "inactive_max_abs_silenced":
                        silenced["inactive_max_abs"],
                    "inactive_max_abs_normalized":
                        normalized["inactive_max_abs"],

                    "status": status,
                    "error": error,
                })

                print(
                    f"  {branch}: {status}"
                )

                print(
                    "    Selected channel: "
                    f"{selected_channel}"
                )

                print(
                    "    RMS: "
                    f"{trimmed['active_rms_dbfs']:.3f}"
                    " → "
                    f"{filtered['active_rms_dbfs']:.3f}"
                    " → "
                    f"{mono['active_rms_dbfs']:.3f}"
                    " → "
                    f"{silenced['active_rms_dbfs']:.3f}"
                    " → "
                    f"{normalized['active_rms_dbfs']:.3f}"
                    " dBFS"
                )

                print(
                    "    Mono/resampling change: "
                    f"{rms_mono_change:+.4f} dB"
                )

            except Exception as exc:
                report_row["error"] = str(exc)

                print(
                    f"  {branch}: ERROR — {exc}"
                )

            rows.append(report_row)

    report = pd.DataFrame(rows)

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_REPORT,
        index=False,
    )

    ok_count = int(
        (
            report["status"] == "OK"
        ).sum()
    )

    fail_count = int(
        (
            report["status"] == "FAIL"
        ).sum()
    )

    error_count = int(
        (
            report["status"] == "ERROR"
        ).sum()
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"OK:    {ok_count}")
    print(f"FAIL:  {fail_count}")
    print(f"ERROR: {error_count}")
    print(
        f"Report: {OUTPUT_REPORT.resolve()}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
