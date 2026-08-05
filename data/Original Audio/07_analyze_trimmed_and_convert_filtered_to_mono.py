from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

TRIMMED_FOLDER = ROOT / "Trimmed"

METADATA_CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

CHANNEL_REPORT_CSV = (
    ROOT
    / "Frequency Range Results"
    / "mono_channel_selection_report.csv"
)

BRANCHES = {
    "with_harmonics": {
        "input_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Audio"
        ),
        "mono_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Mono_Audio"
        ),
    },
    "without_harmonics": {
        "input_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Audio"
        ),
        "mono_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Mono_Audio"
        ),
    },
}


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_SUBTYPE = "PCM_16"
TARGET_SAMPLE_RATE_HZ = 44100

MIN_WAVEFORM_CORRELATION_FOR_AVERAGE = 0.85
MAX_AVERAGING_LOSS_DB = 1.0
MAX_LEFT_RIGHT_RMS_DIFFERENCE_DB = 3.0


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


def find_audio_file(folder: Path, bird_name: str) -> Path:
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
            f"No WAV file found for '{bird_name}' in {folder.resolve()}"
        )

    raise ValueError(
        f"More than one WAV matched '{bird_name}' in {folder.resolve()}: "
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
            raise ValueError(f"Invalid interval '{part}'.")

        start, end = map(float, match.groups())

        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval '{part}'.")

        intervals.append((start, end))

    if not intervals:
        raise ValueError("No active intervals were found.")

    return intervals


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"{path.name} has unsupported shape {audio.shape}."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{path.name} contains invalid samples.")

    return audio, int(sample_rate)


def extract_active_samples(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    duration = len(audio) / sample_rate
    pieces: list[np.ndarray] = []

    for start, end in intervals:
        if start >= duration:
            continue

        end = min(end, duration)

        start_sample = int(round(start * sample_rate))
        end_sample = int(round(end * sample_rate))

        if end_sample > start_sample:
            pieces.append(audio[start_sample:end_sample])

    if not pieces:
        raise ValueError("No active samples overlapped the audio.")

    return np.concatenate(pieces, axis=0)


def rms_dbfs(audio: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if rms <= 0:
        return 0.0, -math.inf

    return rms, 20.0 * math.log10(rms)


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if left.size < 2 or right.size < 2:
        return math.nan

    if np.std(left) <= 0 or np.std(right) <= 0:
        return math.nan

    return float(np.corrcoef(left, right)[0, 1])


def analyze_trimmed_file(
    *,
    input_path: Path,
    intervals: list[tuple[float, float]],
) -> dict[str, object]:
    audio, sample_rate = load_audio(input_path)

    original_channels = (
        1
        if audio.ndim == 1
        else int(audio.shape[1])
    )

    if original_channels == 1:
        return {
            "input_filename": input_path.name,
            "sample_rate_hz": sample_rate,
            "original_channels": 1,
            "was_non_mono": "no",
            "left_active_rms_dbfs": np.nan,
            "right_active_rms_dbfs": np.nan,
            "middle_active_rms_dbfs": np.nan,
            "left_right_rms_difference_db": np.nan,
            "waveform_correlation": np.nan,
            "averaging_loss_db": np.nan,
            "selected_channel": "already_mono",
            "selection_reason": (
                "Trimmed source was already mono."
            ),
            "status": "OK",
            "error": "",
        }

    if original_channels != 2:
        raise ValueError(
            f"{input_path.name} has {original_channels} channels. "
            "Only mono and stereo files are supported."
        )

    left = audio[:, 0]
    right = audio[:, 1]
    middle = (left + right) / 2.0

    left_active = extract_active_samples(
        left,
        sample_rate,
        intervals,
    )
    right_active = extract_active_samples(
        right,
        sample_rate,
        intervals,
    )
    middle_active = extract_active_samples(
        middle,
        sample_rate,
        intervals,
    )

    _, left_db = rms_dbfs(left_active)
    _, right_db = rms_dbfs(right_active)
    _, middle_db = rms_dbfs(middle_active)

    correlation = safe_correlation(
        left_active,
        right_active,
    )

    louder_channel_db = max(left_db, right_db)

    left_right_difference_db = abs(
        left_db - right_db
    )

    averaging_loss_db = (
        middle_db - louder_channel_db
    )

    use_middle = (
        np.isfinite(correlation)
        and correlation >= MIN_WAVEFORM_CORRELATION_FOR_AVERAGE
        and left_right_difference_db
        <= MAX_LEFT_RIGHT_RMS_DIFFERENCE_DB
        and averaging_loss_db
        >= -MAX_AVERAGING_LOSS_DB
    )

    if use_middle:
        selected_channel = "middle"
        reason = (
            "Channels were strongly correlated, had similar active RMS, "
            "and averaging caused little level loss."
        )
    elif left_db >= right_db:
        selected_channel = "left"
        reason = (
            "Averaging was not considered safe; left had the stronger "
            "active-call RMS."
        )
    else:
        selected_channel = "right"
        reason = (
            "Averaging was not considered safe; right had the stronger "
            "active-call RMS."
        )

    return {
        "input_filename": input_path.name,
        "sample_rate_hz": sample_rate,
        "original_channels": original_channels,
        "was_non_mono": "yes",
        "left_active_rms_dbfs": left_db,
        "right_active_rms_dbfs": right_db,
        "middle_active_rms_dbfs": middle_db,
        "left_right_rms_difference_db":
            left_right_difference_db,
        "waveform_correlation": correlation,
        "averaging_loss_db": averaging_loss_db,
        "selected_channel": selected_channel,
        "selection_reason": reason,
        "status": "OK",
        "error": "",
    }


def convert_using_choice(
    *,
    input_path: Path,
    output_path: Path,
    selected_channel: str,
) -> None:
    audio, sample_rate = load_audio(input_path)

    if audio.ndim == 1:
        mono = audio.copy()

    elif audio.ndim == 2 and audio.shape[1] == 2:
        if selected_channel == "left":
            mono = audio[:, 0].copy()

        elif selected_channel == "right":
            mono = audio[:, 1].copy()

        elif selected_channel == "middle":
            mono = np.mean(audio, axis=1)

        elif selected_channel == "already_mono":
            raise ValueError(
                f"{input_path.name} is stereo but report says already_mono."
            )

        else:
            raise ValueError(
                f"Unsupported selected_channel '{selected_channel}'."
            )

    else:
        raise ValueError(
            f"{input_path.name} has unsupported shape {audio.shape}."
        )

    # --------------------------------------------------------
    # Standardize the final mono file before normalization:
    # WAV, mono, PCM 16-bit, 44,100 Hz.
    #
    # resample_poly applies anti-alias filtering automatically
    # when converting 48 kHz or 96 kHz audio to 44.1 kHz.
    # Files already at 44.1 kHz are not resampled.
    # --------------------------------------------------------
    if sample_rate != TARGET_SAMPLE_RATE_HZ:
        common_divisor = math.gcd(
            sample_rate,
            TARGET_SAMPLE_RATE_HZ,
        )

        up_factor = (
            TARGET_SAMPLE_RATE_HZ
            // common_divisor
        )

        down_factor = (
            sample_rate
            // common_divisor
        )

        mono = resample_poly(
            mono,
            up_factor,
            down_factor,
        )

    sf.write(
        output_path,
        mono,
        TARGET_SAMPLE_RATE_HZ,
        format="WAV",
        subtype=OUTPUT_SUBTYPE,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not TRIMMED_FOLDER.exists():
        raise FileNotFoundError(
            f"Trimmed folder not found: {TRIMMED_FOLDER.resolve()}"
        )

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {METADATA_CSV.resolve()}"
        )

    metadata = pd.read_csv(METADATA_CSV)

    required_columns = {
        "name",
        "id",
        "active_intervals_seconds",
        "harmonic_test_required",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            "Metadata CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    print("=" * 78)
    print("ANALYZE TRIMMED FILES, CONVERT TO MONO, AND STANDARDIZE TO 44.1 KHZ")
    print("=" * 78)

    rows: list[dict[str, object]] = []

    # --------------------------------------------------------
    # Stage 1: analyze the 20 Trimmed source files once.
    # --------------------------------------------------------
    for index, (_, row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):
        name = str(row["name"]).strip()
        bird_id = row.get("id", "")

        print(
            f"\n[{index:02d}/{len(metadata)}] Analyze: {name}"
        )

        report_row: dict[str, object] = {
            "name": name,
            "id": bird_id,
        }

        try:
            intervals = parse_intervals(
                row["active_intervals_seconds"]
            )

            trimmed_path = find_audio_file(
                TRIMMED_FOLDER,
                name,
            )

            analysis = analyze_trimmed_file(
                input_path=trimmed_path,
                intervals=intervals,
            )

            report_row.update(analysis)

            print(
                f"  Channels: {analysis['original_channels']}"
            )
            print(
                f"  Choice:   {analysis['selected_channel']}"
            )

        except Exception as error:
            report_row.update({
                "status": "ERROR",
                "error": str(error),
            })

            print(f"  ERROR: {error}")

        rows.append(report_row)

    report_columns = [
        "name",
        "id",
        "input_filename",
        "sample_rate_hz",
        "original_channels",
        "was_non_mono",
        "left_active_rms_dbfs",
        "right_active_rms_dbfs",
        "middle_active_rms_dbfs",
        "left_right_rms_difference_db",
        "waveform_correlation",
        "averaging_loss_db",
        "selected_channel",
        "selection_reason",
        "status",
        "error",
    ]

    report = pd.DataFrame(
        rows,
        columns=report_columns,
    )

    CHANNEL_REPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        CHANNEL_REPORT_CSV,
        index=False,
    )

    print(
        f"\nSaved one channel-selection report: "
        f"{CHANNEL_REPORT_CSV.resolve()}"
    )

    # --------------------------------------------------------
    # Stage 2: apply each selected choice to both silenced
    # branches without changing Silenced_Audio.
    # --------------------------------------------------------
    good_report = report[
        report["status"] == "OK"
    ].copy()

    choices = {
        normalize_name(row["name"]):
            str(row["selected_channel"])
        for _, row in good_report.iterrows()
    }

    for branch, paths in BRANCHES.items():
        input_folder = paths["input_folder"]
        mono_folder = paths["mono_folder"]

        if not input_folder.exists():
            raise FileNotFoundError(input_folder)

        mono_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"\nConvert branch: {branch}")

        converted = 0
        failed = 0

        for _, row in metadata.iterrows():
            if (
                branch == "with_harmonics"
                and not harmonic_test_required(row)
            ):
                continue

            name = str(row["name"]).strip()
            key = normalize_name(name)

            if key not in choices:
                print(
                    f"  SKIP {name}: no valid channel choice."
                )
                failed += 1
                continue

            try:
                input_path = find_audio_file(
                    input_folder,
                    name,
                )

                output_path = (
                    mono_folder
                    / input_path.name
                )

                convert_using_choice(
                    input_path=input_path,
                    output_path=output_path,
                    selected_channel=choices[key],
                )

                print(
                    f"  {name}: {choices[key]} -> {output_path.name}"
                )

                converted += 1

            except Exception as error:
                print(
                    f"  ERROR {name}: {error}"
                )
                failed += 1

        print(
            f"  Converted: {converted}"
        )
        print(
            f"  Failed:    {failed}"
        )

    print("\n" + "=" * 78)
    print("MONO CONVERSION AND 44.1 KHZ STANDARDIZATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
