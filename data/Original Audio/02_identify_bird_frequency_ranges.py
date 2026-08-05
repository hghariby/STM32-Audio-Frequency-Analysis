from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing the 20 original WAV files.
AUDIO_FOLDER = Path("Trimmed")

# CSV created earlier:
# columns: name, id, active_intervals_seconds
INPUT_CSV = Path("bird_active_call_ranges.csv")

# Output folders/files.
OUTPUT_FOLDER = Path("Frequency Range Results")
PLOT_FOLDER = OUTPUT_FOLDER / "spectrograms"
RESULTS_CSV = OUTPUT_FOLDER / "bird_frequency_ranges.csv"

# Ignore frequencies below this value.
# This removes DC and very-low-frequency rumble from consideration.
MIN_ANALYSIS_HZ = 100.0

# Set to None to use the WAV file's Nyquist frequency.
# Because your STM32 records at about 17,857 Hz, frequencies above
# roughly 8,929 Hz cannot later be captured by the STM32.
MAX_ANALYSIS_HZ = 15000.0


# The detected range will contain the middle 98% of accepted energy.
LOW_ENERGY_PERCENTILE = 0.01
HIGH_ENERGY_PERCENTILE = 0.99

# Ignore PSD bins that are extremely weak compared with the strongest bin.
# -40 dB means bins below 1/10,000 of peak power are excluded.
RELATIVE_POWER_THRESHOLD_DB = -40.0

# Suggested filter margins around the detected bird-call range.
LOW_FILTER_MARGIN_HZ = 500.0
HIGH_FILTER_MARGIN_HZ = 500.0

# Welch power-spectrum settings.
WELCH_WINDOW_SECONDS = 0.10
WELCH_OVERLAP = 0.50

# Spectrogram settings.
SPECTROGRAM_WINDOW_SECONDS = 0.025
SPECTROGRAM_OVERLAP = 0.75


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_filename_text(text: str) -> str:
    """
    Normalize names so that:
      House Finch.wav
      House Finch(1).wav
      house_finch.wav
    can still be matched to 'House Finch'.
    """
    text = Path(text).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def find_audio_file(
    audio_folder: Path,
    bird_name: str,
    bird_id: str,
) -> Path:
    """
    Find the WAV file using either the bird name or Macaulay ID.
    """

    wav_files = sorted(
        path
        for path in audio_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files were found in: {audio_folder.resolve()}"
        )

    target_name = normalize_filename_text(bird_name)
    target_id = normalize_filename_text(bird_id)

    exact_name_matches = [
        path
        for path in wav_files
        if normalize_filename_text(path.name) == target_name
    ]

    if len(exact_name_matches) == 1:
        return exact_name_matches[0]

    id_matches = [
        path
        for path in wav_files
        if target_id
        and target_id in normalize_filename_text(path.name)
    ]

    if len(id_matches) == 1:
        return id_matches[0]

    name_matches = [
        path
        for path in wav_files
        if target_name in normalize_filename_text(path.name)
    ]

    if len(name_matches) == 1:
        return name_matches[0]

    possible = exact_name_matches or id_matches or name_matches

    if len(possible) > 1:
        names = ", ".join(path.name for path in possible)
        raise ValueError(
            f"More than one WAV file matches {bird_name}: {names}"
        )

    raise FileNotFoundError(
        f"No WAV file matched bird '{bird_name}' "
        f"with ID '{bird_id}'."
    )


def parse_active_intervals(interval_text: str) -> list[tuple[float, float]]:
    """
    Convert:
        0.45-3.15;5.45-8.00

    into:
        [(0.45, 3.15), (5.45, 8.00)]
    """
    if pd.isna(interval_text):
        raise ValueError("Active interval entry is empty.")

    intervals: list[tuple[float, float]] = []

    for part in str(interval_text).split(";"):
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*",
            part,
        )

        if not match:
            raise ValueError(
                f"Invalid interval '{part}'. Use format start-end;start-end"
            )

        start = float(match.group(1))
        end = float(match.group(2))

        if start < 0:
            raise ValueError(f"Interval start cannot be negative: {part}")

        if end <= start:
            raise ValueError(f"Interval end must exceed start: {part}")

        intervals.append((start, end))

    if not intervals:
        raise ValueError("No valid active intervals were found.")

    return intervals


# ============================================================
# AUDIO PROCESSING
# ============================================================

def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load WAV as floating-point mono audio.
    Stereo files are converted to mono by averaging their channels.
    """
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.size == 0:
        raise ValueError("The WAV file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The WAV contains invalid sample values.")

    return audio, int(sample_rate)


def remove_dc_offset(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Remove DC offset by subtracting the waveform mean.
    """
    dc_offset = float(np.mean(audio))
    corrected = audio - dc_offset
    return corrected, dc_offset


def extract_active_audio(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Extract every listed interval and concatenate the samples.

    The intervals remain separate in the original file. They are joined
    only temporarily for RMS and average-spectrum calculations.
    """
    duration = len(audio) / sample_rate
    pieces: list[np.ndarray] = []
    accepted_intervals: list[tuple[float, float]] = []

    for start, end in intervals:
        if start >= duration:
            raise ValueError(
                f"Interval starts at {start:.3f}s, but file duration is "
                f"only {duration:.3f}s."
            )

        clipped_end = min(end, duration)

        start_sample = int(round(start * sample_rate))
        end_sample = int(round(clipped_end * sample_rate))

        piece = audio[start_sample:end_sample]

        if piece.size < 2:
            continue

        pieces.append(piece)
        accepted_intervals.append((start, clipped_end))

    if not pieces:
        raise ValueError("None of the listed intervals contains audio.")

    return np.concatenate(pieces), accepted_intervals


def calculate_rms(audio: np.ndarray) -> tuple[float, float]:
    """
    Return linear RMS and RMS in dBFS.
    """
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if rms <= 0:
        return 0.0, -math.inf

    return rms, 20.0 * math.log10(rms)


# ============================================================
# FREQUENCY ANALYSIS
# ============================================================

def calculate_welch_psd(
    active_audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the average power spectral density of all active calls.
    """
    requested_length = int(round(WELCH_WINDOW_SECONDS * sample_rate))
    nperseg = min(max(requested_length, 256), len(active_audio))

    if nperseg < 32:
        raise ValueError("The active audio is too short for analysis.")

    noverlap = int(round(nperseg * WELCH_OVERLAP))
    noverlap = min(noverlap, nperseg - 1)

    frequencies, psd = signal.welch(
        active_audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    return frequencies, psd


def identify_frequency_range(
    frequencies: np.ndarray,
    psd: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """
    Estimate the bird-call frequency range.

    Method:
      1. Restrict analysis to MIN_ANALYSIS_HZ through MAX_ANALYSIS_HZ.
      2. Ignore PSD bins more than RELATIVE_POWER_THRESHOLD_DB below peak.
      3. Find the frequency limits containing the middle 98% of the
         remaining spectral energy.
    """
    nyquist = sample_rate / 2.0
    maximum_hz = nyquist

    if MAX_ANALYSIS_HZ is not None:
        maximum_hz = min(maximum_hz, MAX_ANALYSIS_HZ)

    valid = (
        np.isfinite(psd)
        & (psd >= 0)
        & (frequencies >= MIN_ANALYSIS_HZ)
        & (frequencies <= maximum_hz)
    )

    f = frequencies[valid]
    p = psd[valid]

    if f.size == 0 or np.sum(p) <= 0:
        raise ValueError("No usable spectral energy was found.")

    peak_power = float(np.max(p))
    threshold_ratio = 10.0 ** (RELATIVE_POWER_THRESHOLD_DB / 10.0)
    power_threshold = peak_power * threshold_ratio

    accepted = p >= power_threshold

    if np.count_nonzero(accepted) < 3:
        accepted = np.ones_like(p, dtype=bool)

    accepted_f = f[accepted]
    accepted_p = p[accepted]

    order = np.argsort(accepted_f)
    accepted_f = accepted_f[order]
    accepted_p = accepted_p[order]

    cumulative = np.cumsum(accepted_p)
    cumulative /= cumulative[-1]

    low_index = int(
        np.searchsorted(
            cumulative,
            LOW_ENERGY_PERCENTILE,
            side="left",
        )
    )
    high_index = int(
        np.searchsorted(
            cumulative,
            HIGH_ENERGY_PERCENTILE,
            side="left",
        )
    )
    median_index = int(
        np.searchsorted(cumulative, 0.50, side="left")
    )

    low_index = int(np.clip(low_index, 0, len(accepted_f) - 1))
    high_index = int(np.clip(high_index, low_index, len(accepted_f) - 1))
    median_index = int(np.clip(median_index, 0, len(accepted_f) - 1))

    low_hz = float(accepted_f[low_index])
    high_hz = float(accepted_f[high_index])
    median_hz = float(accepted_f[median_index])

    centroid_hz = float(
        np.sum(accepted_f * accepted_p) / np.sum(accepted_p)
    )

    peak_hz = float(f[np.argmax(p)])

    suggested_low_hz = max(
        MIN_ANALYSIS_HZ,
        low_hz - LOW_FILTER_MARGIN_HZ,
    )
    suggested_high_hz = min(
        maximum_hz,
        high_hz + HIGH_FILTER_MARGIN_HZ,
    )

    return {
        "peak_frequency_hz": peak_hz,
        "median_frequency_hz": median_hz,
        "spectral_centroid_hz": centroid_hz,
        "detected_low_hz": low_hz,
        "detected_high_hz": high_hz,
        "detected_bandwidth_hz": high_hz - low_hz,
        "suggested_filter_low_hz": suggested_low_hz,
        "suggested_filter_high_hz": suggested_high_hz,
        "final_filter_low_hz": None,
        "final_filter_high_with_harmonics_hz": None,
        "final_filter_high_without_harmonics_hz": None,
        "harmonic_test_required": None,
        "decision_notes": None,

    }


# ============================================================
# PLOTS
# ============================================================

def make_analysis_plot(
    full_audio: np.ndarray,
    sample_rate: int,
    bird_name: str,
    accepted_intervals: list[tuple[float, float]],
    frequency_results: dict[str, float],
    output_path: Path,
) -> None:
    """
    Make a full-file spectrogram with:
      - active time intervals highlighted
      - detected frequency range marked
      - suggested future bandpass range marked
    """
    nperseg = int(round(SPECTROGRAM_WINDOW_SECONDS * sample_rate))
    nperseg = min(max(nperseg, 128), len(full_audio))

    noverlap = int(round(nperseg * SPECTROGRAM_OVERLAP))
    noverlap = min(noverlap, nperseg - 1)

    frequencies, times, spectrogram_psd = signal.spectrogram(
        full_audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    spectrogram_db = 10.0 * np.log10(
        spectrogram_psd + np.finfo(float).eps
    )

    figure, axis = plt.subplots(figsize=(14, 7))

    image = axis.pcolormesh(
        times,
        frequencies,
        spectrogram_db,
        shading="auto",
        cmap="magma",
    )

    for index, (start, end) in enumerate(accepted_intervals):
        label = "Active-call interval" if index == 0 else None
        axis.axvspan(
            start,
            end,
            alpha=0.16,
            label=label,
        )

    axis.axhline(
        frequency_results["detected_low_hz"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f'Detected low: '
            f'{frequency_results["detected_low_hz"]:.0f} Hz'
        ),
    )
    axis.axhline(
        frequency_results["detected_high_hz"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f'Detected high: '
            f'{frequency_results["detected_high_hz"]:.0f} Hz'
        ),
    )
    axis.axhline(
        frequency_results["suggested_filter_low_hz"],
        linestyle=":",
        linewidth=1.4,
        label=(
            f'Suggested filter low: '
            f'{frequency_results["suggested_filter_low_hz"]:.0f} Hz'
        ),
    )
    axis.axhline(
        frequency_results["suggested_filter_high_hz"],
        linestyle=":",
        linewidth=1.4,
        label=(
            f'Suggested filter high: '
            f'{frequency_results["suggested_filter_high_hz"]:.0f} Hz'
        ),
    )

    MICROPHONE_MAX_HZ = 15000.0

    maximum_plot_hz = min(
        sample_rate / 2.0,
        MICROPHONE_MAX_HZ,
    )

    axis.set_ylim(0, maximum_plot_hz)
    axis.set_xlim(0, len(full_audio) / sample_rate)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(f"{bird_name}: active calls and detected frequency range")
    axis.legend(loc="upper right")

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Power spectral density (dB)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_bird(row: pd.Series) -> dict[str, object]:
    bird_name = str(row["name"]).strip()
    bird_id = str(row["id"]).strip()
    interval_text = str(row["active_intervals_seconds"]).strip()

    file_path = find_audio_file(
        AUDIO_FOLDER,
        bird_name,
        bird_id,
    )

    intervals = parse_active_intervals(interval_text)

    audio, sample_rate = load_mono_audio(file_path)

    # DC-offset removal is intentionally disabled.
    # Temporarily concatenate only the manually selected active calls.
    active_audio, accepted_intervals = extract_active_audio(
        audio,
        sample_rate,
        intervals,
    )

    active_rms, active_rms_dbfs = calculate_rms(active_audio)

    frequencies, psd = calculate_welch_psd(
        active_audio,
        sample_rate,
    )

    frequency_results = identify_frequency_range(
        frequencies,
        psd,
        sample_rate,
    )

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", bird_name).strip("_")
    plot_path = PLOT_FOLDER / f"{safe_name}_frequency_range.png"

    make_analysis_plot(
        full_audio=audio,
        sample_rate=sample_rate,
        bird_name=bird_name,
        accepted_intervals=accepted_intervals,
        frequency_results=frequency_results,
        output_path=plot_path,
    )

    return {
        "name": bird_name,
        "id": bird_id,
        "audio_filename": file_path.name,
        "sample_rate_hz": sample_rate,
        "file_duration_seconds": len(audio) / sample_rate,
        "active_intervals_seconds": ";".join(
            f"{start:.3f}-{end:.3f}"
            for start, end in accepted_intervals
        ),
        "total_active_duration_seconds": sum(
            end - start for start, end in accepted_intervals
        ),
        "active_call_rms_linear": active_rms,
        "active_call_rms_dbfs": active_rms_dbfs,
        **frequency_results,
        "plot_file": str(plot_path),
        "status": "OK",
        "error": "",
    }


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not AUDIO_FOLDER.exists():
        raise FileNotFoundError(
            f"Audio folder was not found: {AUDIO_FOLDER.resolve()}"
        )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV was not found: {INPUT_CSV.resolve()}"
        )

    metadata = pd.read_csv(INPUT_CSV)

    required_columns = {
        "name",
        "id",
        "active_intervals_seconds",
    }

    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    results: list[dict[str, object]] = []

    print("=" * 78)
    print("BIRD-CALL FREQUENCY-RANGE IDENTIFICATION")
    print("=" * 78)

    for _, row in metadata.iterrows():
        bird_name = str(row["name"]).strip()

        print(f"\nProcessing: {bird_name}")

        try:
            result = process_bird(row)
            results.append(result)

            print(f'  File: {result["audio_filename"]}')
            print(
                "  Detected call range: "
                f'{result["detected_low_hz"]:.0f}-'
                f'{result["detected_high_hz"]:.0f} Hz'
            )
            print(
                "  Suggested filter range: "
                f'{result["suggested_filter_low_hz"]:.0f}-'
                f'{result["suggested_filter_high_hz"]:.0f} Hz'
            )
            print(
                "  Active-call RMS: "
                f'{result["active_call_rms_dbfs"]:.2f} dBFS'
            )

        except Exception as error:
            print(f"  ERROR: {error}")

            results.append(
                {
                    "name": bird_name,
                    "id": str(row.get("id", "")).strip(),
                    "active_intervals_seconds": str(
                        row.get("active_intervals_seconds", "")
                    ),
                    "status": "ERROR",
                    "error": str(error),
                }
            )

    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(RESULTS_CSV, index=False)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot write {RESULTS_CSV}. Close this CSV in Excel and run again."
        ) from error

    successful = int((results_df["status"] == "OK").sum())
    failed = len(results_df) - successful

    print("\n" + "=" * 78)
    print("COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Results:    {RESULTS_CSV.resolve()}")
    print(f"Plots:      {PLOT_FOLDER.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
