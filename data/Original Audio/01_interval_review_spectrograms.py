from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

AUDIO_FOLDER = ROOT / "Trimmed"
INTERVALS_CSV = ROOT / "bird_active_call_ranges.csv"
OUTPUT_FOLDER = ROOT / "interval_review_spectrograms"


# ============================================================
# SPECTROGRAM SETTINGS
# ============================================================

WINDOW_SECONDS = 0.025
OVERLAP_FRACTION = 0.75
MAX_DISPLAY_HZ = 10000.0
DYNAMIC_RANGE_DB = 80.0

INTERVAL_PADDING_SECONDS = 0.0
AUDIO_DURATION_SECONDS = 10.0


# ============================================================
# HELPERS
# ============================================================

def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def parse_intervals(text: str) -> list[tuple[float, float]]:
    """
    Read active intervals, add 0.1 seconds before and after each one,
    clip them to the 0-10 second file, and merge overlaps.
    """
    padded_intervals = []

    for part in str(text).split(";"):
        part = part.strip()

        if not part:
            continue

        start_text, end_text = part.split("-")

        start = float(start_text.strip())
        end = float(end_text.strip())

        if end <= start:
            continue

        padded_start = max(
            0.0,
            start - INTERVAL_PADDING_SECONDS,
        )

        padded_end = min(
            AUDIO_DURATION_SECONDS,
            end + INTERVAL_PADDING_SECONDS,
        )

        padded_intervals.append(
            (padded_start, padded_end)
        )

    if not padded_intervals:
        return []

    padded_intervals.sort(
        key=lambda interval: interval[0]
    )

    merged = [
        padded_intervals[0]
    ]

    for start, end in padded_intervals[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged

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
        raise FileNotFoundError(f"No WAV found for {bird_name}")

    raise ValueError(
        f"More than one WAV matched {bird_name}: "
        + ", ".join(path.name for path in matches)
    )


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return audio, int(sample_rate)


def calculate_spectrogram(audio: np.ndarray, sample_rate: int):
    nperseg = int(round(WINDOW_SECONDS * sample_rate))
    nperseg = min(max(nperseg, 128), len(audio))
    noverlap = min(int(round(nperseg * OVERLAP_FRACTION)), nperseg - 1)

    frequencies, times, power = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    power_db = 10.0 * np.log10(power + np.finfo(float).eps)
    return frequencies, times, power_db


def save_plot(
    frequencies,
    times,
    power_db,
    bird_name: str,
    intervals: list[tuple[float, float]],
    output_path: Path,
    sample_rate: int,
) -> None:
    vmax = float(np.max(power_db))
    vmin = vmax - DYNAMIC_RANGE_DB
    max_display_hz = min(MAX_DISPLAY_HZ, sample_rate / 2.0)

    fig, ax = plt.subplots(figsize=(14, 7))

    image = ax.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    for start, end in intervals:
        ax.axvspan(start, end, alpha=0.18)
        ax.axvline(start, linestyle="--", linewidth=1)
        ax.axvline(end, linestyle="--", linewidth=1)

    ax.set_xlim(0, 10.0)

    ax.set_xticks(
        np.arange(0.0, 10.1, 1.0)
    )

    ax.set_xticks(
        np.arange(0.0, 10.01, 0.1),
        minor=True,
    )

    ax.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.8,
        alpha=0.55,
    )

    ax.grid(
        axis="x",
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.35,
    )

    ax.set_ylim(0, max_display_hz)
    ax.set_ylim(0, max_display_hz)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{bird_name} - Playback_10s with Active Intervals")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power spectral density (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    intervals_df = pd.read_csv(INTERVALS_CSV)

    print("=" * 78)
    print("GENERATE SPECTROGRAMS WITH ACTIVE-INTERVAL MARKERS")
    print("=" * 78)

    for _, row in intervals_df.iterrows():
        bird_name = str(row["name"]).strip()
        intervals = parse_intervals(row["active_intervals_seconds"])

        try:
            audio_path = find_audio_file(AUDIO_FOLDER, bird_name)
            audio, sample_rate = load_mono_audio(audio_path)
            frequencies, times, power_db = calculate_spectrogram(audio, sample_rate)

            output_path = (
                OUTPUT_FOLDER
                / f"{safe_filename(bird_name)}_interval_review.png"
            )

            save_plot(
                frequencies,
                times,
                power_db,
                bird_name,
                intervals,
                output_path,
                sample_rate,
            )

            print(f"Saved: {output_path}")

        except Exception as error:
            print(f"{bird_name} -> ERROR: {error}")

    print("\nDone.")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()