from __future__ import annotations

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

# Folder containing the original WAV files.
AUDIO_FOLDER = Path("Trimmed")

# CSV containing the approved cutoff columns.
INPUT_CSV = Path("Frequency Range Results") / "bird_frequency_ranges.csv"

# Output folder for the cutoff-review spectrograms.
OUTPUT_FOLDER = Path("Frequency Range Results") / "filtering_boundary_review"

# Plot all 20 birds.
# Birds marked "yes" show cyan + green upper cutoffs.
# Birds marked "no" show only the cyan final/no-harmonic upper cutoff.
ONLY_HARMONIC_TEST_BIRDS = False

# Fixed comparison ceiling. The plot will automatically use a lower
# ceiling if a WAV file's Nyquist frequency is below 15 kHz.
MAX_PLOT_HZ = 15000.0

# Spectrogram settings.
SPECTROGRAM_WINDOW_SECONDS = 0.025
SPECTROGRAM_OVERLAP = 0.75

# Fixed color scale so all birds are comparable.
COLOR_MIN_DB = -150.0
COLOR_MAX_DB = -30.0

# Cutoff-line colors.
WITH_HARMONICS_COLOR = "lime"
WITHOUT_HARMONICS_COLOR = "cyan"
LOW_CUTOFF_COLOR = "white"


# ============================================================
# HELPERS
# ============================================================

def normalize_filename_text(text: str) -> str:
    """Normalize names for matching bird names to WAV filenames."""
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def find_audio_file(
    audio_folder: Path,
    bird_name: str,
    bird_id: str,
    csv_audio_filename: str | None,
) -> Path:
    """Find exactly one WAV file for the CSV row."""
    if csv_audio_filename:
        candidate = audio_folder / str(csv_audio_filename)
        if candidate.exists():
            return candidate

    wav_files = sorted(
        path
        for path in audio_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    target_name = normalize_filename_text(bird_name)
    target_id = normalize_filename_text(bird_id)

    exact_name_matches = [
        path for path in wav_files
        if normalize_filename_text(path.name) == target_name
    ]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]

    id_matches = [
        path for path in wav_files
        if target_id and target_id in normalize_filename_text(path.name)
    ]
    if len(id_matches) == 1:
        return id_matches[0]

    partial_name_matches = [
        path for path in wav_files
        if target_name in normalize_filename_text(path.name)
    ]
    if len(partial_name_matches) == 1:
        return partial_name_matches[0]

    possible = exact_name_matches or id_matches or partial_name_matches

    if len(possible) > 1:
        raise ValueError(
            f"More than one WAV file matched '{bird_name}': "
            + ", ".join(path.name for path in possible)
        )

    raise FileNotFoundError(
        f"No WAV file matched '{bird_name}' in {audio_folder.resolve()}."
    )


def parse_active_intervals(
    interval_text: str,
) -> list[tuple[float, float]]:
    """Convert '0.5-2.0;4.0-5.0' to a list of time intervals."""
    if pd.isna(interval_text):
        return []

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
                f"Invalid active interval '{part}'. "
                "Use start-end;start-end."
            )

        start = float(match.group(1))
        end = float(match.group(2))

        if end <= start:
            raise ValueError(f"Invalid interval '{part}'.")

        intervals.append((start, end))

    return intervals


def load_analysis_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio as floating point.

    Stereo is averaged only for this visualization. This script does not
    modify or save the source audio.
    """
    audio, sample_rate = sf.read(path, always_2d=True)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{path.name} contains invalid samples.")

    mono = np.mean(audio, axis=1)
    return mono, int(sample_rate)


def read_required_number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")

    if pd.isna(value):
        raise ValueError(
            f"Missing numeric value in CSV column '{column}'."
        )

    return float(value)


# ============================================================
# PLOTTING
# ============================================================

def make_cutoff_review_plot(
    audio: np.ndarray,
    sample_rate: int,
    bird_name: str,
    active_intervals: list[tuple[float, float]],
    low_cutoff_hz: float,
    without_harmonics_hz: float,
    with_harmonics_hz: float | None,
    output_path: Path,
) -> None:
    """
    Create one cutoff-review spectrogram.

    Birds included in the harmonic comparison show both upper cutoffs.
    Other birds show only the regular/no-harmonic upper cutoff.
    """
    nperseg = int(round(SPECTROGRAM_WINDOW_SECONDS * sample_rate))
    nperseg = min(max(nperseg, 128), len(audio))

    noverlap = int(round(nperseg * SPECTROGRAM_OVERLAP))
    noverlap = min(noverlap, nperseg - 1)

    frequencies, times, psd = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    spectrogram_db = 10.0 * np.log10(
        psd + np.finfo(float).eps
    )

    nyquist_hz = sample_rate / 2.0
    maximum_plot_hz = min(MAX_PLOT_HZ, nyquist_hz)

    figure, axis = plt.subplots(figsize=(15, 8))

    image = axis.pcolormesh(
        times,
        frequencies,
        spectrogram_db,
        shading="auto",
        cmap="magma",
        vmin=COLOR_MIN_DB,
        vmax=COLOR_MAX_DB,
    )

    for index, (start, end) in enumerate(active_intervals):
        axis.axvspan(
            start,
            min(end, len(audio) / sample_rate),
            color="white",
            alpha=0.08,
            label="Active-call interval" if index == 0 else None,
        )

    axis.axhline(
        low_cutoff_hz,
        color=LOW_CUTOFF_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"Common low cutoff: {low_cutoff_hz:.0f} Hz",
    )

    axis.axhline(
        without_harmonics_hz,
        color=WITHOUT_HARMONICS_COLOR,
        linestyle="-.",
        linewidth=2.2,
        label=(
            "Upper cutoff — upper harmonics removed: "
            f"{without_harmonics_hz:.0f} Hz"
        ),
    )

    if with_harmonics_hz is not None:
        axis.axhline(
            with_harmonics_hz,
            color=WITH_HARMONICS_COLOR,
            linestyle="-",
            linewidth=2.2,
            label=(
                "Upper cutoff — upper harmonics preserved: "
                f"{with_harmonics_hz:.0f} Hz"
            ),
        )

    axis.set_xlim(0, len(audio) / sample_rate)
    axis.set_ylim(0, maximum_plot_hz)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(
        f"{bird_name}: review of harmonic and non-harmonic cutoffs"
    )

    axis.legend(
        loc="upper right",
        framealpha=0.92,
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Power spectral density (dB)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not AUDIO_FOLDER.exists():
        raise FileNotFoundError(
            f"Audio folder not found: {AUDIO_FOLDER.resolve()}"
        )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"CSV not found: {INPUT_CSV.resolve()}"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(INPUT_CSV)

    required_columns = {
        "name",
        "id",
        "active_intervals_seconds",
        "final_filter_low_hz",
        "final_filter_high_without_harmonics_hz",
        "final_filter_high_with_harmonics_hz",
        "harmonic_test_required",
    }

    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Intentionally process every CSV row so all 20 birds receive plots.
    if table.empty:
        raise ValueError(
            "No birds were selected for harmonic cutoff review."
        )

    results: list[dict[str, object]] = []

    print("=" * 78)
    print("HARMONIC CUTOFF-REVIEW SPECTROGRAMS")
    print(f"CSV rows to process: {len(table)}")
    print("=" * 78)

    for index, (_, row) in enumerate(table.iterrows(), start=1):
        bird_name = str(row["name"]).strip()
        bird_id = str(row.get("id", "")).strip()

        print(f"\n[{index:02d}/{len(table)}] {bird_name}")

        try:
            csv_filename = row.get("audio_filename")
            if pd.isna(csv_filename):
                csv_filename = None

            audio_path = find_audio_file(
                AUDIO_FOLDER,
                bird_name,
                bird_id,
                csv_filename,
            )

            audio, sample_rate = load_analysis_audio(audio_path)

            intervals = parse_active_intervals(
                row["active_intervals_seconds"]
            )

            low_hz = read_required_number(
                row,
                "final_filter_low_hz",
            )
            without_harmonics_hz = read_required_number(
                row,
                "final_filter_high_without_harmonics_hz",
            )
            harmonic_required = (
                str(row.get("harmonic_test_required", "no"))
                .strip()
                .lower()
                in {"yes", "y", "true", "1"}
            )

            with_harmonics_hz: float | None = None

            if harmonic_required:
                with_harmonics_hz = read_required_number(
                    row,
                    "final_filter_high_with_harmonics_hz",
                )

            nyquist_hz = sample_rate / 2.0

            if not 0 < low_hz < without_harmonics_hz:
                raise ValueError(
                    "Expected low cutoff < final/no-harmonic upper cutoff."
                )

            if without_harmonics_hz >= nyquist_hz:
                raise ValueError(
                    f"Upper cutoff {without_harmonics_hz:.0f} Hz "
                    f"is not below Nyquist {nyquist_hz:.1f} Hz."
                )

            if with_harmonics_hz is not None:
                if without_harmonics_hz >= with_harmonics_hz:
                    raise ValueError(
                        "The with-harmonics upper cutoff must be higher "
                        "than the without-harmonics upper cutoff."
                    )

                if with_harmonics_hz >= nyquist_hz:
                    raise ValueError(
                        f"With-harmonics cutoff "
                        f"{with_harmonics_hz:.0f} Hz is not below "
                        f"Nyquist {nyquist_hz:.1f} Hz."
                    )

            safe_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                bird_name,
            ).strip("_")

            output_path = (
                OUTPUT_FOLDER
                / f"{safe_name}_cutoff_review.png"
            )

            make_cutoff_review_plot(
                audio=audio,
                sample_rate=sample_rate,
                bird_name=bird_name,
                active_intervals=intervals,
                low_cutoff_hz=low_hz,
                without_harmonics_hz=without_harmonics_hz,
                with_harmonics_hz=with_harmonics_hz,
                output_path=output_path,
            )

            results.append(
                {
                    "name": bird_name,
                    "audio_filename": audio_path.name,
                    "sample_rate_hz": sample_rate,
                    "nyquist_hz": nyquist_hz,
                    "final_filter_low_hz": low_hz,
                    "final_filter_high_without_harmonics_hz":
                        without_harmonics_hz,
                    "final_filter_high_with_harmonics_hz":
                        with_harmonics_hz,
                    "harmonic_test_required":
                        "yes" if harmonic_required else "no",
                    "plot_file": str(output_path),
                    "status": "OK",
                    "error": "",
                }
            )

            print(f"  Audio: {audio_path.name}")
            print(
                f"  Final/no-harmonic range: "
                f"{low_hz:.0f}-{without_harmonics_hz:.0f} Hz"
            )

            if with_harmonics_hz is not None:
                print(
                    f"  With upper harmonics:    "
                    f"{low_hz:.0f}-{with_harmonics_hz:.0f} Hz"
                )
            else:
                print("  Harmonic comparison:     not included")

            print(f"  Saved: {output_path}")

        except Exception as error:
            print(f"  ERROR: {error}")

            results.append(
                {
                    "name": bird_name,
                    "status": "ERROR",
                    "error": str(error),
                }
            )

    report_path = OUTPUT_FOLDER / "cutoff_review_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    successful = sum(
        result.get("status") == "OK"
        for result in results
    )
    failed = len(results) - successful

    print("\n" + "=" * 78)
    print("COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Plots:      {OUTPUT_FOLDER.resolve()}")
    print(f"Report:     {report_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
