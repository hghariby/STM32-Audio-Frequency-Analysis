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
# PATHS
# ============================================================

ROOT = Path(".")

METADATA_CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

BRANCHES = {
    "with_harmonics": {
        "before_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Mono_Audio"
        ),
        "after_folder": (
            ROOT
            / "Filtered_With_Harmonics"
            / "Silenced_Audio"
        ),
        "report_csv": (
            ROOT
            / "Filtered_With_Harmonics"
            / "silencing_validation_report.csv"
        ),
    },
    "without_harmonics": {
        "before_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Mono_Audio"
        ),
        "after_folder": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "Silenced_Audio"
        ),
        "report_csv": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "silencing_validation_report.csv"
        ),
    },
}

SPECTROGRAM_ROOT = (
    ROOT
    / "Frequency Range Results"
    / "silenced_comparison_review"
)


# ============================================================
# SPECTROGRAM SETTINGS
# ============================================================

WINDOW_SECONDS = 0.025
OVERLAP_FRACTION = 0.75
MAX_DISPLAY_HZ = 15000.0

# Each before/after pair uses the same scale.
DISPLAY_DYNAMIC_RANGE_DB = 100.0
COLORMAP = "magma"


# ============================================================
# HELPERS
# ============================================================

def normalized_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def safe_filename(text: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        Path(str(text)).stem,
    ).strip("_")


def harmonic_test_required(row: pd.Series) -> bool:
    return (
        str(row.get("harmonic_test_required", "no"))
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def find_wav(folder: Path, bird_name: str) -> Path:
    target = normalized_name(bird_name)

    matches = [
        path
        for path in folder.glob("*.wav")
        if normalized_name(path.name) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No WAV found for '{bird_name}' in "
            f"{folder.resolve()}"
        )

    raise ValueError(
        f"Multiple WAV files matched '{bird_name}' in "
        f"{folder.resolve()}: "
        + ", ".join(path.name for path in matches)
    )


def parse_intervals(
    text: str,
) -> list[tuple[float, float]]:
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
            raise ValueError(
                f"Invalid active interval: {part}"
            )

        start, end = map(float, match.groups())

        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid active interval: {part}"
            )

        intervals.append((start, end))

    if not intervals:
        raise ValueError(
            "No active intervals were found."
        )

    return intervals


def load_audio(
    path: Path,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.size == 0:
        raise ValueError(
            f"{path.name} is empty."
        )

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"{path.name} has unsupported shape "
            f"{audio.shape}."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"{path.name} contains invalid samples."
        )

    return audio, int(sample_rate)

def channel_count(audio: np.ndarray) -> int:
    return (
        1
        if audio.ndim == 1
        else int(audio.shape[1])
    )


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

        start_sample = int(
            round(start * sample_rate)
        )
        end_sample = int(
            round(end * sample_rate)
        )

        if end_sample > start_sample:
            pieces.append(
                audio[start_sample:end_sample]
            )

    if not pieces:
        raise ValueError(
            "No active samples overlapped the audio."
        )

    return np.concatenate(
        pieces,
        axis=0,
    )


def make_inactive_mask(
    length: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    active_mask = np.zeros(
        length,
        dtype=bool,
    )

    duration = length / sample_rate

    for start, end in intervals:
        if start >= duration:
            continue

        end = min(end, duration)

        start_sample = max(
            0,
            min(
                length,
                int(round(start * sample_rate)),
            ),
        )

        end_sample = max(
            0,
            min(
                length,
                int(round(end * sample_rate)),
            ),
        )

        if end_sample > start_sample:
            active_mask[
                start_sample:end_sample
            ] = True

    return ~active_mask


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    if rms <= 0:
        return float("-inf")

    return float(
        20.0 * math.log10(rms)
    )


def to_mono_for_plot(
    audio: np.ndarray,
) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    return np.mean(
        audio,
        axis=1,
    )


def calculate_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    mono = to_mono_for_plot(audio)

    nperseg = int(
        round(
            WINDOW_SECONDS
            * sample_rate
        )
    )

    nperseg = min(
        max(nperseg, 128),
        len(mono),
    )

    noverlap = min(
        int(
            round(
                nperseg
                * OVERLAP_FRACTION
            )
        ),
        nperseg - 1,
    )

    frequencies, times, power = (
        signal.spectrogram(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
    )

    power_db = (
        10.0
        * np.log10(
            power
            + np.finfo(float).eps
        )
    )

    return (
        frequencies,
        times,
        power_db,
    )


def draw_interval_boundaries(
    axis,
    intervals: list[tuple[float, float]],
    duration: float,
) -> None:
    for start, end in intervals:
        if start >= duration:
            continue

        end = min(end, duration)

        axis.axvline(
            start,
            color="white",
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
        )

        axis.axvline(
            end,
            color="white",
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
        )


def save_comparison_plot(
    *,
    bird_name: str,
    branch: str,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    output_path: Path,
) -> None:
    before_f, before_t, before_db = (
        calculate_spectrogram(
            before_audio,
            sample_rate,
        )
    )

    after_f, after_t, after_db = (
        calculate_spectrogram(
            after_audio,
            sample_rate,
        )
    )

    # Same scale for both panels.
    pair_max = float(
        max(
            np.max(before_db),
            np.max(after_db),
        )
    )

    vmax = pair_max
    vmin = (
        pair_max
        - DISPLAY_DYNAMIC_RANGE_DB
    )

    before_duration = (
        len(before_audio)
        / sample_rate
    )

    after_duration = (
        len(after_audio)
        / sample_rate
    )

    maximum_duration = max(
        before_duration,
        after_duration,
    )

    maximum_frequency = min(
        MAX_DISPLAY_HZ,
        sample_rate / 2.0,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            before_t,
            before_f,
            before_db,
            "Before silencing — filtered audio",
            before_duration,
        ),
        (
            axes[1],
            after_t,
            after_f,
            after_db,
            "After silencing and fading",
            after_duration,
        ),
    ]

    last_image = None

    for (
        axis,
        times,
        frequencies,
        spectrogram_db,
        title,
        duration,
    ) in panels:
        last_image = axis.pcolormesh(
            times,
            frequencies,
            spectrogram_db,
            shading="auto",
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
        )

        draw_interval_boundaries(
            axis,
            intervals,
            duration,
        )

        axis.set_title(title)
        axis.set_ylabel(
            "Frequency (Hz)"
        )
        axis.set_xlim(
            0,
            maximum_duration,
        )
        axis.set_ylim(
            0,
            maximum_frequency,
        )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    figure.suptitle(
        f"{bird_name} — "
        f"{branch.replace('_', ' ')}",
        fontsize=15,
    )

    colorbar = figure.colorbar(
        last_image,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        shrink=0.96,
    )

    colorbar.set_label(
        "Power spectral density (dB)"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: "
            f"{METADATA_CSV.resolve()}"
        )

    table = pd.read_csv(
        METADATA_CSV
    )

    required_columns = {
        "name",
        "active_intervals_seconds",
        "harmonic_test_required",
    }

    missing = (
        required_columns
        - set(table.columns)
    )

    if missing:
        raise ValueError(
            "Metadata CSV is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    SPECTROGRAM_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "SILENCED-AUDIO SPECTROGRAM "
        "AND NUMERICAL VALIDATION"
    )
    print("=" * 78)

    for branch, paths in BRANCHES.items():
        before_folder = paths[
            "before_folder"
        ]
        after_folder = paths[
            "after_folder"
        ]
        report_csv = paths[
            "report_csv"
        ]

        if not before_folder.exists():
            raise FileNotFoundError(
                before_folder
            )

        if not after_folder.exists():
            raise FileNotFoundError(
                after_folder
            )

        plot_folder = (
            SPECTROGRAM_ROOT
            / branch
        )

        plot_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_rows: list[
            dict[str, object]
        ] = []

        print(
            f"\nBranch: {branch}"
        )

        for index, (_, row) in enumerate(
            table.iterrows(),
            start=1,
        ):
            name = str(
                row["name"]
            ).strip()

            if (
                branch == "with_harmonics"
                and not harmonic_test_required(
                    row
                )
            ):
                continue

            print(
                f"[{index:02d}/{len(table)}] "
                f"{name}"
            )

            report_row: dict[
                str,
                object,
            ] = {
                "name": name,
                "branch": branch,
                "sample_rate_hz": np.nan,
                "channels": np.nan,
                "duration_before_seconds":
                    np.nan,
                "duration_after_seconds":
                    np.nan,
                "active_rms_before_dbfs":
                    np.nan,
                "active_rms_after_dbfs":
                    np.nan,
                "active_rms_change_db":
                    np.nan,
                "inactive_max_abs_after":
                    np.nan,
                "status": "ERROR",
                "error": "",
            }

            try:
                active_intervals = (
                    parse_intervals(
                        row[
                            "active_intervals_seconds"
                        ]
                    )
                )

                before_path = find_wav(
                    before_folder,
                    name,
                )

                after_path = find_wav(
                    after_folder,
                    name,
                )

                before_audio, before_sr = (
                    load_audio(
                        before_path
                    )
                )

                after_audio, after_sr = (
                    load_audio(
                        after_path
                    )
                )


                if before_sr != after_sr:
                    raise ValueError(
                        "Sample-rate mismatch: "
                        f"{before_sr} vs "
                        f"{after_sr} Hz."
                    )

                before_channels = (
                    channel_count(
                        before_audio
                    )
                )

                after_channels = (
                    channel_count(
                        after_audio
                    )
                )

                if (
                    before_channels
                    != after_channels
                ):
                    raise ValueError(
                        "Channel-count mismatch: "
                        f"{before_channels} vs "
                        f"{after_channels}."
                    )

                before_duration = (
                    len(before_audio)
                    / before_sr
                )

                after_duration = (
                    len(after_audio)
                    / after_sr
                )

                active_before = (
                    extract_active_samples(
                        before_audio,
                        before_sr,
                        active_intervals,
                    )
                )

                active_after = (
                    extract_active_samples(
                        after_audio,
                        after_sr,
                        active_intervals,
                    )
                )

                active_before_dbfs = (
                    rms_dbfs(
                        active_before
                    )
                )

                active_after_dbfs = (
                    rms_dbfs(
                        active_after
                    )
                )

                active_change_db = (
                    active_after_dbfs
                    - active_before_dbfs
                )

                inactive_mask = (
                    make_inactive_mask(
                        len(after_audio),
                        after_sr,
                        active_intervals,
                    )
                )

                if after_audio.ndim == 1:
                    inactive_after = (
                        after_audio[
                            inactive_mask
                        ]
                    )
                else:
                    inactive_after = (
                        after_audio[
                            inactive_mask,
                            :,
                        ]
                    )

                inactive_max = (
                    float(
                        np.max(
                            np.abs(
                                inactive_after
                            )
                        )
                    )
                    if inactive_after.size
                    else 0.0
                )

                output_path = (
                    plot_folder
                    / (
                        safe_filename(name)
                        + "_silencing_comparison.png"
                    )
                )

                save_comparison_plot(
                    bird_name=name,
                    branch=branch,
                    before_audio=before_audio,
                    after_audio=after_audio,
                    sample_rate=before_sr,
                    intervals=active_intervals,
                    output_path=output_path,
                )

                report_row.update({
                    "sample_rate_hz":
                        before_sr,
                    "channels":
                        before_channels,
                    "duration_before_seconds":
                        before_duration,
                    "duration_after_seconds":
                        after_duration,
                    "active_rms_before_dbfs":
                        active_before_dbfs,
                    "active_rms_after_dbfs":
                        active_after_dbfs,
                    "active_rms_change_db":
                        active_change_db,
                    "inactive_max_abs_after":
                        inactive_max,
                    "status": "OK",
                    "error": "",
                })

                print(
                    "  Active RMS change: "
                    f"{active_change_db:.4f} dB"
                )

                print(
                    "  Inactive max after: "
                    f"{inactive_max:.10f}"
                )

                print(
                    f"  Plot: {output_path}"
                )

            except Exception as error:
                report_row["error"] = str(
                    error
                )

                print(
                    f"  ERROR: {error}"
                )

            report_rows.append(
                report_row
            )

        report = pd.DataFrame(
            report_rows,
            columns=[
                "name",
                "branch",
                "sample_rate_hz",
                "channels",
                "duration_before_seconds",
                "duration_after_seconds",
                "active_rms_before_dbfs",
                "active_rms_after_dbfs",
                "active_rms_change_db",
                "inactive_max_abs_after",
                "status",
                "error",
            ],
        )

        report.to_csv(
            report_csv,
            index=False,
        )

        successful = int(
            (
                report["status"]
                == "OK"
            ).sum()
        )

        print(
            f"Saved report: "
            f"{report_csv.resolve()}"
        )

        print(
            f"Successful: "
            f"{successful}/"
            f"{len(report)}"
        )

    print("\n" + "=" * 78)
    print("VALIDATION COMPLETE")
    print(
        f"Spectrograms: "
        f"{SPECTROGRAM_ROOT.resolve()}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
