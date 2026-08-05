from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


ROOT = Path(".")

METADATA_CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

BRANCHES = {
    "with_harmonics": {
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
        "normalization_report_csv": (
            ROOT
            / "Filtered_With_Harmonics"
            / "normalization_report.csv"
        ),
    },
    "without_harmonics": {
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
        "normalization_report_csv": (
            ROOT
            / "Filtered_Without_Harmonics"
            / "normalization_report.csv"
        ),
    },
}

SPECTROGRAM_ROOT = (
    ROOT
    / "Frequency Range Results"
    / "normalization_comparison_review"
)

TARGET_RMS_DBFS = -25.0
MAX_ALLOWED_PEAK_DBFS = -1.0
OUTPUT_SUBTYPE = "PCM_16"

WINDOW_SECONDS = 0.025
OVERLAP_FRACTION = 0.75
MAX_DISPLAY_HZ = 15000.0
DISPLAY_DYNAMIC_RANGE_DB = 100.0
COLORMAP = "magma"


def normalize_name(text: str) -> str:
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


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.ndim != 1:
        raise ValueError(
            f"{path.name} is not mono."
        )

    if audio.size == 0:
        raise ValueError(
            f"{path.name} is empty."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"{path.name} contains invalid samples."
        )

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

        clipped_end = min(
            end,
            duration,
        )

        start_sample = int(
            round(start * sample_rate)
        )

        end_sample = int(
            round(clipped_end * sample_rate)
        )

        if end_sample > start_sample:
            pieces.append(
                audio[start_sample:end_sample]
            )

    if not pieces:
        raise ValueError(
            "No active samples overlapped the audio."
        )

    return np.concatenate(pieces)


def rms_dbfs(
    audio: np.ndarray,
) -> tuple[float, float]:
    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    if rms <= 0:
        return 0.0, -math.inf

    return (
        rms,
        20.0 * math.log10(rms),
    )


def peak_dbfs(
    audio: np.ndarray,
) -> tuple[float, float]:
    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak <= 0:
        return 0.0, -math.inf

    return (
        peak,
        20.0 * math.log10(peak),
    )


def normalize_file(
    *,
    input_path: Path,
    output_path: Path,
    intervals: list[tuple[float, float]],
) -> dict[str, object]:
    audio, sample_rate = load_mono_audio(
        input_path
    )

    active_before = extract_active_samples(
        audio,
        sample_rate,
        intervals,
    )

    before_rms_linear, before_rms_dbfs = (
        rms_dbfs(active_before)
    )

    if before_rms_linear <= 0:
        raise ValueError(
            "Active RMS is zero."
        )

    target_linear = (
        10.0
        ** (
            TARGET_RMS_DBFS
            / 20.0
        )
    )

    gain_linear = (
        target_linear
        / before_rms_linear
    )

    gain_db = (
        20.0
        * math.log10(
            gain_linear
        )
    )

    normalized = (
        audio
        * gain_linear
    )

    peak_after_linear, peak_after_dbfs = (
        peak_dbfs(normalized)
    )

    if (
        peak_after_dbfs
        > MAX_ALLOWED_PEAK_DBFS
        + 1e-9
    ):
        raise ValueError(
            f"Required gain would produce a peak of "
            f"{peak_after_dbfs:.2f} dBFS, above "
            f"{MAX_ALLOWED_PEAK_DBFS:.2f} dBFS."
        )

    active_after = extract_active_samples(
        normalized,
        sample_rate,
        intervals,
    )

    after_rms_linear, after_rms_dbfs = (
        rms_dbfs(active_after)
    )

    sf.write(
        output_path,
        normalized,
        sample_rate,
        subtype=OUTPUT_SUBTYPE,
    )

    return {
        "input_filename": input_path.name,
        "normalized_filename": output_path.name,
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "active_rms_before_linear":
            before_rms_linear,
        "active_rms_before_dbfs":
            before_rms_dbfs,
        "gain_linear":
            gain_linear,
        "gain_db":
            gain_db,
        "active_rms_after_linear":
            after_rms_linear,
        "active_rms_after_dbfs":
            after_rms_dbfs,
        "full_file_peak_after_linear":
            peak_after_linear,
        "full_file_peak_after_dbfs":
            peak_after_dbfs,
        "status": "OK",
        "error": "",
    }


def calculate_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    nperseg = int(
        round(
            WINDOW_SECONDS
            * sample_rate
        )
    )

    nperseg = min(
        max(nperseg, 128),
        len(audio),
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
            audio,
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

    maximum_frequency = min(
        MAX_DISPLAY_HZ,
        sample_rate / 2.0,
    )

    duration = (
        len(before_audio)
        / sample_rate
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
            "Mono audio before normalization",
        ),
        (
            axes[1],
            after_t,
            after_f,
            after_db,
            "Normalized mono audio",
        ),
    ]

    last_image = None

    for (
        axis,
        times,
        frequencies,
        spectrogram_db,
        title,
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

        for start, end in intervals:
            axis.axvline(
                start,
                color="white",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

            axis.axvline(
                min(end, duration),
                color="white",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

        axis.set_title(title)
        axis.set_ylabel(
            "Frequency (Hz)"
        )
        axis.set_xlim(
            0,
            duration,
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


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: "
            f"{METADATA_CSV.resolve()}"
        )

    metadata = pd.read_csv(
        METADATA_CSV
    )

    required_columns = {
        "name",
        "id",
        "active_intervals_seconds",
        "harmonic_test_required",
    }

    missing = (
        required_columns
        - set(metadata.columns)
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
    print("MONO NORMALIZATION AND SPECTROGRAM REVIEW")
    print(
        f"Target active RMS: "
        f"{TARGET_RMS_DBFS:.2f} dBFS"
    )
    print(
        f"Maximum allowed peak: "
        f"{MAX_ALLOWED_PEAK_DBFS:.2f} dBFS"
    )
    print("=" * 78)

    for branch, paths in BRANCHES.items():
        silenced_folder = paths[
            "silenced_folder"
        ]

        normalized_folder = paths[
            "normalized_folder"
        ]

        if not silenced_folder.exists():
            raise FileNotFoundError(
                silenced_folder
            )

        normalized_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_folder = (
            SPECTROGRAM_ROOT
            / branch
        )

        plot_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows: list[
            dict[str, object]
        ] = []

        print(
            f"\nBRANCH: {branch}"
        )

        for index, (_, row) in enumerate(
            metadata.iterrows(),
            start=1,
        ):
            if (
                branch == "with_harmonics"
                and not harmonic_test_required(
                    row
                )
            ):
                continue

            name = str(
                row["name"]
            ).strip()

            bird_id = row.get(
                "id",
                "",
            )

            print(
                f"[{index:02d}/{len(metadata)}] "
                f"{name}"
            )

            try:
                intervals = parse_intervals(
                    row[
                        "active_intervals_seconds"
                    ]
                )

                input_path = find_audio_file(
                    silenced_folder,
                    name,
                )

                output_path = (
                    normalized_folder
                    / input_path.name
                )

                result = normalize_file(
                    input_path=input_path,
                    output_path=output_path,
                    intervals=intervals,
                )

                result = {
                    "name": name,
                    "id": bird_id,
                    **result,
                }

                rows.append(
                    result
                )

                before_audio, before_sr = (
                    load_mono_audio(
                        input_path
                    )
                )

                after_audio, after_sr = (
                    load_mono_audio(
                        output_path
                    )
                )

                if before_sr != after_sr:
                    raise ValueError(
                        "Sample-rate mismatch "
                        "after normalization."
                    )

                plot_path = (
                    plot_folder
                    / (
                        safe_filename(name)
                        + "_normalization_comparison.png"
                    )
                )

                save_comparison_plot(
                    bird_name=name,
                    branch=branch,
                    before_audio=before_audio,
                    after_audio=after_audio,
                    sample_rate=before_sr,
                    intervals=intervals,
                    output_path=plot_path,
                )

                print(
                    f"  RMS before: "
                    f"{result['active_rms_before_dbfs']:.2f} dBFS"
                )

                print(
                    f"  Gain: "
                    f"{result['gain_db']:+.2f} dB"
                )

                print(
                    f"  RMS after: "
                    f"{result['active_rms_after_dbfs']:.2f} dBFS"
                )

                print(
                    f"  Peak after: "
                    f"{result['full_file_peak_after_dbfs']:.2f} dBFS"
                )

                print(
                    f"  Plot: "
                    f"{plot_path}"
                )

            except Exception as error:
                print(
                    f"  ERROR: {error}"
                )

                rows.append({
                    "name": name,
                    "id": bird_id,
                    "status": "ERROR",
                    "error": str(error),
                })

        report_columns = [
            "name",
            "id",
            "input_filename",
            "normalized_filename",
            "sample_rate_hz",
            "channels",
            "target_rms_dbfs",
            "active_rms_before_linear",
            "active_rms_before_dbfs",
            "gain_linear",
            "gain_db",
            "active_rms_after_linear",
            "active_rms_after_dbfs",
            "full_file_peak_after_linear",
            "full_file_peak_after_dbfs",
            "status",
            "error",
        ]

        report = pd.DataFrame(
            rows,
            columns=report_columns,
        )

        report.to_csv(
            paths[
                "normalization_report_csv"
            ],
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
            f"{paths['normalization_report_csv'].resolve()}"
        )

        print(
            f"Successful: "
            f"{successful}/"
            f"{len(report)}"
        )

    print("\n" + "=" * 78)
    print("NORMALIZATION COMPLETE")
    print(
        f"Spectrograms: "
        f"{SPECTROGRAM_ROOT.resolve()}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
