from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pandas as pd


# ============================ SETTINGS ============================

FOLDER = Path("Frequency_Test")

EXPECTED_SAMPLE_RATE = 35714
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2

REPORT_NAME = "dc_offset_analysis.csv"

# =================================================================


def to_dbfs(value: float) -> float:
    """Convert a linear PCM16 value to dBFS."""
    if value <= 0:
        return -math.inf

    return 20.0 * math.log10(
        value / 32768.0
    )


def analyze_wav(path: Path) -> dict:
    """Analyze one WAV file for DC offset and basic signal measurements."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            raw_audio = wav_file.readframes(frame_count)

        # ----------------------------------------------------------
        # Basic format checks
        # ----------------------------------------------------------

        format_warnings = []

        if sample_rate != EXPECTED_SAMPLE_RATE:
            format_warnings.append(
                f"sample rate is {sample_rate}, "
                f"expected {EXPECTED_SAMPLE_RATE}"
            )

        if channels != EXPECTED_CHANNELS:
            format_warnings.append(
                f"{channels} channels, "
                f"expected {EXPECTED_CHANNELS}"
            )

        if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
            format_warnings.append(
                f"{sample_width}-byte samples, "
                f"expected {EXPECTED_SAMPLE_WIDTH_BYTES}"
            )

        # This analysis assumes the recordings are PCM16 mono.
        if sample_width != 2:
            raise ValueError(
                "DC analysis currently expects 16-bit PCM audio."
            )

        samples = np.frombuffer(
            raw_audio,
            dtype="<i2",
        ).astype(np.int32)

        if samples.size == 0:
            raise ValueError("No audio samples found.")

        # ----------------------------------------------------------
        # DC OFFSET
        # ----------------------------------------------------------

        mean_dc_offset = float(
            np.mean(samples)
        )

        absolute_dc_offset = abs(
            mean_dc_offset
        )

        dc_percent_full_scale = (
            absolute_dc_offset
            / 32768.0
            * 100.0
        )

        # ----------------------------------------------------------
        # Other useful measurements
        # ----------------------------------------------------------

        rms_counts = float(
            np.sqrt(
                np.mean(
                    samples.astype(
                        np.float64
                    ) ** 2
                )
            )
        )

        peak_counts = int(
            np.max(
                np.abs(samples)
            )
        )

        minimum_sample = int(
            np.min(samples)
        )

        maximum_sample = int(
            np.max(samples)
        )

        duration_seconds = (
            frame_count / sample_rate
            if sample_rate > 0
            else math.nan
        )

        return {
            "file": path.name,

            "sample_rate_hz":
                sample_rate,

            "channels":
                channels,

            "sample_width_bytes":
                sample_width,

            "frame_count":
                frame_count,

            "duration_seconds":
                duration_seconds,

            # Main DC measurements
            "mean_dc_offset_counts":
                mean_dc_offset,

            "absolute_dc_offset_counts":
                absolute_dc_offset,

            "dc_offset_percent_full_scale":
                dc_percent_full_scale,

            # Helpful signal measurements
            "rms_counts":
                rms_counts,

            "rms_dbfs":
                to_dbfs(rms_counts),

            "peak_counts":
                peak_counts,

            "peak_dbfs":
                to_dbfs(peak_counts),

            "minimum_sample":
                minimum_sample,

            "maximum_sample":
                maximum_sample,

            "format_warning":
                " | ".join(
                    format_warnings
                ),

            "status":
                "OK",

            "error":
                "",
        }

    except Exception as exc:

        return {
            "file": path.name,
            "status": "ERROR",
            "error": str(exc),
        }


def print_summary(report: pd.DataFrame) -> None:
    """Print summary statistics for successful recordings."""

    good = report[
        report["status"] == "OK"
    ].copy()

    if good.empty:
        print(
            "\nNo successful WAV files "
            "were available for summary."
        )
        return

    dc = good[
        "absolute_dc_offset_counts"
    ].dropna()

    print("\n" + "=" * 60)
    print("DC OFFSET SUMMARY")
    print("=" * 60)

    print(
        f"Files analyzed: {len(good)}"
    )

    print(
        f"Mean absolute DC offset: "
        f"{dc.mean():.2f} counts"
    )

    print(
        f"Median absolute DC offset: "
        f"{dc.median():.2f} counts"
    )

    print(
        f"Minimum absolute DC offset: "
        f"{dc.min():.2f} counts"
    )

    print(
        f"Maximum absolute DC offset: "
        f"{dc.max():.2f} counts"
    )

    print(
        f"90th percentile: "
        f"{dc.quantile(0.90):.2f} counts"
    )

    print(
        f"95th percentile: "
        f"{dc.quantile(0.95):.2f} counts"
    )

    print(
        f"99th percentile: "
        f"{dc.quantile(0.99):.2f} counts"
    )

    print("\nLargest DC offsets:")

    largest = good.sort_values(
        "absolute_dc_offset_counts",
        ascending=False,
    )[
        [
            "file",
            "mean_dc_offset_counts",
            "absolute_dc_offset_counts",
            "dc_offset_percent_full_scale",
        ]
    ].head(10)

    print(
        largest.to_string(
            index=False
        )
    )


def main() -> None:

    if not FOLDER.exists():
        raise FileNotFoundError(
            f"Folder not found:\n"
            f"{FOLDER.resolve()}"
        )

    wav_files = sorted(
        FOLDER.glob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in:\n"
            f"{FOLDER.resolve()}"
        )

    print(
        f"Found {len(wav_files)} WAV files.\n"
    )

    results = []

    for index, path in enumerate(
        wav_files,
        start=1,
    ):

        print(
            f"[{index}/{len(wav_files)}] "
            f"{path.name}"
        )

        results.append(
            analyze_wav(path)
        )

    report = pd.DataFrame(
        results
    )

    report_path = (
        FOLDER / REPORT_NAME
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print_summary(report)

    print(
        "\nFull report saved to:"
    )

    print(
        report_path.resolve()
    )


if __name__ == "__main__":
    main()