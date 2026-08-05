from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


ROOT = Path(".")

CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

ORIGINAL = ROOT / "Trimmed"
WITH = ROOT / "Filtered_With_Harmonics" / "Audio"
WITHOUT = ROOT / "Filtered_Without_Harmonics" / "Audio"

OUTPUT = (
    ROOT
    / "Frequency Range Results"
    / "filter_bandpower_validation.csv"
)


def key(text: str) -> str:
    text = Path(str(text)).stem.lower().replace("’", "'")
    text = re.sub(r"\(\d+\)$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def find_wav(folder: Path, name: str) -> Path:
    matches = [
        path
        for path in folder.glob("*.wav")
        if key(path.name) == key(name)
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one WAV for '{name}' in "
            f"{folder}, found {len(matches)}."
        )

    return matches[0]


def intervals(
    text: str,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []

    for part in str(text).split(";"):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*-\s*"
            r"(\d+(?:\.\d+)?)\s*",
            part.strip(),
        )

        if match:
            start, end = map(float, match.groups())

            if end <= start:
                raise ValueError(
                    f"Invalid active interval: {part}"
                )

            result.append((start, end))

    if not result:
        raise ValueError("No active intervals.")

    return result


def load_active(
    path: Path,
    selected: list[tuple[float, float]],
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        always_2d=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    mono = np.mean(
        audio,
        axis=1,
    )

    pieces: list[np.ndarray] = []
    duration = len(mono) / sample_rate

    for start, end in selected:
        if start >= duration:
            continue

        start_sample = round(
            start * sample_rate
        )
        end_sample = round(
            min(end, duration) * sample_rate
        )

        if end_sample - start_sample >= 2:
            pieces.append(
                mono[start_sample:end_sample]
            )

    if not pieces:
        raise ValueError(
            f"No active samples in {path.name}"
        )

    return np.concatenate(pieces), int(sample_rate)


def psd(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(
        max(
            round(0.10 * sample_rate),
            256,
        ),
        len(audio),
    )

    noverlap = min(
        round(0.50 * nperseg),
        nperseg - 1,
    )

    return signal.welch(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )


def bandpower(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
        & np.isfinite(power)
    )

    if np.count_nonzero(mask) < 2:
        return 0.0

    return float(
        np.trapezoid(
            power[mask],
            frequencies[mask],
        )
    )


def change_db(
    after: float,
    before: float,
) -> float:
    if before <= 0:
        return math.nan

    if after <= 0:
        return -math.inf

    return 10.0 * math.log10(
        after / before
    )


def number(
    row: pd.Series,
    column: str,
) -> float:
    value = pd.to_numeric(
        row.get(column),
        errors="coerce",
    )

    if pd.isna(value):
        raise ValueError(
            f"Missing {column}"
        )

    return float(value)


def is_yes(value: object) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def add_band_results(
    result: dict[str, object],
    *,
    band_name: str,
    band_low: float,
    band_high: float,
    original_frequencies: np.ndarray,
    original_psd: np.ndarray,
    without_frequencies: np.ndarray,
    without_psd: np.ndarray,
    with_frequencies: np.ndarray | None,
    with_psd: np.ndarray | None,
) -> None:
    original_power = bandpower(
        original_frequencies,
        original_psd,
        band_low,
        band_high,
    )

    without_power = bandpower(
        without_frequencies,
        without_psd,
        band_low,
        band_high,
    )

    result[f"{band_name}_low_hz"] = band_low
    result[f"{band_name}_high_hz"] = band_high

    result[
        f"{band_name}_original_power"
    ] = original_power

    result[
        f"{band_name}_without_power"
    ] = without_power

    result[
        f"{band_name}_without_change_db"
    ] = change_db(
        without_power,
        original_power,
    )

    if (
        with_frequencies is not None
        and with_psd is not None
    ):
        with_power = bandpower(
            with_frequencies,
            with_psd,
            band_low,
            band_high,
        )

        result[
            f"{band_name}_with_power"
        ] = with_power

        result[
            f"{band_name}_with_change_db"
        ] = change_db(
            with_power,
            original_power,
        )

        result[
            f"{band_name}_without_vs_with_db"
        ] = change_db(
            without_power,
            with_power,
        )

    else:
        result[
            f"{band_name}_with_power"
        ] = np.nan

        result[
            f"{band_name}_with_change_db"
        ] = np.nan

        result[
            f"{band_name}_without_vs_with_db"
        ] = np.nan


def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(CSV)

    for folder in [
        ORIGINAL,
        WITHOUT,
    ]:
        if not folder.exists():
            raise FileNotFoundError(folder)

    table = pd.read_csv(CSV)
    rows: list[dict[str, object]] = []

    for i, row in table.iterrows():
        name = str(row["name"]).strip()
        harmonic_required = is_yes(
            row.get(
                "harmonic_test_required",
                "no",
            )
        )

        print(
            f"[{i + 1:02d}/{len(table)}] "
            f"{name}"
        )

        try:
            active_intervals = intervals(
                row[
                    "active_intervals_seconds"
                ]
            )

            original, original_sr = load_active(
                find_wav(
                    ORIGINAL,
                    name,
                ),
                active_intervals,
            )

            without_audio, without_sr = load_active(
                find_wav(
                    WITHOUT,
                    name,
                ),
                active_intervals,
            )

            if original_sr != without_sr:
                raise ValueError(
                    "Original and without-harmonics "
                    "sample rates do not match."
                )

            original_f, original_power = psd(
                original,
                original_sr,
            )

            without_f, without_power = psd(
                without_audio,
                without_sr,
            )

            with_audio = None
            with_sr = None
            with_f = None
            with_power = None

            if harmonic_required:
                with_audio, with_sr = load_active(
                    find_wav(
                        WITH,
                        name,
                    ),
                    active_intervals,
                )

                if with_sr != original_sr:
                    raise ValueError(
                        "With-harmonics sample rate "
                        "does not match the original."
                    )

                with_f, with_power = psd(
                    with_audio,
                    with_sr,
                )

            low = number(
                row,
                "final_filter_low_hz",
            )

            high_without = number(
                row,
                "final_filter_high_without_harmonics_hz",
            )

            if harmonic_required:
                high_with = number(
                    row,
                    "final_filter_high_with_harmonics_hz",
                )
            else:
                high_with = np.nan

            result: dict[str, object] = {
                "name": name,
                "harmonic_test_required": (
                    "yes"
                    if harmonic_required
                    else "no"
                ),
                "sample_rate_hz": original_sr,
                "low_cutoff_hz": low,
                "without_upper_cutoff_hz":
                    high_without,
                "with_upper_cutoff_hz":
                    high_with,
                "status": "OK",
                "error": "",
            }

            # Both branches can be evaluated in the
            # below-low and core regions only when a
            # with-harmonics file exists.
            add_band_results(
                result,
                band_name="below_low",
                band_low=0.0,
                band_high=low,
                original_frequencies=original_f,
                original_psd=original_power,
                without_frequencies=without_f,
                without_psd=without_power,
                with_frequencies=with_f,
                with_psd=with_power,
            )

            add_band_results(
                result,
                band_name="core",
                band_low=low,
                band_high=high_without,
                original_frequencies=original_f,
                original_psd=original_power,
                without_frequencies=without_f,
                without_psd=without_power,
                with_frequencies=with_f,
                with_psd=with_power,
            )

            # Upper-band comparison exists only for
            # harmonic-test birds.
            if harmonic_required:
                upper_high = min(
                    high_with,
                    original_sr / 2.0,
                )

                add_band_results(
                    result,
                    band_name="upper",
                    band_low=high_without,
                    band_high=upper_high,
                    original_frequencies=original_f,
                    original_psd=original_power,
                    without_frequencies=without_f,
                    without_psd=without_power,
                    with_frequencies=with_f,
                    with_psd=with_power,
                )
            else:
                for column in [
                    "upper_low_hz",
                    "upper_high_hz",
                    "upper_original_power",
                    "upper_with_power",
                    "upper_without_power",
                    "upper_with_change_db",
                    "upper_without_change_db",
                    "upper_without_vs_with_db",
                ]:
                    result[column] = np.nan

            rows.append(result)

            if harmonic_required:
                print(
                    "  Below-low attenuation: "
                    f"with="
                    f"{result['below_low_with_change_db']:.2f} dB, "
                    f"without="
                    f"{result['below_low_without_change_db']:.2f} dB"
                )

                print(
                    "  Core change: "
                    f"with="
                    f"{result['core_with_change_db']:.2f} dB, "
                    f"without="
                    f"{result['core_without_change_db']:.2f} dB"
                )

                print(
                    "  Upper band, without vs with: "
                    f"{result['upper_without_vs_with_db']:.2f} dB"
                )

            else:
                print(
                    "  Harmonic comparison skipped."
                )

                print(
                    "  Without-harmonics below-low "
                    f"attenuation: "
                    f"{result['below_low_without_change_db']:.2f} dB"
                )

                print(
                    "  Without-harmonics core change: "
                    f"{result['core_without_change_db']:.2f} dB"
                )

        except Exception as error:
            print(f"  ERROR: {error}")

            rows.append({
                "name": name,
                "harmonic_test_required": (
                    "yes"
                    if harmonic_required
                    else "no"
                ),
                "status": "ERROR",
                "error": str(error),
            })

    report = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"Saved: {OUTPUT.resolve()}"
    )

    print(
        report["status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()