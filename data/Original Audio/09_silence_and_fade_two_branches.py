from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

CSV = (
    ROOT
    / "Frequency Range Results"
    / "bird_frequency_ranges.csv"
)

WITH_INPUT = (
    ROOT
    / "Filtered_With_Harmonics"
    / "Mono_Audio"
)

WITHOUT_INPUT = (
    ROOT
    / "Filtered_Without_Harmonics"
    / "Mono_Audio"
)

WITH_OUTPUT = (
    ROOT
    / "Filtered_With_Harmonics"
    / "Silenced_Audio"
)

WITHOUT_OUTPUT = (
    ROOT
    / "Filtered_Without_Harmonics"
    / "Silenced_Audio"
)

REPORT = (
    ROOT
    / "Frequency Range Results"
    / "silencing_two_branches_report.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FADE_MS = 8.0
OUTPUT_SUBTYPE = "PCM_16"

# ============================================================
# HELPERS
# ============================================================

def key(text: str) -> str:
    text = Path(str(text)).stem
    text = text.lower().replace("’", "'")
    text = re.sub(r"\(\d+\)$", "", text)

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def harmonic_test_required(
    row: pd.Series,
) -> bool:
    return (
        str(
            row.get(
                "harmonic_test_required",
                "no",
            )
        )
        .strip()
        .lower()
        in {"yes", "y", "true", "1"}
    )


def find_wav(
    folder: Path,
    name: str,
) -> Path:
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


def parse_intervals(
    text: str,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []

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
                f"Invalid interval: {part}"
            )

        start, end = map(
            float,
            match.groups(),
        )

        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid interval: {part}"
            )

        result.append(
            (start, end)
        )

    if not result:
        raise ValueError(
            "No active intervals."
        )

    return result

def make_envelope(
    length: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    envelope = np.zeros(
        length,
        dtype=np.float64,
    )

    duration = length / sample_rate

    requested_fade = max(
        1,
        round(
            FADE_MS
            * sample_rate
            / 1000.0
        ),
    )

    for start, end in intervals:
        if start >= duration:
            raise ValueError(
                f"Interval begins at {start:.3f}s; "
                f"duration is {duration:.3f}s."
            )

        clipped_end = min(
            end,
            duration,
        )

        start_sample = max(
            0,
            min(
                length,
                round(
                    start
                    * sample_rate
                ),
            ),
        )

        end_sample = max(
            0,
            min(
                length,
                round(
                    clipped_end
                    * sample_rate
                ),
            ),
        )

        interval_length = (
            end_sample
            - start_sample
        )

        if interval_length < 2:
            continue

        envelope[
            start_sample:end_sample
        ] = 1.0

        fade_samples = min(
            requested_fade,
            interval_length // 2,
        )

        if fade_samples:
            envelope[
                start_sample:
                start_sample + fade_samples
            ] = np.linspace(
                0.0,
                1.0,
                fade_samples,
                endpoint=False,
            )

            envelope[
                end_sample - fade_samples:
                end_sample
            ] = np.linspace(
                1.0,
                0.0,
                fade_samples,
                endpoint=False,
            )

    if not np.any(
        envelope > 0
    ):
        raise ValueError(
            "No intervals overlapped the audio."
        )

    return envelope


def process_one(
    *,
    name: str,
    branch: str,
    input_folder: Path,
    output_folder: Path,
    intervals: list[tuple[float, float]],
) -> dict[str, object]:
    source = find_wav(
        input_folder,
        name,
    )

    audio, sample_rate = sf.read(
        source,
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.size == 0:
        raise ValueError(
            f"{source.name} is empty."
        )

    if audio.ndim not in {1, 2}:
        raise ValueError(
            f"Unsupported audio shape "
            f"{audio.shape}."
        )


    envelope = make_envelope(
        len(audio),
        int(sample_rate),
        intervals,
    )
    
    if audio.ndim == 1:
        output = (
            audio
            * envelope
        )
    else:
        output = (
            audio
            * envelope[:, None]
        )

    destination = (
        output_folder
        / source.name
    )

    sf.write(
        destination,
        output,
        int(sample_rate),
        subtype=OUTPUT_SUBTYPE,
    )

    inactive = (
        envelope == 0
    )

    if np.any(inactive):
        inactive_after_max = float(
            np.max(
                np.abs(
                    output[inactive]
                )
            )
        )
    else:
        inactive_after_max = 0.0

    return {
        "name": name,
        "branch": branch,
        "input_file": source.name,
        "output_file": destination.name,
        "sample_rate_hz": int(sample_rate),
        "channels": (
            1
            if audio.ndim == 1
            else audio.shape[1]
        ),
        "fade_ms": FADE_MS,
        "inactive_after_max_abs":
            inactive_after_max,
        "status": "OK",
        "error": "",
    }

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(CSV)

    if not WITH_INPUT.exists():
        raise FileNotFoundError(
            WITH_INPUT
        )

    if not WITHOUT_INPUT.exists():
        raise FileNotFoundError(
            WITHOUT_INPUT
        )

    WITH_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WITHOUT_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pd.read_csv(CSV)

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
            "Missing CSV columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    rows: list[dict[str, object]] = []

    selected_harmonic_count = int(
        table["harmonic_test_required"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "yes",
                "y",
                "true",
                "1",
            }
        )
        .sum()
    )

    print("=" * 78)
    print("SILENCING AND FADING")
    print(
        f"With-harmonics expected: "
        f"{selected_harmonic_count}"
    )
    print(
        f"Without-harmonics expected: "
        f"{len(table)}"
    )
    print("=" * 78)

    for index, (_, row) in enumerate(
        table.iterrows(),
        start=1,
    ):
        name = str(
            row["name"]
        ).strip()

        include_harmonic_test = (
            harmonic_test_required(row)
        )

        print(
            f"\n[{index:02d}/{len(table)}] "
            f"{name}"
        )

        try:
            active_intervals = (
                parse_intervals(
                    row[
                        "active_intervals_seconds"
                    ]
                )
            )

            # Always process the without-harmonics branch.
            without_result = process_one(
                name=name,
                branch="without_harmonics",
                input_folder=WITHOUT_INPUT,
                output_folder=WITHOUT_OUTPUT,
                intervals=active_intervals,
            )

            rows.append(
                without_result
            )

            print(
                "  Without harmonics saved: "
                f"{WITHOUT_OUTPUT / without_result['output_file']}"
            )

        except Exception as error:
            rows.append({
                "name": name,
                "branch": "without_harmonics",
                "status": "ERROR",
                "error": str(error),
            })

            print(
                "  Without harmonics ERROR: "
                f"{error}"
            )

        if include_harmonic_test:
            try:
                with_result = process_one(
                    name=name,
                    branch="with_harmonics",
                    input_folder=WITH_INPUT,
                    output_folder=WITH_OUTPUT,
                    intervals=active_intervals,
                )

                rows.append(
                    with_result
                )

                print(
                    "  With harmonics saved:    "
                    f"{WITH_OUTPUT / with_result['output_file']}"
                )

            except Exception as error:
                rows.append({
                    "name": name,
                    "branch": "with_harmonics",
                    "status": "ERROR",
                    "error": str(error),
                })

                print(
                    "  With harmonics ERROR: "
                    f"{error}"
                )

        else:
            print(
                "  With harmonics skipped "
                "(harmonic_test_required = no)"
            )

    report = pd.DataFrame(rows)

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT,
        index=False,
    )

    with_rows = report[
        report["branch"]
        == "with_harmonics"
    ]

    without_rows = report[
        report["branch"]
        == "without_harmonics"
    ]

    with_ok = int(
        (
            with_rows["status"]
            == "OK"
        ).sum()
    )

    without_ok = int(
        (
            without_rows["status"]
            == "OK"
        ).sum()
    )

    print("\n" + "=" * 78)
    print("SILENCING COMPLETE")
    print(
        f"With harmonics:    "
        f"{with_ok}/"
        f"{selected_harmonic_count}"
    )
    print(
        f"Without harmonics: "
        f"{without_ok}/"
        f"{len(table)}"
    )
    print(
        f"Report: "
        f"{REPORT.resolve()}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()