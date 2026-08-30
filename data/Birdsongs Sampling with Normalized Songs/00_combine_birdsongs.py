from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

INPUT_FOLDER = ROOT / "Normalized_Audio"

COMBINED_OUTPUT = ROOT / "Combined_Bird_Calls.wav"


# ============================================================
# OUTPUT FORMAT
# ============================================================

TARGET_DURATION_SECONDS = 10.0

# Silence before the first bird and between all birds.
INITIAL_SILENCE_SECONDS = 5.0
SILENCE_BETWEEN_SECONDS = 5.0
FINAL_SILENCE_SECONDS = 5.0

# Short fades prevent clicks and abrupt transitions.
FADE_IN_MS = 20.0
FADE_OUT_MS = 30.0

TARGET_SAMPLE_RATE = 44100
OUTPUT_SUBTYPE = "PCM_16"


# ============================================================
# PLAYBACK ORDER
# ============================================================

BIRD_ORDER = [
    "American Robin",
    "Anna's Hummingbird",
    "Barn Swallow",
    "Bewick's Wren",
    "Black Phoebe",
    "California Towhee",
    "Dark-eyed Junco",
    "European Starling",
    "Hooded Oriole",
    "House Finch",
    "House Sparrow",
    "Lesser Goldfinch",
    "Mourning Dove",
    "Northern House Wren",
    "Northern Mockingbird",
    "Northern Yellow Warbler",
    "Oak Titmouse",
    "Rock Pigeon",
    "Song Sparrow",
    "Spotted Towhee",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"\(\d+\)$", "", text)
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def find_audio_file(folder: Path, bird_name: str) -> Path:
    target = normalize_name(bird_name)

    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and normalize_name(path.name) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No WAV file found for '{bird_name}' in {folder.resolve()}"
        )

    raise ValueError(
        f"More than one WAV file matched '{bird_name}': "
        + ", ".join(path.name for path in matches)
    )



def load_and_validate_audio(
    input_path: Path,
) -> np.ndarray:
    """
    Load one already-prepared WAV and verify that it is:
    - mono,
    - 44.1 kHz,
    - exactly 10 seconds,
    - nonempty,
    - below digital full scale.
    """
    audio, sample_rate = sf.read(
        input_path,
        dtype="float64",
        always_2d=False,
    )

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"{input_path.name}: sample rate is "
            f"{sample_rate} Hz, expected "
            f"{TARGET_SAMPLE_RATE} Hz."
        )

    if audio.ndim != 1:
        raise ValueError(
            f"{input_path.name}: expected mono audio, "
            f"but shape is {audio.shape}."
        )

    if len(audio) == 0:
        raise ValueError(
            f"{input_path.name}: file is empty."
        )

    expected_samples = int(
        round(
            TARGET_DURATION_SECONDS
            * TARGET_SAMPLE_RATE
        )
    )

    if len(audio) != expected_samples:
        actual_duration = (
            len(audio)
            / sample_rate
        )

        raise ValueError(
            f"{input_path.name}: duration is "
            f"{actual_duration:.6f} s, expected exactly "
            f"{TARGET_DURATION_SECONDS:.3f} s.\n"
            f"Samples: {len(audio):,}, "
            f"expected {expected_samples:,}."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"{input_path.name}: contains NaN or infinite values."
        )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1.0:
        raise ValueError(
            f"{input_path.name}: exceeds digital full scale. "
            f"Peak={peak:.6f}"
        )

    return audio


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER.resolve()}"
        )

    bird_samples = int(
        round(
            TARGET_DURATION_SECONDS
            * TARGET_SAMPLE_RATE
        )
    )

    initial_silence_samples = int(
        round(
            INITIAL_SILENCE_SECONDS
            * TARGET_SAMPLE_RATE
        )
    )

    between_silence_samples = int(
        round(
            SILENCE_BETWEEN_SECONDS
            * TARGET_SAMPLE_RATE
        )
    )

    final_silence_samples = int(
        round(
            FINAL_SILENCE_SECONDS
            * TARGET_SAMPLE_RATE
        )
    )

    initial_silence = np.zeros(
        initial_silence_samples,
        dtype=np.float64,
    )

    between_silence = np.zeros(
        between_silence_samples,
        dtype=np.float64,
    )

    final_silence = np.zeros(
        final_silence_samples,
        dtype=np.float64,
    )


    combined_parts: list[np.ndarray] = [
            initial_silence
        ]

    print("=" * 78)
    print("COMBINE PREPARED BIRD-CALL FILES")
    print("=" * 78)

    for index, bird_name in enumerate(
        BIRD_ORDER,
        start=1,
    ):
        input_path = find_audio_file(
            INPUT_FOLDER,
            bird_name,
        )

        print(
            f"[{index:02d}/{len(BIRD_ORDER)}] "
            f"{bird_name}: {input_path.name}"
        )

        audio = load_and_validate_audio(
            input_path
        )

        combined_parts.append(audio)


        if index < len(BIRD_ORDER):
            combined_parts.append(between_silence)

    combined_parts.append(final_silence)

    combined_audio = np.concatenate( 
        combined_parts
    )

    expected_samples = (
        initial_silence_samples
        + len(BIRD_ORDER) * bird_samples
        + (len(BIRD_ORDER) - 1)
        * between_silence_samples
        + final_silence_samples
    )

    if len(combined_audio) != expected_samples:
        raise RuntimeError(
            "Combined sample count is incorrect.\n"
            f"Actual:   {len(combined_audio):,}\n"
            f"Expected: {expected_samples:,}"
        )

    combined_peak = float(
        np.max(
            np.abs(combined_audio)
        )
    )

    sf.write(
        COMBINED_OUTPUT,
        combined_audio,
        TARGET_SAMPLE_RATE,
        subtype=OUTPUT_SUBTYPE,
    )

    actual_duration = (
        len(combined_audio)
        / TARGET_SAMPLE_RATE
    )

    expected_duration = (
        INITIAL_SILENCE_SECONDS
        + len(BIRD_ORDER)
        * TARGET_DURATION_SECONDS
        + (len(BIRD_ORDER) - 1)
        * SILENCE_BETWEEN_SECONDS
        + FINAL_SILENCE_SECONDS
    )

    print("\n" + "=" * 78)
    print("COMPLETE")
    print("=" * 78)
    print(
        f"Combined file:    "
        f"{COMBINED_OUTPUT.resolve()}"
    )
    print(
        f"Expected duration: "
        f"{expected_duration:.3f} s"
    )
    print(
        f"Actual duration:   "
        f"{actual_duration:.3f} s"
    )
    print(
        f"Total samples:     "
        f"{len(combined_audio):,}"
    )
    print(
        f"Sample rate:       "
        f"{TARGET_SAMPLE_RATE:,} Hz"
    )
    print(
        f"Peak amplitude:    "
        f"{combined_peak:.6f}"
    )
    print(
        f"Output subtype:    "
        f"{OUTPUT_SUBTYPE}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()