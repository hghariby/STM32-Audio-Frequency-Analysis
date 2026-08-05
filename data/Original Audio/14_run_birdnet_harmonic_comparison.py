from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

from birdnet_analyzer import analyze


ROOT = Path(".")

BRANCHES = {
    "with_harmonics": (
        ROOT / "Filtered_With_Harmonics" / "Normalized_Audio"
    ),
    "without_harmonics": (
        ROOT / "Filtered_Without_Harmonics" / "Normalized_Audio"
    ),
}

RESULTS_ROOT = (
    ROOT / "BirdNET_Harmonic_Comparison" / "Raw_Results"
)

LA_LAT = 34.0522
LA_LON = -118.2437
WEEK = 27

EXPERIMENTS = {
    "E1_no_metadata": {
        "lat": -1,
        "lon": -1,
        "week": -1,
    },
    "E2_week_only": {
        "lat": -1,
        "lon": -1,
        "week": WEEK,
    },
    "E3_location_only": {
        "lat": LA_LAT,
        "lon": LA_LON,
        "week": -1,
    },
    "E4_week_and_location": {
        "lat": LA_LAT,
        "lon": LA_LON,
        "week": WEEK,
    },
}

TOP_N = 20
MIN_CONFIDENCE = 0.01
THREADS = 8
BATCH_SIZE = 1


def count_wavs(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*.wav")
        if path.is_file()
    )


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BIRDNET WITH/WITHOUT-HARMONICS COMPARISON")
    print("=" * 80)

    for branch, audio_folder in BRANCHES.items():
        if not audio_folder.exists():
            raise FileNotFoundError(
                f"Audio folder not found: {audio_folder.resolve()}"
            )

        wav_count = count_wavs(audio_folder)

        if wav_count == 0:
            raise FileNotFoundError(
                f"No WAV files found in {audio_folder.resolve()}"
            )

        print(f"\nBranch: {branch}")
        print(f"Audio folder: {audio_folder.resolve()}")
        print(f"WAV files: {wav_count}")

        for experiment, metadata in EXPERIMENTS.items():
            output_folder = (
                RESULTS_ROOT / branch / experiment
            )
            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(f"\n  Running {experiment}")
            print(
                f"  lat={metadata['lat']}, "
                f"lon={metadata['lon']}, "
                f"week={metadata['week']}"
            )

            analyze(
                audio_input=str(audio_folder),
                output=str(output_folder),
                lat=metadata["lat"],
                lon=metadata["lon"],
                week=metadata["week"],
                rtype="csv",
                top_n=TOP_N,
                min_conf=MIN_CONFIDENCE,
                threads=THREADS,
                batch_size=BATCH_SIZE,
                combine_results=True,
                skip_existing_results=False,
                additional_columns=[
                    "lat",
                    "lon",
                    "week",
                    "min_conf",
                ],
            )

            print(f"  Finished: {output_folder.resolve()}")

    print("\n" + "=" * 80)
    print("BIRDNET ANALYSIS COMPLETE")
    print(f"Raw results: {RESULTS_ROOT.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    freeze_support()
    main()
