from pathlib import Path
from multiprocessing import freeze_support
from birdnet_analyzer import analyze

ROOT = Path(".")
AUDIO_ROOT = ROOT / 'Birdsongs Recordings'
OUT_ROOT = ROOT / "BirdNET_Results" / "01_raw"

LA_LAT = 34.0522
LA_LON = -118.2437
WEEK = 27

EXPERIMENTS = {
    "E1_no_metadata": {"lat": -1, "lon": -1, "week": -1},
    "E2_week_only": {"lat": -1, "lon": -1, "week": WEEK},
    "E3_location_only": {"lat": LA_LAT, "lon": LA_LON, "week": -1},
    "E4_week_and_location": {"lat": LA_LAT, "lon": LA_LON, "week": WEEK},
}

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    wavs = list(AUDIO_ROOT.rglob("*.wav"))
    print(f"Found {len(wavs)} WAV files")

    for exp_name, meta in EXPERIMENTS.items():
        print(f"\nRunning {exp_name}...")

        out_dir = OUT_ROOT / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        analyze(
            audio_input=str(AUDIO_ROOT),
            output=str(out_dir),
            lat=meta["lat"],
            lon=meta["lon"],
            week=meta["week"],
            rtype="csv",
            top_n=20,
            min_conf=0.01,
            threads=8,
            batch_size=1,
            combine_results=True,
            skip_existing_results=False,
            additional_columns=["lat", "lon", "week", "min_conf"],
        )

        print(f"Finished {exp_name}")

    print("\nDONE")
    print("Results saved to:")
    print(OUT_ROOT)

if __name__ == "__main__":
    freeze_support()
    main()