import os
import re
import numpy as np
import wave
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\hasmi\OneDrive\Desktop\STM32\data\8Khz Sampling\Experiment1_FrequencyAccuracy"
OUTPUT_DIR = Path("Results")
SAMPLE_RATE = 8000
RECORD_SECONDS = 5
TARGET_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

RUNS = ["13", "14", "15"]

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# PARSE FREQUENCY
# =========================
def parse_freq(filename):
    match = re.search(r"(\d+)\s*hz", filename.lower())
    if match:
        return int(match.group(1))
    return None


# =========================
# LOAD WAV
# =========================
def load_wav(path):
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        audio = w.readframes(frames)

    audio = np.frombuffer(audio, dtype=np.int16)

    # FIX LENGTH (IMPORTANT)
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))

    return audio


# =========================
# FFT ANALYSIS
# =========================
def get_dominant_freq(audio):
    audio = audio - np.mean(audio)
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1 / SAMPLE_RATE)

    spectrum[freqs < 50] = 0

    idx = np.argmax(spectrum)
    return freqs[idx]


# =========================
# STORAGE
# =========================
results = []


# =========================
# MAIN LOOP
# =========================
for run in RUNS:
    run_path = Path(BASE_DIR) / run

    if not run_path.exists():
        continue

    for distance_folder in os.listdir(run_path):
        dist_path = run_path / distance_folder

        if not dist_path.is_dir():
            continue

        # output folders
        fft_dir = OUTPUT_DIR / run / distance_folder / "fft_plots"
        fft_dir.mkdir(parents=True, exist_ok=True)

        target_list = []
        detected_list = []

        for i, file in enumerate(os.listdir(dist_path)):

            if not file.endswith(".wav"):
                continue

            full_path = dist_path / file

            target_freq = parse_freq(file)
            if target_freq is None:
                continue

            audio = load_wav(full_path)
            detected = get_dominant_freq(audio)

            error = detected - target_freq
            percent_error = (error / target_freq) * 100 if target_freq else 0

            results.append({
                "run": run,
                "distance": distance_folder,
                "file": file,
                "target_freq": target_freq,
                "detected_freq": detected,
                "error_hz": error,
                "percent_error": percent_error
            })

            # store for plots
            target_list.append(target_freq)
            detected_list.append(detected)

            # =========================
            # FFT PLOT PER FILE
            # =========================
            plt.figure()
            plt.plot(np.abs(np.fft.rfft(audio)))
            plt.title(f"{file}")
            plt.xlabel("FFT bins")
            plt.ylabel("Magnitude")
            plt.savefig(fft_dir / f"fft_{i}.png")
            plt.close()

        # =========================
        # COMPARISON PLOT
        # =========================
        if len(target_list) > 0:

            plt.figure()
            plt.scatter(target_list, detected_list)

            max_f = max(target_list + detected_list)

            plt.plot([0, max_f], [0, max_f], "r--")  # ideal line

            plt.xlabel("Target Frequency")
            plt.ylabel("Detected Frequency")
            plt.title(f"{run} - {distance_folder}")

            comp_dir = OUTPUT_DIR / run / distance_folder
            comp_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(comp_dir / "comparison.png")
            plt.close()

            # error plot
            errors = np.array(detected_list) - np.array(target_list)

            plt.figure()
            plt.plot(target_list, errors, "o")
            plt.axhline(0, color="r")

            plt.xlabel("Target Frequency")
            plt.ylabel("Error (Hz)")
            plt.title("Frequency Error")

            plt.savefig(comp_dir / "error_plot.png")
            plt.close()


# =========================
# SAVE CSV
# =========================
csv_path = OUTPUT_DIR / "results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run",
        "distance",
        "file",
        "target_freq",
        "detected_freq",
        "error_hz",
        "percent_error"
    ])

    writer.writeheader()
    writer.writerows(results)


print("\nDONE")
print("Saved to:", OUTPUT_DIR)
print("CSV:", csv_path)