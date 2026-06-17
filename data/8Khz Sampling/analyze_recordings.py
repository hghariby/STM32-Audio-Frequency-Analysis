import os
import re
import wave
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "./Experiment2_HighFrequencyDetectability"
OUTPUT_CSV = "analysis_results.csv"

VALID_SHIFTS = ["13", "14", "15"]

def parse_filename(filename):
    match = re.match(r"test_(\d+)in_(\d+)hz\.wav", filename)
    if not match:
        return None

    distance_in = int(match.group(1))
    target_freq = int(match.group(2))
    return distance_in, target_freq

def read_wav(path):
    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio, sample_rate

def analyze_audio(audio, sample_rate, target_freq):
    if len(audio) == 0:
        return None

    # Remove DC offset
    audio = audio - np.mean(audio)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    # FFT
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1 / sample_rate)

    # Ignore DC / very low frequencies
    spectrum[freqs < 20] = 0

    dominant_index = np.argmax(spectrum)
    detected_freq = freqs[dominant_index]
    frequency_error = abs(detected_freq - target_freq)

    # Strength at target frequency
    target_index = np.argmin(np.abs(freqs - target_freq))
    target_power = spectrum[target_index]
    dominant_power = spectrum[dominant_index]

    return {
        "rms": rms,
        "peak": peak,
        "detected_freq": detected_freq,
        "frequency_error": frequency_error,
        "target_power": target_power,
        "dominant_power": dominant_power,
    }

results = []

for shift in VALID_SHIFTS:
    shift_path = os.path.join(ROOT_DIR, shift)

    if not os.path.isdir(shift_path):
        continue

    for distance_folder in os.listdir(shift_path):
        distance_path = os.path.join(shift_path, distance_folder)

        if not os.path.isdir(distance_path):
            continue

        for filename in os.listdir(distance_path):
            if not filename.endswith(".wav"):
                continue

            parsed = parse_filename(filename)

            # Skip bad/duplicate files like test_4_100hz.wav
            if parsed is None:
                print(f"Skipping invalid filename: {filename}")
                continue

            distance_in, target_freq = parsed
           # if target_freq > 4100:
            #   continue

            file_path = os.path.join(distance_path, filename)

            try:
                audio, sample_rate = read_wav(file_path)
                metrics = analyze_audio(audio, sample_rate, target_freq)

                if metrics is None:
                    print(f"Empty file: {file_path}")
                    continue

                results.append({
                    "shift": shift,
                    "distance_in": distance_in,
                    "target_freq": target_freq,
                    "detected_freq": metrics["detected_freq"],
                    "frequency_error": metrics["frequency_error"],
                    "rms": metrics["rms"],
                    "peak": metrics["peak"],
                    "target_power": metrics["target_power"],
                    "dominant_power": metrics["dominant_power"],
                    "filename": file_path
                })

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Save CSV
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "shift",
        "distance_in",
        "target_freq",
        "detected_freq",
        "frequency_error",
        "rms",
        "peak",
        "target_power",
        "dominant_power",
        "filename"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {OUTPUT_CSV}")
print(f"Analyzed {len(results)} files.")

# Summary by shift
print("\nAverage results by shift:")

for shift in VALID_SHIFTS:
    shift_results = [r for r in results if r["shift"] == shift]

    if not shift_results:
        continue

    avg_error = np.mean([r["frequency_error"] for r in shift_results])
    avg_rms = np.mean([r["rms"] for r in shift_results])
    avg_peak = np.mean([r["peak"] for r in shift_results])

    print(f"\nShift >>{shift}")
    print(f"Average frequency error: {avg_error:.2f} Hz")
    print(f"Average RMS amplitude:   {avg_rms:.2f}")
    print(f"Average peak amplitude:  {avg_peak:.2f}")

# Plot RMS vs distance for each shift
for shift in VALID_SHIFTS:
    shift_results = [r for r in results if r["shift"] == shift]

    if not shift_results:
        continue

    distances = sorted(set(r["distance_in"] for r in shift_results))
    avg_rms_by_distance = []

    for d in distances:
        vals = [r["rms"] for r in shift_results if r["distance_in"] == d]
        avg_rms_by_distance.append(np.mean(vals))

    plt.figure()
    plt.plot(distances, avg_rms_by_distance, marker="o")
    plt.xlabel("Distance (inches)")
    plt.ylabel("Average RMS Amplitude")
    plt.title(f"Average RMS vs Distance for Shift >>{shift}")
    plt.grid(True)
    plt.savefig(f"rms_vs_distance_shift_{shift}.png", dpi=300)
    plt.close()

# Plot average frequency error by shift
shifts = []
avg_errors = []

for shift in VALID_SHIFTS:
    shift_results = [r for r in results if r["shift"] == shift]
    if shift_results:
        shifts.append(f">>{shift}")
        avg_errors.append(np.mean([r["frequency_error"] for r in shift_results]))

plt.figure()
plt.bar(shifts, avg_errors)
plt.xlabel("Shift")
plt.ylabel("Average Frequency Error (Hz)")
plt.title("Average Frequency Error by Shift")
plt.grid(axis="y")
plt.savefig("average_frequency_error_by_shift.png", dpi=300)
plt.close()

print("\nGenerated plots:")
print("rms_vs_distance_shift_13.png")
print("rms_vs_distance_shift_14.png")
print("rms_vs_distance_shift_15.png")
print("average_frequency_error_by_shift.png")