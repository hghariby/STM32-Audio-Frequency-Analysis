from pathlib import Path
import re
import wave
import csv

import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("birdcall_recordings")
RESULTS_DIR = Path("analysis_results")
FFT_DIR = RESULTS_DIR / "fft_plots"
WAVEFORM_DIR = RESULTS_DIR / "waveform_plots"

RESULTS_DIR.mkdir(exist_ok=True)
FFT_DIR.mkdir(exist_ok=True)
WAVEFORM_DIR.mkdir(exist_ok=True)


def parse_filename(filename):
    """
    Expected format:
    test_1ft_1000hz.wav
    test_10ft_8000hz.wav
    """
    match = re.match(r"test_(\d+)ft_(\d+)hz\.wav", filename.lower())

    if not match:
        return None

    distance_ft = int(match.group(1))
    expected_freq = int(match.group(2))

    return distance_ft, expected_freq


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        raw = wav.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"{path.name}: expected 16-bit WAV, got {sample_width * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples[:, 0]

    return samples, sample_rate, num_frames


def analyze_signal(samples, sample_rate, expected_freq):
    
    print(f"Raw mean = {np.mean(samples):.2f}")
    print(f"Raw min  = {np.min(samples)}")
    print(f"Raw max  = {np.max(samples)}")
    # Remove DC offset
    x = samples.astype(np.float64)
    x = x - np.mean(x)

    duration = len(x) / sample_rate

    # RMS amplitude
    rms = np.sqrt(np.mean(x ** 2))

    # Normalize RMS relative to int16 max
    rms_normalized = rms / 32768.0

    # Windowing for cleaner FFT
    window = np.hanning(len(x))
    x_windowed = x * window

    fft_values = np.fft.rfft(x_windowed)
    fft_freqs = np.fft.rfftfreq(len(x_windowed), d=1 / sample_rate)
    magnitude = np.abs(fft_values)

    # Ignore very low-frequency/DC region
    min_freq = 50
    valid = fft_freqs >= min_freq

    dominant_index = np.argmax(magnitude[valid])
    dominant_freq = fft_freqs[valid][dominant_index]
    dominant_magnitude = magnitude[valid][dominant_index]

    freq_error = abs(dominant_freq - expected_freq)

    return {
        "duration_sec": duration,
        "rms": rms,
        "rms_normalized": rms_normalized,
        "dominant_freq": dominant_freq,
        "dominant_magnitude": dominant_magnitude,
        "freq_error": freq_error,
        "fft_freqs": fft_freqs,
        "magnitude": magnitude,
    }


def save_waveform_plot(samples, sample_rate, output_path, title):
    time_axis = np.arange(len(samples)) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, samples)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_fft_plot(fft_freqs, magnitude, expected_freq, dominant_freq, output_path, title):
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs, magnitude)

    plt.axvline(expected_freq, linestyle="--", label=f"Expected: {expected_freq} Hz")
    plt.axvline(dominant_freq, linestyle=":", label=f"Detected: {dominant_freq:.1f} Hz")

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 9000)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


rows = []

wav_files = sorted(DATA_DIR.glob("*.wav"))

for wav_path in wav_files:
    parsed = parse_filename(wav_path.name)

    if parsed is None:
        print(f"Skipping file with unexpected name: {wav_path.name}")
        continue

    distance_ft, expected_freq = parsed

    try:
        samples, sample_rate, num_frames = read_wav(wav_path)
        result = analyze_signal(samples, sample_rate, expected_freq)

        base_name = wav_path.stem

        waveform_path = WAVEFORM_DIR / f"{base_name}_waveform.png"
        fft_path = FFT_DIR / f"{base_name}_fft.png"

        save_waveform_plot(
            samples,
            sample_rate,
            waveform_path,
            f"Waveform: {distance_ft} ft, {expected_freq} Hz"
        )

        save_fft_plot(
            result["fft_freqs"],
            result["magnitude"],
            expected_freq,
            result["dominant_freq"],
            fft_path,
            f"FFT Spectrum: {distance_ft} ft, {expected_freq} Hz"
        )

        rows.append({
            "file": wav_path.name,
            "distance_ft": distance_ft,
            "expected_freq_hz": expected_freq,
            "sample_rate_hz": sample_rate,
            "num_frames": num_frames,
            "duration_sec": round(result["duration_sec"], 4),
            "rms": round(result["rms"], 2),
            "rms_normalized": round(result["rms_normalized"], 6),
            "dominant_freq_hz": round(result["dominant_freq"], 2),
            "freq_error_hz": round(result["freq_error"], 2),
            "dominant_magnitude": round(result["dominant_magnitude"], 2),
        })

        print(f"Analyzed {wav_path.name}")

    except Exception as e:
        print(f"Error analyzing {wav_path.name}: {e}")


csv_path = RESULTS_DIR / "frequency_analysis_results.csv"

with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "file",
        "distance_ft",
        "expected_freq_hz",
        "sample_rate_hz",
        "num_frames",
        "duration_sec",
        "rms",
        "rms_normalized",
        "dominant_freq_hz",
        "freq_error_hz",
        "dominant_magnitude",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in sorted(rows, key=lambda r: (r["expected_freq_hz"], r["distance_ft"])):
        writer.writerow(row)


print("\nDone.")
print(f"CSV saved to: {csv_path}")
print(f"FFT plots saved to: {FFT_DIR}")
print(f"Waveform plots saved to: {WAVEFORM_DIR}")