import os
import re
import wave
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import pandas as pd

# --------------------------
# Folder setup
# --------------------------
ROOT_DIR = Path("./birdcall_recordings")
ANALYSIS_DIR = Path("./analysis_results")

# Create main analysis folder
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = ANALYSIS_DIR / "analysis_results_with_waveform.csv"

# Define plot types
PLOT_TYPES = ["FFT_Plots", "Linear_Spectrograms", "Mel-Frequency_Spectrogram",
              "waveform_over_samples", "waveform_over_seconds", "RMS_vs_Distance"]

# Create main plot folders
for plot_type in PLOT_TYPES:
    (ANALYSIS_DIR / plot_type).mkdir(parents=True, exist_ok=True)

# --------------------------
# Helper functions
# --------------------------
def parse_filename(filename):
    """Extract species and distance from filename like 'Allens_Hummingbird_1ft.wav'"""
    parts = filename.replace(".wav", "").split("_")
    if len(parts) < 2:
        return None
    species = "_".join(parts[:-1])
    try:
        distance = int(parts[-1].replace("ft",""))
    except ValueError:
        return None
    return species, distance

def read_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).copy()
    return audio, sample_rate

# --------------------------
# Process WAV files
# --------------------------
results = []

for file_path in ROOT_DIR.glob("**/*.wav"):
    parsed = parse_filename(file_path.name)
    if parsed is None:
        print(f"Skipping invalid filename: {file_path.name}")
        continue
    species, distance = parsed

    # Create species subfolders inside each plot type
    species_dirs = {}
    for plot_type in PLOT_TYPES:
        species_dir = ANALYSIS_DIR / plot_type / species
        species_dir.mkdir(parents=True, exist_ok=True)
        species_dirs[plot_type] = species_dir

    # Read audio
    audio, sample_rate = read_wav(file_path)

    # FFT
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1/sample_rate)

    # RMS & peak
    audio_float = audio.astype(np.float64)
    rms = np.sqrt(np.mean(audio_float**2)) if len(audio_float) > 0 else 0
    peak = np.max(np.abs(audio_float)) if len(audio_float) > 0 else 0

    results.append({
        "species": species,
        "distance": distance,
        "filename": file_path.name,
        "rms": rms,
        "peak": peak
    })

    # --------------------------
    # Waveform over first 1000 samples
    # --------------------------
    plt.figure(figsize=(8,3))
    plt.plot(audio[:1000])
    plt.title(f"Waveform (first 1000 samples): {file_path.name}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(species_dirs["waveform_over_samples"] / f"{file_path.stem}_waveform_samples.png", dpi=300)
    plt.close()

    # --------------------------
    # Waveform over seconds
    # --------------------------
    plt.figure(figsize=(10,3))
    time_axis = np.arange(len(audio)) / sample_rate
    plt.plot(time_axis, audio, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform over seconds: {file_path.name}")
    plt.tight_layout()
    plt.savefig(species_dirs["waveform_over_seconds"] / f"{file_path.stem}_waveform_seconds.png", dpi=300)
    plt.close()

    # --------------------------
    # Linear Spectrogram
    # --------------------------
    plt.figure(figsize=(10,4))
    plt.specgram(audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Linear Spectrogram: {file_path.name}")
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig(species_dirs["Linear_Spectrograms"] / f"{file_path.stem}_linear_spectrogram.png", dpi=300)
    plt.close()

    # --------------------------
    # Mel Spectrogram with waveform
    # --------------------------
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)    
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    librosa.display.waveshow(y, sr=sr, color="blue")
    plt.title(f"Waveform: {file_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(2,1,2)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', fmax=8000)    
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Frequency Spectrogram")
    plt.tight_layout()
    plt.savefig(species_dirs["Mel-Frequency_Spectrogram"] / f"{file_path.stem}_mel_spectrogram.png", dpi=300)
    plt.close()

    # --------------------------
    # FFT plot
    # --------------------------
    plt.figure(figsize=(8,3))
    plt.plot(freqs, spectrum)
    plt.title(f"FFT: {file_path.name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(species_dirs["FFT_Plots"] / f"{file_path.stem}_fft.png", dpi=300)
    plt.close()

# --------------------------
# Save CSV
# --------------------------
fieldnames = ["species", "distance", "filename", "rms", "peak"]
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"CSV file saved: {OUTPUT_CSV}")

# --------------------------
# RMS vs Distance per species
# --------------------------
df = pd.DataFrame(results)
df = df.sort_values(['species','distance'])

for species_name in df['species'].unique():
    subset = df[df['species'] == species_name]
    plt.figure()
    plt.plot(subset['distance'], subset['rms'], marker='o', label='RMS')
    plt.xlabel("Distance (ft)")
    plt.ylabel("RMS Amplitude")
    plt.title(f"RMS vs Distance - {species_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "RMS_vs_Distance" / species_name / f"RMS_vs_Distance.png", dpi=300)
    plt.close()

print("All analysis and plots saved in 'analysis_results/' folder.")