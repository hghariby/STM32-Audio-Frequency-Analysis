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
ROOT_DIR = Path("./birdcallrange_recordings")
ANALYSIS_DIR = Path("./analysis_results")
FFT_DIR = ANALYSIS_DIR / "FFT_Plots"
LINEAR_SPEC_DIR = ANALYSIS_DIR / "Linear Spectrograms"
MEL_SPEC_DIR = ANALYSIS_DIR / "Mel-Frequency Spectrogram"
WAVEFORM_DIR = ANALYSIS_DIR / "waveform_over_samples"
WAVEFORM_SEC_DIR = ANALYSIS_DIR / "waveform_over_seconds"
RMS_DIR = ANALYSIS_DIR / "RMS_vs_Distance"
ERROR_DIR = ANALYSIS_DIR / "Frequency_Error_vs_Distance"
TVD_DIR = ANALYSIS_DIR / "Target_vs_Detected"

for folder in [ANALYSIS_DIR, FFT_DIR, LINEAR_SPEC_DIR, MEL_SPEC_DIR, WAVEFORM_DIR, WAVEFORM_SEC_DIR, RMS_DIR, ERROR_DIR, TVD_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = ANALYSIS_DIR / "analysis_results_with_waveform.csv"

# --------------------------
# Helper functions
# --------------------------
def parse_filename(filename):
    match = re.match(r"test_(\d+ft)_(\d+)hz\.wav", filename)
    if not match:
        return None
    distance = match.group(1)
    target_freq = int(match.group(2))
    return distance, target_freq

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

for file_path in ROOT_DIR.glob("*.wav"):
    parsed = parse_filename(file_path.name)
    if parsed is None:
        print(f"Skipping invalid filename: {file_path.name}")
        continue
    distance, target_freq = parsed

    audio, sample_rate = read_wav(file_path)

    # FFT
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1/sample_rate)
    spectrum[freqs < 20] = 0
    dominant_index = np.argmax(spectrum)
    detected_freq = freqs[dominant_index]
    freq_error = abs(detected_freq - target_freq)

    # RMS & peak
    audio_float = audio.astype(np.float64)
    rms = np.sqrt(np.mean(audio_float**2))
    peak = np.max(np.abs(audio_float))

    results.append({
        "distance": distance,
        "filename": file_path.name,
        "target_freq": target_freq,
        "detected_freq": detected_freq,
        "frequency_error": freq_error,
        "rms": rms,
        "peak": peak
    })

    # =========================
    # Plot waveform of first 1000 samples
    # =========================
    plt.figure(figsize=(8,3))
    plt.plot(audio[:1000])
    plt.title(f"Waveform over first 1000 samples: {file_path.name}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # save to correct folder
    plt.savefig(WAVEFORM_DIR / f"{file_path.name}_waveform.png", dpi=300)
    plt.close()
    #print("Waveform over samples folder:", WAVEFORM_DIR)

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
    plt.savefig(WAVEFORM_SEC_DIR / f"{file_path.stem}_waveform_seconds.png", dpi=300)
    plt.close()
    #print("Waveform over seconds folder:", WAVEFORM_DIR)

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
    plt.savefig(LINEAR_SPEC_DIR / f"{file_path.stem}_linear_spectrogram.png", dpi=300)
    plt.close()

    # --------------------------
    # Mel Spectrogram with waveform
    # --------------------------
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    librosa.display.waveshow(y, sr=sr, color="blue")
    plt.title(f"Waveform: {file_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(2,1,2)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Frequency Spectrogram")
    plt.tight_layout()
    plt.savefig(MEL_SPEC_DIR / f"{file_path.stem}_mel_spectrogram.png", dpi=300)
    plt.close()

    # --------------------------
    # FFT plot
    # --------------------------
    plt.figure(figsize=(8,3))
    plt.plot(freqs, spectrum)
    plt.title(f"FFT: {file_path.name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 2*target_freq)
    plt.tight_layout()
    plt.savefig(FFT_DIR / f"{file_path.stem}_fft.png", dpi=300)
    plt.close()

# --------------------------
# Save CSV
# --------------------------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = ["distance", "filename", "target_freq", "detected_freq", "frequency_error", "rms", "peak"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"CSV file saved: {OUTPUT_CSV}")

# --------------------------
# Summary Analysis Plots
# --------------------------
df = pd.DataFrame(results)
df['distance_num'] = df['distance'].str.replace("ft","").astype(int)
df = df.sort_values('distance_num')

# Target vs Detected Frequency
for distance in df['distance'].unique():
    plt.figure()
    subset = df[df['distance']==distance]
    plt.scatter(subset['target_freq'], subset['detected_freq'], color='blue')
    plt.plot([0, max(df['target_freq'])], [0, max(df['target_freq'])], 'r--')
    plt.xlabel("Target Frequency (Hz)")
    plt.ylabel("Detected Frequency (Hz)")
    plt.title(f"Target vs Detected Frequency - {distance}")
    plt.tight_layout()
    plt.savefig(TVD_DIR / f"TVD_{distance}.png", dpi=300)
    plt.close()

# RMS vs Distance
plt.figure()
for freq in df['target_freq'].unique():
    subset = df[df['target_freq']==freq]
    plt.plot(subset['distance_num'], subset['rms'], marker='o', label=f'{freq} Hz')
plt.xlabel("Distance (ft)")
plt.ylabel("RMS Amplitude")
plt.title("RMS vs Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RMS_DIR / "RMS_vs_Distance.png", dpi=300)
plt.close()

# Frequency Error vs Distance
plt.figure()
for freq in df['target_freq'].unique():
    subset = df[df['target_freq']==freq]
    plt.plot(subset['distance_num'], subset['frequency_error'], marker='o', label=f'{freq} Hz')
plt.xlabel("Distance (ft)")
plt.ylabel("Frequency Error (Hz)")
plt.title("Frequency Error vs Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ERROR_DIR / "Frequency_Error_vs_Distance.png", dpi=300)
plt.close()

# Optional trend line plots
summary = df.groupby('distance_num').agg({
    'detected_freq':'mean',
    'frequency_error':'mean',
    'rms':'mean'
}).reset_index()

plt.figure()
plt.plot(summary['distance_num'], summary['frequency_error'], marker='o', color='red', label='Average Frequency Error')
plt.xlabel("Distance (ft)")
plt.ylabel("Average Frequency Error (Hz)")
plt.title("Average Frequency Error Trend vs Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig(ERROR_DIR / "Frequency_Error_Trend_vs_Distance.png", dpi=300)
plt.close()

plt.figure()
plt.plot(summary['distance_num'], summary['rms'], marker='o', color='green', label='Average RMS')
plt.xlabel("Distance (ft)")
plt.ylabel("Average RMS")
plt.title("Average RMS Trend vs Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig(RMS_DIR / "RMS_Trend_vs_Distance.png", dpi=300)
plt.close()

print("All analysis and plots saved in 'analysis_results/' folder.")