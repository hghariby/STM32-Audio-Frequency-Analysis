import os
import wave

import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 35714
DISTANCE = "1in"
FREQUENCIES = [4000, 6000, 8000, 10000, 12000, 14000]


def detect_dominant_frequency(filename: str) -> float:
    with wave.open(filename, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()

        if channels != 1:
            raise ValueError(f"{filename}: expected mono audio.")

        if sample_width != 2:
            raise ValueError(f"{filename}: expected 16-bit audio.")

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{filename}: WAV sample rate is {sample_rate} Hz, "
                f"but {SAMPLE_RATE} Hz was expected."
            )

        raw_audio = wav_file.readframes(frame_count)

    samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float64)

    # Remove DC offset.
    samples -= np.mean(samples)

    # Apply a Hann window to reduce spectral leakage.
    windowed_samples = samples * np.hanning(len(samples))

    spectrum = np.fft.rfft(windowed_samples)
    magnitudes = np.abs(spectrum)
    frequency_axis = np.fft.rfftfreq(
        len(windowed_samples),
        d=1 / sample_rate,
    )

    # Ignore DC.
    magnitudes[0] = 0

    peak_index = int(np.argmax(magnitudes))
    return float(frequency_axis[peak_index])


expected_frequencies = []
detected_frequencies = []

print(
    f"{'Expected (Hz)':>14}"
    f"{'Detected (Hz)':>15}"
    f"{'Error (Hz)':>13}"
    f"{'Error (%)':>12}"
)
print("-" * 54)

for expected_frequency in FREQUENCIES:
    filename = (
        f"test_{DISTANCE}_{expected_frequency}hz.wav"
    )

    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    detected_frequency = detect_dominant_frequency(filename)

    error_hz = detected_frequency - expected_frequency
    error_percent = (
        abs(error_hz) / expected_frequency * 100
    )

    expected_frequencies.append(expected_frequency)
    detected_frequencies.append(detected_frequency)

    print(
        f"{expected_frequency:14.2f}"
        f"{detected_frequency:15.2f}"
        f"{error_hz:13.2f}"
        f"{error_percent:12.4f}"
    )


if not expected_frequencies:
    raise RuntimeError("No WAV files were found.")


# Ideal comparison line.
minimum_frequency = min(expected_frequencies)
maximum_frequency = max(expected_frequencies)

margin = 500
line_start = minimum_frequency - margin
line_end = maximum_frequency + margin


plt.figure(figsize=(8, 6))

plt.scatter(
    expected_frequencies,
    detected_frequencies,
    s=80,
    label="Recorded frequencies",
)

plt.plot(
    [line_start, line_end],
    [line_start, line_end],
    linestyle="--",
    label="Ideal: detected = expected",
)

# Label each point.
for expected, detected in zip(
    expected_frequencies,
    detected_frequencies,
):
    plt.annotate(
        f"{detected:.1f} Hz",
        (expected, detected),
        xytext=(6, 7),
        textcoords="offset points",
    )

plt.xlabel("Expected Frequency (Hz)")
plt.ylabel("Detected Frequency (Hz)")
plt.title(
    f"Expected vs. Detected Frequency at {DISTANCE}"
)
plt.xlim(line_start, line_end)
plt.ylim(line_start, line_end)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_filename = (
    f"expected_vs_detected_{DISTANCE}.png"
)
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"\nPlot saved as: {output_filename}")