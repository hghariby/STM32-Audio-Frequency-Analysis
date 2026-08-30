from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from scipy.signal import spectrogram


COMBINED_FILE = Path("Combined_Bird_Calls.wav")
RECORDING_FILE = Path("Distance_Recordings/D07_48FT.WAV")
OUTPUT_FILE = Path("Segmentation_Results/Combined_vs_48ft_full_spectrogram.png")

MAX_FREQUENCY = 16000
MAJOR_TICK_SECONDS = 5
MINOR_TICK_SECONDS = 0.1


def load_audio(path):
    audio, sample_rate = sf.read(path)

    if audio.ndim == 2:
        if audio.shape[1] != 1:
            raise ValueError(f"{path.name} must be mono.")
        audio = audio[:, 0]

    return audio, sample_rate


def make_spectrogram(audio, sample_rate):
    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=2048,
        noverlap=1792,
        mode="magnitude",
    )

    power_db = 20 * np.log10(power + 1e-12)
    keep = frequencies <= min(MAX_FREQUENCY, sample_rate / 2)

    return frequencies[keep], times, power_db[keep]


combined_audio, combined_rate = load_audio(COMBINED_FILE)
recording_audio, recording_rate = load_audio(RECORDING_FILE)

combined = make_spectrogram(combined_audio, combined_rate)
recording = make_spectrogram(recording_audio, recording_rate)

fig, axes = plt.subplots(2, 1, figsize=(22, 10), constrained_layout=True)

for axis, data, title in [
    (axes[0], combined, "Original combined playback"),
    (axes[1], recording, "STM32 recording at 48 ft"),
]:
    frequencies, times, power_db = data

    image = axis.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="auto",
        cmap="magma",
    )

    axis.set_title(title)
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0, MAX_FREQUENCY)

    axis.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_SECONDS))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axis.xaxis.set_minor_locator(MultipleLocator(MINOR_TICK_SECONDS))
    axis.yaxis.grid(False)
    
    axis.grid(
        axis="x",
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )
    axis.grid(
        axis="x",
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.8,
    )
    
    axis.tick_params(axis="x", which="major", labelrotation=45)
    axis.tick_params(axis="x", which="minor", length=2)

axes[1].set_xlabel("Time (seconds)")
fig.colorbar(image, ax=axes, label="Magnitude (dB)")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUTPUT_FILE.resolve()}")
