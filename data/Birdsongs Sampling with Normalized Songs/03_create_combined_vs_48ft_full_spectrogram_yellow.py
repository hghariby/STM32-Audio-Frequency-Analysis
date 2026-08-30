from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from scipy import signal

ROOT = Path('.')
COMBINED_FILE = ROOT / 'Combined_Bird_Calls.wav'
RECORDING_48FT_FILE = ROOT / 'Distance_Recordings' / 'D07_48FT.WAV'
OUTPUT_FOLDER = ROOT / 'Segmentation_Results'
OUTPUT_PNG = OUTPUT_FOLDER / 'Combined_vs_48ft_full_spectrogram.png'

MIN_FREQUENCY_HZ = 0
MAX_FREQUENCY_HZ = 16000
NPERSEG = 2048
HOP_SAMPLES = 256
NOVERLAP = NPERSEG - HOP_SAMPLES
COLOR_MIN_PERCENTILE = 5.0
COLOR_MAX_PERCENTILE = 99.7
MAJOR_TICK_SECONDS = 5.0
MINOR_TICK_SECONDS = 0.1
FIGURE_WIDTH_INCHES = 22
FIGURE_HEIGHT_INCHES = 11
OUTPUT_DPI = 220


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f'Audio file not found: {path.resolve()}')

    audio, sample_rate = sf.read(path, dtype='float64', always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        if audio.shape[1] != 1:
            raise ValueError(
                f'{path.name} is stereo or multichannel. '
                f'Expected mono, but shape is {audio.shape}.'
            )
        audio = audio[:, 0]

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f'Invalid or empty WAV file: {path.resolve()}')

    if not np.all(np.isfinite(audio)):
        raise ValueError(f'Non-finite samples found in {path.name}')

    return audio, int(sample_rate)


def calculate_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = audio - np.mean(audio)

    frequencies, times, stft = signal.stft(
        centered,
        fs=sample_rate,
        window='hann',
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NPERSEG,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(stft)
    floor = np.finfo(np.float64).tiny
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, floor))

    frequency_mask = (
        (frequencies >= MIN_FREQUENCY_HZ)
        & (frequencies <= min(MAX_FREQUENCY_HZ, sample_rate / 2.0))
    )

    return frequencies[frequency_mask], times, magnitude_db[frequency_mask, :]


def microsecond_formatter(value: float, position: int) -> str:
    return f'{value:.1f} s'


def add_spectrogram(
    axis: plt.Axes,
    frequencies: np.ndarray,
    times: np.ndarray,
    magnitude_db: np.ndarray,
    title: str,
    duration_seconds: float,
    vmin: float,
    vmax: float,
):
    image = axis.pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading='auto',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    axis.set_title(title, fontsize=13)
    axis.set_ylabel('Frequency (Hz)')
    axis.set_xlim(0.0, duration_seconds)
    axis.set_ylim(MIN_FREQUENCY_HZ, min(MAX_FREQUENCY_HZ, frequencies[-1]))
    axis.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_SECONDS))
    axis.xaxis.set_major_formatter(FuncFormatter(microsecond_formatter))
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.tick_params(axis='x', labelrotation=45)
    axis.grid(axis='x', which='major', alpha=0.30, linewidth=0.7)
    axis.grid(axis='x', which='minor', alpha=0.12, linewidth=0.4)

    return image


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    combined_audio, combined_rate = load_mono(COMBINED_FILE)
    recording_audio, recording_rate = load_mono(RECORDING_48FT_FILE)

    combined_duration = len(combined_audio) / combined_rate
    recording_duration = len(recording_audio) / recording_rate

    print('Loading and calculating spectrograms...')
    print(
        f'Combined: {combined_duration:.6f} s, '
        f'{combined_rate} Hz, {len(combined_audio)} samples'
    )
    print(
        f'48 ft:    {recording_duration:.6f} s, '
        f'{recording_rate} Hz, {len(recording_audio)} samples'
    )

    combined_f, combined_t, combined_db = calculate_spectrogram(
        combined_audio, combined_rate
    )
    recording_f, recording_t, recording_db = calculate_spectrogram(
        recording_audio, recording_rate
    )

    all_values = np.concatenate(
        (
            combined_db[np.isfinite(combined_db)].ravel(),
            recording_db[np.isfinite(recording_db)].ravel(),
        )
    )
    vmin = float(np.percentile(all_values, COLOR_MIN_PERCENTILE))
    vmax = float(np.percentile(all_values, COLOR_MAX_PERCENTILE))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        constrained_layout=True,
    )

    image = add_spectrogram(
        axis=axes[0],
        frequencies=combined_f,
        times=combined_t,
        magnitude_db=combined_db,
        title=(
            'Original combined playback — '
            f'{combined_duration:.6f} s, {combined_rate} Hz'
        ),
        duration_seconds=combined_duration,
        vmin=vmin,
        vmax=vmax,
    )

    add_spectrogram(
        axis=axes[1],
        frequencies=recording_f,
        times=recording_t,
        magnitude_db=recording_db,
        title=(
            'STM32 recording at 48 ft — '
            f'{recording_duration:.6f} s, {recording_rate} Hz'
        ),
        duration_seconds=recording_duration,
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_xlabel(
        'Time from beginning of each WAV file '
        '(seconds shown to six decimal places)'
    )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        location='right',
        shrink=0.92,
        pad=0.01,
    )
    colorbar.set_label('Magnitude (dB)')

    fig.suptitle(
        'Full-length spectrogram comparison: combined playback vs. 48-ft recording',
        fontsize=16,
    )

    fig.savefig(OUTPUT_PNG, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)

    combined_hop_us = HOP_SAMPLES / combined_rate * 1_000_000.0
    recording_hop_us = HOP_SAMPLES / recording_rate * 1_000_000.0

    print('\nSaved:')
    print(OUTPUT_PNG.resolve())
    print('\nActual spectrogram time spacing:')
    print(f'Combined: {combined_hop_us:.3f} microseconds per time step')
    print(f'48 ft:    {recording_hop_us:.3f} microseconds per time step')
    print(
        '\nImportant: the axis labels show six decimal places, but the real '
        'time resolution is determined by HOP_SAMPLES/sample_rate.'
    )


if __name__ == '__main__':
    main()
