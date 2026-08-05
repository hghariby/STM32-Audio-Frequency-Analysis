from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

ROOT = Path('.')
CSV = ROOT / 'Frequency Range Results' / 'bird_frequency_ranges.csv'

BRANCHES = {
    'without_harmonics': {
        'before_folder': ROOT / 'Filtered_Without_Harmonics' / 'Silenced_Audio',
        'after_folder': ROOT / 'Filtered_Without_Harmonics' / 'Normalized_Audio',
    },
    'with_harmonics': {
        'before_folder': ROOT / 'Filtered_With_Harmonics' / 'Silenced_Audio',
        'after_folder': ROOT / 'Filtered_With_Harmonics' / 'Normalized_Audio',
    },
}

OUTPUT_REPORT = ROOT / 'Frequency Range Results' / 'normalization_validation_report.csv'
PLOT_ROOT = ROOT / 'Frequency Range Results' / 'normalization_validation_review'

TARGET_RMS_DBFS = -25.0
MAX_ALLOWED_PEAK_DBFS = -1.0
RMS_TOLERANCE_DB = 0.10
DURATION_TOLERANCE_SECONDS = 1e-6
INACTIVE_ZERO_TOLERANCE = 0.0

MAX_PLOT_HZ = 15000.0
WINDOW_SECONDS = 0.025
OVERLAP_FRACTION = 0.75
DISPLAY_DYNAMIC_RANGE_DB = 80.0
COLORMAP = 'magma'


def key(text: str) -> str:
    text = Path(str(text)).stem.lower().replace('’', "'")
    text = re.sub(r'\(\d+\)$', '', text)
    return re.sub(r'[^a-z0-9]+', '', text)


def safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')


def harmonic_test_required(row: pd.Series) -> bool:
    return str(row.get('harmonic_test_required', 'no')).strip().lower() in {'yes', 'y', 'true', '1'}


def find_wav(folder: Path, name: str) -> Path:
    matches = [p for p in folder.glob('*.wav') if key(p.name) == key(name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one WAV for '{name}' in {folder}, found {len(matches)}.")
    return matches[0]


def parse_intervals(text: str) -> list[tuple[float, float]]:
    intervals = []
    for part in str(text).split(';'):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*', part)
        if not match:
            raise ValueError(f'Invalid active interval: {part}')
        start, end = map(float, match.groups())
        if start < 0 or end <= start:
            raise ValueError(f'Invalid active interval: {part}')
        intervals.append((start, end))
    if not intervals:
        raise ValueError('No active intervals were found.')
    return intervals


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        raise ValueError(f'{path.name} is empty.')
    if not np.all(np.isfinite(audio)):
        raise ValueError(f'{path.name} contains NaN or infinite samples.')
    if audio.ndim not in {1, 2}:
        raise ValueError(f'Unsupported audio shape: {audio.shape}')
    return audio, int(sample_rate)


def channel_count(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def to_display_mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else np.mean(audio, axis=1)


def active_sample_mask(frame_count: int, sample_rate: int, intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    duration = frame_count / sample_rate
    for start, end in intervals:
        if start >= duration:
            continue
        clipped_end = min(end, duration)
        a = max(0, min(frame_count, round(start * sample_rate)))
        b = max(0, min(frame_count, round(clipped_end * sample_rate)))
        mask[a:b] = True
    if not np.any(mask):
        raise ValueError('No active samples were found inside the audio duration.')
    return mask


def active_rms_dbfs(audio: np.ndarray, sample_rate: int, intervals: list[tuple[float, float]]) -> float:
    mask = active_sample_mask(len(audio), sample_rate, intervals)
    samples = audio[mask] if audio.ndim == 1 else audio[mask, :].reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return float('-inf') if rms <= 0 else float(20.0 * np.log10(rms))


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)))
    return float('-inf') if peak <= 0 else float(20.0 * np.log10(peak))


def inactive_max_abs(audio: np.ndarray, sample_rate: int, intervals: list[tuple[float, float]]) -> float:
    mask = active_sample_mask(len(audio), sample_rate, intervals)
    inactive = ~mask
    if not np.any(inactive):
        return 0.0
    values = audio[inactive] if audio.ndim == 1 else audio[inactive, :]
    return float(np.max(np.abs(values)))


def calculate_spectrogram(audio_mono: np.ndarray, sample_rate: int):
    nperseg = min(max(int(round(WINDOW_SECONDS * sample_rate)), 128), len(audio_mono))
    noverlap = min(int(round(nperseg * OVERLAP_FRACTION)), nperseg - 1)
    f, t, power = signal.spectrogram(
        audio_mono, fs=sample_rate, window='hann', nperseg=nperseg,
        noverlap=noverlap, detrend='constant', scaling='density', mode='psd'
    )
    return f, t, 10.0 * np.log10(power + np.finfo(float).eps)


def mark_active_intervals(axis, intervals, duration):
    for start, end in intervals:
        if start < duration:
            axis.axvspan(start, min(end, duration), color='white', alpha=0.05)


def save_comparison_plot(bird_name, branch, before_audio, before_sr, after_audio, after_sr, intervals, output_path):
    if before_sr != after_sr:
        raise ValueError(f'Sample-rate mismatch: before={before_sr}, after={after_sr}')

    before_mono = to_display_mono(before_audio)
    after_mono = to_display_mono(after_audio)
    bf, bt, bdb = calculate_spectrogram(before_mono, before_sr)
    af, at, adb = calculate_spectrogram(after_mono, after_sr)

    vmax = float(max(np.max(bdb), np.max(adb)))
    vmin = vmax - DISPLAY_DYNAMIC_RANGE_DB
    max_hz = min(MAX_PLOT_HZ, before_sr / 2.0)
    before_duration = len(before_mono) / before_sr
    after_duration = len(after_mono) / after_sr
    max_time = max(before_duration, after_duration)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True, constrained_layout=True)

    axes[0].pcolormesh(bt, bf, bdb, shading='auto', cmap=COLORMAP, vmin=vmin, vmax=vmax)
    mark_active_intervals(axes[0], intervals, before_duration)
    axes[0].set_title(f'{bird_name} — {branch} — Before normalization')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlim(0, max_time)
    axes[0].set_ylim(0, max_hz)

    image = axes[1].pcolormesh(at, af, adb, shading='auto', cmap=COLORMAP, vmin=vmin, vmax=vmax)
    mark_active_intervals(axes[1], intervals, after_duration)
    axes[1].set_title(f'{bird_name} — {branch} — After normalization')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlim(0, max_time)
    axes[1].set_ylim(0, max_hz)

    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, shrink=0.96)
    cbar.set_label('Power spectral density (dB)')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def determine_status(before_sr, after_sr, before_channels, after_channels,
                     duration_before, duration_after, active_rms_after_dbfs,
                     peak_after_dbfs, inactive_max_abs_after):
    errors = []
    if before_sr != after_sr:
        errors.append('sample rate changed')
    if before_channels != after_channels:
        errors.append('channel count changed')
    if abs(duration_after - duration_before) > DURATION_TOLERANCE_SECONDS:
        errors.append('duration changed')
    if not np.isfinite(active_rms_after_dbfs):
        errors.append('normalized active RMS is invalid')
    elif abs(active_rms_after_dbfs - TARGET_RMS_DBFS) > RMS_TOLERANCE_DB:
        errors.append(f'active RMS is not within ±{RMS_TOLERANCE_DB:.2f} dB of {TARGET_RMS_DBFS:.2f} dBFS')
    if peak_after_dbfs > MAX_ALLOWED_PEAK_DBFS + 1e-9:
        errors.append(f'peak exceeds {MAX_ALLOWED_PEAK_DBFS:.2f} dBFS')
    if inactive_max_abs_after > INACTIVE_ZERO_TOLERANCE:
        errors.append('inactive samples are not zero')
    return ('FAIL', '; '.join(errors)) if errors else ('OK', '')


def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(CSV)

    table = pd.read_csv(CSV)
    required = {'name', 'active_intervals_seconds', 'harmonic_test_required'}
    missing = required - set(table.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))

    for config in BRANCHES.values():
        if not config['before_folder'].exists():
            raise FileNotFoundError(config['before_folder'])
        if not config['after_folder'].exists():
            raise FileNotFoundError(config['after_folder'])

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    print('=' * 80)
    print('NORMALIZATION VALIDATION AND SPECTROGRAM REVIEW')
    print('=' * 80)
    print(f'Target active RMS: {TARGET_RMS_DBFS:.2f} dBFS')
    print(f'Peak limit:        {MAX_ALLOWED_PEAK_DBFS:.2f} dBFS')

    for i, (_, row) in enumerate(table.iterrows(), start=1):
        bird_name = str(row['name']).strip()
        branches = ['without_harmonics']
        if harmonic_test_required(row):
            branches.append('with_harmonics')
        print(f'\n[{i:02d}/{len(table)}] {bird_name}')

        for branch in branches:
            try:
                before_path = find_wav(BRANCHES[branch]['before_folder'], bird_name)
                after_path = find_wav(BRANCHES[branch]['after_folder'], bird_name)
                before_audio, before_sr = load_audio(before_path)
                after_audio, after_sr = load_audio(after_path)
                intervals = parse_intervals(row['active_intervals_seconds'])

                before_channels = channel_count(before_audio)
                after_channels = channel_count(after_audio)
                duration_before = len(before_audio) / before_sr
                duration_after = len(after_audio) / after_sr
                rms_before = active_rms_dbfs(before_audio, before_sr, intervals)
                rms_after = active_rms_dbfs(after_audio, after_sr, intervals)
                rms_change = float(rms_after - rms_before)
                final_peak = peak_dbfs(after_audio)
                inactive_after = inactive_max_abs(after_audio, after_sr, intervals)

                status, error = determine_status(
                    before_sr, after_sr, before_channels, after_channels,
                    duration_before, duration_after, rms_after,
                    final_peak, inactive_after
                )

                plot_folder = PLOT_ROOT / branch
                plot_folder.mkdir(parents=True, exist_ok=True)
                plot_path = plot_folder / f'{safe_name(bird_name)}_{branch}_normalization_comparison.png'
                save_comparison_plot(
                    bird_name, branch, before_audio, before_sr,
                    after_audio, after_sr, intervals, plot_path
                )

                rows.append({
                    'name': bird_name,
                    'branch': branch,
                    'sample_rate_hz': after_sr,
                    'channels': after_channels,
                    'duration_before_seconds': duration_before,
                    'duration_after_seconds': duration_after,
                    'active_rms_before_dbfs': rms_before,
                    'active_rms_after_dbfs': rms_after,
                    'active_rms_change_db': rms_change,
                    'target_rms_dbfs': TARGET_RMS_DBFS,
                    'final_peak_dbfs': final_peak,
                    'peak_limit_dbfs': MAX_ALLOWED_PEAK_DBFS,
                    'inactive_max_abs_after': inactive_after,
                    'status': status,
                    'error': error,
                })
                print(f'  {branch}: {status}, RMS {rms_before:.3f} -> {rms_after:.3f} dBFS, peak {final_peak:.3f} dBFS')

            except Exception as exc:
                rows.append({
                    'name': bird_name,
                    'branch': branch,
                    'sample_rate_hz': np.nan,
                    'channels': np.nan,
                    'duration_before_seconds': np.nan,
                    'duration_after_seconds': np.nan,
                    'active_rms_before_dbfs': np.nan,
                    'active_rms_after_dbfs': np.nan,
                    'active_rms_change_db': np.nan,
                    'target_rms_dbfs': TARGET_RMS_DBFS,
                    'final_peak_dbfs': np.nan,
                    'peak_limit_dbfs': MAX_ALLOWED_PEAK_DBFS,
                    'inactive_max_abs_after': np.nan,
                    'status': 'ERROR',
                    'error': str(exc),
                })
                print(f'  {branch}: ERROR — {exc}')

    pd.DataFrame(rows).to_csv(OUTPUT_REPORT, index=False)
    print('\n' + '=' * 80)
    print('COMPLETE')
    print(f'Report: {OUTPUT_REPORT.resolve()}')
    print(f'Plots:  {PLOT_ROOT.resolve()}')
    print('=' * 80)


if __name__ == '__main__':
    main()
