from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

ROOT = Path('.')
ORIGINAL_FOLDER = ROOT / 'Trimmed'
WITH_HARMONICS_FOLDER = ROOT / 'Filtered_With_Harmonics' / 'Audio'
WITHOUT_HARMONICS_FOLDER = ROOT / 'Filtered_Without_Harmonics' / 'Audio'
INPUT_CSV = ROOT / 'Frequency Range Results' / 'bird_frequency_ranges.csv'
OUTPUT_FOLDER = ROOT / 'Frequency Range Results' / 'filtering_comparison_review'
REPORT_CSV = OUTPUT_FOLDER / 'filtering_comparison_report.csv'

MAX_PLOT_HZ = 15000.0
SPECTROGRAM_WINDOW_SECONDS = 0.025
SPECTROGRAM_OVERLAP = 0.75
COLORMAP = 'magma'
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 99.5
LOW_CUTOFF_COLOR = 'white'
WITH_HARMONICS_COLOR = 'lime'
WITHOUT_HARMONICS_COLOR = 'cyan'


def normalize_name(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r'\(\d+\)$', '', text)
    text = text.lower().replace('’', "'")
    return re.sub(r'[^a-z0-9]+', '', text)


def find_audio_file(folder: Path, bird_name: str) -> Path:
    target = normalize_name(bird_name)
    matches = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == '.wav'
        and normalize_name(p.name) == target
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No WAV file found for '{bird_name}' in {folder.resolve()}"
        )
    raise ValueError(
        f"More than one WAV file matched '{bird_name}' in {folder.resolve()}: "
        + ', '.join(p.name for p in matches)
    )


def parse_active_intervals(text: str) -> list[tuple[float, float]]:
    if pd.isna(text):
        return []
    intervals: list[tuple[float, float]] = []
    for part in str(text).split(';'):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(
            r'\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*', part
        )
        if not match:
            raise ValueError(
                f"Invalid active interval '{part}'. Expected start-end;start-end"
            )
        start = float(match.group(1))
        end = float(match.group(2))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid active interval '{part}'.")
        intervals.append((start, end))
    return intervals


def load_audio_for_plot(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        raise ValueError(f'{path.name} is empty.')
    if not np.all(np.isfinite(audio)):
        raise ValueError(f'{path.name} contains invalid samples.')
    return np.mean(audio, axis=1), int(sr)


def get_required_number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors='coerce')
    if pd.isna(value):
        raise ValueError(f"Missing numeric cutoff in column '{column}'.")
    return float(value)


def harmonic_test_required(row: pd.Series) -> bool:
    return (
        str(row.get('harmonic_test_required', 'no')).strip().lower()
        in {'yes', 'y', 'true', '1'}
    )


def compute_spectrogram(audio: np.ndarray, sr: int):
    nperseg = int(round(SPECTROGRAM_WINDOW_SECONDS * sr))
    nperseg = min(max(nperseg, 128), len(audio))
    noverlap = int(round(nperseg * SPECTROGRAM_OVERLAP))
    noverlap = min(noverlap, nperseg - 1)
    f, t, psd = signal.spectrogram(
        audio,
        fs=sr,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
        mode='psd',
    )
    db = 10.0 * np.log10(psd + np.finfo(float).eps)
    return f, t, db


def draw_cutoff_lines(axis, low_hz, without_hz, with_hz, labels=False):
    axis.axhline(
        low_hz,
        color=LOW_CUTOFF_COLOR,
        linestyle='--',
        linewidth=1.4,
        label=f'Low cutoff: {low_hz:.0f} Hz' if labels else None,
    )
    axis.axhline(
        without_hz,
        color=WITHOUT_HARMONICS_COLOR,
        linestyle='-.',
        linewidth=1.8,
        label=(
            f'Upper cutoff without upper harmonics: {without_hz:.0f} Hz'
            if labels else None
        ),
    )
    if with_hz is not None:
        axis.axhline(
            with_hz,
            color=WITH_HARMONICS_COLOR,
            linestyle='-',
            linewidth=1.8,
            label=(
                f'Upper cutoff with upper harmonics: {with_hz:.0f} Hz'
                if labels else None
            ),
        )


def shade_active_intervals(axis, intervals, duration):
    for start, end in intervals:
        if start >= duration:
            continue
        axis.axvspan(start, min(end, duration), color='white', alpha=0.05)


def make_plot(
    bird_name,
    original_audio,
    original_sr,
    with_audio,
    with_sr,
    without_audio,
    without_sr,
    intervals,
    low_hz,
    without_hz,
    with_hz,
    output_path,
):
    if original_sr != without_sr:
        raise ValueError(
            'Original and without-harmonics files have different sample rates.'
        )
    if with_audio is not None and with_sr != original_sr:
        raise ValueError(
            'The with-harmonics file has a different sample rate.'
        )

    of, ot, odb = compute_spectrogram(original_audio, original_sr)
    nf, nt, ndb = compute_spectrogram(without_audio, without_sr)

    if with_audio is not None:
        wf, wt, wdb = compute_spectrogram(with_audio, with_sr)
        values = np.concatenate([odb.ravel(), wdb.ravel(), ndb.ravel()])
    else:
        wf = wt = wdb = None
        values = np.concatenate([odb.ravel(), ndb.ravel()])

    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('Spectrogram contained no finite values.')

    vmin = -150.0
    vmax = -30.0
    maximum_plot_hz = min(MAX_PLOT_HZ, original_sr / 2.0)

    original_duration = len(original_audio) / original_sr
    without_duration = len(without_audio) / without_sr
    with_duration = len(with_audio) / with_sr if with_audio is not None else 0.0
    maximum_time = max(original_duration, without_duration, with_duration)

    fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True, sharey=True)

    last_image = axes[0].pcolormesh(
        ot, of, odb, shading='auto', cmap=COLORMAP, vmin=vmin, vmax=vmax
    )
    shade_active_intervals(axes[0], intervals, original_duration)
    draw_cutoff_lines(axes[0], low_hz, without_hz, with_hz, labels=True)
    axes[0].set_title('Original — Trimmed')

    if with_audio is not None:
        last_image = axes[1].pcolormesh(
            wt, wf, wdb, shading='auto', cmap=COLORMAP, vmin=vmin, vmax=vmax
        )
        shade_active_intervals(axes[1], intervals, with_duration)
        # Green cutoff only on the with-harmonics panel.
        axes[1].axhline(
            low_hz, color=LOW_CUTOFF_COLOR, linestyle='--', linewidth=1.4
        )
        axes[1].axhline(
            with_hz, color=WITH_HARMONICS_COLOR, linestyle='-', linewidth=1.8
        )
    else:
        axes[1].set_facecolor('black')
        axes[1].text(
            0.5, 0.5, 'Not included in harmonic comparison',
            transform=axes[1].transAxes, ha='center', va='center',
            color='white', fontsize=15, fontweight='bold'
        )
    axes[1].set_title('Filtered — upper harmonics preserved')

    last_image = axes[2].pcolormesh(
        nt, nf, ndb, shading='auto', cmap=COLORMAP, vmin=vmin, vmax=vmax
    )
    shade_active_intervals(axes[2], intervals, without_duration)
    # Cyan cutoff only on the without-harmonics panel.
    axes[2].axhline(
        low_hz, color=LOW_CUTOFF_COLOR, linestyle='--', linewidth=1.4
    )
    axes[2].axhline(
        without_hz, color=WITHOUT_HARMONICS_COLOR, linestyle='-.', linewidth=1.8
    )
    axes[2].set_title('Filtered — upper harmonics removed')

    for axis in axes:
        axis.set_ylabel('Frequency (Hz)')
        axis.set_xlim(0, maximum_time)
        axis.set_ylim(0, maximum_plot_hz)

    axes[0].legend(loc='upper right', framealpha=0.92)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'{bird_name}: original and filtered comparison', fontsize=16)
    cbar = fig.colorbar(
        last_image, ax=axes, fraction=0.025, pad=0.02, shrink=0.96
    )
    cbar.set_label('Power spectral density (dB)')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    for folder in [ORIGINAL_FOLDER, WITH_HARMONICS_FOLDER, WITHOUT_HARMONICS_FOLDER]:
        if not folder.exists():
            raise FileNotFoundError(f'Required folder not found: {folder.resolve()}')
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f'CSV not found: {INPUT_CSV.resolve()}')

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(INPUT_CSV)

    required_columns = {
        'name',
        'active_intervals_seconds',
        'final_filter_low_hz',
        'final_filter_high_without_harmonics_hz',
        'final_filter_high_with_harmonics_hz',
        'harmonic_test_required',
    }
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError('CSV is missing required columns: ' + ', '.join(sorted(missing)))

    results = []
    print('=' * 78)
    print('ORIGINAL / OPTIONAL WITH-HARMONICS / WITHOUT-HARMONICS REVIEW')
    print(f'CSV rows to process: {len(table)}')
    print('=' * 78)

    for index, (_, row) in enumerate(table.iterrows(), start=1):
        bird_name = str(row['name']).strip()
        print(f'\n[{index:02d}/{len(table)}] {bird_name}')
        try:
            original_path = find_audio_file(ORIGINAL_FOLDER, bird_name)
            without_path = find_audio_file(
                WITHOUT_HARMONICS_FOLDER, bird_name
            )

            original_audio, original_sr = load_audio_for_plot(original_path)
            without_audio, without_sr = load_audio_for_plot(without_path)

            intervals = parse_active_intervals(row['active_intervals_seconds'])
            low_hz = get_required_number(row, 'final_filter_low_hz')
            without_hz = get_required_number(
                row, 'final_filter_high_without_harmonics_hz'
            )

            include_harmonic_test = harmonic_test_required(row)
            with_hz = None
            with_path = None
            with_audio = None
            with_sr = None

            if include_harmonic_test:
                with_hz = get_required_number(
                    row, 'final_filter_high_with_harmonics_hz'
                )
                with_path = find_audio_file(
                    WITH_HARMONICS_FOLDER, bird_name
                )
                with_audio, with_sr = load_audio_for_plot(with_path)

            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', bird_name).strip('_')
            output_path = OUTPUT_FOLDER / f'{safe_name}_filtering_comparison.png'

            make_plot(
                bird_name,
                original_audio,
                original_sr,
                with_audio,
                with_sr,
                without_audio,
                without_sr,
                intervals,
                low_hz,
                without_hz,
                with_hz,
                output_path,
            )

            results.append({
                'name': bird_name,
                'original_file': original_path.name,
                'with_harmonics_file': with_path.name if with_path is not None else '',
                'without_harmonics_file': without_path.name,
                'sample_rate_hz': original_sr,
                'harmonic_test_required': 'yes' if include_harmonic_test else 'no',
                'final_filter_low_hz': low_hz,
                'final_filter_high_without_harmonics_hz': without_hz,
                'final_filter_high_with_harmonics_hz': with_hz,
                'plot_file': str(output_path),
                'status': 'OK',
                'error': '',
            })
            print(f'  Saved: {output_path}')

        except Exception as error:
            print(f'  ERROR: {error}')
            results.append({
                'name': bird_name,
                'status': 'ERROR',
                'error': str(error),
            })

    report = pd.DataFrame(results)
    report.to_csv(REPORT_CSV, index=False)
    successful = int((report['status'] == 'OK').sum())
    failed = len(report) - successful

    print('\n' + '=' * 78)
    print('REVIEW COMPLETE')
    print(f'Successful: {successful}')
    print(f'Failed:     {failed}')
    print(f'Plots:      {OUTPUT_FOLDER.resolve()}')
    print(f'Report:     {REPORT_CSV.resolve()}')
    print('=' * 78)


if __name__ == '__main__':
    main()