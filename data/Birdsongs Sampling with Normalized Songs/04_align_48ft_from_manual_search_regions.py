from pathlib import Path
import re

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


ROOT = Path(".")
COMBINED_FILE = ROOT / "Combined_Bird_Calls.wav"
RECORDING_FILE = ROOT / "Distance_Recordings" /  "D07_48FT.WAV"
SEARCH_CSV = ROOT / "search_regions.csv"

OUTPUT_FOLDER = ROOT / "Birdsongs Recordings" / "48ft"
REPORT_FILE = ROOT / "Segmentation_Results" / "48ft_manual_region_alignment_report.csv"

INITIAL_SILENCE_SECONDS = 5.0
BIRD_DURATION_SECONDS = 10.0
SLOT_SECONDS = 15.0

FRAME_MS = 30.0
HOP_MS = 10.0


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_mono(path):
    audio, sample_rate = sf.read(path, dtype="float64")
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        if audio.shape[1] != 1:
            raise ValueError(f"{path.name} must be mono.")
        audio = audio[:, 0]

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"{path.name} is empty or invalid.")

    return audio, int(sample_rate)


def rms_envelope(audio, sample_rate):
    audio = audio - np.mean(audio)

    frame = max(1, round(FRAME_MS * sample_rate / 1000))
    hop = max(1, round(HOP_MS * sample_rate / 1000))

    if len(audio) < frame:
        raise ValueError("Audio is shorter than one RMS frame.")

    squared = audio * audio
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))

    starts = np.arange(0, len(audio) - frame + 1, hop)
    energy = cumulative[starts + frame] - cumulative[starts]
    envelope = np.sqrt(energy / frame)

    envelope -= np.mean(envelope)
    std = np.std(envelope)
    if std > 0:
        envelope /= std

    return envelope


def normalized_correlation(search_envelope, reference_envelope):
    if len(search_envelope) < len(reference_envelope):
        raise ValueError("Search region is shorter than the 10-second reference.")

    reference = reference_envelope - np.mean(reference_envelope)
    reference_norm = np.linalg.norm(reference)

    numerator = signal.correlate(
        search_envelope,
        reference,
        mode="valid",
        method="fft",
    )

    n = len(reference)
    squared = search_envelope * search_envelope
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    window_energy = cumulative[n:] - cumulative[:-n]

    denominator = reference_norm * np.sqrt(
        np.maximum(window_energy, np.finfo(float).tiny)
    )

    return numerator / denominator


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    combined_audio, combined_rate = load_mono(COMBINED_FILE)
    recording_audio, recording_rate = load_mono(RECORDING_FILE)

    table = pd.read_csv(SEARCH_CSV)

    required = {"name", "start_seconds", "end_seconds"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            "CSV is missing columns: " + ", ".join(sorted(missing))
        )

    report_rows = []
    recording_duration = len(recording_audio) / recording_rate

    for index, row in table.iterrows():
        bird = str(row["name"]).strip()
        search_start = float(row["start_seconds"])
        search_end = float(row["end_seconds"])

        reference_start = INITIAL_SILENCE_SECONDS + index * SLOT_SECONDS
        reference_end = reference_start + BIRD_DURATION_SECONDS

        if search_start < 0 or search_end > recording_duration:
            raise ValueError(
                f"{bird}: search region {search_start}-{search_end} s "
                f"is outside the recording duration {recording_duration:.3f} s."
            )

        if search_end - search_start < BIRD_DURATION_SECONDS:
            raise ValueError(
                f"{bird}: search region is only "
                f"{search_end - search_start:.3f} s. "
                f"It must be at least {BIRD_DURATION_SECONDS:.1f} s."
            )

        reference = combined_audio[
            round(reference_start * combined_rate):
            round(reference_end * combined_rate)
        ]

        search_audio = recording_audio[
            round(search_start * recording_rate):
            round(search_end * recording_rate)
        ]

        reference_envelope = rms_envelope(reference, combined_rate)
        search_envelope = rms_envelope(search_audio, recording_rate)

        scores = normalized_correlation(
            search_envelope,
            reference_envelope,
        )

        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        aligned_start = search_start + best_index * HOP_MS / 1000.0
        aligned_end = aligned_start + BIRD_DURATION_SECONDS

        start_sample = round(aligned_start * recording_rate)
        end_sample = start_sample + round(
            BIRD_DURATION_SECONDS * recording_rate
        )

        if end_sample > len(recording_audio):
            raise ValueError(
                f"{bird}: aligned segment extends past the recording end."
            )

        segment = recording_audio[start_sample:end_sample]

        output_file = OUTPUT_FOLDER / f"{safe_name(bird)}_48ft.wav"
        sf.write(
            output_file,
            segment,
            recording_rate,
            subtype="PCM_16",
        )

        report_rows.append({
            "bird_index": index + 1,
            "name": bird,
            "reference_start_seconds": reference_start,
            "reference_end_seconds": reference_end,
            "start_seconds": search_start,
            "end_seconds": search_end,
            "aligned_start_seconds": aligned_start,
            "aligned_end_seconds": aligned_end,
            "correlation_score": best_score,
            "output_file": str(output_file),
        })

        print(
            f"{index + 1:02d}. {bird}: "
            f"{aligned_start:.3f}-{aligned_end:.3f} s, "
            f"correlation={best_score:.4f}"
        )

    pd.DataFrame(report_rows).to_csv(REPORT_FILE, index=False)

    print("\nDone.")
    print(f"Segments: {OUTPUT_FOLDER.resolve()}")
    print(f"Report:   {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()
