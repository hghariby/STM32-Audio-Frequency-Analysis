from __future__ import annotations

import math
import re
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ============================ USER SETTINGS ============================
ROOT = Path(".")
RECORDINGS_ROOT = ROOT / "Birdsongs Recordings"
OUTPUT_DIR = ROOT / "Recording_Validation"

EXPECTED_SAMPLE_RATE = 35714
EXPECTED_DURATION_SECONDS = 10.0
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2
DISTANCE_ORDER = ["1ft", "4ft", "8ft", "12ft", "24ft", "36ft", "48ft"]

DURATION_TOLERANCE_SECONDS = 0.005
CLIPPING_SAMPLE_FRACTION_LIMIT = 0.001
NEAR_FULL_SCALE_FRACTION_LIMIT = 0.01
NEAR_FULL_SCALE_THRESHOLD = 32000
MIN_UNIQUE_LOW_BYTES = 32
# =====================================================================


def parse_identity(path: Path) -> tuple[str, str]:
    match = re.search(r"(?<!\d)(\d+)ft(?!\w)", path.stem, flags=re.I)
    if match:
        distance = f"{int(match.group(1))}ft"
        bird_text = re.sub(r"[_\-\s]*\d+ft.*$", "", path.stem, flags=re.I)
    else:
        parent_match = re.fullmatch(r"(\d+)ft", path.parent.name, flags=re.I)
        distance = f"{int(parent_match.group(1))}ft" if parent_match else ""
        bird_text = path.stem
    bird = re.sub(r"[_\-]+", " ", bird_text)
    return re.sub(r"\s+", " ", bird).strip(), distance


def to_dbfs(linear: float) -> float:
    return -math.inf if linear <= 0 else 20.0 * math.log10(linear)


def inspect_wav(path: Path) -> dict:
    bird, distance = parse_identity(path)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()

        duration = frame_count / sample_rate if sample_rate else math.nan
        audio, decoded_rate = sf.read(path, dtype="int16", always_2d=True)
        mono = audio[:, 0].astype(np.int32)

        if decoded_rate != sample_rate:
            errors.append(f"decoded sample rate {decoded_rate} differs from header {sample_rate}")
        if audio.shape[1] != channels:
            errors.append(f"decoded channels {audio.shape[1]} differs from header {channels}")

        minimum = int(np.min(mono)) if mono.size else 0
        maximum = int(np.max(mono)) if mono.size else 0
        mean = float(np.mean(mono)) if mono.size else math.nan
        rms_counts = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
        peak_counts = int(np.max(np.abs(mono))) if mono.size else 0
        clipping_fraction = float(np.mean((mono == 32767) | (mono == -32768))) if mono.size else 0.0
        near_full_fraction = float(np.mean(np.abs(mono) >= NEAR_FULL_SCALE_THRESHOLD)) if mono.size else 0.0

        # Low-byte diversity is a useful warning for the one-byte corruption pattern.
        pcm_bytes = mono.astype("<i2", copy=False).tobytes()
        unique_low_bytes = len(np.unique(np.frombuffer(pcm_bytes, dtype=np.uint8)[::2])) if pcm_bytes else 0

        if compression != "NONE":
            errors.append(f"compressed WAV ({compression})")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            errors.append(f"sample rate {sample_rate}; expected {EXPECTED_SAMPLE_RATE}")
        if channels != EXPECTED_CHANNELS:
            errors.append(f"channels {channels}; expected {EXPECTED_CHANNELS}")
        if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
            errors.append(f"sample width {sample_width}; expected {EXPECTED_SAMPLE_WIDTH_BYTES}")
        if abs(duration - EXPECTED_DURATION_SECONDS) > DURATION_TOLERANCE_SECONDS:
            errors.append(f"duration {duration:.6f}s; expected {EXPECTED_DURATION_SECONDS:.6f}s")

        expected_frames = round(EXPECTED_SAMPLE_RATE * EXPECTED_DURATION_SECONDS)
        if frame_count != expected_frames:
            errors.append(f"frame count {frame_count}; expected {expected_frames}")
        if mono.size == 0:
            errors.append("no audio samples")
        if peak_counts == 0:
            errors.append("completely silent")
        if clipping_fraction > CLIPPING_SAMPLE_FRACTION_LIMIT:
            errors.append(f"clipping fraction {clipping_fraction:.6%} exceeds limit")
        if near_full_fraction > NEAR_FULL_SCALE_FRACTION_LIMIT:
            errors.append(f"near-full-scale fraction {near_full_fraction:.6%} exceeds limit")
        if unique_low_bytes < MIN_UNIQUE_LOW_BYTES:
            warnings.append(f"only {unique_low_bytes} unique low-byte values; inspect for byte corruption")
        if abs(mean) > 7000:
            warnings.append(f"large DC offset: {mean:.2f} counts")

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        return {
            "file": path.name,
            "relative_path": str(path.relative_to(RECORDINGS_ROOT)),
            "bird": bird,
            "distance": distance,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "minimum_sample": minimum,
            "maximum_sample": maximum,
            "mean_sample_dc_offset": mean,
            "rms_counts": rms_counts,
            "rms_dbfs": to_dbfs(rms_counts / 32768.0),
            "peak_counts": peak_counts,
            "peak_dbfs": to_dbfs(peak_counts / 32768.0),
            "clipping_fraction": clipping_fraction,
            "near_full_scale_fraction": near_full_fraction,
            "unique_low_byte_values": unique_low_bytes,
            "status": status,
            "errors": " | ".join(errors),
            "warnings": " | ".join(warnings),
        }
    except Exception as exc:
        return {
            "file": path.name,
            "relative_path": str(path),
            "bird": bird,
            "distance": distance,
            "status": "FAIL",
            "errors": str(exc),
            "warnings": "",
        }


def main() -> None:
    if not RECORDINGS_ROOT.exists():
        raise FileNotFoundError(f"Recording folder not found:\n{RECORDINGS_ROOT.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(RECORDINGS_ROOT.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under:\n{RECORDINGS_ROOT.resolve()}")

    rows = []
    for index, path in enumerate(wav_paths, start=1):
        print(f"[{index}/{len(wav_paths)}] {path}")
        rows.append(inspect_wav(path))

    report = pd.DataFrame(rows)
    report["distance"] = pd.Categorical(report["distance"], categories=DISTANCE_ORDER, ordered=True)
    report = report.sort_values(["distance", "bird", "file"], na_position="last")

    full_path = OUTPUT_DIR / "recording_validation.csv"
    failed_path = OUTPUT_DIR / "failed_recordings.csv"
    report.to_csv(full_path, index=False)
    report[report["status"] == "FAIL"].to_csv(failed_path, index=False)

    counts = report["status"].value_counts().to_dict()
    print("\nVALIDATION COMPLETE")
    print(f"PASS: {counts.get('PASS', 0)} | WARNING: {counts.get('WARNING', 0)} | FAIL: {counts.get('FAIL', 0)}")
    print(f"Full report: {full_path.resolve()}")
    print(f"Failed files: {failed_path.resolve()}")


if __name__ == "__main__":
    main()
