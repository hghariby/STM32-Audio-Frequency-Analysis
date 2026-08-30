from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

# ============================ USER SETTINGS ============================
ROOT = Path(".")
RECORDINGS_ROOT = ROOT / "Birdsongs Recordings"
ENVIRONMENT_AUDIO_ROOT = (
    ROOT
    / ".."
    / "Original Audio"
    / "Filtered_With_Harmonics"
    / "Normalized_Audio_for_Environment_SNR"
)
ACTIVE_INTERVALS_CSV = ROOT / "bird_active_call_ranges.csv"
FREQUENCY_RANGES_CSV = ROOT / "bird_frequency_ranges.csv"
OUTPUT_DIR = ROOT / "Acoustic_Results"

DISTANCE_ORDER = ["1ft", "4ft", "8ft", "12ft", "24ft", "36ft", "48ft"]
BAND_METHOD = "equal_energy"  # or "equal_width"
AUTO_ALIGN = False
MAX_ALIGNMENT_SHIFT_SECONDS = 0.50
WELCH_WINDOW_SECONDS = 0.10
WELCH_OVERLAP = 0.50
NOISE_GUARD_SECONDS = 0.10
BANDWIDTH_LOW_PERCENTILE = 0.01
BANDWIDTH_HIGH_PERCENTILE = 0.99
EPSILON = 1e-20

# Leave as None for the first run. In that case:
#   - FDI is NOISE_LIMITED when active power <= noise power
#   - otherwise FDI is REVIEW_PENDING
# After reviewing the actual SNR distribution, set a defensible threshold
# (for example, 3.0 or 6.0 dB) and rerun.
FDI_RELIABLE_SNR_THRESHOLD_DB: float | None = 3.0

ALIASES = {
    "Annas Hummingbird": "Anna's Hummingbird",
    "Bewicks Wren": "Bewick's Wren",
    "Dark eyed Junco": "Dark-eyed Junco",
}
# =====================================================================


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).replace("’", "'").lower())


def parse_intervals(text: str) -> list[tuple[float, float]]:
    intervals = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", item)
        if not match:
            raise ValueError(f"Invalid interval: {item}")
        start, end = float(match.group(1)), float(match.group(2))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval: {item}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError("No active intervals found.")
    return intervals


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
    bird = re.sub(r"\s+", " ", bird).strip()
    return ALIASES.get(bird, bird), distance


def numeric_value(row: pd.Series, candidates: list[str]) -> float | None:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return None


def load_metadata() -> dict[str, dict]:
    active_df = pd.read_csv(ACTIVE_INTERVALS_CSV)
    frequency_df = pd.read_csv(FREQUENCY_RANGES_CSV)
    required = {"name", "active_intervals_seconds"}
    missing = required - set(active_df.columns)
    if missing:
        raise ValueError("Active-call CSV missing: " + ", ".join(sorted(missing)))
    if "name" not in frequency_df.columns:
        raise ValueError("Frequency-range CSV must contain a name column.")

    frequency_lookup = {normalize(row["name"]): row for _, row in frequency_df.iterrows()}
    metadata = {}
    for _, row in active_df.iterrows():
        bird = str(row["name"]).strip()
        key = normalize(bird)
        if key not in frequency_lookup:
            raise KeyError(f"No frequency-range row found for {bird}.")
        frequency_row = frequency_lookup[key]
        low_hz = numeric_value(frequency_row, ["final_filter_low_hz", "suggested_filter_low_hz", "detected_low_hz", "low_hz"])
        high_hz = numeric_value(frequency_row, ["final_filter_high_hz", "suggested_filter_high_hz", "detected_high_hz", "high_hz"])
        if low_hz is None or high_hz is None or high_hz <= low_hz:
            raise ValueError(f"Invalid frequency range for {bird}: {low_hz}–{high_hz}")
        metadata[key] = {
            "bird": bird,
            "intervals": parse_intervals(row["active_intervals_seconds"]),
            "analysis_low_hz": low_hz,
            "analysis_high_hz": high_hz,
        }
    return metadata


def find_environment_audio(folder: Path, bird_name: str) -> Path:
    """Find the normalized environment-reference WAV for one bird."""

    environment_aliases = {
        "Barn Swallow": "Barn Sparrow",
    }

    environment_name = environment_aliases.get(
        bird_name,
        bird_name,
    )

    key = normalize(environment_name)
    matches = [
        path
        for path in folder.glob("*.wav")
        if normalize(path.stem) == key
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No environment-reference WAV found for {bird_name} in {folder.resolve()}."
        )
    raise ValueError(
        f"More than one environment-reference WAV matched {bird_name}: "
        + ", ".join(path.name for path in matches)
    )


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        dtype="float64",
        always_2d=False,
    )

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Invalid audio: {path}")

    # Remove the constant electronic DC offset.
    audio = audio - np.mean(audio)

    return audio, int(sample_rate)


def shift_intervals(intervals: list[tuple[float, float]], offset: float, duration: float) -> list[tuple[float, float]]:
    shifted = []
    for start, end in intervals:
        a, b = max(0.0, start + offset), min(duration, end + offset)
        if b > a:
            shifted.append((a, b))
    if not shifted:
        raise ValueError("No active intervals remain after alignment.")
    return shifted


def build_mask(length: int, sample_rate: int, intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        a = max(0, int(round(start * sample_rate)))
        b = min(length, int(round(end * sample_rate)))
        if b > a:
            mask[a:b] = True
    return mask


def extract_active(audio: np.ndarray, sample_rate: int, intervals: list[tuple[float, float]]) -> np.ndarray:
    active = audio[build_mask(len(audio), sample_rate, intervals)]
    if active.size < 2:
        raise ValueError("No active samples extracted.")
    return active


def rms_linear_and_dbfs(audio: np.ndarray) -> tuple[float, float]:
    """Return RMS amplitude and RMS level in dBFS."""
    rms_linear = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms_linear <= 0:
        return 0.0, -math.inf
    return rms_linear, 20.0 * math.log10(rms_linear)


def peak_linear_and_dbfs(audio: np.ndarray) -> tuple[float, float]:
    """Return peak absolute amplitude and peak level in dBFS."""
    peak_linear = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_linear <= 0:
        return 0.0, -math.inf
    return peak_linear, 20.0 * math.log10(peak_linear)


def alignment_offset(reference: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    length = min(len(reference), len(target))
    reference = reference[:length] - np.mean(reference[:length])
    target = target[:length] - np.mean(target[:length])

    desired_rate = min(sample_rate, 4000)
    step = max(1, int(round(sample_rate / desired_rate)))
    ref_ds, target_ds = reference[::step], target[::step]
    effective_rate = sample_rate / step
    max_lag = round(MAX_ALIGNMENT_SHIFT_SECONDS * effective_rate)

    correlation = signal.correlate(target_ds, ref_ds, mode="full", method="fft")
    lags = signal.correlation_lags(len(target_ds), len(ref_ds), mode="full")
    valid = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(valid):
        return 0.0
    best_lag = int(lags[valid][np.argmax(correlation[valid])])
    return best_lag / effective_rate


def calculate_psd(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(len(audio), max(256, round(WELCH_WINDOW_SECONDS * sample_rate)))
    if nperseg < 16:
        raise ValueError("Not enough audio for PSD.")
    noverlap = min(round(WELCH_OVERLAP * nperseg), nperseg - 1)
    frequencies, psd = signal.welch(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return frequencies, np.maximum(psd, EPSILON)


def restrict(frequencies: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray]:
    high_hz = min(high_hz, frequencies[-1])
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"Too few PSD bins in {low_hz}–{high_hz} Hz.")
    return frequencies[mask], psd[mask]


def integrate(frequencies: np.ndarray, psd: np.ndarray, low_hz: float | None = None, high_hz: float | None = None) -> float:
    mask = np.ones(len(frequencies), dtype=bool)
    if low_hz is not None:
        mask &= frequencies >= low_hz
    if high_hz is not None:
        mask &= frequencies <= high_hz
    selected_f, selected_psd = frequencies[mask], psd[mask]
    return float(np.trapezoid(selected_psd, selected_f)) if len(selected_f) >= 2 else 0.0

def percentile_frequency(frequencies: np.ndarray, psd: np.ndarray, percentile: float) -> float:
    delta_f = np.diff(frequencies)
    segment_energy = 0.5 * (psd[:-1] + psd[1:]) * delta_f
    cumulative = np.concatenate([[0.0], np.cumsum(segment_energy)])
    if cumulative[-1] <= 0:
        return float(frequencies[0])
    cumulative /= cumulative[-1]
    return float(np.interp(percentile, cumulative, frequencies))


def equal_energy_boundaries(frequencies: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    return percentile_frequency(frequencies, psd, 1 / 3), percentile_frequency(frequencies, psd, 2 / 3)


def centroid(frequencies: np.ndarray, psd: np.ndarray) -> float:
    denominator = float(np.sum(psd))
    return float(np.sum(frequencies * psd) / denominator) if denominator > 0 else math.nan


def bandwidth(frequencies: np.ndarray, psd: np.ndarray) -> tuple[float, float, float]:
    low = percentile_frequency(frequencies, psd, BANDWIDTH_LOW_PERCENTILE)
    high = percentile_frequency(frequencies, psd, BANDWIDTH_HIGH_PERCENTILE)
    return low, high, high - low


def fdi_db(reference_psd: np.ndarray, target_psd: np.ndarray) -> float:
    reference_db = 10 * np.log10(np.maximum(reference_psd, EPSILON))
    target_db = 10 * np.log10(np.maximum(target_psd, EPSILON))
    reference_db -= np.mean(reference_db)
    target_db -= np.mean(target_db)
    return float(np.sqrt(np.mean((target_db - reference_db) ** 2)))


def db_ratio(target_power: float, reference_power: float) -> float:
    return 10 * math.log10(target_power / reference_power) if target_power > 0 and reference_power > 0 else math.nan


def calculate_snr(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> dict[str, float | str]:
    """Estimate background noise from non-active regions."""
    active_mask = build_mask(len(audio), sample_rate, intervals)
    duration = len(audio) / sample_rate
    guarded = [
        (
            max(0.0, start - NOISE_GUARD_SECONDS),
            min(duration, end + NOISE_GUARD_SECONDS),
        )
        for start, end in intervals
    ]
    noise_mask = ~build_mask(len(audio), sample_rate, guarded)

    active_audio = audio[active_mask]
    noise_audio = audio[noise_mask]

    active_duration_seconds = active_audio.size / sample_rate
    noise_duration_seconds = noise_audio.size / sample_rate

    if active_audio.size < 2 or noise_audio.size < 2:
        return {
            "active_power": math.nan,
            "noise_power": math.nan,
            "active_region_to_noise_ratio_db": math.nan,
            "noise_subtracted_snr_db": math.nan,
            "snr_status": "INSUFFICIENT_DATA",
            "active_duration_seconds": active_duration_seconds,
            "noise_duration_seconds": noise_duration_seconds,
            "noise_rms_linear": math.nan,
            "noise_rms_dbfs": math.nan,
            "noise_peak_linear": math.nan,
            "noise_peak_dbfs": math.nan,
        }

    active_power = float(np.mean(active_audio ** 2))
    noise_power = float(np.mean(noise_audio ** 2))
    noise_rms_linear, noise_rms_dbfs = rms_linear_and_dbfs(noise_audio)
    noise_peak_linear, noise_peak_dbfs = peak_linear_and_dbfs(noise_audio)

    if noise_power <= 0:
        active_region_to_noise_ratio_db = math.inf
        noise_subtracted_snr_db = math.inf
        snr_status = "NOISE_POWER_ZERO"
    else:
        active_region_to_noise_ratio_db = 10.0 * math.log10(
            max(active_power, EPSILON) / noise_power
        )
        if active_power <= noise_power:
            # Do not replace a non-positive signal estimate with epsilon.
            # That would create an extremely negative dB value that looks
            # more precise than the measurement supports.
            noise_subtracted_snr_db = math.nan
            snr_status = "BELOW_NOISE_FLOOR"
        else:
            signal_power = active_power - noise_power
            noise_subtracted_snr_db = 10.0 * math.log10(
                signal_power / noise_power
            )
            snr_status = "OK"

    return {
        "active_power": active_power,
        "noise_power": noise_power,
        "active_region_to_noise_ratio_db": active_region_to_noise_ratio_db,
        "noise_subtracted_snr_db": noise_subtracted_snr_db,
        "snr_status": snr_status,
        "active_duration_seconds": active_duration_seconds,
        "noise_duration_seconds": noise_duration_seconds,
        "noise_rms_linear": noise_rms_linear,
        "noise_rms_dbfs": noise_rms_dbfs,
        "noise_peak_linear": noise_peak_linear,
        "noise_peak_dbfs": noise_peak_dbfs,
    }


def calculate_environment_adjusted_snr(
    active_power: float,
    recording_noise_power: float,
    environment_noise_power: float,
) -> dict[str, float | str]:
    """
    User-requested combined-noise SNR:

        combined_noise = recording_noise + environment_noise
        estimated_signal = active_power - combined_noise
        SNR = 10*log10(estimated_signal / combined_noise)

    All subtraction/addition is done in linear power before conversion to dB.
    """
    values = [active_power, recording_noise_power, environment_noise_power]
    if not all(math.isfinite(value) for value in values):
        return {
            "combined_noise_power": math.nan,
            "estimated_signal_power": math.nan,
            "environment_adjusted_snr_db": math.nan,
            "environment_adjusted_snr_status": "INSUFFICIENT_DATA",
        }

    combined_noise_power = recording_noise_power + environment_noise_power
    estimated_signal_power = active_power - combined_noise_power

    if combined_noise_power <= 0:
        return {
            "combined_noise_power": combined_noise_power,
            "estimated_signal_power": estimated_signal_power,
            "environment_adjusted_snr_db": math.inf,
            "environment_adjusted_snr_status": "NOISE_POWER_ZERO",
        }

    if estimated_signal_power <= 0:
        return {
            "combined_noise_power": combined_noise_power,
            "estimated_signal_power": estimated_signal_power,
            "environment_adjusted_snr_db": math.nan,
            "environment_adjusted_snr_status": "BELOW_NOISE_FLOOR",
        }

    snr_db = 10.0 * math.log10(
        estimated_signal_power / combined_noise_power
    )
    return {
        "combined_noise_power": combined_noise_power,
        "estimated_signal_power": estimated_signal_power,
        "environment_adjusted_snr_db": snr_db,
        "environment_adjusted_snr_status": "OK",
    }


def classify_snr_status(
    raw_status: str,
    noise_subtracted_snr_db: float,
) -> str:
    """Convert the calculation status into an interpretable SNR class."""
    if raw_status == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    if raw_status == "NOISE_POWER_ZERO":
        return "NOISE_POWER_ZERO"
    if raw_status == "BELOW_NOISE_FLOOR":
        return "BELOW_NOISE_FLOOR"
    if not math.isfinite(noise_subtracted_snr_db):
        return "UNDEFINED"
    if noise_subtracted_snr_db >= 6.0:
        return "RELIABLE"
    if noise_subtracted_snr_db >= 3.0:
        return "MARGINAL"
    if noise_subtracted_snr_db >= 0.0:
        return "POOR"
    return "NOISE_DOMINATED"


def classify_environment_adjusted_snr(
    raw_status: str,
    environment_adjusted_snr_db: float,
) -> str:
    """Classify the new environment-adjusted SNR using the empirical 3 dB cutoff."""
    if raw_status == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    if raw_status == "NOISE_POWER_ZERO":
        return "NOISE_POWER_ZERO"
    if raw_status == "BELOW_NOISE_FLOOR":
        return "BELOW_NOISE_FLOOR"
    if not math.isfinite(environment_adjusted_snr_db):
        return "UNDEFINED"
    if environment_adjusted_snr_db >= 3.0:
        return "RELIABLE"
    if environment_adjusted_snr_db >= 0.0:
        return "POOR"
    return "NOISE_DOMINATED"


def fdi_reliability_status(
    environment_adjusted_status: str,
    environment_adjusted_snr_db: float,
) -> str:
    """Judge FDI reliability using the new environment-adjusted SNR."""
    if environment_adjusted_status in {
        "BELOW_NOISE_FLOOR",
        "INSUFFICIENT_DATA",
        "UNDEFINED",
    }:
        return "NOISE_LIMITED"
    if FDI_RELIABLE_SNR_THRESHOLD_DB is None:
        return "REVIEW_PENDING"
    if not math.isfinite(environment_adjusted_snr_db):
        return "NOISE_LIMITED"
    return (
        "RELIABLE"
        if environment_adjusted_snr_db >= FDI_RELIABLE_SNR_THRESHOLD_DB
        else "NOISE_LIMITED"
    )


def discover_recordings(metadata: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for path in sorted(RECORDINGS_ROOT.rglob("*.wav")):
        bird_text, distance = parse_identity(path)
        key = normalize(bird_text)
        if key not in metadata:
            raise KeyError(f"'{bird_text}' from {path.name} does not match metadata.")
        rows.append({"bird": metadata[key]["bird"], "bird_key": key, "distance": distance, "file": path.name, "path": path})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No WAV files found under {RECORDINGS_ROOT.resolve()}")
    if df.duplicated(["bird_key", "distance"], keep=False).any():
        raise ValueError("More than one recording exists for a bird/distance pair.")
    return df


def main() -> None:
    for path in [
        RECORDINGS_ROOT,
        ENVIRONMENT_AUDIO_ROOT,
        ACTIVE_INTERVALS_CSV,
        FREQUENCY_RANGES_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    recordings = discover_recordings(metadata)
    result_rows, band_rows = [], []

    for bird_key, group in recordings.groupby("bird_key"):
        meta = metadata[bird_key]
        bird = meta["bird"]
        reference_rows = group[group["distance"] == "1ft"]
        if len(reference_rows) != 1:
            raise ValueError(f"Expected exactly one 1-ft recording for {bird}; found {len(reference_rows)}.")

        reference_path = reference_rows.iloc[0]["path"]
        reference_audio, reference_rate = load_audio_mono(reference_path)
        ref_intervals = shift_intervals(
            meta["intervals"],
            0.0,
            len(reference_audio) / reference_rate,
        )
        reference_active = extract_active(reference_audio, reference_rate, ref_intervals)

        # Environment-reference audio for this bird.
        # Use the same active-call intervals and the same non-active/noise logic
        # already used for the STM32 recordings.
        environment_path = find_environment_audio(
            ENVIRONMENT_AUDIO_ROOT,
            bird,
        )
        environment_audio, environment_rate = load_audio_mono(environment_path)
        environment_intervals = shift_intervals(
            meta["intervals"],
            0.0,
            len(environment_audio) / environment_rate,
        )
        environment_metrics = calculate_snr(
            environment_audio,
            environment_rate,
            environment_intervals,
        )
        environment_noise_power = float(environment_metrics["noise_power"])
        environment_noise_rms_dbfs = float(environment_metrics["noise_rms_dbfs"])
        reference_active_rms_linear, reference_active_rms_dbfs = (
            rms_linear_and_dbfs(reference_active)
        )
        reference_active_peak_linear, reference_active_peak_dbfs = (
            peak_linear_and_dbfs(reference_active)
        )
        ref_f, ref_psd_full = calculate_psd(reference_active, reference_rate)
        ref_f, ref_psd = restrict(ref_f, ref_psd_full, meta["analysis_low_hz"], meta["analysis_high_hz"])

        if BAND_METHOD == "equal_energy":
            low_mid, mid_high = equal_energy_boundaries(ref_f, ref_psd)
        elif BAND_METHOD == "equal_width":
            width = (meta["analysis_high_hz"] - meta["analysis_low_hz"]) / 3
            low_mid = meta["analysis_low_hz"] + width
            mid_high = meta["analysis_low_hz"] + 2 * width
        else:
            raise ValueError("BAND_METHOD must be equal_energy or equal_width.")

        ref_total = integrate(ref_f, ref_psd)
        ref_low = integrate(ref_f, ref_psd, meta["analysis_low_hz"], low_mid)
        ref_mid = integrate(ref_f, ref_psd, low_mid, mid_high)
        ref_high = integrate(ref_f, ref_psd, mid_high, meta["analysis_high_hz"])
        ref_centroid = centroid(ref_f, ref_psd)
        ref_bw_low, ref_bw_high, ref_bw = bandwidth(ref_f, ref_psd)

        band_rows.append({
            "bird": bird,
            "reference_file": reference_path.name,
            "band_method": BAND_METHOD,
            "analysis_low_hz": meta["analysis_low_hz"],
            "low_mid_boundary_hz": low_mid,
            "mid_high_boundary_hz": mid_high,
            "analysis_high_hz": meta["analysis_high_hz"],
            "reference_active_rms_linear": reference_active_rms_linear,
            "reference_active_rms_dbfs": reference_active_rms_dbfs,
            "reference_active_peak_linear": reference_active_peak_linear,
            "reference_active_peak_dbfs": reference_active_peak_dbfs,
            "reference_usable_low_edge_hz": ref_bw_low,
            "reference_usable_high_edge_hz": ref_bw_high,
            "reference_usable_bandwidth_hz": ref_bw,
            "reference_total_power": ref_total,
            "reference_low_power": ref_low,
            "reference_mid_power": ref_mid,
            "reference_high_power": ref_high,
        })

        for _, row in group.iterrows():
            path, distance = row["path"], row["distance"]
            print(f"Analyzing {bird} at {distance}: {path.name}")
            try:
                audio, sample_rate = load_audio_mono(path)
                if sample_rate != reference_rate:
                    raise ValueError(f"Sample rate {sample_rate} differs from 1-ft rate {reference_rate}.")

                offset = alignment_offset(reference_audio, audio, sample_rate) if AUTO_ALIGN and distance != "1ft" else 0.0
                intervals = shift_intervals(meta["intervals"], offset, len(audio) / sample_rate)
                active = extract_active(audio, sample_rate, intervals)
                active_rms_linear, active_rms_dbfs = rms_linear_and_dbfs(active)
                active_peak_linear, active_peak_dbfs = peak_linear_and_dbfs(active)
                active_rms_attenuation_db = (
                    active_rms_dbfs - reference_active_rms_dbfs
                )
                active_peak_attenuation_db = (
                    active_peak_dbfs - reference_active_peak_dbfs
                )
                f, psd_full = calculate_psd(active, sample_rate)
                f, psd = restrict(f, psd_full, meta["analysis_low_hz"], meta["analysis_high_hz"])
                target_psd = np.interp(ref_f, f, psd, left=EPSILON, right=EPSILON)

                total = integrate(ref_f, target_psd)
                low = integrate(ref_f, target_psd, meta["analysis_low_hz"], low_mid)
                mid = integrate(ref_f, target_psd, low_mid, mid_high)
                high = integrate(ref_f, target_psd, mid_high, meta["analysis_high_hz"])
                total_att = db_ratio(total, ref_total)
                low_att = db_ratio(low, ref_low)
                mid_att = db_ratio(mid, ref_mid)
                high_att = db_ratio(high, ref_high)
                current_centroid = centroid(ref_f, target_psd)
                bw_low, bw_high, current_bw = bandwidth(ref_f, target_psd)
                snr_metrics = calculate_snr(audio, sample_rate, intervals)

                # Combine the non-active noise from the STM32 recording with
                # the non-active noise from the normalized environment audio.
                recording_noise_power = float(snr_metrics["noise_power"])
                adjusted_snr = calculate_environment_adjusted_snr(
                    active_power=float(snr_metrics["active_power"]),
                    recording_noise_power=recording_noise_power,
                    environment_noise_power=environment_noise_power,
                )

                fdi_value = fdi_db(ref_psd, target_psd)
                raw_snr_status = str(snr_metrics["snr_status"])
                noise_subtracted_snr_db = float(
                    snr_metrics["noise_subtracted_snr_db"]
                )
                snr_quality = classify_snr_status(
                    raw_snr_status,
                    noise_subtracted_snr_db,
                )

                environment_adjusted_status = str(
                    adjusted_snr["environment_adjusted_snr_status"]
                )
                environment_adjusted_snr_db = float(
                    adjusted_snr["environment_adjusted_snr_db"]
                )
                environment_adjusted_snr_quality = (
                    classify_environment_adjusted_snr(
                        environment_adjusted_status,
                        environment_adjusted_snr_db,
                    )
                )
                fdi_status = fdi_reliability_status(
                    environment_adjusted_status,
                    environment_adjusted_snr_db,
                )

                result_rows.append({
                    "bird": bird,
                    "distance": distance,
                    "file": path.name,
                    "reference_file": reference_path.name,
                    "sample_rate_hz": sample_rate,
                    "alignment_offset_seconds": offset,
                    "analysis_low_hz": meta["analysis_low_hz"],
                    "low_mid_boundary_hz": low_mid,
                    "mid_high_boundary_hz": mid_high,
                    "analysis_high_hz": meta["analysis_high_hz"],
                    "active_rms_linear": active_rms_linear,
                    "active_rms_dbfs": active_rms_dbfs,
                    "active_peak_linear": active_peak_linear,
                    "active_peak_dbfs": active_peak_dbfs,
                    "reference_active_rms_linear": reference_active_rms_linear,
                    "reference_active_rms_dbfs": reference_active_rms_dbfs,
                    "reference_active_peak_linear": reference_active_peak_linear,
                    "reference_active_peak_dbfs": reference_active_peak_dbfs,
                    "active_rms_attenuation_db": active_rms_attenuation_db,
                    "active_peak_attenuation_db": active_peak_attenuation_db,
                    "total_power": total,
                    "total_attenuation_db": total_att,
                    "low_band_power": low,
                    "low_attenuation_db": low_att,
                    "mid_band_power": mid,
                    "mid_attenuation_db": mid_att,
                    "high_band_power": high,
                    "high_attenuation_db": high_att,
                    "high_minus_low_db": high_att - low_att,
                    "frequency_distortion_index_db": fdi_value,
                    "fdi_status": fdi_status,
                    "active_power_time_domain": snr_metrics["active_power"],
                    "noise_power_time_domain": snr_metrics["noise_power"],
                    "active_region_to_noise_ratio_db":
                        snr_metrics["active_region_to_noise_ratio_db"],
                    "noise_subtracted_snr_db": noise_subtracted_snr_db,
                    "snr_calculation_status": raw_snr_status,
                    "snr_status": snr_quality,

                    # Environment-adjusted SNR inputs and result.
                    "environment_noise_file": environment_path.name,
                    "environment_noise_power": environment_noise_power,
                    "environment_noise_rms_dbfs": environment_noise_rms_dbfs,
                    "recording_noise_power": recording_noise_power,
                    "recording_noise_rms_dbfs": snr_metrics["noise_rms_dbfs"],
                    "combined_noise_power": adjusted_snr["combined_noise_power"],
                    "estimated_signal_power": adjusted_snr["estimated_signal_power"],
                    "environment_adjusted_snr_db":
                        environment_adjusted_snr_db,
                    "environment_adjusted_snr_status":
                        environment_adjusted_status,
                    "environment_adjusted_snr_quality":
                        environment_adjusted_snr_quality,
                    "active_duration_seconds":
                        snr_metrics["active_duration_seconds"],
                    "noise_duration_seconds":
                        snr_metrics["noise_duration_seconds"],
                    "spectral_centroid_hz": current_centroid,
                    "reference_centroid_hz": ref_centroid,
                    "centroid_shift_hz": current_centroid - ref_centroid,
                    "usable_low_edge_hz": bw_low,
                    "usable_high_edge_hz": bw_high,
                    "usable_bandwidth_hz": current_bw,
                    "reference_usable_low_edge_hz": ref_bw_low,
                    "reference_usable_high_edge_hz": ref_bw_high,
                    "reference_bandwidth_hz": ref_bw,
                    "usable_low_edge_change_hz": bw_low - ref_bw_low,
                    "usable_high_edge_change_hz": bw_high - ref_bw_high,
                    "bandwidth_change_hz": current_bw - ref_bw,
                    "noise_rms_linear": snr_metrics["noise_rms_linear"],
                    "noise_rms_dbfs": snr_metrics["noise_rms_dbfs"],
                    "noise_peak_linear": snr_metrics["noise_peak_linear"],
                    "noise_peak_dbfs": snr_metrics["noise_peak_dbfs"],
                    "status": "OK",
                    "error": "",
                })
            except Exception as exc:
                result_rows.append({
                    "bird": bird,
                    "distance": distance,
                    "file": path.name,
                    "reference_file": reference_path.name,
                    "status": "ERROR",
                    "error": str(exc),
                })

    results = pd.DataFrame(result_rows)
    bands = pd.DataFrame(band_rows)

    results["distance"] = pd.Categorical(
        results["distance"], categories=DISTANCE_ORDER, ordered=True
    )
    results = results.sort_values(["distance", "bird"], na_position="last")
    bands = bands.sort_values("bird")

    results_path = OUTPUT_DIR / "acoustic_by_recording.csv"
    bands_path = OUTPUT_DIR / "band_definitions_1ft.csv"
    noise_recording_path = OUTPUT_DIR / "background_noise_by_recording.csv"
    noise_distance_path = OUTPUT_DIR / "background_noise_by_distance.csv"

    results.to_csv(results_path, index=False)
    bands.to_csv(bands_path, index=False)

    valid_noise = results[results["status"] == "OK"].copy()
    noise_columns = [
        "bird", "distance", "file", "sample_rate_hz",
        "active_duration_seconds", "noise_duration_seconds",
        "noise_rms_linear", "noise_rms_dbfs",
        "noise_power_time_domain", "noise_peak_linear", "noise_peak_dbfs",
        "active_rms_linear", "active_rms_dbfs",
        "active_peak_linear", "active_peak_dbfs",
        "active_power_time_domain",
        "active_region_to_noise_ratio_db", "noise_subtracted_snr_db",
        "snr_calculation_status", "snr_status",
        "environment_noise_file", "environment_noise_power",
        "environment_noise_rms_dbfs", "recording_noise_power",
        "recording_noise_rms_dbfs", "combined_noise_power",
        "estimated_signal_power", "environment_adjusted_snr_db",
        "environment_adjusted_snr_status",
        "environment_adjusted_snr_quality",
        "status", "error",
    ]
    noise_by_recording = valid_noise[noise_columns].copy()
    noise_by_recording.to_csv(noise_recording_path, index=False)

    noise_by_distance = (
        valid_noise.groupby("distance", observed=True)
        .agg(
            segment_count=("file", "count"),
            mean_noise_rms_dbfs=("noise_rms_dbfs", "mean"),
            median_noise_rms_dbfs=("noise_rms_dbfs", "median"),
            std_noise_rms_db=("noise_rms_dbfs", "std"),
            minimum_noise_rms_dbfs=("noise_rms_dbfs", "min"),
            maximum_noise_rms_dbfs=("noise_rms_dbfs", "max"),
            mean_noise_power=("noise_power_time_domain", "mean"),
            median_noise_power=("noise_power_time_domain", "median"),
            mean_noise_duration_seconds=("noise_duration_seconds", "mean"),
            below_noise_floor_count=(
                "snr_calculation_status",
                lambda values: int((values == "BELOW_NOISE_FLOOR").sum()),
            ),
        )
        .reset_index()
    )
    noise_by_distance.to_csv(noise_distance_path, index=False)

    print("\nACOUSTIC ANALYSIS COMPLETE")
    print(f"Rows: {len(results)}")
    print(
        f"OK: {(results['status'] == 'OK').sum()} | "
        f"ERROR: {(results['status'] == 'ERROR').sum()}"
    )
    print(f"Results: {results_path.resolve()}")
    print(f"Bands: {bands_path.resolve()}")
    print(f"Noise by recording: {noise_recording_path.resolve()}")
    print(f"Noise by distance: {noise_distance_path.resolve()}")


if __name__ == "__main__":
    main()
