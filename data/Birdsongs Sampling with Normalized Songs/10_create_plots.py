from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

ACOUSTIC_CSV = ROOT / "Acoustic_Results" / "acoustic_by_recording.csv"
BIRDNET_CSV = ROOT / "BirdNET_Results" / "02_evaluated" / "birdnet_by_recording.csv"

RECORDINGS_ROOT = ROOT / "Birdsongs Recordings"
ACTIVE_INTERVALS_CSV = ROOT / "bird_active_call_ranges.csv"
FREQUENCY_RANGES_CSV = ROOT / "bird_frequency_ranges.csv"

OUTPUT_ROOT = ROOT / "Final_Plots"
PER_BIRD =  OUTPUT_ROOT / "Per_Species"
OVERALL = OUTPUT_ROOT / "Overall"
SPECIES_SUMMARY_DIR = OUTPUT_ROOT / "Species_Summaries"
PLOT_DATA_DIR = OUTPUT_ROOT / "Plot_Data"

CONF_DIR = OUTPUT_ROOT / "Confidence"
CONF_DISTANCE_DIR = CONF_DIR / "Confidence_vs_Distance_Bird_Color_Distance_Shape"
CONF_SNR_EXPERIMENT_DIR = CONF_DIR / "Confidence_vs_SNR_By_Experiment"
CONF_SNR_BIRD_DISTSHAPE_DIR = CONF_DIR / "Confidence_vs_SNR_Bird_Color_Distance_Shape"


# ============================================================
# SETTINGS
# ============================================================

DISTANCE_ORDER = ["1ft", "4ft", "8ft", "12ft", "24ft", "36ft", "48ft"]
DISTANCE_FEET = {
    "1ft": 1.0,
    "4ft": 4.0,
    "8ft": 8.0,
    "12ft": 12.0,
    "24ft": 24.0,
    "36ft": 36.0,
    "48ft": 48.0,
}

EXPERIMENT_ORDER = [
    "E1_no_metadata",
    "E2_week_only",
    "E3_location_only",
    "E4_week_and_location",
]

EXPERIMENT_LABELS = {
    "E1_no_metadata": "E1: No metadata",
    "E2_week_only": "E2: Week only",
    "E3_location_only": "E3: Location only",
    "E4_week_and_location": "E4: Week + location",
}

SNR_COLUMN = "environment_adjusted_snr_db"
SNR_QUALITY_COLUMN = "environment_adjusted_snr_quality"
SNR_RELIABILITY_THRESHOLD_DB = 3.0

CONFIDENCE_COLUMN = "expected_mean_active_window_confidence"
CORRECT_TOP1_COLUMN = "mean_correct_top1"

DISTANCE_MARKERS = {
    "1ft": "^",
    "4ft": "o",
    "8ft": "s",
    "12ft": "D",
    "24ft": "v",
    "36ft": "P",
    "48ft": "X",
}

BIRD_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "<", ">", "*",
    "h", "H", "p", "8", "d", "1", "2", "3", "4", "+"
]

# Known species-name mismatch in the current dataset.
# Resolve it once here instead of scattering special cases through plots.
BIRD_ALIASES = {
    "barnswallow": "barnsparrow",
}

WELCH_WINDOW_SECONDS = 0.10
WELCH_OVERLAP = 0.50
EPSILON = 1e-20

DPI = 300
POINT_SIZE = 62
JITTER_STD = 0.055


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(text).replace("’", "'").lower(),
    )


def canonical_bird_key(text: object) -> str:
    return normalize(text)


def safe_name(text: object) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(text))
    return re.sub(r"\s+", "_", value.strip())


def safe_filename(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return value or "unnamed"


def distance_ft(value: str) -> int:
    match = re.fullmatch(r"(\d+)ft", str(value).strip(), flags=re.I)
    if not match:
        raise ValueError(f"Invalid distance: {value}")
    return int(match.group(1))


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def save_figure(path: Path, note: str | None = None, right: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if note:
        plt.gcf().text(
            0.5,
            0.012,
            "Note: " + note,
            ha="center",
            va="bottom",
            fontsize=8,
            wrap=True,
        )
        plt.tight_layout(rect=[0.0, 0.075, right, 1.0])
    else:
        plt.tight_layout(rect=[0.0, 0.0, right, 1.0])

    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def format_distance_axis(ax) -> None:
    x = [distance_ft(d) for d in DISTANCE_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(DISTANCE_ORDER)
    ax.set_xlabel("Distance")


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.3)


def make_bird_colors(birds: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab20")
    return {
        bird: cmap(i % 20)
        for i, bird in enumerate(birds)
    }


def make_bird_markers(birds: list[str]) -> dict[str, str]:
    return {
        bird: BIRD_MARKERS[i % len(BIRD_MARKERS)]
        for i, bird in enumerate(birds)
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ACOUSTIC_CSV.exists():
        raise FileNotFoundError(f"Missing acoustic CSV: {ACOUSTIC_CSV.resolve()}")

    if not BIRDNET_CSV.exists():
        raise FileNotFoundError(f"Missing BirdNET CSV: {BIRDNET_CSV.resolve()}")

    acoustic = pd.read_csv(ACOUSTIC_CSV)
    birdnet = pd.read_csv(BIRDNET_CSV)

    if "status" in acoustic.columns:
        acoustic = acoustic[
            acoustic["status"].astype(str).str.upper().eq("OK")
        ].copy()

    if "status" in birdnet.columns:
        birdnet = birdnet[
            birdnet["status"].astype(str).str.upper().eq("OK")
        ].copy()

    required_acoustic = {
        "bird",
        "distance",
        "active_rms_dbfs",
        "active_rms_attenuation_db",
        "total_attenuation_db",
        "low_attenuation_db",
        "mid_attenuation_db",
        "high_attenuation_db",
        "high_minus_low_db",
        "frequency_distortion_index_db",
        "fdi_status",
        SNR_COLUMN,
        SNR_QUALITY_COLUMN,
        "usable_low_edge_hz",
        "usable_high_edge_hz",
        "bandwidth_change_hz",
    }

    required_birdnet = {
        "bird",
        "distance",
        "experiment",
        CONFIDENCE_COLUMN,
        CORRECT_TOP1_COLUMN,
    }

    require_columns(acoustic, required_acoustic, "Acoustic CSV")
    require_columns(birdnet, required_birdnet, "BirdNET CSV")

    acoustic["distance_ft"] = acoustic["distance"].map(distance_ft)
    birdnet["distance_ft"] = birdnet["distance"].map(distance_ft)

    numeric_acoustic = [
        "active_rms_dbfs",
        "active_rms_attenuation_db",
        "total_attenuation_db",
        "low_attenuation_db",
        "mid_attenuation_db",
        "high_attenuation_db",
        "high_minus_low_db",
        "frequency_distortion_index_db",
        SNR_COLUMN,
        "usable_low_edge_hz",
        "usable_high_edge_hz",
        "bandwidth_change_hz",
    ]

    for column in numeric_acoustic:
        acoustic[column] = pd.to_numeric(acoustic[column], errors="coerce")

    birdnet[CONFIDENCE_COLUMN] = pd.to_numeric(
        birdnet[CONFIDENCE_COLUMN],
        errors="coerce",
    )
    birdnet["correct_top1_bool"] = to_boolean(
        birdnet[CORRECT_TOP1_COLUMN]
    )

    merged = birdnet.merge(
        acoustic,
        on=["bird", "distance"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_acoustic"),
    )

    return acoustic, birdnet, merged


# ============================================================
# PSD SUPPORT
# ============================================================

def load_metadata() -> dict[str, dict]:
    active = pd.read_csv(ACTIVE_INTERVALS_CSV)
    freq = pd.read_csv(FREQUENCY_RANGES_CSV)

    freq_lookup = {
        normalize(row["name"]): row
        for _, row in freq.iterrows()
    }

    metadata = {}

    for _, row in active.iterrows():
        bird = str(row["name"]).strip()
        key = normalize(bird)

        intervals = []

        for item in str(row["active_intervals_seconds"]).split(";"):
            item = item.strip()
            if not item:
                continue

            match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*",
                item,
            )

            if not match:
                raise ValueError(f"Invalid active interval for {bird}: {item}")

            intervals.append(
                (float(match.group(1)), float(match.group(2)))
            )

        if key not in freq_lookup:
            raise KeyError(f"No frequency-range row found for {bird}.")

        fr = freq_lookup[key]

        low_candidates = [
            "final_filter_low_hz",
            "suggested_filter_low_hz",
            "detected_low_hz",
            "low_hz",
        ]

        high_candidates = [
            "final_filter_high_hz",
            "suggested_filter_high_hz",
            "detected_high_hz",
            "high_hz",
        ]

        low_hz = next(
            (
                float(fr[c])
                for c in low_candidates
                if c in fr.index and pd.notna(fr[c])
            ),
            None,
        )

        high_hz = next(
            (
                float(fr[c])
                for c in high_candidates
                if c in fr.index and pd.notna(fr[c])
            ),
            None,
        )

        if low_hz is None or high_hz is None:
            raise ValueError(f"Missing frequency range for {bird}.")

        metadata[key] = {
            "bird": bird,
            "intervals": intervals,
            "low_hz": low_hz,
            "high_hz": high_hz,
        }

    return metadata


def parse_recording_identity(path: Path) -> tuple[str, str]:
    match = re.search(r"(?<!\d)(\d+)ft(?!\w)", path.stem, flags=re.I)

    if match:
        distance = f"{int(match.group(1))}ft"
        bird_text = re.sub(
            r"[_\-\s]*\d+ft.*$",
            "",
            path.stem,
            flags=re.I,
        )
    else:
        parent_match = re.fullmatch(
            r"(\d+)ft",
            path.parent.name,
            flags=re.I,
        )
        distance = (
            f"{int(parent_match.group(1))}ft"
            if parent_match
            else ""
        )
        bird_text = path.stem

    bird = re.sub(r"[_\-]+", " ", bird_text)
    bird = re.sub(r"\s+", " ", bird).strip()

    return bird, distance


def discover_recordings() -> dict[tuple[str, str], Path]:
    lookup = {}

    for path in sorted(RECORDINGS_ROOT.rglob("*.wav")):
        bird, distance = parse_recording_identity(path)
        key = (canonical_bird_key(bird), distance)

        if key in lookup:
            raise RuntimeError(
                "Duplicate recording identity found:\n"
                f"  {lookup[key]}\n"
                f"  {path}\n"
                f"Both resolve to bird={bird!r}, distance={distance!r}."
            )

        lookup[key] = path

    return lookup


def extract_active(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:

    pieces = []
    duration = len(audio) / sample_rate

    for start, end in intervals:
        start = max(0.0, start)
        end = min(duration, end)

        a = int(round(start * sample_rate))
        b = int(round(end * sample_rate))

        if b > a:
            pieces.append(audio[a:b])

    if not pieces:
        raise ValueError("No active audio samples found.")

    return np.concatenate(pieces)


def calculate_psd(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:

    nperseg = min(
        len(audio),
        max(256, round(WELCH_WINDOW_SECONDS * sample_rate)),
    )

    noverlap = min(
        round(WELCH_OVERLAP * nperseg),
        nperseg - 1,
    )

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


def get_psd_curves(
    bird: str,
    metadata: dict[str, dict],
    recordings: dict[tuple[str, str], Path],
) -> list[tuple[str, np.ndarray, np.ndarray]]:

    key = canonical_bird_key(bird)

    if key not in metadata:
        raise KeyError(f"No metadata found for {bird}.")

    meta = metadata[key]
    curves = []

    for distance in DISTANCE_ORDER:

        path = recordings.get((key, distance))

        if path is None:
            continue

        audio, sample_rate = sf.read(
            path,
            dtype="float64",
            always_2d=False,
        )

        audio = np.asarray(audio, dtype=np.float64)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        audio = audio - np.mean(audio)

        active = extract_active(
            audio,
            int(sample_rate),
            meta["intervals"],
        )

        f, psd = calculate_psd(
            active,
            int(sample_rate),
        )

        mask = (
            (f >= meta["low_hz"])
            & (f <= min(meta["high_hz"], f[-1]))
        )

        curves.append(
            (distance, f[mask], psd[mask])
        )

    return curves


# ============================================================
# PER-BIRD PLOTS
# ============================================================

def plot_single_metric_vs_distance(
    bird_df: pd.DataFrame,
    bird: str,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    horizontal_zero: bool = False,
) -> None:

    data = bird_df.sort_values("distance_ft")

    plt.figure(figsize=(8.5, 5.2))

    plt.plot(
        data["distance_ft"],
        data[column],
        marker="o",
        linewidth=2,
    )

    if horizontal_zero:
        plt.axhline(
            0.0,
            linestyle="--",
            linewidth=1,
        )

    plt.title(f"{bird}\n{title}")
    plt.ylabel(ylabel)

    format_distance_axis(plt.gca())
    style_axis(plt.gca())

    save_figure(output_path)


def plot_low_mid_high(
    bird_df: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    data = bird_df.sort_values("distance_ft")

    plt.figure(figsize=(9, 5.5))

    for column, label, marker, linestyle in [
        ("low_attenuation_db", "Low band", "o", "-"),
        ("mid_attenuation_db", "Mid band", "s", "--"),
        ("high_attenuation_db", "High band", "^", "-."),
    ]:
        plt.plot(
            data["distance_ft"],
            data[column],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
            label=label,
        )

    plt.axhline(0.0, linestyle=":", linewidth=1)

    plt.title(
        f"{bird}\nLow / Mid / High Band Attenuation vs Distance"
    )

    plt.ylabel(
        "Attenuation relative to 1 ft (dB)"
    )

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def plot_psd(
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    bird: str,
    output_path: Path,
    normalized: bool,
) -> None:

    if not curves:
        return

    plt.figure(figsize=(10, 6))

    line_styles = [
        "-", "--", "-.", ":", "-", "--", "-."
    ]

    for index, (distance, f, psd) in enumerate(curves):

        psd_db = 10.0 * np.log10(
            np.maximum(psd, EPSILON)
        )

        if normalized:
            psd_db = psd_db - np.mean(psd_db)

        plt.plot(
            f,
            psd_db,
            linestyle=line_styles[index % len(line_styles)],
            linewidth=1.8,
            alpha=max(0.45, 1.0 - index * 0.07),
            label=distance,
        )

    if normalized:
        title = "Level-Normalized PSD vs Frequency — All Distances"
        ylabel = "Relative PSD shape (dB)"
    else:
        title = "Raw PSD vs Frequency — All Distances"
        ylabel = "PSD (dB/Hz)"

    plt.title(f"{bird}\n{title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)

    style_axis(plt.gca())
    plt.legend(ncol=2)

    save_figure(output_path)


def plot_fdi(
    bird_df: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    data = bird_df.sort_values("distance_ft")

    reliable = (
        data["fdi_status"]
        .astype(str)
        .str.upper()
        .eq("RELIABLE")
    )

    plt.figure(figsize=(8.5, 5.2))

    plt.plot(
        data["distance_ft"],
        data["frequency_distortion_index_db"],
        linewidth=1.5,
        alpha=0.6,
    )

    plt.scatter(
        data.loc[reliable, "distance_ft"],
        data.loc[reliable, "frequency_distortion_index_db"],
        marker="o",
        s=55,
        label="Reliable",
        zorder=3,
    )

    if (~reliable).any():
        plt.scatter(
            data.loc[~reliable, "distance_ft"],
            data.loc[~reliable, "frequency_distortion_index_db"],
            marker="x",
            s=70,
            label="Noise-limited",
            zorder=4,
        )

    plt.title(
        f"{bird}\nFrequency Distortion Index vs Distance"
    )

    plt.ylabel("FDI (dB)")

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def plot_snr(
    bird_df: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    data = bird_df.sort_values("distance_ft")

    plt.figure(figsize=(8.5, 5.2))

    plt.plot(
        data["distance_ft"],
        data[SNR_COLUMN],
        marker="o",
        linewidth=2,
    )

    plt.axhline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1.5,
        label="3 dB reliability threshold",
    )

    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
        label="0 dB",
    )

    plt.title(
        f"{bird}\nEnvironment-Adjusted SNR vs Distance"
    )

    plt.ylabel("Environment-adjusted SNR (dB)")

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def plot_frequency_edges(
    bird_df: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    data = bird_df.sort_values("distance_ft")

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        data["distance_ft"],
        data["usable_low_edge_hz"],
        marker="o",
        linewidth=2,
        label="Low edge",
    )

    plt.plot(
        data["distance_ft"],
        data["usable_high_edge_hz"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="High edge",
    )

    plt.title(
        f"{bird}\n1–99% Spectral-Energy Edges vs Distance"
    )

    plt.ylabel("Frequency (Hz)")

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def plot_confidence_vs_snr_bird(
    acoustic_bird: pd.DataFrame,
    birdnet_bird: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    merged = birdnet_bird.merge(
        acoustic_bird[["distance", SNR_COLUMN]],
        on="distance",
        how="left",
        validate="many_to_one",
    )

    plt.figure(figsize=(8.5, 5.5))

    markers = ["o", "s", "^", "D"]

    for experiment, marker in zip(
        EXPERIMENT_ORDER,
        markers,
    ):

        data = merged[
            merged["experiment"] == experiment
        ].dropna(
            subset=[
                SNR_COLUMN,
                CONFIDENCE_COLUMN,
            ]
        )

        if data.empty:
            continue

        plt.scatter(
            data[SNR_COLUMN],
            data[CONFIDENCE_COLUMN],
            marker=marker,
            s=55,
            label=EXPERIMENT_LABELS[experiment],
        )

        data = data.sort_values("distance_ft")

        plt.plot(
            data[SNR_COLUMN],
            data[CONFIDENCE_COLUMN],
            linewidth=1,
            alpha=0.5,
        )

    plt.axvline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1,
        label="3 dB threshold",
    )

    plt.title(
        f"{bird}\nExpected-Bird Confidence vs Environment-Adjusted SNR"
    )

    plt.xlabel("Environment-adjusted SNR (dB)")
    plt.ylabel("Mean expected-bird confidence")
    plt.ylim(-0.05, 1.05)

    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def plot_confidence_vs_distance_bird(
    birdnet_bird: pd.DataFrame,
    bird: str,
    output_path: Path,
) -> None:

    plt.figure(figsize=(9, 5.5))

    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", "-.", ":"]

    for experiment, marker, linestyle in zip(
        EXPERIMENT_ORDER,
        markers,
        styles,
    ):

        data = (
            birdnet_bird[
                birdnet_bird["experiment"] == experiment
            ]
            .sort_values("distance_ft")
        )

        if data.empty:
            continue

        plt.plot(
            data["distance_ft"],
            data[CONFIDENCE_COLUMN],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=EXPERIMENT_LABELS[experiment],
        )

    plt.title(
        f"{bird}\nExpected-Bird Confidence vs Distance"
    )

    plt.ylabel("Mean expected-bird confidence")
    plt.ylim(-0.05, 1.05)

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(output_path)


def create_per_bird_plots(
    acoustic: pd.DataFrame,
    birdnet: pd.DataFrame,
    metadata: dict[str, dict],
    recordings: dict[tuple[str, str], Path],
) -> None:

    birds = sorted(
        set(acoustic["bird"].dropna().astype(str))
        & set(birdnet["bird"].dropna().astype(str))
    )

    for index, bird in enumerate(birds, start=1):

        print(f"  [{index}/{len(birds)}] {bird}")

        acoustic_bird = acoustic[
            acoustic["bird"] == bird
        ].copy()

        birdnet_bird = birdnet[
            birdnet["bird"] == bird
        ].copy()

        folder = PER_BIRD / safe_name(bird)

        plot_single_metric_vs_distance(
            acoustic_bird,
            bird,
            "active_rms_dbfs",
            "Active RMS (dBFS)",
            "Active Bird-Call RMS vs Distance",
            folder / "01_active_rms_dbfs_vs_distance.png",
        )

        plot_single_metric_vs_distance(
            acoustic_bird,
            bird,
            "active_rms_attenuation_db",
            "RMS attenuation relative to 1 ft (dB)",
            "Active RMS Attenuation vs Distance",
            folder / "02_active_rms_attenuation_vs_distance.png",
            horizontal_zero=True,
        )

        plot_low_mid_high(
            acoustic_bird,
            bird,
            folder / "03_low_mid_high_attenuation_vs_distance.png",
        )

        plot_single_metric_vs_distance(
            acoustic_bird,
            bird,
            "total_attenuation_db",
            "Total spectral attenuation relative to 1 ft (dB)",
            "Total Spectral Attenuation vs Distance",
            folder / "04_total_spectral_attenuation_vs_distance.png",
            horizontal_zero=True,
        )

        try:
            curves = get_psd_curves(
                bird,
                metadata,
                recordings,
            )

            plot_psd(
                curves,
                bird,
                folder / "05_raw_psd_all_distances.png",
                normalized=False,
            )

            plot_psd(
                curves,
                bird,
                folder / "06_level_normalized_psd_all_distances.png",
                normalized=True,
            )

        except Exception as exc:
            print(f"    PSD skipped: {exc}")

        plot_fdi(
            acoustic_bird,
            bird,
            folder / "07_fdi_vs_distance.png",
        )

        plot_snr(
            acoustic_bird,
            bird,
            folder / "08_snr_vs_distance.png",
        )

        plot_frequency_edges(
            acoustic_bird,
            bird,
            folder / "09_spectral_edges_vs_distance.png",
        )

        plot_single_metric_vs_distance(
            acoustic_bird,
            bird,
            "bandwidth_change_hz",
            "Bandwidth change from 1 ft (Hz)",
            "1–99% Spectral-Energy Bandwidth Change vs Distance",
            folder / "10_bandwidth_change_vs_distance.png",
            horizontal_zero=True,
        )

        plot_confidence_vs_snr_bird(
            acoustic_bird,
            birdnet_bird,
            bird,
            folder / "11_confidence_vs_snr.png",
        )

        plot_confidence_vs_distance_bird(
            birdnet_bird,
            bird,
            folder / "12_confidence_vs_distance.png",
        )


# ============================================================
# OVERALL PLOTS 05–10
# ============================================================

def plot_overall_total_attenuation(
    acoustic: pd.DataFrame,
) -> None:

    summary = (
        acoustic
        .groupby("distance_ft", as_index=False)
        .agg(
            mean_total_attenuation_db=("total_attenuation_db", "mean"),
            median_total_attenuation_db=("total_attenuation_db", "median"),
        )
        .sort_values("distance_ft")
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        summary["distance_ft"],
        summary["mean_total_attenuation_db"],
        marker="o",
        linewidth=2,
        label="Mean across birds",
    )

    plt.plot(
        summary["distance_ft"],
        summary["median_total_attenuation_db"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Median across birds",
    )

    plt.axhline(0.0, linestyle=":", linewidth=1)

    plt.title(
        "Overall Total Spectral Attenuation vs Distance"
    )

    plt.ylabel(
        "Attenuation relative to each bird's 1-ft reference (dB)"
    )

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(
        OVERALL / "01_Total_Attenuation_vs_Distance.png"
    )


def plot_overall_low_mid_high(
    acoustic: pd.DataFrame,
) -> None:

    summary = (
        acoustic
        .groupby("distance_ft", as_index=False)
        .agg(
            low=("low_attenuation_db", "mean"),
            mid=("mid_attenuation_db", "mean"),
            high=("high_attenuation_db", "mean"),
        )
        .sort_values("distance_ft")
    )

    plt.figure(figsize=(9, 5.5))

    for column, label, marker, linestyle in [
        ("low", "Mean low band", "o", "-"),
        ("mid", "Mean mid band", "s", "--"),
        ("high", "Mean high band", "^", "-."),
    ]:
        plt.plot(
            summary["distance_ft"],
            summary[column],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    plt.axhline(0.0, linestyle=":", linewidth=1)

    plt.title(
        "Overall Low / Mid / High Attenuation vs Distance"
    )

    plt.ylabel(
        "Mean attenuation relative to 1 ft (dB)"
    )

    format_distance_axis(plt.gca())
    style_axis(plt.gca())
    plt.legend()

    save_figure(
        OVERALL / "02_Low_Mid_High_Attenuation_vs_Distance.png"
    )


def plot_high_minus_low_distribution(
    acoustic: pd.DataFrame,
) -> None:

    groups = []
    labels = []

    for distance in DISTANCE_ORDER:
        values = acoustic.loc[
            acoustic["distance"] == distance,
            "high_minus_low_db",
        ].dropna()

        if len(values):
            groups.append(values.to_numpy())
            labels.append(distance)

    plt.figure(figsize=(10, 5.8))

    positions = np.arange(1, len(groups) + 1)

    plt.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        showfliers=False,
    )

    rng = np.random.default_rng(12345)

    for x, values in zip(positions, groups):
        jitter = rng.normal(
            0.0,
            0.055,
            len(values),
        )

        plt.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=28,
            alpha=0.7,
            zorder=3,
        )

    plt.axhline(0.0, linestyle=":", linewidth=1)

    plt.xticks(positions, labels)

    plt.title(
        "High-minus-Low Attenuation Distribution by Distance"
    )

    plt.xlabel("Distance")
    plt.ylabel(
        "High attenuation - low attenuation (dB)"
    )

    style_axis(plt.gca())

    save_figure(
        OVERALL / "03_TOTAL_High_Minus_Low_Distribution_by_Distance.png"
    )


def plot_overall_snr_distribution(
    acoustic: pd.DataFrame,
) -> None:

    groups = []
    labels = []

    for distance in DISTANCE_ORDER:

        values = pd.to_numeric(
            acoustic.loc[
                acoustic["distance"] == distance,
                SNR_COLUMN,
            ],
            errors="coerce",
        )

        values = values[np.isfinite(values)]

        if len(values):
            groups.append(values.to_numpy())
            labels.append(distance)

    plt.figure(figsize=(10, 5.8))

    positions = np.arange(1, len(groups) + 1)

    plt.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        showfliers=False,
    )

    rng = np.random.default_rng(23456)

    for x, values in zip(positions, groups):
        jitter = rng.normal(
            0.0,
            0.055,
            len(values),
        )

        plt.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=28,
            alpha=0.7,
            zorder=3,
        )

    plt.axhline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1.5,
        label="3 dB reliability threshold",
    )

    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
        label="0 dB",
    )

    plt.xticks(positions, labels)

    plt.title(
        "Environment-Adjusted SNR Distribution by Distance"
    )

    plt.xlabel("Distance")
    plt.ylabel("Environment-adjusted SNR (dB)")

    style_axis(plt.gca())
    plt.legend()

    save_figure(
        OVERALL / "04_TOTAL_SNR_Distribution_by_Distance.png"
    )


def plot_overall_fdi_vs_snr(
    acoustic: pd.DataFrame,
) -> None:

    data = acoustic[
        np.isfinite(acoustic[SNR_COLUMN])
        & np.isfinite(acoustic["frequency_distortion_index_db"])
    ].copy()

    plt.figure(figsize=(9, 5.8))

    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for distance, marker in zip(
        DISTANCE_ORDER,
        markers,
    ):

        part = data[
            data["distance"] == distance
        ]

        if part.empty:
            continue

        plt.scatter(
            part[SNR_COLUMN],
            part["frequency_distortion_index_db"],
            marker=marker,
            s=45,
            alpha=0.75,
            label=distance,
        )

    plt.axvline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1.5,
        label="3 dB reliability threshold",
    )

    plt.title(
        "FDI vs Environment-Adjusted SNR — All Birds"
    )

    plt.xlabel("Environment-adjusted SNR (dB)")
    plt.ylabel("FDI (dB)")

    style_axis(plt.gca())
    plt.legend(ncol=2)

    save_figure(
        OVERALL / "05_TOTAL_FDI_vs_SNR.png"
    )


def plot_overall_confidence_distribution(
    birdnet: pd.DataFrame,
) -> None:

    groups = []
    labels = []

    for distance in DISTANCE_ORDER:

        values = birdnet.loc[
            birdnet["distance"] == distance,
            CONFIDENCE_COLUMN,
        ].dropna()

        if len(values):
            groups.append(values.to_numpy())
            labels.append(distance)

    plt.figure(figsize=(11, 6))

    positions = np.arange(1, len(groups) + 1)

    plt.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        showfliers=False,
    )

    rng = np.random.default_rng(12345)

    for x, values in zip(positions, groups):

        jitter = rng.normal(
            0.0,
            0.055,
            len(values),
        )

        plt.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=28,
            alpha=0.55,
            zorder=3,
        )

    plt.xticks(positions, labels)
    plt.ylim(-0.05, 1.05)

    plt.title(
        "Overall BirdNET Expected-Bird Confidence by Distance"
    )

    plt.xlabel("Distance")
    plt.ylabel("Mean expected-bird confidence")

    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    save_figure(
        OVERALL / "06_TOTAL_Confidence_Distribution_by_Distance.png"
    )


# ============================================================
# SIGNAL QUALITY OVERVIEW
# ============================================================

def mean_ci95(
    values: pd.Series,
) -> tuple[float, float]:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return math.nan, math.nan

    mean = float(values.mean())

    if len(values) < 2:
        return mean, math.nan

    ci = 1.96 * float(
        values.std(ddof=1) / math.sqrt(len(values))
    )

    return mean, ci


def plot_signal_quality_overview(
    acoustic: pd.DataFrame,
) -> None:

    rows = []

    for distance in DISTANCE_ORDER:

        subset = acoustic[
            acoustic["distance"] == distance
        ]

        mean, ci = mean_ci95(
            subset[SNR_COLUMN]
        )

        rows.append(
            {
                "distance": distance,
                "mean": mean,
                "ci95": ci,
                "count": int(
                    np.isfinite(
                        pd.to_numeric(
                            subset[SNR_COLUMN],
                            errors="coerce",
                        )
                    ).sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.4),
    )

    x = [
        DISTANCE_FEET[d]
        for d in summary["distance"]
    ]

    axes[0].errorbar(
        x,
        summary["mean"],
        yerr=summary["ci95"],
        marker="o",
        capsize=4,
    )

    axes[0].axhline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1.3,
        label="3 dB reliability threshold",
    )

    axes[0].axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
        label="0 dB",
    )

    axes[0].set_title(
        "Environment-Adjusted SNR by Distance"
    )

    axes[0].set_xlabel("Distance (ft)")
    axes[0].set_ylabel(
        "Environment-adjusted SNR (dB)"
    )

    axes[0].set_xticks(
        list(DISTANCE_FEET.values())
    )

    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    quality = (
        acoustic.assign(
            snr_quality=
                acoustic[SNR_QUALITY_COLUMN]
                .astype(str)
                .str.upper()
        )
        .groupby(
            ["distance", "snr_quality"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(DISTANCE_ORDER)
    )

    bottom = np.zeros(
        len(quality)
    )

    quality_order = [
        "RELIABLE",
        "POOR",
        "NOISE_DOMINATED",
        "BELOW_NOISE_FLOOR",
        "INSUFFICIENT_DATA",
        "UNDEFINED",
    ]

    for status in quality_order:

        if status not in quality.columns:
            continue

        values = quality[status].to_numpy()

        axes[1].bar(
            np.arange(len(quality)),
            values,
            bottom=bottom,
            label=status.replace("_", " ").title(),
        )

        bottom += values

    axes[1].set_title(
        "Signal Quality Classification by Distance"
    )

    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Species count")

    axes[1].set_xticks(
        np.arange(len(quality)),
        DISTANCE_ORDER,
    )

    axes[1].set_ylim(
        0,
        acoustic["bird"].nunique(),
    )

    axes[1].legend(fontsize="small")

    axes[1].grid(
        True,
        axis="y",
        alpha=0.3,
    )

    summary.to_csv(
        PLOT_DATA_DIR / "environment_adjusted_snr_summary.csv",
        index=False,
    )

    quality.reset_index().to_csv(
        PLOT_DATA_DIR / "environment_adjusted_snr_quality_counts.csv",
        index=False,
    )

    save_figure(
        OVERALL / "07_Signal_Quality_Overview.png"
    )


# ============================================================
# HEATMAPS
# ============================================================

def pivot_metric(
    acoustic: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:

    birds = sorted(
        acoustic["bird"]
        .dropna()
        .unique()
    )

    table = acoustic.pivot(
        index="bird",
        columns="distance",
        values=metric,
    )

    return table.reindex(
        index=birds,
        columns=DISTANCE_ORDER,
    )


def draw_heatmap(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    label: str,
) -> None:

    values = table.to_numpy(
        dtype=float
    )

    masked = np.ma.masked_invalid(
        values
    )

    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title(title)

    axis.set_xticks(
        np.arange(len(table.columns)),
        table.columns,
    )

    axis.set_yticks(
        np.arange(len(table.index)),
        table.index,
        fontsize="small",
    )

    axis.set_xlabel("Distance")

    colorbar = plt.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )

    colorbar.set_label(label)


def plot_species_distance_acoustic_heatmaps(
    acoustic: pd.DataFrame,
) -> None:

    reliable = acoustic[
        acoustic["fdi_status"]
        .astype(str)
        .str.upper()
        .eq("RELIABLE")
    ].copy()

    total = pivot_metric(
        acoustic,
        "total_attenuation_db",
    )

    snr = pivot_metric(
        acoustic,
        SNR_COLUMN,
    )

    fdi = pivot_metric(
        reliable,
        "frequency_distortion_index_db",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 9),
    )

    draw_heatmap(
        axes[0],
        total,
        "Total Attenuation",
        "dB relative to 1 ft",
    )

    draw_heatmap(
        axes[1],
        snr,
        "Environment-Adjusted SNR",
        "SNR (dB)",
    )

    draw_heatmap(
        axes[2],
        fdi,
        "Reliable FDI Only",
        "FDI (dB)",
    )

    total.to_csv(
        PLOT_DATA_DIR / "heatmap_total_attenuation.csv"
    )

    snr.to_csv(
        PLOT_DATA_DIR / "heatmap_environment_adjusted_snr.csv"
    )

    fdi.to_csv(
        PLOT_DATA_DIR / "heatmap_reliable_fdi.csv"
    )

    save_figure(
        OVERALL / "08_Species_Distance_Acoustic_Heatmaps.png"
    )


# ============================================================
# SPECIES SUMMARIES
# ============================================================

def plot_species_summary(
    acoustic: pd.DataFrame,
    birdnet: pd.DataFrame,
    bird: str,
) -> None:

    acoustic_bird = (
        acoustic[
            acoustic["bird"] == bird
        ]
        .sort_values("distance_ft")
    )

    birdnet_bird = birdnet[
        birdnet["bird"] == bird
    ].copy()

    if acoustic_bird.empty or birdnet_bird.empty:
        return

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        sharex=True,
    )

    for metric, label in [
        ("total_attenuation_db", "Total"),
        ("low_attenuation_db", "Low"),
        ("mid_attenuation_db", "Middle"),
        ("high_attenuation_db", "High"),
    ]:

        axes[0].plot(
            acoustic_bird["distance_ft"],
            acoustic_bird[metric],
            marker="o",
            label=label,
        )

    axes[0].axhline(
        0.0,
        linewidth=1,
    )

    axes[0].set_ylabel(
        "Attenuation (dB)"
    )

    axes[0].set_title(
        "Acoustic Level and Frequency-Band Loss"
    )

    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        acoustic_bird["distance_ft"],
        acoustic_bird[SNR_COLUMN],
        marker="o",
        label="Environment-adjusted SNR",
    )

    axes[1].axhline(
        SNR_RELIABILITY_THRESHOLD_DB,
        linestyle="--",
        linewidth=1.2,
        label="3 dB reliability threshold",
    )

    axes[1].axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
        label="0 dB",
    )

    axes[1].set_ylabel(
        "Environment-adjusted SNR (dB)"
    )

    axes[1].set_title(
        "Signal Quality"
    )

    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    markers = ["o", "s", "^", "D"]

    for experiment, marker in zip(
        EXPERIMENT_ORDER,
        markers,
    ):

        subset = (
            birdnet_bird[
                birdnet_bird["experiment"] == experiment
            ]
            .sort_values("distance_ft")
        )

        if subset.empty:
            continue

        axes[2].plot(
            subset["distance_ft"],
            subset[CONFIDENCE_COLUMN],
            marker=marker,
            label=EXPERIMENT_LABELS[experiment],
        )

        failures = subset[
            ~subset["correct_top1_bool"]
        ]

        axes[2].scatter(
            failures["distance_ft"],
            failures[CONFIDENCE_COLUMN],
            marker="x",
            s=70,
        )

    axes[2].set_xlabel("Distance (ft)")
    axes[2].set_ylabel(
        "Expected-species confidence"
    )

    axes[2].set_ylim(-0.05, 1.05)

    axes[2].set_title(
        "BirdNET Performance (× marks incorrect mean top-1)"
    )

    axes[2].set_xticks(
        list(DISTANCE_FEET.values())
    )

    axes[2].legend(
        ncol=2,
        fontsize="small",
    )

    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"{bird}: Acoustic Degradation and BirdNET Response",
        y=1.01,
    )

    save_figure(
        SPECIES_SUMMARY_DIR / f"{safe_filename(bird)}_summary.png"
    )


def create_species_summaries(
    acoustic: pd.DataFrame,
    birdnet: pd.DataFrame,
) -> None:

    for bird in sorted(
        acoustic["bird"]
        .dropna()
        .unique()
    ):

        plot_species_summary(
            acoustic,
            birdnet,
            bird,
        )


# ============================================================
# CONFIDENCE VS DISTANCE
# Bird = color, distance = marker shape
# ============================================================

def create_confidence_distance_bird_color_distance_shape(
    birdnet: pd.DataFrame,
) -> None:

    birds = sorted(
        birdnet["bird"]
        .dropna()
        .astype(str)
        .unique()
    )

    bird_colors = make_bird_colors(birds)

    distance_to_x = {
        distance: i
        for i, distance in enumerate(DISTANCE_ORDER, start=1)
    }

    for experiment in EXPERIMENT_ORDER:

        data = birdnet[
            birdnet["experiment"] == experiment
        ].copy()

        fig, ax = plt.subplots(figsize=(16, 9))

        groups = []
        positions = []

        for distance in DISTANCE_ORDER:

            values = data.loc[
                data["distance"] == distance,
                CONFIDENCE_COLUMN,
            ].dropna()

            if len(values):
                groups.append(values.to_numpy())
                positions.append(distance_to_x[distance])

        if groups:
            ax.boxplot(
                groups,
                positions=positions,
                widths=0.55,
                showfliers=False,
            )

        rng = np.random.default_rng(12345)

        for distance in DISTANCE_ORDER:

            part = data[
                data["distance"] == distance
            ]

            if part.empty:
                continue

            x_center = distance_to_x[distance]
            marker = DISTANCE_MARKERS[distance]

            jitters = rng.normal(
                0.0,
                JITTER_STD,
                len(part),
            )

            for jitter, (_, row) in zip(
                jitters,
                part.iterrows(),
            ):

                bird = str(row["bird"])

                ax.scatter(
                    x_center + jitter,
                    row[CONFIDENCE_COLUMN],
                    marker=marker,
                    s=POINT_SIZE,
                    color=bird_colors[bird],
                    edgecolors="black",
                    linewidths=0.45,
                    alpha=0.85,
                    zorder=3,
                )

                ax.annotate(
                    bird,
                    (
                        x_center + jitter,
                        row[CONFIDENCE_COLUMN],
                    ),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=6,
                    color=bird_colors[bird],
                    alpha=0.75,
                )

        ax.set_xticks(
            list(distance_to_x.values()),
            DISTANCE_ORDER,
        )

        ax.set_ylim(-0.05, 1.05)

        ax.set_title(
            f"{EXPERIMENT_LABELS[experiment]}\n"
            "BirdNET Expected-Bird Confidence by Distance",
            fontsize=16,
        )

        ax.set_xlabel("Distance")
        ax.set_ylabel("Mean expected-bird confidence")
        ax.grid(True, axis="y", alpha=0.25)

        bird_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=bird_colors[bird],
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=bird,
            )
            for bird in birds
        ]

        bird_legend = ax.legend(
            handles=bird_handles,
            title="Bird species = color",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
        )

        ax.add_artist(bird_legend)

        distance_handles = [
            Line2D(
                [0],
                [0],
                marker=DISTANCE_MARKERS[d],
                linestyle="None",
                markersize=8,
                markerfacecolor="gray",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=d,
            )
            for d in DISTANCE_ORDER
        ]

        ax.legend(
            handles=distance_handles,
            title="Distance = marker shape",
            bbox_to_anchor=(1.02, 0.34),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
        )

        save_figure(
            CONF_DISTANCE_DIR
            / f"{experiment}_confidence_vs_distance_bird_color_distance_shape.png",
            right=0.79,
        )


# ============================================================
# CONFIDENCE VS SNR
# Bird = color, distance = marker shape
# ============================================================

def create_confidence_snr_bird_color_distance_shape(
    merged: pd.DataFrame,
) -> None:

    birds = sorted(
        merged["bird"]
        .dropna()
        .astype(str)
        .unique()
    )

    bird_colors = make_bird_colors(birds)

    for experiment in EXPERIMENT_ORDER:

        data = merged[
            merged["experiment"] == experiment
        ].copy()

        finite = data[
            np.isfinite(data[SNR_COLUMN])
            & np.isfinite(data[CONFIDENCE_COLUMN])
        ].copy()

        if finite.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 9))

        for bird in birds:

            bird_data = finite[
                finite["bird"] == bird
            ].copy()

            if bird_data.empty:
                continue

            bird_data["distance_order"] = bird_data["distance"].map(
                {
                    d: i
                    for i, d in enumerate(DISTANCE_ORDER)
                }
            )

            bird_data = bird_data.sort_values(
                "distance_order"
            )

            # Connect the same bird across distance using its stable color.
            ax.plot(
                bird_data[SNR_COLUMN],
                bird_data[CONFIDENCE_COLUMN],
                linewidth=1.0,
                alpha=0.45,
                color=bird_colors[bird],
            )

            # Marker shape represents distance.
            for _, row in bird_data.iterrows():

                distance = str(row["distance"])

                ax.scatter(
                    row[SNR_COLUMN],
                    row[CONFIDENCE_COLUMN],
                    marker=DISTANCE_MARKERS[distance],
                    s=POINT_SIZE,
                    color=bird_colors[bird],
                    edgecolors="black",
                    linewidths=0.45,
                    alpha=0.85,
                    zorder=3,
                )

        ax.axvline(
            SNR_RELIABILITY_THRESHOLD_DB,
            linestyle="--",
            linewidth=1.5,
        )

        ax.axvline(
            0.0,
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_ylim(-0.05, 1.05)

        ax.set_title(
            f"{EXPERIMENT_LABELS[experiment]}\n"
            "Expected-Bird Confidence vs Environment-Adjusted SNR",
            fontsize=16,
        )

        ax.set_xlabel(
            "Environment-adjusted SNR (dB)"
        )

        ax.set_ylabel(
            "Mean expected-bird confidence"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        bird_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=bird_colors[bird],
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=bird,
            )
            for bird in birds
        ]

        bird_legend = ax.legend(
            handles=bird_handles,
            title="Bird species = color",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
        )

        ax.add_artist(
            bird_legend
        )

        distance_handles = [
            Line2D(
                [0],
                [0],
                marker=DISTANCE_MARKERS[d],
                linestyle="None",
                markersize=8,
                markerfacecolor="gray",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=d,
            )
            for d in DISTANCE_ORDER
        ]

        distance_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="black",
                    label=f"{SNR_RELIABILITY_THRESHOLD_DB:g} dB reliability threshold",
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle=":",
                    color="black",
                    label="0 dB",
                ),
            ]
        )

        ax.legend(
            handles=distance_handles,
            title="Distance = marker shape",
            bbox_to_anchor=(1.02, 0.34),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
        )

        save_figure(
            CONF_SNR_BIRD_DISTSHAPE_DIR
            / f"{experiment}_confidence_vs_snr_bird_color_distance_shape.png",
            right=0.79,
        )


# ============================================================
# CONFIDENCE WITH VALID SNR BY DISTANCE
# ============================================================

def create_confidence_valid_snr_by_experiment(
    merged: pd.DataFrame,
) -> None:

    for experiment in EXPERIMENT_ORDER:

        data = merged[
            merged["experiment"] == experiment
        ].copy()

        groups = []
        labels = []

        for distance in DISTANCE_ORDER:

            part = data[
                data["distance"] == distance
            ]

            valid = part[
                np.isfinite(part[SNR_COLUMN])
                & np.isfinite(part[CONFIDENCE_COLUMN])
            ]

            if len(valid):

                groups.append(
                    valid[
                        CONFIDENCE_COLUMN
                    ].to_numpy()
                )

                labels.append(
                    distance
                )

        if not groups:
            continue

        plt.figure(
            figsize=(11, 6)
        )

        positions = np.arange(
            1,
            len(groups) + 1,
        )

        plt.boxplot(
            groups,
            positions=positions,
            widths=0.55,
            showfliers=False,
        )

        rng = np.random.default_rng(
            12345
        )

        for x, values in zip(
            positions,
            groups,
        ):

            jitter = rng.normal(
                0.0,
                0.055,
                len(values),
            )

            plt.scatter(
                np.full(
                    len(values),
                    x,
                ) + jitter,
                values,
                s=30,
                alpha=0.65,
                zorder=3,
            )

        plt.xticks(
            positions,
            labels,
        )

        plt.ylim(
            -0.05,
            1.05,
        )

        plt.title(
            f"{EXPERIMENT_LABELS[experiment]}\n"
            "BirdNET Expected-Bird Confidence for Recordings "
            "with Valid Environment-Adjusted SNR"
        )

        plt.xlabel("Distance")

        plt.ylabel(
            "Mean expected-bird confidence"
        )

        plt.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        save_figure(
            CONF_SNR_EXPERIMENT_DIR
            / f"{experiment}_confidence_with_valid_snr_by_distance.png"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    for directory in [
        OUTPUT_ROOT,
        PER_BIRD,
        SPECIES_SUMMARY_DIR,
        OVERALL,
        PLOT_DATA_DIR,
        CONF_DIR,
        CONF_DISTANCE_DIR,
        CONF_SNR_EXPERIMENT_DIR,
        CONF_SNR_BIRD_DISTSHAPE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    acoustic, birdnet, merged = load_tables()

    print(
        f"Acoustic rows: {len(acoustic)}"
    )

    print(
        f"BirdNET rows: {len(birdnet)}"
    )

    print(
        f"Species: {acoustic['bird'].nunique()}"
    )

    print(
        "\nLoading metadata / recordings for PSD plots..."
    )

    metadata = load_metadata()
    recordings = discover_recordings()

    print(
        "\nCreating per-bird plot folders..."
    )

    create_per_bird_plots(
        acoustic,
        birdnet,
        metadata,
        recordings,
    )

    print(
        "\nCreating overall plots..."
    )

    plot_overall_total_attenuation(
        acoustic
    )

    plot_overall_low_mid_high(
        acoustic
    )

    plot_high_minus_low_distribution(
        acoustic
    )

    plot_overall_snr_distribution(
        acoustic
    )

    plot_overall_fdi_vs_snr(
        acoustic
    )

    plot_overall_confidence_distribution(
        birdnet
    )


    plot_signal_quality_overview(
        acoustic
    )


    plot_species_distance_acoustic_heatmaps(
        acoustic
    )

    print(
        "Creating species summary plots..."
    )

    create_species_summaries(
        acoustic,
        birdnet,
    )

    print(
        "Creating Confidence plots..."
    )

    create_confidence_distance_bird_color_distance_shape(
        birdnet
    )

    create_confidence_snr_bird_color_distance_shape(
        merged
    )

    create_confidence_valid_snr_by_experiment(
        merged
    )

    print(
        "\nFINAL PLOTTING COMPLETE"
    )

    print(
        f"Saved under: {OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()
