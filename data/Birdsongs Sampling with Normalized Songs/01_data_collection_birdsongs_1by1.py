import sys
import time
import wave
from pathlib import Path

import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM5"
BAUD = 921600

# Must match the actual STM32 recording sample rate.
SAMPLE_RATE = 35714

RECORD_SECONDS = 10
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # PCM16

# Exact amount expected from the STM32 for every recording.
TOTAL_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
TOTAL_BYTES = TOTAL_SAMPLES * SAMPLE_WIDTH_BYTES

# Allow slightly longer than the nominal recording time because
# serial buffering and operating-system scheduling can add delays.
AUDIO_TIMEOUT_SECONDS = 30.0
MARKER_TIMEOUT_SECONDS = 5.0

BIRDS = [
    "American Robin",
    "Anna's Hummingbird",
    "Barn Swallow",
    "Bewick's Wren",
    "Black Phoebe",
    "California Towhee",
    "Dark-eyed Junco",
    "European Starling",
    "Hooded Oriole",
    "House Finch",
    "House Sparrow",
    "Lesser Goldfinch",
    "Mourning Dove",
    "Northern House Wren",
    "Northern Mockingbird",
    "Northern Yellow Warbler",
    "Oak Titmouse",
    "Rock Pigeon",
    "Song Sparrow",
    "Spotted Towhee",
]

DISTANCES = [
    "1ft",
    "4ft",
    "8ft",
    "12ft",
    "24ft",
    "36ft",
    "48ft",
    "50ft",
]

OUTPUT_DIR = Path("Birdsongs Recordings")


# ============================================================
# FILE-NAME HELPERS
# ============================================================

def clean_filename(name: str) -> str:
    """Convert a bird name into a safe WAV filename."""
    return (
        name.replace("'", "")
        .replace("’", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# SERIAL MARKER FUNCTIONS
# ============================================================

def wait_for_text_marker(
    ser: serial.Serial,
    expected_marker: bytes,
    timeout_seconds: float,
) -> None:
    """
    Wait for a newline-terminated marker such as START or STOP.

    The STM32 must send:
        START\\r\\n
        raw PCM16 bytes
        STOP\\r\\n
    """
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        line = ser.readline()

        if not line:
            continue

        cleaned = line.strip()

        if cleaned == expected_marker:
            return

        # This should not normally happen, but it helps reveal unwanted
        # debug output on the same UART.
        print(f"  Ignored UART line: {cleaned!r}")

    raise TimeoutError(
        f"Timed out waiting for marker {expected_marker!r}."
    )


def wait_for_start_signal(ser: serial.Serial) -> None:
    """Wait for the STM32 START marker."""
    wait_for_text_marker(
        ser=ser,
        expected_marker=b"START",
        timeout_seconds=MARKER_TIMEOUT_SECONDS,
    )


def wait_for_stop_signal(ser: serial.Serial) -> None:
    """Wait for the STM32 STOP marker after the PCM data."""
    wait_for_text_marker(
        ser=ser,
        expected_marker=b"STOP",
        timeout_seconds=MARKER_TIMEOUT_SECONDS,
    )


# ============================================================
# AUDIO RECEPTION
# ============================================================

def record_audio(
    ser: serial.Serial,
    total_bytes: int,
    timeout_seconds: float,
) -> bytes:
    """
    Read exactly total_bytes of PCM16 audio.

    The function stops only after the required byte count has been
    received or when the timeout is reached.
    """
    audio = bytearray()
    deadline = time.monotonic() + timeout_seconds

    while len(audio) < total_bytes:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Audio reception timed out.\n"
                f"Received: {len(audio):,} bytes\n"
                f"Expected: {total_bytes:,} bytes"
            )

        remaining = total_bytes - len(audio)

        # Read in moderately large chunks to reduce Python overhead.
        chunk = ser.read(min(8192, remaining))

        if chunk:
            audio.extend(chunk)

    return bytes(audio)


# ============================================================
# AUDIO VALIDATION
# ============================================================

def validate_pcm16(audio: bytes) -> dict:
    """
    Validate the received little-endian signed PCM16 stream.

    Returns a dictionary of useful signal statistics.
    """
    if not audio:
        raise RuntimeError("No audio data was received.")

    if len(audio) != TOTAL_BYTES:
        raise RuntimeError(
            f"Incorrect audio size: received {len(audio):,} bytes, "
            f"expected {TOTAL_BYTES:,} bytes."
        )

    if len(audio) % SAMPLE_WIDTH_BYTES != 0:
        raise RuntimeError(
            f"Odd PCM byte count: {len(audio):,}. "
            "The 16-bit sample stream may be misaligned."
        )

    samples = np.frombuffer(audio, dtype="<i2")

    if len(samples) != TOTAL_SAMPLES:
        raise RuntimeError(
            f"Incorrect sample count: received {len(samples):,}, "
            f"expected {TOTAL_SAMPLES:,}."
        )

    samples32 = samples.astype(np.int32)
    samples_float = samples32.astype(np.float64)

    minimum = int(samples.min())
    maximum = int(samples.max())
    mean = float(np.mean(samples_float))
    rms = float(np.sqrt(np.mean(samples_float**2)))
    peak = int(np.max(np.abs(samples32)))

    near_full_scale_fraction = float(
        np.mean(np.abs(samples32) >= 32000)
    )

    exact_clipping_fraction = float(
        np.mean(
            (samples32 == 32767)
            | (samples32 == -32768)
        )
    )

    unique_low_bytes = len(np.unique(np.frombuffer(audio, dtype=np.uint8)[::2]))

    stats = {
        "samples": len(samples),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "rms": rms,
        "peak": peak,
        "near_full_scale_percent": near_full_scale_fraction * 100.0,
        "exact_clipping_percent": exact_clipping_fraction * 100.0,
        "unique_low_bytes": unique_low_bytes,
    }

    print(f"  Samples:                  {stats['samples']:,}")
    print(f"  Minimum:                  {stats['minimum']}")
    print(f"  Maximum:                  {stats['maximum']}")
    print(f"  Mean/DC offset:           {stats['mean']:.2f}")
    print(f"  RMS:                      {stats['rms']:.2f}")
    print(f"  Peak:                     {stats['peak']}")
    print(
        "  Near full-scale samples: "
        f"{stats['near_full_scale_percent']:.4f}%"
    )
    print(
        "  Exact clipping samples:  "
        f"{stats['exact_clipping_percent']:.4f}%"
    )
    print(
        "  Unique low-byte values:  "
        f"{stats['unique_low_bytes']}"
    )

    # Reject obviously unsafe recordings.
    if near_full_scale_fraction > 0.01:
        raise RuntimeError(
            "More than 1% of samples are near full scale. "
            "Possible clipping or PCM byte corruption."
        )

    if exact_clipping_fraction > 0.001:
        raise RuntimeError(
            "Too many samples are exactly at the PCM limits. "
            "Possible clipping."
        )

    # A very small number of unique low-byte values can indicate the
    # one-byte alignment corruption seen in earlier recordings.
    if unique_low_bytes < 32:
        raise RuntimeError(
            "The low byte uses unusually few distinct values. "
            "Possible one-byte PCM alignment corruption."
        )

    return stats


# ============================================================
# WAV OUTPUT
# ============================================================

def save_wav(filename: Path, audio: bytes) -> None:
    """Save raw PCM16 bytes as a mono WAV file."""
    filename.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio)


# ============================================================
# ONE RECORDING
# ============================================================

def record_one_bird(
    ser: serial.Serial,
    bird_name: str,
    distance: str,
    output_path: Path,
) -> None:
    """
    Wait for one button press, receive exactly 10 seconds of audio,
    validate it, save it, and wait for the automatic STOP marker.
    """
    print("\n" + "-" * 72)
    print(f"Bird:     {bird_name}")
    print(f"Distance: {distance}")
    print(f"Output:   {output_path}")
    print("Press the Nucleo button once to start.")

    wait_for_start_signal(ser)

    print("START received.")
    print(
        f"Receiving {TOTAL_SAMPLES:,} samples "
        f"({TOTAL_BYTES:,} bytes)..."
    )

    audio = record_audio(
        ser=ser,
        total_bytes=TOTAL_BYTES,
        timeout_seconds=AUDIO_TIMEOUT_SECONDS,
    )

    print(f"Received exactly {len(audio):,} PCM bytes.")
    print("Validating recording...")

    validate_pcm16(audio)

    # The STM32 sends STOP immediately after the last PCM sample.
    print("Waiting for automatic STOP marker...")
    wait_for_stop_signal(ser)
    print("STOP received.")

    save_wav(output_path, audio)

    print(f"Saved successfully: {output_path}")

    # The START/PCM/STOP transaction is now complete. Clearing any
    # unrelated remaining bytes is safe at this point.
    time.sleep(0.1)
    ser.reset_input_buffer()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STM32 BIRDSONG DATA COLLECTION")
    print("=" * 72)
    print(f"Serial port:         {PORT}")
    print(f"Baud rate:           {BAUD:,}")
    print(f"Sample rate:         {SAMPLE_RATE:,} Hz")
    print(f"Recording duration:  {RECORD_SECONDS} seconds")
    print(f"Samples per file:    {TOTAL_SAMPLES:,}")
    print(f"PCM bytes per file:  {TOTAL_BYTES:,}")
    print(f"Output folder:       {OUTPUT_DIR.resolve()}")
    print("=" * 72)

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            timeout=0.1,
            write_timeout=1.0,
        )
    except serial.SerialException as exc:
        print(f"Could not open {PORT}: {exc}")
        sys.exit(1)

    try:
        # Give the serial connection and ST-LINK virtual COM port
        # time to stabilize.
        time.sleep(1.0)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("\nSerial connection ready.")

        for distance_index, distance in enumerate(DISTANCES, start=1):
            distance_dir = OUTPUT_DIR / distance
            distance_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 72)
            print(
                f"DISTANCE {distance_index}/{len(DISTANCES)}: "
                f"{distance}"
            )
            print("=" * 72)

            input(
                f"Set the speaker/microphone distance to {distance}. "
                "Press ENTER when ready..."
            )

            for bird_index, bird_name in enumerate(BIRDS, start=1):
                bird_clean = clean_filename(bird_name)

                output_path = (
                    distance_dir
                    / f"{bird_clean}_{distance}.wav"
                )

                print(
                    f"\nRecording bird {bird_index}/{len(BIRDS)} "
                    f"at {distance}"
                )

                # Prevent accidental overwriting without confirmation.
                if output_path.exists():
                    answer = input(
                        f"{output_path.name} already exists. "
                        "Overwrite it? [y/N]: "
                    ).strip().lower()

                    if answer != "y":
                        print("Skipped existing file.")
                        continue

                try:
                    record_one_bird(
                        ser=ser,
                        bird_name=bird_name,
                        distance=distance,
                        output_path=output_path,
                    )

                except (
                    TimeoutError,
                    RuntimeError,
                    serial.SerialException,
                ) as exc:
                    print("\nRECORDING FAILED")
                    print(f"Reason: {exc}")

                    # Remove a partial or previously overwritten file.
                    if output_path.exists():
                        output_path.unlink()

                    ser.reset_input_buffer()

                    action = input(
                        "Press ENTER to retry this bird, "
                        "or type S to skip it: "
                    ).strip().lower()

                    if action == "s":
                        print(f"Skipped: {bird_name} at {distance}")
                        continue

                    # Retry the same bird until it succeeds or is skipped.
                    while True:
                        try:
                            record_one_bird(
                                ser=ser,
                                bird_name=bird_name,
                                distance=distance,
                                output_path=output_path,
                            )
                            break

                        except (
                            TimeoutError,
                            RuntimeError,
                            serial.SerialException,
                        ) as retry_exc:
                            print("\nRETRY FAILED")
                            print(f"Reason: {retry_exc}")

                            if output_path.exists():
                                output_path.unlink()

                            ser.reset_input_buffer()

                            retry_action = input(
                                "Press ENTER to retry again, "
                                "or type S to skip: "
                            ).strip().lower()

                            if retry_action == "s":
                                print(
                                    f"Skipped: {bird_name} "
                                    f"at {distance}"
                                )
                                break

            print(f"\nCompleted distance: {distance}")

        print("\n" + "=" * 72)
        print("ALL REQUESTED RECORDINGS ARE COMPLETE")
        print(f"Files saved under: {OUTPUT_DIR.resolve()}")
        print("=" * 72)

    except KeyboardInterrupt:
        print("\nRecording stopped by the user.")

    finally:
        if ser.is_open:
            ser.close()

        print("Serial port closed.")


if __name__ == "__main__":
    main()