import re
import time
import wave

import serial


PORT = "COM5"
BAUD = 921600
SAMPLE_RATE = 35714
RECORD_SECONDS = 5
SAMPLE_WIDTH_BYTES = 2

FREQUENCIES = [4000, 6000, 8000, 10000, 12000, 14000]

EXPECTED_AUDIO_BYTES = (
    SAMPLE_RATE * RECORD_SECONDS * SAMPLE_WIDTH_BYTES
)


def clean_distance(text: str) -> str:
    """Convert the entered distance into a filename-safe value."""
    cleaned = text.strip().lower().replace(" ", "")
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)

    if not cleaned:
        raise ValueError("The recording distance cannot be empty.")

    return cleaned


def wait_for_start(ser: serial.Serial) -> None:
    """Wait until the STM32 sends the START line."""
    while True:
        line = ser.readline()

        if not line:
            continue

        if line.strip() == b"START":
            return

        # Optional: display unexpected text messages from the STM32.
        try:
            message = line.decode("utf-8").strip()
            if message:
                print(f"STM32: {message}")
        except UnicodeDecodeError:
            pass


def read_exact_audio(
    ser: serial.Serial,
    required_bytes: int,
    timeout_seconds: float = 10.0,
) -> bytearray:
    """Read exactly the required number of raw PCM audio bytes."""
    audio = bytearray()
    deadline = time.monotonic() + timeout_seconds

    while len(audio) < required_bytes:
        remaining = required_bytes - len(audio)
        chunk = ser.read(min(4096, remaining))

        if chunk:
            audio.extend(chunk)
            deadline = time.monotonic() + timeout_seconds
        elif time.monotonic() >= deadline:
            raise TimeoutError(
                f"Audio transfer stopped after {len(audio)} of "
                f"{required_bytes} expected bytes."
            )

    return audio


def save_wav(filename: str, audio: bytes) -> None:
    """Save mono, signed 16-bit PCM audio."""
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio)


def main() -> None:
    distance = clean_distance(
        input("Enter recording distance, for example 4in or 10cm: ")
    )

    try:
        with serial.Serial(
            PORT,
            BAUD,
            timeout=0.1,
            write_timeout=2,
        ) as ser:
            time.sleep(1)
            ser.reset_input_buffer()

            print("\nReady.")
            print(
                "For each frequency, play the indicated tone and press "
                "the Nucleo button once.\n"
            )

            for frequency in FREQUENCIES:
                filename = f"test_{distance}_{frequency}hz.wav"

                print(f"Prepare the {frequency} Hz tone.")
                print(f"Waiting to record {filename}...")

                wait_for_start(ser)

                print(f"Recording {filename}...")
                audio = read_exact_audio(
                    ser,
                    EXPECTED_AUDIO_BYTES,
                    timeout_seconds=10.0,
                )

                save_wav(filename, audio)

                print(f"Saved: {filename}")
                print(f"Bytes recorded: {len(audio)}")
                print(
                    f"Samples recorded: "
                    f"{len(audio) // SAMPLE_WIDTH_BYTES}\n"
                )

                # Clear only text or unused data remaining after the
                # exact audio payload has already been received.
                time.sleep(0.1)
                ser.reset_input_buffer()

    except serial.SerialException as error:
        print(f"Serial-port error: {error}")
    except TimeoutError as error:
        print(f"Recording error: {error}")
    except ValueError as error:
        print(f"Input error: {error}")

    print("Recording process complete.")


if __name__ == "__main__":
    main()