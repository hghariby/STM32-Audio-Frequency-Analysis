import serial
import wave
import time
import re
from pathlib import Path

# =========================
# CONFIG
# =========================
PORT = "COM5"
BAUD = 921600

# IMPORTANT FIX (REAL STM32 RATE)
SAMPLE_RATE = 17857   # 17.857 kHz

RECORD_SECONDS = 5

DISTANCES = ["1ft", "2ft", "4ft", "8ft", "10ft"]

OUTPUT_DIR = Path("birdcallrange_recordings")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# CLEAN INPUT
# =========================
def clean_frequency(text):
    text = text.strip().lower()
    text = re.sub(r"[^0-9]", "", text)
    if not text:
        raise ValueError("Enter the species of bir")
    return text


freq = clean_frequency(input("Enter bird : "))


# =========================
# SERIAL INIT
# =========================
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(1)
ser.reset_input_buffer()

print("\nReady.")
print("Press Nucleo button to START recording each sample.")


# =========================
# MAIN LOOP
# =========================
for distance in DISTANCES:

    filename = OUTPUT_DIR / f"test_{distance}_{freq}hz.wav"
    audio = bytearray()

    input(f"\nSet distance to {distance}. Press ENTER when ready...")

    print(f"Waiting for START signal for {filename.name}...")

    # Wait for START
    while True:
        data = ser.read(1024)

        if b"START\r\n" in data:
            print("Recording started...")
            ser.reset_input_buffer()
            break

    # =========================
    # FIXED BYTE LENGTH
    # =========================
    TOTAL_BYTES = int(SAMPLE_RATE * RECORD_SECONDS * 2)

    print(f"Collecting {TOTAL_BYTES} bytes...")

    # =========================
    # DATA COLLECTION
    # =========================
    while len(audio) < TOTAL_BYTES:

        remaining = TOTAL_BYTES - len(audio)
        chunk = ser.read(min(4096, remaining))

        if chunk:
            audio.extend(chunk)

    print(f"Received {len(audio)} bytes")

    # =========================
    # SAVE WAV
    # =========================
    with wave.open(str(filename), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio)

    print(f"Saved: {filename}")

    # small delay for stability
    time.sleep(0.5)
    ser.reset_input_buffer()

    print("Ready for next distance.")


ser.close()
print("\nAll recordings complete.")