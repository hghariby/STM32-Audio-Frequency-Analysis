import serial
import wave
import time
import re
from pathlib import Path

PORT = "COM5"
BAUD = 921600
SAMPLE_RATE = 16000   # change to 8000 if STM32 SAI is 8 kHz
RECORD_SECONDS = 5
DISTANCES = ["1ft", "2ft", "4ft", "8ft", "10ft"]

OUTPUT_DIR = Path("birdcall_recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_frequency(text):
    text = text.strip().lower().replace("hz", "")
    text = re.sub(r"[^0-9]", "", text)
    if not text:
        raise ValueError("Enter a number like 4000 or 4000hz.")
    return text

freq = clean_frequency(input("Enter test frequency in Hz, for example 4000: "))

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(1)
ser.reset_input_buffer()

print("\nReady.")
print("For each distance:")
print("1. Press Nucleo button to START.")
print("2. Python records exactly 5 seconds.")
print("3. Press Nucleo button again to STOP before next distance.\n")

for distance in DISTANCES:
    filename = OUTPUT_DIR / f"test_{distance}_{freq}hz.wav"
    audio = bytearray()

    input(f"Set distance to {distance}. Press ENTER when ready...")

    print(f"Waiting for START for {filename.name}...")

    while True:
        data = ser.read(1024)

        if b"START\r\n" in data:
            print(f"Recording {filename.name}...")
            ser.reset_input_buffer()
            break

    TOTAL_BYTES = SAMPLE_RATE * RECORD_SECONDS * 2

    print(f"Waiting for {TOTAL_BYTES} bytes...")

    while len(audio) < TOTAL_BYTES:

        remaining = TOTAL_BYTES - len(audio)

        chunk = ser.read(min(4096, remaining))

        if chunk:
            audio.extend(chunk)

    print(f"Received {len(audio)} bytes")

    with wave.open(str(filename), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio)

    print(f"Saved {filename}")
    print(f"Bytes recorded: {len(audio)}")
    input("Now press Nucleo button again to STOP streaming, then press ENTER here...")

    time.sleep(0.5)
    ser.reset_input_buffer()
    print("Ready for next distance.\n")
    

ser.close()
print("All recordings complete.")

# FREQUENCIES = [1000, 1700, 2400, 3100, 3800, 4500, 5200, 5900, 6600, 7300, 8000]
