import serial
import wave
import time
import re

PORT = "COM5"
BAUD = 921600
SAMPLE_RATE = 8000
RECORD_SECONDS = 5

FREQUENCIES = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000]
# FREQUENCIES = [1000, 1700, 2400, 3100, 3800, 4500, 5200, 5900, 6600, 7300, 8000]


def clean_distance(text):
    text = text.strip().lower().replace(" ", "")
    return re.sub(r"[^a-z0-9_]", "", text)

distance = clean_distance(input("Enter recording distance, for example 4in or 10cm: "))

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(1)
ser.reset_input_buffer()

print("Ready.")
print("For each frequency: press Nucleo button ONCE to record 5 seconds.\n")

for freq in FREQUENCIES:
    filename = f"test_{distance}_{freq}hz.wav"
    audio = bytearray()

    print(f"Waiting to record {filename}...")

    while True:
        data = ser.read(1024)

        if b"START\r\n" in data:
            print(f"recording {filename}")

            ser.reset_input_buffer()   # remove leftover STOP/audio bytes

            end_time = time.time() + RECORD_SECONDS

            while time.time() < end_time:
                chunk = ser.read(1024)
                if chunk:
                    audio.extend(chunk)

            break

    ser.reset_input_buffer()  # clear leftover STOP before next frequency

    with wave.open(filename, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio)

    print(f"Saved {filename}")
    print(f"Bytes recorded: {len(audio)}\n")

ser.close()
print("All 9 recordings complete.")