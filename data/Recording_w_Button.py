import serial
import wave
import time

PORT = "COM5"
BAUD = 921600
SAMPLE_RATE = 8000
OUTPUT_FILE = "stm32_button_recording.wav"

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(1)
ser.reset_input_buffer()

print("Waiting for START from STM32 button...")

audio = bytearray()
recording = False

while True:
    data = ser.read(1024)

    if not data:
        continue

    if b"START\r\n" in data:
        print("Recording started")
        recording = True
        audio.clear()

        # remove START text from audio data
        data = data.replace(b"START\r\n", b"")

    if b"STOP\r\n" in data:
        print("Recording stopped")

        # keep only data before STOP
        data = data.split(b"STOP\r\n")[0]

        if recording:
            audio.extend(data)

        break

    if recording:
        audio.extend(data)

ser.close()

print("Bytes recorded:", len(audio))

with wave.open(OUTPUT_FILE, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)      # int16 audio
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(audio)

print("Saved", OUTPUT_FILE)