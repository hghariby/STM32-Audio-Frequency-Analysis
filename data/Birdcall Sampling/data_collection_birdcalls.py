import serial
import wave
import time
import sys
from pathlib import Path

# =========================
# CONFIG
# =========================
PORT = "COM5"
BAUD = 921600
SAMPLE_RATE = 17857  # 17.857 kHz
RECORD_SECONDS = 5
DISTANCES = [ "24ft"] #"16ft", "24ft"

# Create target directory
OUTPUT_DIR = Path("birdcall_recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# USER INPUT FOR SPECIES
# =========================
# Ask the user for the bird name at runtime
species_name = input("Enter the name of the bird you are recording: ").strip()

# Safely exit if no name was typed
if not species_name:
    print("Error: No bird name entered. Exiting script.")
    sys.exit()

# Replace spaces with underscores for safer file names
species_clean = species_name.replace(" ", "_")

SPECIES_DIR = OUTPUT_DIR / f"{species_clean}"
SPECIES_DIR.mkdir(exist_ok=True)

# =========================
# SERIAL INIT
# =========================
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(1)
ser.reset_input_buffer()

print("\nReady.")
print(f"Recording sequence started for species: {species_name}")
print("Press Nucleo button to START recording each sample.")

# =========================
# MAIN LOOP
# =========================
for distance in DISTANCES:
    # Uses the entered name (species_clean) directly in the filename 
    filename = SPECIES_DIR / f"{species_clean}_{distance}.wav"
    audio = bytearray()
    
    input(f"\nSet distance to {distance} for {species_name}. Press ENTER when ready...")
    print(f"Waiting for START signal for {filename.name}...")
    
    # Wait for START from STM32 button
    while True:
        data = ser.read(1024)
        if b"START\r\n" in data:
            print("Recording started...")
            ser.reset_input_buffer()
            break
            
    TOTAL_BYTES = int(SAMPLE_RATE * RECORD_SECONDS * 2)
    print(f"Collecting {TOTAL_BYTES} bytes...")
    
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
        
    print(f"Saved: {filename}")
    time.sleep(0.5)
    ser.reset_input_buffer()
    print("Ready for next distance.")

ser.close()
print("\nAll recordings complete.")
