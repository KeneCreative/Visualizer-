import os
import sys
import json
import librosa
import numpy as np

# 1. Folder = wherever this script lives (drop the .wav next to it).
#    Optionally pass the song name as a command-line argument.
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
SONG_BASE_NAME = sys.argv[1] if len(sys.argv) > 1 else "Violin Concerto"

# 2. Check common audio extensions
audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
audio_file = None

for ext in audio_extensions:
    candidate = os.path.join(FOLDER_PATH, SONG_BASE_NAME + ext)
    if os.path.exists(candidate):
        audio_file = candidate
        break

if not audio_file:
    have = [f for f in os.listdir(FOLDER_PATH) if os.path.splitext(f)[1].lower() in audio_extensions]
    raise FileNotFoundError(
        f"No '{SONG_BASE_NAME}' audio ({audio_extensions}) in '{FOLDER_PATH}'.\n"
        f"Audio files that ARE there: {have or 'none'}"
    )

print(f"Loading: {audio_file}")
y, sr = librosa.load(audio_file, sr=None)

# 3. Calculate RMS energy
print("Calculating RMS energy...")
rms = librosa.feature.rms(y=y)[0]

# Generate exact timestamps for each frame
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

# 4. Normalize between 0.0 and 1.0
rms_min, rms_max = np.min(rms), np.max(rms)
if rms_max - rms_min > 0:
    rms_normalized = (rms - rms_min) / (rms_max - rms_min)
else:
    rms_normalized = rms

# 5. Format payload and export JSON
energy_data = [
    {"t": round(float(t), 3), "e": round(float(e), 4)}
    for t, e in zip(times, rms_normalized)
]

output_json = os.path.join(FOLDER_PATH, f"{SONG_BASE_NAME}_energy.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(energy_data, f, indent=2)

print(f"Success! Exported {len(energy_data)} frames to:\n{output_json}")