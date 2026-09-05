import os
import sys
import json
import librosa
import numpy as np

# Reads a recording from  ../audio/  and writes its normalised RMS-energy
# envelope to  ../energy/<name>_energy.json  — the [{t, e}] payload the
# visualizer folds into the ENERGY macro (AUDIO ENERGY -> load JSON).
#
#   python pipeline/extract_rms.py "Violin Concerto"
#
# The argument is matched against audio file names in ../audio/ as a
# case-insensitive substring, so a short fragment is enough. With no
# argument it lists what's available.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
ENERGY_DIR = os.path.join(ROOT, "energy")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

query = sys.argv[1] if len(sys.argv) > 1 else ""

pool = sorted(
    f for f in os.listdir(AUDIO_DIR)
    if os.path.splitext(f)[1].lower() in AUDIO_EXTS
)
matches = [f for f in pool if query.lower() in f.lower()] if query else []

if len(matches) != 1:
    print(f"Audio in {AUDIO_DIR}:")
    for f in pool:
        print(f"  {f}")
    if not query:
        raise SystemExit("\nPass a name fragment to pick one.")
    raise SystemExit(
        f"\n'{query}' matched {len(matches)} of them"
        + (f": {matches}" if matches else "") + " — be more specific."
    )

audio_file = os.path.join(AUDIO_DIR, matches[0])
song_base = os.path.splitext(matches[0])[0]

print(f"Loading: {audio_file}")
y, sr = librosa.load(audio_file, sr=None)

print("Calculating RMS energy...")
rms = librosa.feature.rms(y=y)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

rms_min, rms_max = np.min(rms), np.max(rms)
if rms_max - rms_min > 0:
    rms_normalized = (rms - rms_min) / (rms_max - rms_min)
else:
    rms_normalized = rms

energy_data = [
    {"t": round(float(t), 3), "e": round(float(e), 4)}
    for t, e in zip(times, rms_normalized)
]

os.makedirs(ENERGY_DIR, exist_ok=True)
output_json = os.path.join(ENERGY_DIR, f"{song_base}_energy.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(energy_data, f, indent=2)

print(f"Success! Exported {len(energy_data)} frames to:\n{output_json}")
