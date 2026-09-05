import os

import librosa
import pretty_midi
import numpy as np
from scipy.interpolate import interp1d

# DTW-align a source MIDI to a real recording so its note onsets line up with
# the performance, then write the humanised MIDI out for the visualizer.
#
# Layout this expects (paths below are relative to the repo root, so run it
# from anywhere):
#     audio/<recording>.wav        the performance to match
#     midi/source/<score>.mid      the unaligned MIDI
#     midi/aligned/<name>.mid      <- written here
# Edit the three names in the __main__ block and run:  python pipeline/align_midi.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
MIDI_SRC_DIR = os.path.join(ROOT, "midi", "source")
MIDI_OUT_DIR = os.path.join(ROOT, "midi", "aligned")

# Some MIDIs carry a malformed SMPTE-offset meta event (sub-frame field > 99)
# that makes mido — and therefore pretty_midi — refuse to open the file. That
# field is purely informational for our purposes, so relax mido's range check.
import mido.midifiles.meta as _mido_meta
_mido_meta.check_int = lambda value, low, high: None


def align_midi_to_audio(audio_path, midi_path, output_path, start_time=0.0,
                        clip_duration=None, hop_length=None):
    print("Loading and clipping audio...")
    y_audio, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=clip_duration)

    # DTW cost is O(frames_midi * frames_audio); 512 (~23 ms/frame) is fine for
    # a few minutes but a full slow movement (10-15 min) would need a ~30k x 30k
    # matrix and run out of memory. Scale the hop up with the piece length so
    # the frame count — and the matrix — stays manageable.
    dur = librosa.get_duration(y=y_audio, sr=sr)
    if hop_length is None:
        hop_length = 512 if dur < 360 else 1024 if dur < 720 else 2048
    print(f"  {dur:.0f}s audio, hop_length={hop_length} (~{1000 * hop_length / sr:.0f} ms/frame)")

    chroma_audio = librosa.feature.chroma_cqt(y=y_audio, sr=sr, hop_length=hop_length)

    print("Loading MIDI and synthesizing reference audio...")
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    y_midi = midi_data.synthesize(fs=sr)
    chroma_midi = librosa.feature.chroma_cqt(y=y_midi, sr=sr, hop_length=hop_length)

    print("Running Dynamic Time Warping (DTW)...")
    # Constrain the warp to a diagonal band: a recording and a MIDI of the same
    # piece never drift more than a fraction of the total length apart, and the
    # band keeps the cost matrix cheap for long movements.
    D, wp = librosa.sequence.dtw(X=chroma_midi, Y=chroma_audio, metric='cosine',
                                 global_constraints=True, band_rad=0.2)

    wp = wp[::-1]

    time_midi = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop_length)
    time_audio = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop_length)

    print("Calculating warp function...")
    _, unique_indices = np.unique(time_midi, return_index=True)
    time_midi_unique = time_midi[unique_indices]
    time_audio_unique = time_audio[unique_indices]

    # Force the mapping to be non-decreasing. After de-duplicating on MIDI time
    # the corresponding audio times can still step backwards a little, which
    # would make interp1d hand back note.end < note.start and produce stuck /
    # zero-length notes in the aligned file.
    time_audio_unique = np.maximum.accumulate(time_audio_unique)

    warp_func = interp1d(time_midi_unique, time_audio_unique, kind='linear', fill_value='extrapolate')

    print("Warping MIDI note data...")
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            s = float(warp_func(note.start))
            e = float(warp_func(note.end))
            note.start = s
            note.end = max(e, s + 0.02)   # never let a note collapse to nothing

    print(f"Saving humanized MIDI to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_data.write(output_path)


# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    align_midi_to_audio(
        audio_path=os.path.join(AUDIO_DIR, 'Beethoven, String Quartet op.130 4thMVMT.wav'),
        midi_path=os.path.join(MIDI_SRC_DIR, 'quartet_13_4_(c)edwards (2).mid'),
        output_path=os.path.join(MIDI_OUT_DIR, 'Beethoven13m4aligned.mid'),
        start_time=0.0,
        clip_duration=None
    )
