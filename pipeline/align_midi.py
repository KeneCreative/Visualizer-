import os

import librosa
import pretty_midi
import numpy as np
from scipy.interpolate import interp1d

# Align a source MIDI to a real recording so its note onsets line up with the
# performance, then write the result out for the visualizer.
#
#   method='dtw'    (default) chroma Dynamic Time Warping — for a live
#                   performance with rubato / tempo drift, e.g. a string quartet.
#                   Needs constantly-moving harmony to lock onto.
#   method='linear' constant tempo scale + offset — for a metronomic recording
#                   (drum-machine hip-hop, anything cut to a grid). DTW has
#                   nothing to grip on a static harmonic loop and just mangles
#                   it; a 1% stretch is all these need.
#
# Layout this expects (paths below are relative to the repo root, so run it
# from anywhere):
#     audio/<recording>.wav        the performance to match
#     midi/source/<score>.mid      the unaligned MIDI
#     midi/aligned/<name>.mid      <- written here
# Edit the call in the __main__ block and run:  python pipeline/align_midi.py

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
                        clip_duration=None, hop_length=None, method='dtw',
                        midi_offset=0.0, subseq=None, transpose='auto'):
    print("Loading and clipping audio...")
    y_audio, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=clip_duration)
    dur = librosa.get_duration(y=y_audio, sr=sr)

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    midi_dur = midi_data.get_end_time()

    if method == 'linear':
        warp_func = _linear_warp(y_audio, sr, dur, midi_data, midi_offset)
        _apply_warp(midi_data, warp_func, output_path)
        return

    # DTW cost is O(frames_midi * frames_audio); 512 (~23 ms/frame) is fine for
    # a few minutes but a full slow movement (10-15 min) would need a ~30k x 30k
    # matrix and run out of memory. Scale the hop up with the piece length so
    # the frame count — and the matrix — stays manageable.
    if hop_length is None:
        hop_length = 512 if dur < 360 else 1024 if dur < 720 else 2048
    print(f"  {dur:.0f}s audio, hop_length={hop_length} (~{1000 * hop_length / sr:.0f} ms/frame)")

    print("Synthesizing MIDI reference audio...")
    # Build the reference from pitched instruments only — a synthesised drum kit
    # is just broadband noise in a chroma and drags the alignment around.
    ref = pretty_midi.PrettyMIDI()
    ref.instruments = [i for i in midi_data.instruments if not i.is_drum] or midi_data.instruments
    y_midi = ref.synthesize(fs=sr)

    def chroma(y):
        C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        C = np.nan_to_num(C)
        # a fully-silent frame (a rest in the MIDI, a gap between movements in
        # the recording) is an all-zero column, and cosine distance to that is
        # 0/0 = NaN, which blows up the DTW cost matrix. Make silence a flat
        # non-zero vector so it's simply "far from everything" instead.
        C[:, C.sum(axis=0) < 1e-8] = 1e-4
        return C

    chroma_audio = chroma(y_audio)
    chroma_midi = chroma(y_midi)

    # A pitched-up rip or a period-instrument recording at a different concert
    # pitch sits a semitone or two off the MIDI, and chroma DTW then matches
    # nothing (it's key-relative). Detect the shift from the mean chroma
    # profiles and rotate the reference to match — the notes written out keep
    # their original pitch, only the matching is transposed.
    if transpose == 'auto':
        pa = chroma_audio.mean(axis=1)
        pm = chroma_midi.mean(axis=1)
        corr = [float(np.corrcoef(pa, np.roll(pm, k))[0, 1]) for k in range(12)]
        transpose = int(np.argmax(corr))
        if transpose:
            up = transpose if transpose <= 6 else transpose - 12
            print(f"  recording sits {up:+d} semitone(s) from the MIDI "
                  f"(profile corr {corr[transpose]:.2f} vs {corr[0]:.2f} at unison) — "
                  f"matching transposed")
    if transpose:
        chroma_midi = np.roll(chroma_midi, transpose, axis=0)

    # When the MIDI only covers part of the recording (a loop, or a cover that
    # skips the outro), a plain DTW smears it across the whole thing. Subsequence
    # DTW instead finds WHERE that chunk sits. Auto when the MIDI is clearly
    # shorter; pass subseq=True/False to force it either way.
    if subseq is None:
        subseq = midi_dur < 0.85 * dur

    print("Running Dynamic Time Warping (DTW)...")
    if subseq:
        print(f"  MIDI is {midi_dur:.0f}s vs {dur:.0f}s audio -> subsequence match")
        D, wp = librosa.sequence.dtw(X=chroma_midi, Y=chroma_audio, metric='cosine',
                                     subseq=True, backtrack=True)
    else:
        # Constrain the warp to a diagonal band: a recording and a MIDI of the
        # same piece never drift more than a fraction of the total length apart,
        # and the band keeps the cost matrix cheap for long movements.
        D, wp = librosa.sequence.dtw(X=chroma_midi, Y=chroma_audio, metric='cosine',
                                     global_constraints=True, band_rad=0.2)

    end_cost = D[wp[0, 0], wp[0, 1]] / len(wp)   # mean cosine cost along the path

    wp = wp[::-1]

    time_midi = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop_length)
    time_audio = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop_length)

    print("Calculating warp function...")
    _, unique_indices = np.unique(time_midi, return_index=True)
    time_midi_unique = time_midi[unique_indices]
    time_audio_unique = time_audio[unique_indices]

    # The raw DTW path is jittery frame-to-frame — locally it stalls or
    # backtracks, and once that's forced monotonic (below) whole stretches go
    # flat, collapsing every note inside them. Anchor the warp on the median
    # audio time within coarse MIDI-time bins: keeps the global shape, drops
    # the noise. 0.25s bins stay well below any real rubato.
    bin_s = 0.25
    bins = np.round(time_midi_unique / bin_s).astype(np.int64)
    anchor_midi, anchor_audio = [], []
    for b in np.unique(bins):
        sel = bins == b
        anchor_midi.append(float(time_midi_unique[sel].mean()))
        anchor_audio.append(float(np.median(time_audio_unique[sel])))
    anchor_midi = np.asarray(anchor_midi)
    # Force the mapping non-decreasing so interp1d can't hand back
    # note.end < note.start (stuck / zero-length notes).
    anchor_audio = np.maximum.accumulate(np.asarray(anchor_audio))

    print(f"  matched audio {time_audio_unique[0]:.1f}s -> {time_audio_unique[-1]:.1f}s"
          f"   mean path cost {end_cost:.3f} (0 = perfect, ~1 = unrelated)")

    dm = np.diff(anchor_midi)
    da = np.diff(anchor_audio)
    flat_frac = float((da / np.maximum(dm, 1e-6) < 0.15).mean())
    if flat_frac > 0.5:
        # A poor chroma match (thin reduction, very different orchestration)
        # makes the DTW path degenerate into flat runs — whole phrases of MIDI
        # map onto one instant and every note collapses. The path's endpoints
        # are still meaningful; its interior isn't. Straight line between them.
        print(f"  warp is flat over {flat_frac:.0%} of its length (cost "
              f"{end_cost:.2f}) — falling back to a straight endpoint fit")
        anchor_midi, anchor_audio = anchor_midi[[0, -1]], anchor_audio[[0, -1]]
    elif flat_frac > 0.12:
        # Some flat treads — enforce a floor on how much audio time each step
        # covers so notes landing on a tread don't collapse, then rescale to
        # keep the overall span DTW found. Steep parts give back the slack.
        span = anchor_audio[-1] - anchor_audio[0]
        da = np.maximum(da, 0.4 * dm)
        anchor_audio = anchor_audio[0] + np.concatenate([[0], np.cumsum(da)])
        anchor_audio = anchor_audio[0] + (anchor_audio - anchor_audio[0]) * (span / (anchor_audio[-1] - anchor_audio[0]))
        print(f"  {flat_frac:.0%} of the warp was flat — floored the step slope to keep notes from collapsing")

    warp_func = interp1d(anchor_midi, anchor_audio, kind='linear', fill_value='extrapolate')

    _apply_warp(midi_data, warp_func, output_path)


def _linear_warp(y_audio, sr, dur, midi_data, midi_offset):
    """Constant tempo scale + offset. For a grid-locked recording where the
    only difference from the MIDI is a slightly different fixed tempo."""
    audio_tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y_audio, sr=sr)[0])[0])
    midi_tempo = float(midi_data.get_tempo_changes()[1][0])
    scale = midi_tempo / audio_tempo
    print(f"  linear fit: MIDI {midi_tempo:.1f} bpm vs audio {audio_tempo:.1f} bpm "
          f"-> x{scale:.4f}, offset {midi_offset:+.2f}s")
    if abs(scale - 1) > 0.06:
        print("  (large tempo gap — the beat tracker may have picked a half/double "
              "tempo; check the recording's actual bpm)")
    return lambda t: midi_offset + t * scale


def _apply_warp(midi_data, warp_func, output_path):
    print("Warping MIDI note data...")
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            s = float(warp_func(note.start))
            e = float(warp_func(note.end))
            note.start = max(0.0, s)
            note.end = max(e, s + 0.02)   # never let a note collapse to nothing

    print(f"Saving aligned MIDI to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_data.write(output_path)


# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    align_midi_to_audio(
        audio_path=os.path.join(AUDIO_DIR, 'String_Quartet_in_F_Major_Op_135_I_Allegretto.wav'),
        midi_path=os.path.join(MIDI_SRC_DIR, 'quartet_16_1_(c)edwards.mid'),
        output_path=os.path.join(MIDI_OUT_DIR, 'Beethoven135_1_aligned.mid'),
    )
