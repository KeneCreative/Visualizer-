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
# The DTW path matches on chroma PLUS an onset-strength row: chroma alone says
# what is sounding but not when it started, so a passage that holds one harmony
# gives the path nothing to lock onto and it drifts there. It then runs a second
# pass at a fine hop, chunk by chunk, inside the window the first pass found —
# a full fine-hop matrix over a whole movement would be several GB.
#
# Check the result with:  python pipeline/check_alignment.py <audio> <aligned>
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
                        midi_offset=0.0, subseq=None, transpose='auto',
                        onset_weight=0.8, refine=True, refine_hop=256):
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

    print("Building features...")
    chroma_audio, onset_audio = _audio_features(y_audio, sr, hop_length)
    chroma_midi, onset_midi = _midi_features(midi_data, sr, hop_length)

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

    # Chroma says WHAT is sounding, not WHEN it started. Through a passage that
    # holds one harmony the chroma is flat, the DTW has no gradient, and the
    # path is free to wander — that is where an otherwise-good alignment drifts.
    # Stacking an onset-strength row underneath gives it a timing cue wherever
    # there is an attack, which is nearly everywhere.
    feat_audio = _stack(chroma_audio, onset_audio, onset_weight)
    feat_midi = _stack(chroma_midi, onset_midi, onset_weight)

    # When the MIDI only covers part of the recording (a loop, or a cover that
    # skips the outro), a plain DTW smears it across the whole thing. Subsequence
    # DTW instead finds WHERE that chunk sits. Auto when the MIDI is clearly
    # shorter; pass subseq=True/False to force it either way.
    if subseq is None:
        subseq = midi_dur < 0.85 * dur

    print("Running Dynamic Time Warping (DTW)...")
    if subseq:
        print(f"  MIDI is {midi_dur:.0f}s vs {dur:.0f}s audio -> subsequence match")
        D, wp = librosa.sequence.dtw(X=feat_midi, Y=feat_audio, metric='cosine',
                                     subseq=True, backtrack=True)
    else:
        # Constrain the warp to a diagonal band: a recording and a MIDI of the
        # same piece never drift more than a fraction of the total length apart,
        # and the band keeps the cost matrix cheap for long movements.
        D, wp = librosa.sequence.dtw(X=feat_midi, Y=feat_audio, metric='cosine',
                                     global_constraints=True, band_rad=0.2)

    end_cost = D[wp[0, 0], wp[0, 1]] / len(wp)   # mean cosine cost along the path

    wp = wp[::-1]

    time_midi = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop_length)
    time_audio = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop_length)

    # Second pass: the coarse hop above caps how precisely any onset can be
    # placed (a 512 hop is 23 ms of unavoidable slop). Re-run the match at a
    # fine hop, chunk by chunk, each chunk searched only in the small audio
    # window the coarse path already pointed at — so the cost matrices stay
    # tiny while the resolution goes up several times over.
    if refine and refine_hop < hop_length:
        time_midi, time_audio = _refine_path(
            y_audio, midi_data, sr, refine_hop, onset_weight, transpose,
            time_midi, time_audio, midi_dur, dur)

    print("Calculating warp function...")
    _, unique_indices = np.unique(time_midi, return_index=True)
    time_midi_unique = time_midi[unique_indices]
    time_audio_unique = time_audio[unique_indices]

    # The raw DTW path is jittery frame-to-frame — locally it stalls or
    # backtracks, and once that's forced monotonic (below) whole stretches go
    # flat, collapsing every note inside them. Anchor the warp on the median
    # audio time within coarse MIDI-time bins: keeps the global shape, drops
    # the noise. 0.25s bins stay well below any real rubato.
    bin_s = 0.10 if (refine and refine_hop < hop_length) else 0.25
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


def _norm_cols(C):
    """Unit-L2 per frame, and give a silent frame a flat non-zero vector —
    cosine distance to an all-zero column is 0/0 = NaN and poisons the DTW."""
    C = np.nan_to_num(C)
    C[:, C.sum(axis=0) < 1e-8] = 1e-4
    return C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)


def _norm_env(e):
    """Onset envelope scaled to roughly 0..1 by a robust max, so one loud
    attack can't flatten the rest of the piece."""
    e = np.nan_to_num(np.asarray(e, dtype=float))
    hi = np.percentile(e, 95) if e.size else 1.0
    return np.clip(e / (hi + 1e-9), 0.0, 1.0)


def _audio_features(y, sr, hop):
    return (_norm_cols(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)),
            _norm_env(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)))


def _midi_features(midi_data, sr, hop):
    """Built straight off the note data rather than from a synthesised render.
    pretty_midi's synth is pure sine tones — its attacks barely register in an
    onset detector, and its CQT response is nothing like a real instrument.
    The piano roll gives an exact chroma and exact attack times instead."""
    fps = sr / hop
    chroma = _norm_cols(midi_data.get_chroma(fs=fps))
    n = chroma.shape[1]
    env = np.zeros(n)
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            f = int(round(note.start * fps))
            if 0 <= f < n:
                env[f] += (note.velocity or 64) / 127.0
    # smear each attack to ~25 ms so it can meet a real (slower) string attack
    sigma = max(1.0, 0.025 * fps)
    k = int(sigma * 4) | 1
    g = np.exp(-0.5 * ((np.arange(k) - k // 2) / sigma) ** 2)
    return chroma, _norm_env(np.convolve(env, g / g.sum(), mode='same'))


def _stack(chroma, onset, weight):
    """13-row feature: unit-norm chroma plus one weighted onset row."""
    n = min(chroma.shape[1], len(onset))
    return np.vstack([chroma[:, :n], weight * onset[np.newaxis, :n]])


def _refine_path(y_audio, midi_data, sr, hop, onset_weight, transpose,
                 coarse_midi_t, coarse_audio_t, midi_dur, audio_dur,
                 chunk_s=24.0, overlap_s=6.0, pad_s=1.5):
    """Re-match at a fine hop, one chunk of MIDI at a time, each searched only
    inside the audio window the coarse path already found (plus a little pad).
    A full fine-hop DTW over a whole movement would need a multi-GB matrix;
    these chunk matrices are a few tens of MB and run in a moment each."""
    print(f"Refining at hop_length={hop} (~{1000 * hop / sr:.0f} ms/frame)...")
    ca, oa = _audio_features(y_audio, sr, hop)
    cm, om = _midi_features(midi_data, sr, hop)
    if transpose:
        cm = np.roll(cm, transpose, axis=0)
    Fa = _stack(ca, oa, onset_weight)
    Fm = _stack(cm, om, onset_weight)
    fps = sr / hop

    coarse = interp1d(coarse_midi_t, coarse_audio_t, kind='linear', fill_value='extrapolate')
    out_m, out_a = [], []
    step = chunk_s - overlap_s
    starts = np.arange(0.0, max(midi_dur - overlap_s, step), step)
    kept = 0
    for t0 in starts:
        t1 = min(t0 + chunk_s, midi_dur)
        mi0, mi1 = int(t0 * fps), min(int(t1 * fps), Fm.shape[1])
        if mi1 - mi0 < int(2 * fps):
            continue
        a0 = float(coarse(t0)) - pad_s
        a1 = float(coarse(t1)) + pad_s
        ai0 = max(0, int(a0 * fps))
        ai1 = min(Fa.shape[1], int(a1 * fps))
        if ai1 - ai0 < (mi1 - mi0):
            continue                      # not enough audio to hold this chunk
        # free start/end on the audio side: the chunk sits somewhere inside the
        # padded window, and we don't know exactly where yet
        _, wp = librosa.sequence.dtw(X=Fm[:, mi0:mi1], Y=Fa[:, ai0:ai1],
                                     metric='cosine', subseq=True, backtrack=True)
        wp = wp[::-1]
        tm = (mi0 + wp[:, 0]) / fps
        ta = (ai0 + wp[:, 1]) / fps
        # keep the middle of the chunk only — the edges of a subsequence match
        # are the least reliable part of it
        lo = t0 if t0 <= 0 else t0 + overlap_s / 2
        hi = t1 if t1 >= midi_dur else t1 - overlap_s / 2
        sel = (tm >= lo) & (tm <= hi)
        if not sel.any():
            continue
        # sanity: a chunk that lands far from where the coarse pass put it has
        # almost certainly locked onto the wrong thing — keep the coarse answer
        drift = float(np.median(ta[sel] - coarse(tm[sel])))
        if abs(drift) > pad_s:
            continue
        out_m.append(tm[sel])
        out_a.append(ta[sel])
        kept += 1

    if not out_m:
        print("  refinement found nothing usable — keeping the coarse path")
        return coarse_midi_t, coarse_audio_t
    tm = np.concatenate(out_m)
    ta = np.concatenate(out_a)
    order = np.argsort(tm, kind='stable')
    print(f"  {kept}/{len(starts)} chunks refined")
    return tm[order], ta[order]


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
        hop_length=512,   # finer than the auto-picked 1024 (dur>360s tier) — cheap enough at
                          # this length (~8s DTW) and tightens onset accuracy through the piece
        refine_hop=128,   # ~6 ms second pass
    )
