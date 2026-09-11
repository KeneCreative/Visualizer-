"""Measure how well an aligned MIDI actually lines up with its recording.

Mean DTW path cost says how well the *features* matched; it says nothing about
whether a listener hears the notes land on the beat. This measures that
directly, and — more usefully — says WHERE it drifts.

Method: build an onset-strength envelope for the recording and a matching
impulse envelope from the aligned MIDI's note starts, then slide one against
the other inside short windows. The lag that maximises the correlation in a
window is how far the MIDI is off there (negative = MIDI is early).

    python pipeline/check_alignment.py <audio-fragment> <aligned-midi-fragment>

Both arguments are case-insensitive substrings matched against audio/ and
midi/aligned/.
"""
import os
import sys

import librosa
import numpy as np
import pretty_midi
import mido.midifiles.meta as _mido_meta
_mido_meta.check_int = lambda value, low, high: None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
MIDI_OUT_DIR = os.path.join(ROOT, "midi", "aligned")

SR = 22050
# 11.6 ms per frame — the resolution of this measurement. Override with
# CHECK_HOP=128 for a finer (noisier) read when chasing the last few ms.
HOP = int(os.environ.get("CHECK_HOP", 256))
WIN_S = 12.0               # window length for a local lag estimate
STEP_S = 4.0               # window hop
MAX_LAG_S = 0.6            # how far either way we look for the best match
JUMP_PENALTY = 2.0         # cost per second of lag change between neighbouring windows


def _pick(folder, query, exts):
    pool = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)
    hits = [f for f in pool if query.lower() in f.lower()] if query else []
    if len(hits) != 1:
        print(f"In {folder}:")
        for f in pool:
            print(f"  {f}")
        raise SystemExit(f"\n'{query}' matched {len(hits)} — be more specific.")
    return os.path.join(folder, hits[0])


def midi_onset_envelope(midi, n_frames):
    """Impulse train at note starts, smoothed to the width of a real attack.
    Simultaneous notes stack, so a full chord counts for more than one voice."""
    env = np.zeros(n_frames)
    for inst in midi.instruments:
        for n in inst.notes:
            f = int(round(n.start * SR / HOP))
            if 0 <= f < n_frames:
                env[f] += (n.velocity or 64) / 127.0
    # ~25 ms Gaussian: wide enough to tolerate quantisation, narrow enough to
    # keep the correlation peak sharp
    sigma = 0.025 * SR / HOP
    k = int(sigma * 4) | 1
    g = np.exp(-0.5 * ((np.arange(k) - k // 2) / sigma) ** 2)
    return np.convolve(env, g / g.sum(), mode='same')


def local_lags(env_a, env_m, sr_f):
    """Best lag (seconds) per window. Positive = MIDI is LATE vs the audio.

    A plain per-window argmax is unreliable on metrical music: the onset
    pattern repeats every beat, so the correlation has near-equal peaks one
    beat either side and isolated windows flip to a neighbouring beat. So we
    score EVERY candidate lag per window and then pick the path through them
    with a Viterbi pass that charges for changing lag between windows — real
    drift moves smoothly, a beat-slip doesn't."""
    win = int(WIN_S * sr_f)
    step = int(STEP_S * sr_f)
    max_lag = int(MAX_LAG_S * sr_f)
    times, curves = [], []
    for start in range(0, min(len(env_a), len(env_m)) - win, step):
        a = env_a[start:start + win]
        m = env_m[start:start + win]
        if m.sum() < 1e-6 or a.sum() < 1e-6:
            continue                       # silence in one of them — nothing to compare
        a = (a - a.mean()) / (a.std() + 1e-9)
        m = (m - m.mean()) / (m.std() + 1e-9)
        cc = np.correlate(a, m, mode='full')
        mid = len(cc) // 2
        times.append(start / sr_f)
        curves.append(cc[mid - max_lag: mid + max_lag + 1] / win)
    if not curves:
        return []

    lags_s = (np.arange(-max_lag, max_lag + 1)) / sr_f
    # Viterbi over lag candidates, maximising correlation minus jump cost
    score = curves[0].copy()
    back = []
    for c in curves[1:]:
        trans = score[:, None] - JUMP_PENALTY * np.abs(lags_s[:, None] - lags_s[None, :])
        prev = np.argmax(trans, axis=0)
        score = trans[prev, np.arange(len(lags_s))] + c
        back.append(prev)
    idx = [int(np.argmax(score))]
    for prev in reversed(back):
        idx.append(int(prev[idx[-1]]))
    idx.reverse()

    return [(t, float(lags_s[i]), float(c[i]), float(c.max()))
            for t, c, i in zip(times, curves, idx)]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    audio_path = _pick(AUDIO_DIR, sys.argv[1], (".wav", ".mp3", ".flac", ".m4a", ".ogg"))
    midi_path = _pick(MIDI_OUT_DIR, sys.argv[2], (".mid", ".midi"))
    print(f"audio: {os.path.basename(audio_path)}")
    print(f"midi : {os.path.basename(midi_path)}\n")

    y, _ = librosa.load(audio_path, sr=SR)
    env_a = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    midi = pretty_midi.PrettyMIDI(midi_path)
    env_m = midi_onset_envelope(midi, len(env_a))

    sr_f = SR / HOP
    rows = local_lags(env_a, env_m, sr_f)
    if not rows:
        raise SystemExit("No comparable windows — is the MIDI empty?")

    lags = np.array([r[1] for r in rows])
    print(f"{len(rows)} windows of {WIN_S:.0f}s")
    print(f"  median |offset|  {np.median(np.abs(lags)) * 1000:6.0f} ms")
    print(f"  90th pct |offset|{np.percentile(np.abs(lags), 90) * 1000:6.0f} ms")
    print(f"  worst            {np.abs(lags).max() * 1000:6.0f} ms")
    print(f"  within 50 ms     {100 * np.mean(np.abs(lags) < 0.050):5.0f} %")
    print(f"  mean signed      {lags.mean() * 1000:+6.0f} ms  (a constant bias -> use Visual Offset)\n")

    print("  time   offset   conf   (| = 0, one char = 20 ms, + = MIDI late)")
    for t, lag, peak, best in rows:
        n = int(round(lag * 1000 / 20))
        n = max(-24, min(24, n))
        bar = ('.' * (24 + n) + '|' + ' ' * (24 - n)) if n < 0 else (' ' * 24 + '|' + '.' * n)
        flag = '  <-- off' if abs(lag) > 0.08 else ''
        print(f"  {t:5.0f}s {lag * 1000:+5.0f}ms  {peak:4.2f}  {bar}{flag}")


if __name__ == "__main__":
    main()
