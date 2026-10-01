# Geometric MIDI Stage

A music visualizer: MIDI notes travel along geometric paths, timed to a real
recording. Kenneth builds videos with it — a recording plus an aligned MIDI,
captured through OBS or the built-in recorder, cut in Premiere.

Two halves that meet at one file format:

- **`index.html`** — the whole app. ~5900 lines, single file, no build step.
- **`pipeline/*.py`** — offline tools that produce what the app eats: an
  aligned MIDI, and an RMS-energy JSON.

Everything else in the repo is data.

---

## Running it

```bash
./"Start Visualizer.bat"
```

Double-clickable. Serves the folder on `http://localhost:8000/` and opens a
browser once the port answers. The console window it opens **is** the server —
closing it stops it.

It must be served over http, not opened as a file. The dropdowns that list
`midi/aligned/`, `energy/`, `presets/` and `audio/` work by parsing the
server's directory listing, and none of them exist on `file://`.

`legend.html` is a second, standalone page served from the same folder — an
instrument-key builder. It shares no code with `index.html` on purpose; it only
mirrors the shape vocabulary.

---

## Repo layout

```
index.html              the app
legend.html             standalone instrument-key / legend builder
pipeline/
  align_midi.py         MIDI -> recording alignment (the important one)
  check_alignment.py    measures an alignment in sliding windows
  extract_rms.py        RMS envelope -> energy/<name>_energy.json
  fix_wav_headers.py    repairs streamed-WAV headers (see Traps)
audio/                  source recordings          (git-ignored, large)
energy/                 RMS JSON                   (git-ignored, derived)
midi/source/            original transcriptions    (tracked)
midi/aligned/           pipeline output            (tracked)
presets/                saved looks                (tracked)
vendor/                 tone.js, midi.js, tailwindcss.js — vendored to run offline
scratch/                one-off repair scripts     (git-ignored)
```

`midi/aligned/` is what the app loads. `midi/source/` is only an input to the
pipeline — renaming things there affects nothing in the app.

**Third-party source files get neutral names.** A transcription downloaded with
a credit in its filename (`..._(c)harfesoft.mid`, `... [MIDIfind.com].mid`) is
renamed to describe the music instead. The repo is public; there is no reason
for it to advertise someone else's copyright notice.

---

## The app

### cfg is the whole state

One plain object, `cfg`, holds every setting. The render loop re-reads it every
frame and nothing caches it, so **mutating `cfg.anything` takes effect on the
next frame** with no plumbing. Presets are just `cfg` serialised.

Every control carries a `data-cfg` path — `shape.bolt.lenScale`,
`lane.dna.0.color`, or a bare global — which doubles as a stable address for
that setting. `writeConfigToDOM()` pushes `cfg` back onto the panel wholesale.

Loading a preset is:

```js
cfg = repairConfig(Object.assign(defaultConfig(), parsed));
writeConfigToDOM();
```

`repairConfig()` backfills keys added since a preset was saved, so old presets
keep working. **When you add a per-shape or per-lane setting, give it a default
in `defaultConfig()`** and old presets inherit it silently.

> **Trap:** `defaultConfig()` is *not* what seeds `cfg` at boot. The control
> binding runs `applyControl(node)` on startup, which reads each control's DOM
> `value=` attribute into `cfg`. Change a default in the function and you must
> change the matching `value=` in the markup, or a fresh browser gets the old
> one. This bit once already.

### The shape registry

`SHAPES` is an array of objects, each with a `place(t, laneIdx, geo)` that maps
a note's pitch-derived parameter `t` (0..1) to `{x, y, angle}`. Everything else
— the blueprint trace, tapers, reverb echoes, mirroring — is derived from
`place()`. Adding a shape is mostly one object in that array.

The panel is *generated* from `SHAPES`, so a shape that declares a property
gets its control for free. Declaring `twistSpeed` gives you the TWIST SPEED
slider with no UI work.

### Render pipeline, in order

```
fadeTrail()                 decay the persistent trail buffer
renderShapes(t, dt)         note geometry -> instance buffer
flushInstances(trailRT)     one instanced draw call into the trail target
--- screen ---
background   (color | gradient | image | chroma | transparent)
blueprint    (cached canvas -> texture)
trail        blit, additive OR through the halo shader
particles, EQ / voice / bass / sub-waveform -> screen or fxRT
```

Every glyph, glow and wisp is one instanced quad shaded by a signed-distance
function — a single draw call for the lot. `push()` appends one instance.

Three Canvas2D layers are rasterised once and uploaded as textures rather than
retraced per frame: **blueprint** (`bpDirty`), **background** (`bgDirty`),
**path preview** (`previewDirty`). If you change something those depend on, set
the flag — `invalidateBlueprint()` exists for this.

### Compositing — why the halo pass exists

The note layer is blitted **additively**. Adding light to an already-bright
pixel does nothing, so over a photograph the notes vanish into the bright parts
and no colour choice fixes it.

`HALO_FS` widens the note layer's own alpha by a few pixels and uses that as a
mask for how much *background to remove*, producing a dark edge that hugs the
notes and leaves the rest of the image untouched. `NOTE HALO` at 0 skips the
shader entirely and takes the original additive path, so it is exactly
backwards compatible.

Particles and the bass/EQ visualisers draw after the note layer, so they get
the same treatment through their own render target (`fxRT`).

### Scene Timeline

A scene is a whole `cfg` pinned to a timestamp, with morphing between scenes.

**The rule that shapes it: the look at time `t` is computed from `t` every
frame, never accumulated by firing cues as the playhead sweeps past.** Kenneth
scrubs constantly and re-records takes; anything stateful would desync the
moment he dragged backwards. `sceneStateAt(t)` is pure.

Numbers lerp, `#rrggbb` lerps through RGB, everything else steps at the
halfway point. `SCENE_SNAP` forces specific keys to step anyway — counts, and
anything that repaints a cached canvas. `SCENE_SKIP` holds settings a scene has
no business touching (volume, visual offset, frame size).

### Chords

`chordGroups()` groups notes by **attack time within a window**, not by
overlap. A held bass note under a moving line is polyphony but is not a double
stop, and counting it would leave chord effects on permanently. Grouping scans
all lanes of a shape, so chord tones arriving on separate MIDI tracks still
group if they route into the same path.

---

## The alignment pipeline

`align_midi_to_audio()` in `pipeline/align_midi.py` is the entry point. **Drive
it from Python.** Kenneth does not run it himself and does not want a CLI
invocation as the answer — he asks for a piece to be aligned and expects a
verified file in `midi/aligned/`.

```python
import sys; sys.path.insert(0, 'pipeline')
from align_midi import align_midi_to_audio
align_midi_to_audio(audio_path, midi_path, out_path,
                    start_time=0.0, clip_duration=None,
                    hop_length=1024, refine_hop=128, extend_tail=False)
```

### How it works

DTW over **chroma plus a normalised onset-strength row** (`onset_weight`,
default 0.8). Chroma alone says *what* is sounding, not *when* it started, so a
passage holding one harmony gives the path no gradient and it drifts there.
MIDI features are built analytically from the piano roll and note-start
impulses, not from a rendered sine — a synthesised attack barely registers in
an onset detector.

Then a **second pass**: `_refine_path()` re-matches chunk by chunk at
`refine_hop`, each chunk subsequence-matched inside only the audio window the
coarse pass found. A full fine-hop matrix over a whole movement would be
several GB; these are tens of MB.

`subseq=True` engages automatically when the MIDI is under **85%** of the
audio length — and that rule misfires when the gap is tempo rather than
missing music. The Chaconne's transcription is notated at 60 bpm against a
performance 22% slower, which tripped it at 82%; a subsequence match is free
to start anywhere, so it hung the first note 25 s into a recording that
starts playing at 0.37 s. **Pass `subseq=False` for a complete movement**
whenever the MIDI is merely slower than notated. `method='linear'` (constant scale + offset) is for grid-locked
recordings where chroma DTW has nothing to lock onto.

### Known tunable debt

`_refine_path`'s **`pad_s` defaults to 1.5 and is not exposed** by
`align_midi_to_audio`. On a long movement the coarse path can legitimately
wander further than that and the refine pass rejects almost every chunk — on
the Violin Sonata it rejected 34 of 37 until monkey-patched to 3.0, which took
error from 23.7 ms to 13.7 ms. **Make it a real parameter** next time that file
is open.

---

## Verifying an alignment

This is the part that matters, and it is where the hours went. Good-looking
averages routinely hide broken alignments.

### Use more than one metric, and know what each is blind to

**1. Onset-peak matching.** Match each aligned note start to the nearest
detected audio attack; report median |error| and % within 45 ms.

> Blind to whole-note slips. If the whole passage is shifted by exactly one
> note, every note still lands on *some* attack and the metric looks perfect.
> On dense music it is nearly uninformative — the Partita has 1812 detected
> peaks for 1082 notes, so almost anything scores well.

**2. Chroma / pitch agreement.** Compare the MIDI's sounding pitch classes to
the audio chroma at the mapped time. **Always sanity-check it against a shifted
baseline** — real agreement collapses when you shift ±0.5 s; if it doesn't, the
metric isn't discriminating.

> Blind where harmony moves slowly. Two bars of one chord agree with
> themselves anywhere inside those bars. Chroma also *lags* attacks by
> construction (the CQT window has to fill), so a peak at +60 ms does not
> mean the MIDI is 60 ms early — check signed onset error for that.

**3. Spectral evidence for a specific note.** The tiebreaker. Pull the CQT bin
for the note's actual pitch and find where its energy rises. This settled the
Violin Sonata's missing F6 — energy peaked at 99.919 s against a repair that
placed it at 99.910, while both aggregate metrics preferred the broken version.

> **Harmonics contaminate upward.** A pitch's band is fed by every note a
> fifth, an octave or a twelfth below it that is still ringing. Reading the
> Partita's chord, A5 appeared to enter 93 ms *before* the A3 under it —
> impossible on a violin — because A5 is the third harmonic of the D4 that
> precedes the chord. **Two independent instances returning the same number
> is the tell:** a real performance detail varies between takes, an artifact
> does not. Probe the *lowest* tone of a chord, which has nothing beneath it,
> and check what was sounding immediately before.

**4. Lower the onset threshold in quiet passages.** The default
`delta=0.08` finds *nothing* in a soft stretch, which means the aligner had
nothing to hold onto there either — that is where it drifts. The Partita's
post-rest note was 519 ms early for exactly this reason, and the diagnostic
was that a `delta=0.02` pass found 29 attacks where the default found zero.

### Always check structure separately from error

Average error says nothing about whether the picture is watchable. Compute, per
adjacent pair of source onsets, the local ratio `Δaligned / Δsource`:

- **ratio < ~0.3 — a collapse.** Several notes piled on one instant. Reads as a
  flash and a gap. **Collapses make error metrics look *better***, because
  several notes stack onto one correct attack.
- **ratio > ~4 — a stretch.**
- **gaps > 2 s between note starts** — but check whether a long note is still
  sounding before calling it a hole. A held cadence chord is not a hole, and
  the audio being loud there is that chord ringing, not a misalignment.

Also always: note counts source vs aligned, pitch-for-pitch match, and
monotonicity.

### Cross-validate before accepting a repair

Fitting anchors to audio attacks and then measuring error *at those anchors*
reports whatever you want. Hold out ~20% of notes, fit on the rest, score only
the held-out ones. This rejected a full-window re-derivation on the Sonata
(held-out error got *worse*, 15 ms → 23 ms) and a global attack-snap on Op.130
that looked like 2.9 ms but degraded untouched notes.

### Repair surgically

The right fix is usually two or three anchors, not a re-derivation. Keep every
audio-confirmed anchor, re-interpolate only the broken span between confirmed
neighbours, and re-verify. The Sonata's fix moved **18 notes out of 7541**.

### A fix has to be perceptible to be a fix

Under about 100 ms nothing reads on screen — 30 ms is two frames at 60 fps.
A first attempt at the Partita's chord staggered it by 30 ms per string, which
was defensible musically and completely invisible. When he says he sees no
change, check the size of the change before re-examining the reasoning.

Domain fact worth keeping: **a violin cannot sound four strings at once.** A
four-note chord is played as two pairs — the lower pair on the beat, left to
decay, then the upper pair entering separately and sustaining. In this
recording that gap is 580–650 ms, not the few tens of milliseconds a "roll"
implies. Both of the Partita's four-note chords do it, and they agree.

### Source transcriptions drop notes

Compare audio attacks to MIDI onsets per window. The Partita runs 1473
attacks to 1082 onsets overall, but 1.66 attacks per note in one window
against 1.16–1.32 in the well-transcribed stretches — a 32nd-note group
flattened into even 16ths. The aligner then has to stretch three notes across
nine attacks, which reads as the run dragging. **No re-timing fixes a missing
note**, so diagnose this before trying to repair timing.

### Kenneth's ear beats the metrics

He has been right every time he said something looked wrong, including when
every number disagreed. On Op.130 he insisted the opening was late; onset
metrics said centred; chroma agreement found a real −0.16 to −0.58 s lateness.
**Treat a report of visible drift as ground truth and go find the measurement
that shows it.** His timestamps are accurate to the second and worth taking
literally — "very soon after 1:13" was the 73.14 s entry of a chord's upper
pair, which no aggregate metric had flagged.

---

## House style

The app's defaults *are* Kenneth's style, derived from what all 35 presets
settle on. Every settled preset across seven pieces uses the same values:

| | house | original |
|---|---|---|
| TRAIL | 0 | 0.55 |
| LENGTH | 3.75 | 2 |
| THICKNESS | 0.5 | 2 |
| GLOW | 0.45 | 1 |
| BLUEPRINT | off | on |

The originals live in `FACTORY_LOOK` behind the **Original defaults** button.

Beyond those numbers:

- **Effects stay near zero.** Across 195 routed paths: reverb 0 on 154,
  fireworks 0 on 148, petals 0 on 180, vibrato 0 on 185. Bloom appears at
  0.05–0.06 when it appears at all. EQ, voice circle, bass orbit and
  sub-waveform are on in **zero** presets.
- He dislikes stage-wide audio-reactive swells. Effects should be scoped to a
  voice, which is why `AUDIO ENERGY → APPLIES TO` exists.
- Global LENGTH is set high and trimmed per path with LEN SCALE (0.1–0.35 on
  over half of routed paths), not tuned directly.
- 2–9 paths per preset, median 5. Favourites: DNA, Purple Arcs, Ladders,
  Spiral, Clouds, Spiral Staircase. Green Center Ring has never been used.
- **Seven of those eight are columns or centred rosettes.** Sine was the only
  path crossing the stage, which is why he kept reaching for it. Standing
  Wave, Harmonograph, Swag and Ridges were added to fill that gap, and all
  four — plus Sine — steer their motion from the existing TWIST SPEED rather
  than a new control, because declaring `twistSpeed` is what makes the panel
  generator emit that slider.
- Synth muted in 31 of 35 — he always scores to a real recording.
- New features default to **off**, so no existing preset changes appearance.

---

## Traps

Each of these cost real time.

**pretty_midi tempo maps.** Copying `_tick_scales` onto a new object with
shifted times corrupts written durations. Either mutate the original in place
or start from a fresh `PrettyMIDI()`. Symptom: a file that reports a wildly
wrong end time.

**Float-precision anchor boundaries.** `if A0 <= note.start` misses notes whose
stored float sits a hair below the literal. Use an epsilon when sweeping a
region, or boundary notes silently keep stale times.

**MP3 loses attacks, not pitches.** Measured on 30 s of solo violin: an MP3
round trip leaves chroma essentially untouched (0.9996 frame similarity) and
introduces *no* decode offset — librosa and the browser's `decodeAudioData`
both return the same sample count with zero lag — but it drops **3–8% of
detected onsets at every threshold**. That is the onset-strength row the DTW
stacks on chroma, and it is also what the verification method leans on when
it lowers `delta` to find attacks in quiet passages. Prefer WAV (or FLAC,
which is identical and half the size). An MP3 is still worth aligning if it
is the only source; converting one to WAV gains nothing.

**Streamed WAV headers.** ffmpeg/yt-dlp WAVs land with RIFF/data size
`0xFFFFFFFF`. librosa copes; the browser's `decodeAudioData` takes it literally
and the backing track cuts out early. Run `pipeline/fix_wav_headers.py` on any
new recording.

**`check_alignment.py` can cry wolf.** Its Viterbi lag-picking beat-slips at
low confidence. It once reported 337 ms median / 0% within 50 ms on an
alignment that was fine. Confirm against per-note measurement before acting.

**Silent windows flatter the stats.** `check_alignment.py` skips windows where
one side is silent, so an 89-second hole scored well on Mozart. Always check
note distribution and gaps too.

**Sizing a lateral path off `g.H`.** Amplitude as a fraction of stage
*height* is fine for a column and wrong for anything that crosses the stage:
in a 9:16 frame the path swings further than the stage is wide and reads as a
vertical zigzag. Use `Math.min(g.W, g.H)`. Anything that rotates needs two
more guards — cap the drop against the width, and floor the pivot against the
lift the swing itself produces — or it leaves the top of a 21:9 stage and the
sides of a tall one. Check every new geometry at 16:9, 9:16, 1:1 and 21:9
before calling it done; `place()` is pure, so sampling it over a few seconds
of motion in each frame is a dozen lines and catches all of this.

**`populateTrackSelects()` round-robins.** When nothing is routed it assigns
every path a track. Sensible on a fresh MIDI load, surprising after Reset,
which is why Reset leaves all 21 paths wired.

---

## Conventions

- **No build step, no new dependencies.** Everything vendored under `vendor/`
  so it runs offline. Keep it that way.
- **Comments explain why, not what.** The existing ones record the reasoning
  and the rejected alternative — that is the house voice, match it. Explain the
  trap you avoided, not the syntax.
- **Verify in the browser, not by reasoning.** Serve it, drive it, read values
  back. Measure numerically where possible: exact midpoints on a morph, pixel
  alpha for transparency, instance counts for an effect firing.
- Only code is committed by default; `scratch/` and `__pycache__/` are ignored.
