# Geometric MIDI Stage

A WebGL2 music visualizer that maps MIDI tracks onto geometric paths and renders
note events, trails, particles, and audio-reactive shapes for video capture.

## Run

Open `index.html` in a browser (Chrome/Edge — needs WebGL2). No server required;
all dependencies are vendored under `vendor/`.

To avoid `file://` quirks — and to get folder browsing, see below — serve the
folder instead:

```bash
python -m http.server 8000
```

then visit http://localhost:8000/

### Folder browsing

When served over http, the panel adds a dropdown under each of the MIDI,
Energy JSON, and Preset pickers that lists files straight out of
`midi/aligned/`, `energy/`, and `presets/` — pick one instead of hunting for
it in the OS file dialog. It works by reading the server's directory-listing
page (which `python -m http.server` provides for free), so it's not available
over plain `file://` — those dropdowns just say so and the manual file
pickers above them still work either way.

## Layout

| Path | What |
|------|------|
| `index.html` | The whole visualizer — UI, WebGL2 renderer, MIDI/audio, capture |
| `vendor/` | Local copies of Tailwind (Play CDN), Tone.js 14.8.49, @tonejs/midi 2.0.28 |
| `presets/` | Exported control-panel presets (`.json`) |
| `pipeline/align_midi.py` | librosa chroma + DTW to align a source MIDI to a recording |
| `pipeline/extract_rms.py` | RMS-energy extractor → `[{t,e}]` JSON for the ENERGY macro |
| `pipeline/fix_wav_headers.py` | repair `audio/` WAVs with a streamed (unknown-length) header — the browser can't play those as a Backing Track |
| `audio/` | Source recordings (`.wav`, `.mp4`) — git-ignored, local only |
| `midi/source/` | Original / unaligned MIDIs |
| `midi/aligned/` | DTW-aligned MIDIs (output of `align_midi.py`) |
| `energy/` | `*_energy.json` RMS envelopes (output of `extract_rms.py`) — git-ignored |
| `assets/` | Misc images |

The pipeline scripts resolve `audio/`, `midi/`, and `energy/` relative to the
repo root, so they can be run from anywhere:

```bash
# align a MIDI — edit the three file names in the __main__ block first
python pipeline/align_midi.py

# extract RMS energy — arg is a case-insensitive name fragment of a file in audio/
python pipeline/extract_rms.py "Brandenburg 4"
```

If a recording came out of ffmpeg/yt-dlp it may have a streamed WAV header
(`RIFF`/`data` size = `0xFFFFFFFF`). Python reads it fine but the browser's
`decodeAudioData` doesn't, so the Backing Track cuts in and out. Fix in place:

```bash
python pipeline/fix_wav_headers.py            # scan + repair audio/
python pipeline/fix_wav_headers.py --check    # report only
```

## Workflow

1. Drop the recording in `audio/` and the score in `midi/source/`.
2. DTW-align the MIDI to the recording (`pipeline/align_midi.py`) → `midi/aligned/`.
3. Extract RMS energy from the recording (`pipeline/extract_rms.py`) → `energy/`.
4. In the visualizer: load the aligned MIDI, the energy JSON, and — under **Audio Mixer**
   — the recording itself as the **Backing Track**. Route tracks onto shapes and
   tune the panel. The aligned MIDI drives the geometry; the recording is the
   transport audio and is included in Record captures. Use **Visual Offset**
   (Capture & Session) to fine-tune picture vs. sound, and **Backing Delay**
   (Audio Mixer) to slide the recording itself. By default the RMS energy reacts
   stage-wide; set **AUDIO ENERGY → APPLIES TO** to a single path to scope it.
5. Pick a **Frame / Aspect** (Capture & Session) — Fill window, or a fixed
   9:16 / 4:5 / 1:1 / 16:9 stage for social video. Then capture (in-app
   **Record**, or the pre-roll countdown into OBS). Fullscreen a large
   monitor before recording a vertical frame so it renders at a useful size.

The older path — render with the synth as a scratch track and lay the recording
over the video in an editor using the one-frame sync flash — still works if you
leave the Backing Track empty.

## Dependencies

Vendored, so the app works offline. To update:

```bash
curl -sL -o vendor/tone.js        https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js
curl -sL -o vendor/midi.js        https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.28/build/Midi.js
curl -sL -o vendor/tailwindcss.js https://cdn.tailwindcss.com
```
