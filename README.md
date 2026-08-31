# Geometric MIDI Stage

A WebGL2 music visualizer that maps MIDI tracks onto geometric paths and renders
note events, trails, particles, and audio-reactive shapes for video capture.

## Run

Open `index.html` in a browser (Chrome/Edge — needs WebGL2). No server required;
all dependencies are vendored under `vendor/`.

To avoid `file://` quirks you can also serve the folder:

```bash
python -m http.server 8000
```

then visit http://localhost:8000/

## Layout

| Path | What |
|------|------|
| `index.html` | The whole visualizer — UI, WebGL2 renderer, MIDI/audio, capture |
| `vendor/` | Local copies of Tailwind (Play CDN), Tone.js 14.8.49, @tonejs/midi 2.0.28 |
| `presets/` | Exported control-panel presets (`.json`) |
| `Music SYnc/` | `syncscript.py` — librosa chroma + DTW to align a MIDI to a recording |
| `Music RMS for vizualizer/` | RMS-energy extractor → `[{t,e}]` JSON for the ENERGY macro |

## Workflow

1. DTW-align the MIDI to the target recording (`Music SYnc/syncscript.py`).
2. Extract RMS energy from the recording (`Music RMS for vizualizer/`).
3. In the visualizer: load the MIDI, the energy JSON, and — under **Audio Mixer**
   — the recording itself as the **Backing Track**. Route tracks onto shapes and
   tune the panel. The aligned MIDI drives the geometry; the recording is the
   transport audio and is included in Record captures. Use **Visual Offset**
   (Capture & Session) to fine-tune picture vs. sound, and **Backing Delay**
   (Audio Mixer) to slide the recording itself. By default the RMS energy reacts
   stage-wide; set **AUDIO ENERGY → APPLIES TO** to a single path to scope it.
4. Capture (in-app **Record**, or the pre-roll countdown into OBS).

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
