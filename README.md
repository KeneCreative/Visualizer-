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
3. Load both into the visualizer, route tracks onto shapes, tune the panel.
4. Capture (in-app **Record**, or the pre-roll countdown into OBS).
5. Lay the real recording over the captured video in an editor, using the
   one-frame sync flash on the first note to line it up.

## Dependencies

Vendored, so the app works offline. To update:

```bash
curl -sL -o vendor/tone.js        https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js
curl -sL -o vendor/midi.js        https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.28/build/Midi.js
curl -sL -o vendor/tailwindcss.js https://cdn.tailwindcss.com
```
