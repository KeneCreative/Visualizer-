import librosa
import pretty_midi
import numpy as np
from scipy.interpolate import interp1d

def align_midi_to_audio(audio_path, midi_path, output_path, start_time=0.0, clip_duration=None):
    print("Loading and clipping audio...")
    # Restored high sample rate (22050) for maximum precision
    y_audio, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=clip_duration)
    
    # Restored tight hop length (512) for millisecond accuracy
    hop_length = 512
    
    chroma_audio = librosa.feature.chroma_cqt(y=y_audio, sr=sr, hop_length=hop_length)
    
    print("Loading MIDI and synthesizing reference audio...")
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    y_midi = midi_data.synthesize(fs=sr)
    chroma_midi = librosa.feature.chroma_cqt(y=y_midi, sr=sr, hop_length=hop_length)
    
    print("Running Dynamic Time Warping (DTW)...")
    D, wp = librosa.sequence.dtw(X=chroma_midi, Y=chroma_audio, metric='cosine')
    
    wp = wp[::-1]
    
    time_midi = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop_length)
    time_audio = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop_length)
    
    print("Calculating warp function...")
    _, unique_indices = np.unique(time_midi, return_index=True)
    time_midi_unique = time_midi[unique_indices]
    time_audio_unique = time_audio[unique_indices]
    
    warp_func = interp1d(time_midi_unique, time_audio_unique, kind='linear', fill_value='extrapolate')
    
    print("Warping MIDI note data...")
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start = float(warp_func(note.start))
            note.end = float(warp_func(note.end))
            
    print(f"Saving humanized MIDI to: {output_path}")
    midi_data.write(output_path)

# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    align_midi_to_audio(
        audio_path='Beethoven, String Quartet op.130 4thMVMT.wav', # <-- Audio recording here
        midi_path='Beethoven4thMVMT.mid',                          # <-- MIDI file here
        output_path='Beethovensq134th.mid',
        start_time=0.0,        
        clip_duration=None     
    )