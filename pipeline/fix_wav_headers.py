"""Repair WAV files in audio/ that carry a streamed (unknown-length) header.

ffmpeg, when piping its output, writes 0xFFFFFFFF for the RIFF and data chunk
sizes and never backfills them. libsndfile (and this repo's Python tools) read
such files fine by falling back to the file length, but the browser's
decodeAudioData takes the size fields literally and the recording ends up
cutting in and out / stopping early when used as a Backing Track.

This rewrites each affected file in place with correct chunk sizes. The PCM
samples are copied through untouched; frame counts are checked before the
original is replaced.

    python pipeline/fix_wav_headers.py            # scan + fix audio/
    python pipeline/fix_wav_headers.py --check    # report only, change nothing
"""
import os
import sys
import struct

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "audio")
BLOCK = 1 << 20  # frames per copy chunk


def header_is_streamed(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return False
    riff_size = struct.unpack_from("<I", head, 4)[0]
    di = head.find(b"data")
    data_size = struct.unpack_from("<I", head, di + 4)[0] if di >= 0 else None
    return riff_size == 0xFFFFFFFF or data_size == 0xFFFFFFFF


def fix(path):
    info = sf.info(path)
    tmp = path + ".fixed.wav"
    with sf.SoundFile(path) as fin, sf.SoundFile(
        tmp, "w", samplerate=fin.samplerate, channels=fin.channels,
        subtype=fin.subtype, format="WAV"
    ) as fout:
        copied = 0
        while True:
            block = fin.read(BLOCK, dtype="int16" if "PCM_16" in fin.subtype else "float32")
            if not len(block):
                break
            fout.write(block)
            copied += len(block)
    out_frames = sf.info(tmp).frames
    if out_frames != info.frames:
        os.remove(tmp)
        raise RuntimeError(f"frame mismatch ({info.frames} -> {out_frames}), left original alone")
    os.replace(tmp, path)
    return info.frames


def main():
    check_only = "--check" in sys.argv
    wavs = sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav"))
    bad = [f for f in wavs if header_is_streamed(os.path.join(AUDIO_DIR, f))]
    if not bad:
        print(f"All {len(wavs)} WAV(s) in audio/ have a proper header.")
        return
    print(f"{len(bad)} of {len(wavs)} WAV(s) have a streamed (0xFFFFFFFF) header:")
    for f in bad:
        print(f"  {f}")
    if check_only:
        print("\n--check: nothing changed.")
        return
    print()
    for f in bad:
        p = os.path.join(AUDIO_DIR, f)
        n = fix(p)
        print(f"  fixed {f}  ({n} frames)")


if __name__ == "__main__":
    main()
