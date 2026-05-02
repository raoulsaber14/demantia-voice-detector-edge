#!/usr/bin/env python3
"""
Generate test WAV files from the Cookie Theft scripts using gTTS.

Requirements:
    pip install gTTS pydub
    # Also needs ffmpeg on PATH. On Pi 4: sudo apt install ffmpeg
    # On Windows: download from https://ffmpeg.org/download.html

Usage:
    python generate_test_audio.py

Outputs:
    test_audio/normal_speech.wav    -- fluent healthy-speaker script
    test_audio/dementia_speech.wav  -- hesitant dementia-like script (slow rate)

IMPORTANT: TTS audio will almost always score LOW risk regardless of script
content. The model was trained on real elderly human voices and uses acoustic
features (jitter, shimmer, prosody) that TTS does not replicate. Use these
files to verify the pipeline runs end-to-end, not to validate clinical scores.

To get meaningful clinical scores you need real human voice recordings,
ideally from elderly speakers describing the Cookie Theft picture.
"""

import io
import sys
from pathlib import Path


NORMAL_SCRIPT = """\
This picture shows a kitchen scene. A woman is standing at the sink washing \
dishes, but the water is overflowing onto the floor and she does not seem to \
notice. She appears to be looking out the window. On the left side of the \
image, a boy is standing on a stool, reaching up into a cookie jar on a high \
shelf. The stool looks like it is about to tip over. A girl is standing next \
to him, reaching up with her hand, apparently asking for a cookie as well. \
The boy is handing her one or two cookies. The kitchen looks like a typical \
home from the nineteen sixties. There is a window above the sink, and outside \
you can see some trees and bushes. The overall scene suggests a mother who is \
distracted while the children are misbehaving behind her back.\
"""

# Hesitations and repetitions are written out so TTS speaks them naturally.
# Extra commas and short sentence fragments create natural-sounding pauses.
DEMENTIA_SCRIPT = """\
Um. There is a woman. She is, um, standing at the, the kitchen area. \
And the water is, the water is going over. Over the edge there. \
She is not, she is not looking. Um. \
And there is a boy. He is on a, on a thing, a stool. \
And he is getting, getting something from up high. Cookies I think. \
And there is a girl there too. She wants some. She wants some too I think. \
Um. And the woman is still, she is just standing there. \
There is a window. You can see outside, some trees or something. \
Um. It looks like a kitchen from, from a long time ago. \
The boy might fall. Um. That is about all I can see.\
"""


def _check_deps() -> bool:
    missing = []
    try:
        import gtts  # noqa: F401
    except ImportError:
        missing.append("gTTS  →  pip install gTTS")
    try:
        import pydub  # noqa: F401
    except ImportError:
        missing.append("pydub  →  pip install pydub")
    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(f"  {m}")
        return False
    return True


def generate_wav(text: str, out_path: Path, slow: bool = False) -> None:
    from gtts import gTTS
    from pydub import AudioSegment

    print(f"  Synthesising {'(slow) ' if slow else ''}→ {out_path.name} ...", end=" ", flush=True)

    # Synthesise to an in-memory MP3 buffer (avoids temp files on Pi 4 SD card).
    tts = gTTS(text=text, lang="en", slow=slow)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    # Decode and convert to 16 kHz mono WAV (model's expected format).
    audio = AudioSegment.from_mp3(mp3_buf)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(out_path), format="wav")
    duration_s = len(audio) / 1000.0
    print(f"done  ({duration_s:.1f} s)")


def main() -> int:
    if not _check_deps():
        return 1

    out_dir = Path("test_audio")
    print(f"Output directory: {out_dir.resolve()}\n")

    generate_wav(NORMAL_SCRIPT,   out_dir / "normal_speech.wav",   slow=False)
    generate_wav(DEMENTIA_SCRIPT, out_dir / "dementia_speech.wav", slow=True)

    print()
    print("Generated files:")
    for f in sorted(out_dir.glob("*.wav")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")

    print()
    print("Run the screener on each file:")
    print("  python record_and_screen.py --input test_audio/normal_speech.wav  --threads 4")
    print("  python record_and_screen.py --input test_audio/dementia_speech.wav --threads 4")
    print()
    print("NOTE: Both files will likely score LOW risk because TTS audio lacks")
    print("the acoustic signatures (jitter, shimmer, prosody) the model was trained on.")
    print("These files verify the pipeline runs correctly, not that scores are meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())