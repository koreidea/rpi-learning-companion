#!/usr/bin/env python3
"""Generate nursery rhyme WAV files using Piper TTS.

Run on the Pi:
    cd ~/rpi-bot/rpi
    .venv/bin/python scripts/generate_songs.py

Creates WAV files in audio/songs/ from nursery rhyme lyrics.
These are spoken versions (not sung) — replace with actual sung
versions from Suno AI or similar for better quality.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SONGS_DIR = BASE_DIR / "audio" / "songs"
MODELS_DIR = BASE_DIR / "models" / "tts"

# Piper binary and voice model
PIPER_BIN = BASE_DIR / ".venv" / "bin" / "piper"
VOICE_MODEL = MODELS_DIR / "en_US-lessac-medium.onnx"

# Nursery rhymes to generate
SONGS = {
    "twinkle_twinkle": (
        "Twinkle, twinkle, little star, "
        "How I wonder what you are! "
        "Up above the world so high, "
        "Like a diamond in the sky. "
        "Twinkle, twinkle, little star, "
        "How I wonder what you are!"
    ),
    "baa_baa_black_sheep": (
        "Baa, baa, black sheep, have you any wool? "
        "Yes sir, yes sir, three bags full. "
        "One for the master, one for the dame, "
        "And one for the little boy who lives down the lane. "
        "Baa, baa, black sheep, have you any wool? "
        "Yes sir, yes sir, three bags full."
    ),
    "jack_and_jill": (
        "Jack and Jill went up the hill, "
        "To fetch a pail of water. "
        "Jack fell down and broke his crown, "
        "And Jill came tumbling after. "
        "Up Jack got, and home did trot, "
        "As fast as he could caper. "
        "He went to bed to mend his head, "
        "With vinegar and brown paper."
    ),
    "humpty_dumpty": (
        "Humpty Dumpty sat on a wall, "
        "Humpty Dumpty had a great fall. "
        "All the king's horses and all the king's men, "
        "Couldn't put Humpty together again."
    ),
    "mary_had_a_little_lamb": (
        "Mary had a little lamb, little lamb, little lamb, "
        "Mary had a little lamb, its fleece was white as snow. "
        "And everywhere that Mary went, Mary went, Mary went, "
        "Everywhere that Mary went, the lamb was sure to go."
    ),
    "old_macdonald": (
        "Old MacDonald had a farm, E I E I O! "
        "And on his farm he had a cow, E I E I O! "
        "With a moo moo here, and a moo moo there, "
        "Here a moo, there a moo, everywhere a moo moo! "
        "Old MacDonald had a farm, E I E I O!"
    ),
    "itsy_bitsy_spider": (
        "The itsy bitsy spider climbed up the water spout. "
        "Down came the rain and washed the spider out. "
        "Out came the sun and dried up all the rain. "
        "And the itsy bitsy spider climbed up the spout again."
    ),
    "row_row_row_your_boat": (
        "Row, row, row your boat, gently down the stream. "
        "Merrily, merrily, merrily, merrily, life is but a dream. "
        "Row, row, row your boat, gently down the stream. "
        "If you see a crocodile, don't forget to scream!"
    ),
    "london_bridge": (
        "London bridge is falling down, falling down, falling down. "
        "London bridge is falling down, my fair lady. "
        "Build it up with bricks and mortar, bricks and mortar, bricks and mortar. "
        "Build it up with bricks and mortar, my fair lady."
    ),
    "head_shoulders_knees_and_toes": (
        "Head, shoulders, knees and toes, knees and toes. "
        "Head, shoulders, knees and toes, knees and toes. "
        "And eyes, and ears, and mouth, and nose. "
        "Head, shoulders, knees and toes, knees and toes."
    ),
}


def generate_song(name: str, lyrics: str):
    """Generate a WAV file from lyrics using Piper TTS."""
    output_path = SONGS_DIR / f"{name}.wav"
    if output_path.exists():
        print(f"  Skipping {name} (already exists)")
        return

    print(f"  Generating {name}...")
    try:
        proc = subprocess.run(
            [
                str(PIPER_BIN),
                "--model", str(VOICE_MODEL),
                "--output_file", str(output_path),
            ],
            input=lyrics,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            size_kb = output_path.stat().st_size // 1024
            print(f"  Created {name}.wav ({size_kb} KB)")
        else:
            print(f"  ERROR: {proc.stderr.strip()}")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    SONGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SONGS)} nursery rhymes to {SONGS_DIR}/\n")

    for name, lyrics in SONGS.items():
        generate_song(name, lyrics)

    print(f"\nDone! {len(list(SONGS_DIR.glob('*.wav')))} songs available.")


if __name__ == "__main__":
    main()
