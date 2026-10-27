"""Voiceover for the Mira-Clé exhibition video.

Generates each narration line with Kokoro (open-weight neural TTS, Apache 2.0),
places it on the video timeline, ducks soundtrack.wav under the voice and
writes final-mix.wav.

Model files (not committed, ~350 MB) come from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

Usage: python3 voiceover.py MODEL_DIR [voice]
"""
import sys
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from kokoro_onnx import Kokoro

MODEL_DIR = sys.argv[1]
VOICE = sys.argv[2] if len(sys.argv) > 2 else 'af_heart'
SPEED = 0.95
HERE = os.path.dirname(os.path.abspath(__file__))

# Brand name spelt phonetically for the voice: "MEE-ra CLAY".
BRAND = 'Meera Clay'

# (start time in seconds, latest end time, text). Times follow index.html.
LINES = [
    (0.8, 6.6, 'Progress depends on the people who lead an organization, and the structure that supports them.'),
    (7.6, 11.2, f'{BRAND}. The intelligence behind progress.'),
    (12.4, 20.6, 'We work across five areas of practice: structure, people, performance, capability, and resilience.'),
    (21.6, 30.6, 'Structure. We help organizations restructure, design operating models, and align teams and culture behind one direction.'),
    (31.6, 43.6, 'People. Executive development, team effectiveness, and workforce capability, built together with HR and leadership. '
                 'Credit One Al Sayer has worked with us since twenty twelve.'),
    (44.6, 53.6, 'Performance. We build HR systems and performance frameworks, and we stay to support the implementation.'),
    (54.6, 66.6, 'Capability. Custom learning programs, assessments, and digital learning platforms. '
                 'For Aramex, we developed L M S and SCORM content from twenty nineteen to twenty twenty-two.'),
    (67.6, 76.6, 'Resilience. Emergency response frameworks, simulations, and business continuity, '
                 'so your organization is ready before it is tested.'),
    (79.3, 82.6, 'Transform. Evolve. Lead.'),
    (82.8, 86.0, f'{BRAND}.'),
]

kokoro = Kokoro(os.path.join(MODEL_DIR, 'kokoro-v1.0.onnx'), os.path.join(MODEL_DIR, 'voices-v1.0.bin'))

sr, music = wavfile.read(os.path.join(HERE, 'soundtrack.wav'))
music = music.astype(np.float64) / 32767
N = len(music)
vo = np.zeros(N)
active = np.zeros(N)

for start, end, text in LINES:
    speed = SPEED
    for _ in range(6):
        audio, vsr = kokoro.create(text, voice=VOICE, speed=speed, lang='en-us')
        if start + len(audio) / vsr <= end:
            break
        speed += 0.05  # tighten delivery slightly if the line overruns its scene
    audio = resample_poly(audio, sr, vsr) if vsr != sr else audio
    # trim leading/trailing silence
    nz = np.nonzero(np.abs(audio) > 0.01)[0]
    audio = audio[max(nz[0] - 200, 0): nz[-1] + 2000]
    i = int(start * sr)
    audio = audio[: N - i]
    vo[i:i + len(audio)] += audio
    active[i:i + len(audio)] = 1
    print(f'{start:5.1f}-{start + len(audio) / sr:5.1f}s (limit {end:5.1f}) speed {speed:.2f}: {text[:50]}')

# duck the music about 8 dB under the voice, with smooth 250 ms ramps
k = int(0.25 * sr)
ramp = np.convolve(active, np.ones(k) / k, mode='same')
ramp = np.clip(np.convolve(ramp > 0.001, np.ones(k) / k, mode='same'), 0, 1)
duck = 1 - ramp * (1 - 10 ** (-8 / 20))

vo *= 10 ** (-14 / 20) / np.sqrt(np.mean(vo[active > 0] ** 2))  # voice level
mix = music * duck[:, None] + np.stack([vo, vo], 1)
ceiling = 10 ** (-1 / 20)
mix = np.tanh(mix / ceiling) * ceiling

out = os.path.join(HERE, 'final-mix.wav')
wavfile.write(out, sr, (mix * 32767).astype(np.int16))
print('wrote', out)
