"""Voiceover for the Mira-Clé exhibition video.

Generates each narration line with Kokoro (open-weight neural TTS, Apache 2.0)
and places it on the timeline from timeline.js. Writes:
  voice.wav      the narration track (44.1 kHz mono)
  vo-timing.js   when each phrase is spoken, used by index.html for captions
                 and to reveal each service as it is named, and by score.py

Model files (not committed, ~350 MB) come from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

Usage: python3 voiceover.py MODEL_DIR [voice]
Then run score.py to build final-mix.wav.
"""
import json
import os
import re
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from kokoro_onnx import Kokoro

MODEL_DIR = sys.argv[1]
VOICE = sys.argv[2] if len(sys.argv) > 2 else 'af_heart'
SPEED = 0.95
SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))


def load_timeline():
    src = open(os.path.join(HERE, 'timeline.js')).read()
    return json.loads(src[src.index('{'): src.rindex('}') + 1])


TL = load_timeline()
TH = TL['themes']

# Brand name: written as a placeholder, then swapped for exact phonemes so the
# voice says "MEE-ra CLeh" (/ˈmiːrə klɛ/).
BRAND = 'Meera Clay'
BRAND_PHONEMES = ('mˈɪɹə klˈeɪ', 'mˈiːɹə klˈɛ')

# {n} marks the phrase that reveals item n on screen, {q} the testimonial and
# {m} the Leadership Metamorphosis feature. (id, start, latest end, text, captions)
LINES = [
    ('brand', TL['reveal'][0] + 0.9, TL['reveal'][0] + 2.3, f'{BRAND}.', False),
    ('slogan', TL['reveal'][0] + 2.6, TL['reveal'][1] - 0.4, 'Transform. Evolve. Lead.', False),
    ('title', TL['title'][0] + 0.6, TL['title'][0] + 2.8, 'The intelligence behind progress.', False),
    ('intro', TL['title'][0] + 3.0, TL['title'][1] - 0.2,
     'Progress depends on the people who lead an organization, and the structure that supports them.', True),
    ('overview', TL['overview'][0] + 0.6, TL['overview'][1] - 0.4,
     'We work across five areas of practice: {0}structure, {1}people, {2}performance, {3}capability, {4}and resilience.', True),
    ('structure', TH[0][0] + 0.7, TH[0][1] - 0.4,
     'Structure. {0}We restructure organizations, {1}design operating models, {2}align teams behind one direction, '
     '{3}and lead cultural change.', True),
    ('people', TH[1][0] + 0.7, TH[1][1] - 0.4,
     'People. {0}Leadership Metamorphosis, {1}executive development, {2}team effectiveness, {3}workforce capability, '
     '{4}and behavioral change, developed by our team of veteran experts. '
     '{m}Explore the Leadership Metamorphosis program: an executive leadership journey inspired by nature\'s most profound transformation.', True),
    ('performance', TH[2][0] + 0.7, TH[2][1] - 0.4,
     'Performance. {0}We design HR systems, {1}build performance frameworks and reward systems, '
     '{2}align operations around them, {3}and stay to support implementation.', True),
    ('capability', TH[3][0] + 0.7, TH[3][1] - 0.4,
     'Capability. {0}Bespoke learning programs, {1}psychometric and technical assessments, '
     '{2}L M S and e-learning curriculum development, {3}and talent succession. '
     '{q}For Aramex, we developed a learning management system and SCORM content.', True),
    ('resilience', TH[4][0] + 0.7, TH[4][1] - 0.4,
     'Resilience. {0}Emergency response planning, {1}drills and simulations, {2}business continuity, '
     '{3}and organizational readiness, so a crisis meets a prepared team.', True),
    ('close-slogan', TL['close'][0] + 2.3, TL['close'][0] + 4.3, 'Transform. Evolve. Lead.', False),
    ('close-brand', TL['close'][0] + 4.6, TL['close'][0] + 6.0, f'{BRAND}.', False),
    ('close-cta', TL['close'][0] + 6.4, TL['close'][1] - 1.0, f'Talk to our team, or visit {BRAND} dot com.', False),
]

# How captions should spell what the voice is told to say.
CAPTION_FIX = {BRAND: 'Mira-Clé', 'L M S': 'LMS'}


def segments_of(text):
    """Split at punctuation the voice pauses on; keep marker ids per segment."""
    parts = re.split(r'(?<=[,.:;])\s+', text)
    segs = []
    for p in parts:
        marks = re.findall(r'\{(\w+)\}', p)
        clean = re.sub(r'\{\w+\}', '', p)
        segs.append({'text': clean, 'marks': marks})
    return segs


def find_pauses(audio, sr, want):
    """Return the start times of the `want` longest internal pauses, in order."""
    hop = int(0.01 * sr)
    frames = len(audio) // hop
    rms = np.sqrt(np.mean(audio[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    silent = rms < 0.03 * rms.max()
    runs, i = [], 0
    while i < frames:
        if silent[i]:
            j = i
            while j < frames and silent[j]:
                j += 1
            if i > 5 and j < frames - 5 and j - i >= 6:
                runs.append((j - i, i, j))
            i = j
        else:
            i += 1
    runs = sorted(sorted(runs, reverse=True)[:want], key=lambda r: r[1])
    return [r[2] * hop / sr for r in runs]  # speech resumes at the end of each pause


kokoro = Kokoro(os.path.join(MODEL_DIR, 'kokoro-v1.0.onnx'), os.path.join(MODEL_DIR, 'voices-v1.0.bin'))
total = TL['close'][1]
voice = np.zeros(int(total * SR))
timing = []

for line_id, start, end, marked, captions in LINES:
    segs = segments_of(marked)
    text = ' '.join(s['text'] for s in segs)
    speed = SPEED
    for _ in range(6):
        phonemes = kokoro.tokenizer.phonemize(text, 'en-us').replace(*BRAND_PHONEMES)
        audio, vsr = kokoro.create(phonemes, voice=VOICE, speed=speed, lang='en-us', is_phonemes=True)
        if start + len(audio) / vsr <= end:
            break
        speed += 0.05  # tighten delivery slightly if the line overruns its scene
    audio = resample_poly(audio, SR, vsr) if vsr != SR else audio
    nz = np.nonzero(np.abs(audio) > 0.01)[0]
    audio = audio[max(nz[0] - 200, 0): nz[-1] + 2000]
    dur = len(audio) / SR

    pauses = find_pauses(audio, SR, len(segs) - 1)
    if len(pauses) == len(segs) - 1:
        starts = [0.0] + pauses
        how = 'pauses'
    else:  # fall back to spreading segments by length
        lens = np.array([len(s['text']) for s in segs], float)
        starts = list(np.concatenate([[0], np.cumsum(lens)[:-1]]) / lens.sum() * dur)
        how = f'proportional ({len(pauses)} of {len(segs) - 1} pauses found)'

    entry = {'id': line_id, 'start': start, 'end': round(start + dur, 3), 'captions': captions, 'segments': [], 'marks': {}}
    for k, s in enumerate(segs):
        t0 = round(start + starts[k], 3)
        cap = s['text']
        for a, b in CAPTION_FIX.items():
            cap = cap.replace(a, b)
        entry['segments'].append({'t': t0, 'text': cap})
        for m in s['marks']:
            entry['marks'][m] = t0
    timing.append(entry)

    i = int(start * SR)
    voice[i:i + len(audio)] += audio[: len(voice) - i]
    print(f'{start:5.1f}-{start + dur:5.1f}s (limit {end:5.1f}) speed {speed:.2f} [{how}] {text[:48]}')

voice *= 10 ** (-14 / 20) / np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2))
voice = np.clip(voice, -0.98, 0.98)
wavfile.write(os.path.join(HERE, 'voice.wav'), SR, (voice * 32767).astype(np.int16))
with open(os.path.join(HERE, 'vo-timing.js'), 'w') as f:
    f.write('window.VO = ' + json.dumps(timing, indent=1, ensure_ascii=False) + ';\n')
print('wrote voice.wav and vo-timing.js')
