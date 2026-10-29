"""Original soundtrack for the Mira-Clé exhibition video.

Synthesises a warm corporate music bed plus sound accents synced to the
animation (timeline.js) and to the narration (vo-timing.js), ducks the music
under voice.wav and writes final-mix.wav (44.1 kHz stereo).
The music is generated here, so there is no third-party audio to license.

Run voiceover.py first. Usage: python3 score.py [output.wav]
"""
import json
import os
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
rng = np.random.default_rng(7)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_js(name):
    src = open(os.path.join(HERE, name)).read()
    return json.loads(src[src.index('=') + 1: src.rindex(';')])


TL = load_js('timeline.js')
VO = load_js('vo-timing.js')
T_REVEAL = TL['open'][0]
T_OVER = TL['overview'][0]
THEME_STARTS = [a for a, b in TL['themes']]
T_CLOSE = TL['close'][0]
DURATION = TL['close'][1]
VO_BY_ID = {line['id']: line for line in VO}
VO_OVER = VO_BY_ID['overview']
VO_THEMES = [VO_BY_ID[k] for k in ('structure', 'people', 'performance', 'capability', 'resilience')]

# ---- music grid: 96 BPM, bar lines land on the first practice scene ----
BEAT = 60 / 96
BAR = BEAT * 4
OFFSET = THEME_STARTS[0] % BAR

N = int(DURATION * SR)
music = np.zeros((N, 2))
sfx = np.zeros((N, 2))
send = np.zeros((N, 2))  # reverb send


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def add(buf, start, sig, gain=1.0, pan=0.0):
    i = int(start * SR)
    if i >= N:
        return
    if i < 0:  # an accent that would start before the video does is trimmed
        sig, i = sig[-i:], 0
    sig = sig[: N - i]
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        sig = np.stack([sig * l, sig * r], axis=1)
    buf[i:i + len(sig)] += sig * gain


def adsr(n, a, r, sustain_end=None):
    env = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    env[:na] = np.linspace(0, 1, na) if na else env[:na]
    if nr:
        env[-nr:] *= np.linspace(1, 0, nr)
    return env


def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


# ---- instruments ----
def pad_note(m, dur, bright=4.0):
    t = tvec(dur)
    out = np.zeros_like(t)
    for cents in (-7, 0, 7):
        f = midi(m) * 2 ** (cents / 1200)
        for h in range(1, 9):
            out += np.sin(2 * np.pi * f * h * t + rng.random() * 6.28) / h * np.exp(-h / bright)
    return out * adsr(len(t), 1.2, 1.4) * 0.05


def pluck(m, dur=1.2, decay=0.45):
    t = tvec(dur)
    f = midi(m)
    sig = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.15)
           + 0.12 * np.sin(6 * np.pi * f * t) * np.exp(-t / 0.08))
    return sig * np.exp(-t / decay) * adsr(len(t), 0.004, 0.05)


def bell(m, dur=3.0):
    t = tvec(dur)
    f = midi(m)
    sig = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
              for r, a, d in ((1, 1, 1.6), (2.76, 0.4, 0.6), (5.4, 0.2, 0.25), (8.9, 0.08, 0.1)))
    return sig * adsr(len(t), 0.002, 0.2)


def bass(m, dur):
    t = tvec(dur)
    f = midi(m)
    sig = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return sig * np.exp(-t / 0.9) * adsr(len(t), 0.01, 0.12)


def kick():
    t = tvec(0.45)
    f = 45 + 80 * np.exp(-t / 0.04)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.18)


def shaker():
    t = tvec(0.08)
    return filt(rng.standard_normal(len(t)), 'highpass', 6500) * np.exp(-t / 0.018)


def rim():
    t = tvec(0.15)
    return filt(rng.standard_normal(len(t)), 'bandpass', [1200, 3200]) * np.exp(-t / 0.03)


def whoosh(dur=0.9, peak=0.65):
    t = tvec(dur)
    noise = rng.standard_normal(len(t))
    lo, hi = filt(noise, 'bandpass', [300, 1500]), filt(noise, 'bandpass', [1500, 6000])
    x = t / dur
    mix = lo * (1 - x) + hi * x
    env = np.where(x < peak, (x / peak) ** 2, ((1 - x) / (1 - peak)) ** 1.5)
    return mix * env


def riser(dur):
    t = tvec(dur)
    noise = filt(rng.standard_normal(len(t)), 'bandpass', [600, 4500], order=3)
    tone = sum(np.sin(2 * np.pi * midi(62 + k) * (1 + 0.5 * t / dur) * t) for k in (0, 7, 12))
    return (noise * 0.3 + tone * 0.15) * (t / dur) ** 2.5


def boom():
    t = tvec(2.5)
    f = 38 + 50 * np.exp(-t / 0.08)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.9)
    air = filt(rng.standard_normal(len(t)), 'lowpass', 900) * np.exp(-t / 0.25) * 0.35
    return body + air


def tick(m):
    t = tvec(0.25)
    return (np.sin(2 * np.pi * midi(m) * t) * np.exp(-t / 0.06)
            + 0.3 * filt(rng.standard_normal(len(t)), 'highpass', 4000) * np.exp(-t / 0.008))


# ---- harmony: Dmaj9 | Bm9 | Gmaj9 | Asus4-A, one chord per bar ----
CHORDS = [
    (38, [62, 66, 69, 73, 76]),
    (35, [62, 66, 69, 71, 73]),
    (31, [62, 66, 67, 71, 69]),
    (33, [62, 64, 69, 71, 73]),
]
ARP = [[0, 2, 1, 3, 2, 4, 3, 1], [0, 1, 3, 2, 4, 2, 3, 1]]

bar_starts = np.arange(OFFSET - BAR * 3, DURATION, BAR)
for bi, b0 in enumerate(bar_starts):
    if b0 + BAR < 0:
        continue
    root, notes = CHORDS[bi % 4]
    # pad: everywhere, darker in the intro, gone in the last bars
    if b0 < T_CLOSE + 3:
        bright = 2.2 if b0 < T_OVER else 4.0
        for m in notes[:4]:
            add(music, max(b0, 0), pad_note(m - 12 if m > 70 else m, BAR + 1.4, bright), 0.9)
    # sparse piano in the intro and logo reveal
    if b0 < T_OVER:
        for k, m in enumerate((notes[0] + 12, notes[2] + 12)):
            add(music, max(b0, 0) + k * BEAT * 2, pluck(m, 2.5, 0.9) * 0.08, 1, -0.3 + 0.6 * k)
            add(send, max(b0, 0) + k * BEAT * 2, pluck(m, 2.5, 0.9) * 0.08)
    # arpeggio: overview and practice scenes
    if T_OVER - 1 <= b0 < T_CLOSE:
        pat = ARP[(bi // 4) % 2]
        for k in range(8):
            m = notes[pat[k]] + 12
            st = b0 + k * BEAT / 2
            p = pluck(m, 1.0, 0.35) * 0.055
            add(music, st, p, 1, -0.45 if k % 2 else 0.45)
            add(send, st, p * 0.7)
    # bass and drums: practice scenes only
    if THEME_STARTS[0] - 0.01 <= b0 < T_CLOSE:
        for k, d in ((0, 1.5), (1.5, 1.0), (2.5, 1.5)):
            add(music, b0 + k * BEAT, bass(root, d * BEAT + 0.3) * 0.16)
        for k in (0, 2):
            add(music, b0 + k * BEAT, kick() * 0.22)
        for k in (1, 3):
            add(music, b0 + k * BEAT, rim() * 0.05, 1, 0.2)
            add(send, b0 + k * BEAT, rim() * 0.03)
        for k in range(8):
            add(music, b0 + k * BEAT / 2, shaker() * (0.035 if k % 2 else 0.02), 1, 0.35)
    elif T_OVER <= b0 < THEME_STARTS[0]:
        for k in range(8):
            add(music, b0 + k * BEAT / 2, shaker() * (0.025 if k % 2 else 0.012), 1, 0.35)

# ---- accents synced to the animation ----
# a soft whoosh under every wipe transition
for t in [T_OVER] + THEME_STARTS + [T_CLOSE]:
    add(sfx, t - 0.55, whoosh(0.9), 0.05, -0.2)

# opening: the mark lands, then a bell for each word of the slogan
add(sfx, T_REVEAL + 0.4, boom(), 0.26)
for m, d in ((74, 0), (78, 0.04), (81, 0.08), (86, 0.12)):
    add(sfx, T_REVEAL + 0.4 + d, bell(m) * 0.045, 1, (m - 80) / 10)
    add(send, T_REVEAL + 0.4 + d, bell(m) * 0.05)
for k in range(3):  # the slogan writes on under the logo
    add(sfx, T_REVEAL + 1.0 + k * 0.33, bell((78, 81, 86)[k], 2.0) * 0.03, 1, -0.4 + 0.4 * k)
    add(send, T_REVEAL + 1.0 + k * 0.33, bell((78, 81, 86)[k], 2.0) * 0.04)

# title: a bell as "Progress" lands
add(sfx, VO_BY_ID['title']['start'] + 1.4, bell(86, 3.0) * 0.04, 1, 0.3)
add(send, VO_BY_ID['title']['start'] + 1.4, bell(86, 3.0) * 0.05)

# overview: one rising note as each pillar is named
for i, m in enumerate((74, 76, 78, 81, 83)):
    st = VO_OVER['marks'][str(i)]
    add(sfx, st, bell(m, 2.0) * 0.035, 1, -0.6 + 0.3 * i)
    add(send, st, bell(m, 2.0) * 0.04)

# practice scenes: a quiet tick as each service is named, a chime for the testimonial
for vo in VO_THEMES:
    ticks = sorted(k for k in vo['marks'] if k.isdigit())
    for k in ticks:
        add(sfx, vo['marks'][k], tick((81, 83, 85, 86, 88)[int(k)]) * 0.03, 1, 0.5)
    if 'q' in vo['marks']:
        add(sfx, vo['marks']['q'], bell(78, 2.5) * 0.03, 1, -0.4)
        add(send, vo['marks']['q'], bell(78, 2.5) * 0.04)

# Leadership Metamorphosis: a lift into the feature, then a rising shimmer for the butterfly
mt = VO_BY_ID['people']['marks']['m']
add(sfx, mt - 1.6, riser(1.8), 0.06)
for k, m in enumerate((74, 78, 81, 83, 86, 90, 93)):
    add(sfx, mt - 0.1 + k * 0.14, bell(m, 2.5) * 0.028, 1, -0.6 + 0.2 * k)
    add(send, mt - 0.1 + k * 0.14, bell(m, 2.5) * 0.04)

# closing: rise into the logo, final chord as the slogan writes on
add(sfx, T_CLOSE - 1.6, riser(3.0), 0.07)
add(sfx, T_CLOSE + 0.5, boom(), 0.30)
for m in (50, 57, 62, 66, 69, 73, 76):
    add(music, T_CLOSE + 0.5, pad_note(m, DURATION - T_CLOSE, 3.0) * 1.3)
cs = VO_BY_ID['close-slogan']['start']
for m, d in ((74, 0), (78, 0.05), (81, 0.1), (85, 0.15), (86, 0.2)):
    add(sfx, cs + d, bell(m, 4.0) * 0.045, 1, (m - 80) / 10)
    add(send, cs + d, bell(m, 4.0) * 0.06)

# ---- mix ----
music = filt(music, 'highpass', 30)
ir_t = tvec(2.8)
ir = np.stack([filt(rng.standard_normal(len(ir_t)), 'lowpass', 5000) * np.exp(-ir_t / 0.8) for _ in range(2)], 1)
ir /= np.sqrt((ir ** 2).sum(0))
verb = np.stack([fftconvolve(send[:, c] + music[:, c] * 0.25, ir[:, c])[:N] for c in range(2)], 1)
mix = music + sfx + verb * 0.35

# fades: gentle in, full fade over the last 3 s
fade_in = np.clip(np.arange(N) / (0.5 * SR), 0, 1)
fade_out = np.clip((DURATION - np.arange(N) / SR) / 3.0, 0, 1) ** 1.5
mix *= (fade_in * fade_out)[:, None]

# loudness: target about -16 dBFS RMS, soft limit peaks at -1 dBFS
rms = np.sqrt(np.mean(mix ** 2))
mix *= 10 ** (-16 / 20) / rms
ceiling = 10 ** (-1 / 20)

# voice: duck the music about 8 dB while the narrator speaks, with 250 ms ramps
_, voice = wavfile.read(os.path.join(HERE, 'voice.wav'))
voice = voice.astype(np.float64)[:N] / 32767
voice = np.pad(voice, (0, N - len(voice)))
active = np.zeros(N)
for line in VO:
    active[int(line['start'] * SR): int(line['end'] * SR)] = 1
k = int(0.25 * SR)
ramp = np.clip(np.convolve(active, np.ones(2 * k) / k, mode='same'), 0, 1)
mix *= (1 - ramp * (1 - 10 ** (-8 / 20)))[:, None]
mix += voice[:, None]

ceiling = 10 ** (-1 / 20)
mix = np.tanh(mix / ceiling) * ceiling

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'final-mix.wav')
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print(f'wrote {out}: {DURATION:.1f}s, peak {20*np.log10(np.abs(mix).max()):.1f} dBFS')
