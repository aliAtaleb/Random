// Renders index.html frame by frame and encodes a 1920x1080 MP4 with ffmpeg.
// Usage: node render.js [output.mp4] [--stills t1,t2,...]
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');

const FPS = 30;

(async () => {
  const args = process.argv.slice(2);
  const stillsIdx = args.indexOf('--stills');
  const stills = stillsIdx >= 0 ? args[stillsIdx + 1].split(',').map(Number) : null;
  const out = args.find(a => a.endsWith('.mp4')) || path.join(__dirname, 'mira-cle-exhibition.mp4');

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto('file://' + path.join(__dirname, 'index.html'));
  await page.evaluate(() => window.ready);
  const duration = await page.evaluate(() => window.DURATION);

  if (stills) {
    for (const t of stills) {
      await page.evaluate(t => window.seek(t), t);
      await page.screenshot({ path: path.join(process.env.STILLS_DIR || __dirname, `still-${t}.png`) });
    }
    await browser.close();
    return;
  }

  // soundtrack.wav comes from score.py; it is muxed in when present
  const wav = path.join(__dirname, 'soundtrack.wav');
  const audio = require('fs').existsSync(wav)
    ? ['-i', wav, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k']
    : [];
  const ff = spawn('ffmpeg', [
    '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-', ...audio,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', out
  ], { stdio: ['pipe', 'inherit', 'inherit'] });

  const total = Math.round(duration * FPS);
  for (let f = 0; f < total; f++) {
    await page.evaluate(t => window.seek(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 150 === 0) console.error(`frame ${f}/${total}`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.error('done: ' + out);
})();
