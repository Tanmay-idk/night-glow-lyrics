---
name: night-glow-lyrics
description: Make a 16:9 night-blue, glowing-white lyric video that makes people feel the song (style of the "Dooron Dooron" video by Paresh Pahuja ft. Harleen Sethi). Takes a song MP3 (or clip), its lyrics, 2-4 emotional video clips (glowing / dreamy couple clips work best) and the song title + artists. Separates vocals (demucs), force-aligns every word (torchaudio MMS_FA), finds beats (librosa), maps each lyric line to the clip moment that matches its meaning, draws everything in an HTML canvas rendered frame by frame in headless Chrome and encodes with ffmpeg. Use whenever the user runs /night-glow-lyrics or asks for a lyric video, kinetic typography video, Instagram-post-dive intro, or an emotional glowing lyric edit from a song plus clips.
argument-hint: <song.mp3> <lyrics.txt> <clip1.mp4> <clip2.mp4> [clip3 clip4] "<Song Title>" "<Artist ft. Artist>"
---

# night-glow-lyrics

Goal: a 1920x1080, 30 fps lyric video that makes people **feel** the song. One night-blue look, glowing white light, emotional clips that follow the story of each line, an Instagram-post intro that dives into the title, and lyric typography that changes style with the music. The last line turns to light.

Reply in the language the user writes in. Show lyrics on screen exactly as the user supplied them (do not translate on screen unless asked).

## Hard rules (every run)

1. **Everything is drawn in code.** One HTML canvas, rendered frame by frame in headless Chrome, encoded with ffmpeg. No stock footage, no image generators, no online assets except the user's own clips. All randomness is seeded (mulberry32) so the look is identical every run.
2. **Never guess timings.** Every word is timed by forced alignment (torchaudio `MMS_FA`) on the separated vocal track. If alignment fails for a line, stop and tell the user; do not invent times.
3. **Contact sheet before the full render.** Render stills, show them, and wait for approval.
4. **Ask before overwriting** any existing file (outputs, work folder, fonts). List what would be replaced and wait for a yes.
5. **Never use the same clip moment twice.** A moment plays once, forward, as one continuous take (it may span several consecutive lines). No replays, loops, reversals or overlapping source ranges.
6. Outputs: full MP4 with the **original audio**, plus a **720p copy under 20 MB** for WhatsApp.

## Step 0: Inputs and setup

Parse `$ARGUMENTS` and any attached files. Identify: audio (mp3/wav/m4a), lyrics (txt, one line per sung line, blank line between sections if given), 2-4 video clips, title, artists. If anything is missing, ask once for all missing items together. Then:

- Working folder: `night-glow-work/` next to the audio. Slug = lowercase title with dashes. Outputs: `<slug>-lyric-video.mp4`, `<slug>-720p.mp4`. If any exist, ask before overwriting.
- Check tools: `ffmpeg`, `ffprobe`, `python3`, `node`, a Chrome/Chromium binary. Python: `demucs`, `torch`, `torchaudio`, `librosa`, `soundfile`, `numpy`, `Pillow`; `uroman` if lyrics are not Latin script. Node: `puppeteer-core` (use the system Chrome). Tell the user what is missing and ask before installing.
- Fonts: **Inter** (weight 300) and **Instrument Serif Italic**. Keep local `.woff2/.ttf` in `night-glow-work/fonts/` and load with `FontFace` + `await document.fonts.ready`. Ask before downloading. No other font families anywhere.

## Step 1: Audio analysis and the story

**1a. Vocals**
```bash
python -m demucs --two-stems=vocals -o night-glow-work/demucs "<audio>"
```
Use `.../vocals.wav` for alignment only. The final video uses the **original** audio.

**1b. Forced alignment of every word** (MMS_FA expects lowercase a-z and apostrophes):
```python
import torch, torchaudio, re, json
from torchaudio.pipelines import MMS_FA as bundle
wav, sr = torchaudio.load("vocals.wav")
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, bundle.sample_rate)
model = bundle.get_model(with_star=False); tok = bundle.get_tokenizer(); aligner = bundle.get_aligner()
with torch.inference_mode():
    emission, _ = model(wav)
# words: list of normalized strings, one per lyric word (romanize first if the script is not Latin)
norm = lambda w: re.sub(r"[^a-z']", "", w.lower())
spans = aligner(emission[0], tok(words))
ratio = wav.size(1) / emission.size(1) / bundle.sample_rate
# per word: start = spans[i][0].start*ratio, end = spans[i][-1].end*ratio, score = mean(s.score for s in spans[i])
```
- Keep a map: original on-screen word <-> normalized word <-> (start, end, score). Words that normalize to empty need a neighbour-bound time; flag them.
- If the audio is a clip, the lyrics must be the lyrics of that clip only; confirm with the user if counts look off.
- If memory fails on a long song, align section by section using coarse windows from vocal-energy gaps, never by guessing.
- **Sanity checks** (print a table, fix or ask): times strictly increasing; no word <0.04 s or >3 s unless held on purpose; each word lies in a region where vocal-stem RMS is above the noise floor; low-score words flagged. Save `words.json` with `{line, word, text, start, end, score}`.

**1c. Beats and energy**
```python
import librosa
y, sr = librosa.load(audio, sr=22050)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
rms = librosa.feature.rms(y=y)[0]  # normalize to 0-1, resample to 30 fps -> energy[]
```
Save `beats.json` (beat times, downbeats guess, tempo, per-frame energy). Beats drive the glow pulse, mote twinkle and push-in easing; line cuts/crossfades land on or just before a beat when within 0.15 s.

**1d. Read the lyrics for meaning.** If not English, write a short English gloss per line (for your own matching and the shot table). Then **tell the user the story in one line** before building, e.g. "Two people who love each other across distance, wishing the distance would dissolve." Also note: the emotional peak, the hook / title phrase, repeated phrases, contrast pairs, words about distance/light/dust, and the final line.

## Step 2: Watch every clip, list usable moments

For each clip:
```bash
ffmpeg -i clip.mp4 -vf "fps=2,scale=640:-2" night-glow-work/frames/clipN/%04d.jpg
```
Make contact grids (e.g. 5x4 of the 2 fps frames with timestamps burnt in) and **look at them**. Build a moments list per clip with an ID, time range, and what happens: holding hands, looking from behind, reaching, walking apart, alone, embrace, laughing, close-up of eyes, hands, silhouette, light flare... Also record per moment:
- mean luma and 95th-percentile luma (PIL on the 2 fps frames) -> `blown` if mean > 0.6 or p95 > 0.9
- where the people are (left / right / center) and the **empty side** for text
- mood tags (tender, longing, joy, distance, alone, release)

Drop shaky, blurry, off-topic or watermark/UI-bearing moments. Moments must be >= 1.2 s. Save `moments.json`.

## Step 3: Shot table (the match)

Map every lyric line (or consecutive lines sharing one continuous take) to the moment whose meaning fits the line. Present this table to the user and keep it in `shots.json`:

| line | text (gloss) | line time | style | clip | source range | empty side | brightness / contrast | motion |
|---|---|---|---|---|---|---|---|---|

Rules:
- Meaning first: "alone / far" lines get alone or walking-apart moments; "together / near" lines get hands, embrace; the final line gets the most luminous, release-like moment (reaching, light, silhouette).
- No moment reused; source ranges never overlap. A shot may start up to ~0.4 s before its first word and run ~0.5 s past the last (crossfade overlap).
- Source plays at 0.7x-1.0x speed to fill the shot; never freeze-frame. If moments run out, say so and ask for more clips or permission to lengthen shots with a different moment. Do not repeat.
- Run a check script that asserts: every line has a shot, no source overlap, shots cover the whole timeline with crossfades, no gaps.
- Shot motion: slow push-in 1.00 -> 1.06-1.10, alternating anchor point (toward the people, or away from the text block). Crossfade 0.5-0.8 s between shots.
- **Dim blown-out moments** per shot: `brightness = clamp(0.55 / mean_luma, 0.6, 1.0)`, `contrast = 1.05-1.15`; the text must always win. Verify on the contact sheet.

Frames for rendering: extract each shot's source range at 30 fps, cover-fit to 2112x1188 (room for push-in), JPEG q92, into `night-glow-work/shotframes/<shot>/%05d.jpg` (resampled to the playback speed with ffmpeg `setpts`/`fps`). The renderer loads image frames, not `<video>` (video seeking in headless Chrome is unreliable).

## Step 4: One look for the whole video

Palette (CSS variables in the renderer): `--bg #050b1f`, `--deep #0a1a3f`, `--blue #6fa8ff`, `--ice #cfe3ff`, `--white #f6f9ff`, `--dust #8e919d`.

Per frame, in this order:
1. Shot frame(s) with push-in and crossfade, `ctx.filter = brightness(b) contrast(c) saturate(.7)` using the per-shot values.
2. **Night-blue grade**: fill `--deep` with `multiply` at ~0.38; shadows lifted toward blue with `soft-light` of `--bg` at 0.35.
3. **Glowing whites**: blurred copy (`blur(28px) brightness(1.25)`) drawn with `screen` at 0.28 + 0.12 * beat energy.
4. **Light motes**: ~70 seeded particles, drifting up and sideways slowly, 2-7 px soft radial gradients, `lighter` composite, alpha 0.15-0.6, twinkle on beats. Motes stay behind the text.
5. Vignette (radial, edges to `--bg` at 0.55).
6. Text layer (below), always on top.
7. Fine film grain at 3-4% (seeded noise tile shifting per frame).

**Typography** (one family system only): small words in Inter 300, uppercase, tracking 0.32-0.42em; lyrics in Instrument Serif italic. Text is white. Every text draw is three passes: (1) dark shadow `rgba(2,6,20,.7)`, blur 22, offset y+4 so it reads over bright clips; (2) soft blue glow `rgba(111,168,255,.85)`, blur 32-60 scaled with beat energy; (3) crisp white fill. Sizes at 1080p: small caps 30-38 px; key word 300-380 px; medium row 76-96 px; bottom line 60-68 px.

## Step 5: Opening (title sequence)

Fit it to the instrumental intro before the first sung word (if the first word arrives under ~9 s, compress; if vocals start immediately, ask the user how to handle it).

1. **Clip 1, quiet, no text.** Slow push-in, motes only, fades up from `--bg` over ~1 s.
2. **Instagram post** (drawn in canvas, generic UI, no real logos): a rounded card centred on the dark blue grade. Header: circle avatar with the first artist's initial and a handle from the artist name (lowercase, no spaces); square media area playing **clip 2**; action icons (heart filled, comment, share, save) drawn as vector paths; "1,284 likes" (a plausible number, fictional UI); caption = handle + the **lyric** (the hook line, or the first key line), then a hashtag of the title (`#dooron_dooron` style). Card floats in with a gentle rise and fade, hold ~1.5 s with a tiny heart pulse.
3. **The camera dives**: scale and translate into the media area (ease-in-out, ~1.6 s) until the clip-2 image fills the whole 16:9 frame, UI fading away during the dive; crossfade the media into the full-frame graded version.
4. **Title lands**: song title in huge Instrument Serif italic (about 200-260 px), glow blooming in with letter-by-letter fade, artists below in Inter 300 small caps, tracked. Hold >= 2.5 s, then fade out as the first line begins.

## Step 6: Lyric variety, timed to the voice

Classify every line:

- **Key lines (hierarchy block)**: hook / title phrase, emotional peaks, longest held words, repeated chorus hooks, the contrast pair, the ending. Target ~35-45% of lines. Three rows placed in the **empty space beside the people** (use the shot's empty side; left/right third, vertically centred or top-weighted; never over faces or hands):
  1. small tracked caps row (the words before the key word),
  2. **one huge serif key word** (the emotional word),
  3. medium serif row (the words after).
- **In-between lines**: one simple centred line at the bottom (baseline ~y 965), Instrument Serif italic, single row.
- Mix both so it never reads as captions: never more than 2 key lines or 3 bottom lines in a row; the first and last sung lines are key lines.

**Timing**: each word's reveal begins at its aligned start (alpha 0 -> 1 over ~0.18 s, starting ~0.05 s early) and finishes by the time the word is sung. Rows hold until line end + 0.4 s, then fade/drift out over 0.35 s. The key word enters on its own word time.

**Entrances** (rotate; never the same one twice in a row): smooth **slide** (24-60 px, ease-out), **rise** (from below with blur-out), **scale** (0.92 -> 1.0 with glow bloom), **letter-by-letter fade** (each letter staggered 25-45 ms). All easing is `easeOutCubic` / `easeInOutSine`; nothing bounces.

## Step 7: Meaning moments

Detect these in the lyrics and give them special treatment; for other songs, look for the equivalents (a contrast pair, a word of distance, a final line). Default examples from "Dooron Dooron":

- **Contrast pair (e.g. mitti / pari)**: split into two words. The first is **grey and dusty** (`--dust`, no glow, slightly blurred, fine dust specks falling from its edge, gentle downward sag). The second is **glowing**, brightest white, with 4-point **sparkles** twinkling around its letters (seeded positions, size pulses on beats). The glowing word enters on its sung time with a bloom; the dust word is slightly smaller.
- **Distance words (e.g. dooron dooron)**: letters **drift apart one by one**: each letter's x offset grows with an eased curve and a per-letter stagger, spacing widens, alpha thins out toward the edges while the glow stays on the centre letters. The motion should be completed by the end of the second repeat.
- **Final line (e.g. kahan main)**: **dissolves into light**. Words appear normally, then, on the last sung word: letters blur up and apart, scale to ~1.12, bloom into a screen-blended white-blue flare that fills the frame, while the shot brightens (do not dim here) and motes surge upward. Hold the glow ~1.5 s, then fade everything to `--bg` and black. Nothing remains but light.

## Step 8: Contact sheet, then render

Implementation layout in `night-glow-work/`:
- `timeline.json` (merge of words, beats/energy, shots, styles, special moments, title card data, fonts).
- `render.html`: exposes `window.renderFrame(n)`; draws the whole frame as a pure function of `t = n / 30` and the seeded RNG; preloads shot frames lazily (`Image.decode()`); waits for fonts.
- `render.mjs`: puppeteer-core with system Chrome (`--headless=new --disable-gpu-vsync`), viewport 1920x1080, calls `renderFrame(n)`, grabs the canvas as PNG into `frames/%06d.png`. Use `os.cpus().length / 2` parallel pages, each on a contiguous frame range. Resume-safe (skip existing frames only after the user confirmed overwrite rules).

**Contact sheet first.** Render ~12-16 stills at 1920x1080 and tile them (ffmpeg `tile` or PIL) into `contact-sheet.png`: quiet opener, Instagram post, mid-dive, title, at least two key-line blocks (different entrances), two bottom lines, the dusty/glow pair, the drifting-apart word, the final dissolve start and mid-glow. Look at it yourself (open the PNG) and check: text never over faces/hands, text legible on the brightest shots, one consistent look, no blown-out frames, grain/motes not distracting. Fix problems, then **show the contact sheet to the user and wait for approval** before the full render. Ask before overwriting any existing frames or outputs.

**Full render and encode** (frames = `ceil(duration * 30)`; video duration = audio duration; if the last word ends less than 2.5 s before the audio ends, ask whether to add a silent tail):
```bash
ffmpeg -framerate 30 -i frames/%06d.png -i "<original audio>" \
  -map 0:v -map 1:a -vf "format=yuv420p" \
  -af "afade=t=in:d=0.5,afade=t=out:st=$(python3 -c "print(max(0,D-1.5))"):d=1.5" \
  -c:v libx264 -preset slow -crf 16 -r 30 -c:a aac -b:a 256k -movflags +faststart \
  -t D "<slug>-lyric-video.mp4"
```
(D = audio duration in seconds.) Then the **WhatsApp copy under 20 MB**:
```bash
# video kbps = (18.5 * 8192 / D) - 128   (aim for ~18.5 MB total, keep a margin)
ffmpeg -i "<slug>-lyric-video.mp4" -vf scale=1280:720 -c:v libx264 -preset slow \
  -b:v ${VK}k -maxrate ${VK}k -bufsize $((VK*2))k -pix_fmt yuv420p \
  -c:a aac -b:a 128k -movflags +faststart "<slug>-720p.mp4"
```
Check the size with `stat`; if >= 20 MB (20,000,000 bytes to be safe), lower the bitrate and redo. Verify both files with `ffprobe` (resolution, 30 fps, duration, audio present).

## Final checks and report

- Spot-check 5 frames from the finished MP4 (first, title, a key line, a bottom line, last) and confirm audio sync: the first word appears on its aligned time.
- Report to the user: the one-line story, the shot table summary, any flagged alignment words, any dimmed shots, output paths and sizes, and the render time. Offer one round of tweaks (swap a moment, change a key line to a bottom line, change an entrance) and re-render only the changed range.
