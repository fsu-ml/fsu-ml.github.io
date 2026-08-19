# Media, motion and timing

**Covers:** images of text, video and audio (captions, audio description, transcripts, player controls), audio control, pause/stop/hide, flash thresholds, animation from interactions, timing adjustable, and `prefers-reduced-motion`.
**Load when:** `content.has_video`, `has_audio`, `has_animation`, `has_carousel` or `has_images_of_text` is true — **or** the page contains any moving, sounding or auto-updating thing at all, including decorative ones.
**Why "including decorative":** 1.4.2, 2.2.2 and 2.3.1 are **non-interference** criteria. They apply even to content you do not rely on. A decorative autoplaying background video breaks conformance for the whole page (`targets.md`, conformance requirement 5).

---

## 1. Images of text — 1.4.5 (AA), 1.4.9 (AAA)

| SC | Level | Requirement |
|---|---|---|
| 1.4.5 Images of Text | AA | Use real text, not pictures of text. **Logos excepted**, plus cases where a particular presentation is essential. |
| 1.4.9 Images of Text (No Exception) | AAA | No images of text at all except decoration and logos. |

Real text + web fonts, not rendered images. Text in an image does not scale with 1.4.4, does not reflow for 1.4.10, does not respond to 1.4.12 text spacing, cannot be selected or translated, and its contrast cannot be user-adjusted. If an image of text is unavoidable, the text must appear **verbatim** in the `alt` (1.1.1).

Common offenders: hero banners with the headline baked in, "quote card" social images reused on the site, infographics, event posters, price tables exported from a design tool, and email-derived campaign graphics.

**How to verify:** try to select the text with the mouse — if it does not highlight, it is an image. Zoom to 400%: real text stays crisp, an image goes soft. Console sweep for suspicious candidates:
```js
[...document.images].filter(i=>i.naturalWidth>400).map(i=>[i.src, i.alt])
```
Then look at each. Any `alt` containing a full sentence lifted off the image is a 1.4.5 finding as well as an alt-text pass.

---

## 2. Video and audio

| SC | Level | Requirement |
|---|---|---|
| 1.2.1 Audio-only and Video-only (Prerecorded) | A | **Transcript** for audio-only. Transcript **or** an audio track for video-only. |
| 1.2.2 Captions (Prerecorded) | A | Synchronized captions for all prerecorded audio in video. |
| 1.2.3 Audio Description or Media Alternative (Prerecorded) | A | Audio description **or** a full text alternative for prerecorded video. |
| 1.2.4 Captions (Live) | AA | Real-time captions for live audio. |
| 1.2.5 Audio Description (Prerecorded) | AA | Audio description **specifically** — a text alternative is no longer sufficient at AA. |
| 1.2.6 Sign Language (Prerecorded) | AAA | Sign language interpretation. |
| 1.2.7 Extended Audio Description | AAA | Pause the video where needed to fit the description. |
| 1.2.8 Media Alternative (Prerecorded) | AAA | Full text alternative for all prerecorded synchronized media. |
| 1.2.9 Audio-only (Live) | AAA | Text alternative for live audio-only. |
| 1.4.7 Low or No Background Audio | AAA | Background sound ≥ **20 dB** below speech, or separately mutable. |

```html
<video controls>
  <source src="briefing.mp4" type="video/mp4">
  <track kind="captions"     srclang="en" label="English" src="briefing.vtt" default>
  <track kind="descriptions" srclang="en" label="English descriptions" src="briefing-desc.vtt">
</video>
```

**Captions must be human-verified.** Auto-generated captions typically run a **5–15% word error rate** and lack speaker identification and punctuation, which is not "equivalent" and does **not** meet 1.2.2. Accepting YouTube auto-captions as-is is a finding, not a mitigation.

**Audio description** is needed wherever the visual track carries information the audio does not. **The cheapest fix is at scripting time:** if the narration speaks what is on screen ("as you can see in the left column, applications rose to 4,800"), you may avoid needing a separate description track entirely.

**Transcripts** are required for audio-only under 1.2.1 and are useful for everything — they are also the only version that is searchable, translatable and skimmable.

**Player accessibility (2.1.1, 4.1.2, 2.4.7):** the player controls themselves must be keyboard operable and labelled. Many embedded players fail this. Custom-skinned players and lazy-loaded iframe embeds are the usual offenders. Test the actual player on the actual page, not the vendor's demo.

**How to verify:**
- Inventory: `[...document.querySelectorAll('video,audio,iframe[src*=youtube],iframe[src*=vimeo],iframe[src*=wistia]')].map(e=>e.src||e.currentSrc)`
- Caption tracks present: `[...document.querySelectorAll('video')].map(v=>[...v.textTracks].map(t=>t.kind+':'+t.language))`
- Caption **quality**: play 60 seconds with captions on and compare against the audio. Count errors and check for speaker labels. A tool cannot do this.
- Player keyboard: Tab into the player — every control (play, volume, scrub, captions, fullscreen) must be reachable, labelled, and operable with Enter/Space/arrows, and you must be able to Tab back out (2.1.2).
- Transcript: is one linked adjacent to the media, or only in a separate PDF three clicks away? Adjacent counts; buried does not.

---

## 3. Audio control — 1.4.2 (A)

Any audio that plays automatically for **more than 3 seconds** must either stop/pause automatically within 3 seconds, or provide a mechanism to pause/stop it, or to control its volume **independently of the system volume level**.

The pause control must be **within the first few tab stops** — a screen reader user cannot find a control they cannot hear over. The correct answer is: **do not autoplay audio.**

**How to verify:** load the page with headphones on and the system volume up. If anything makes a sound, press Tab up to five times and see whether you reach a control that stops it. Console: `[...document.querySelectorAll('audio,video')].filter(m=>m.autoplay && !m.muted)`.

---

## 4. Pause, Stop, Hide — 2.2.2 (A)

Any moving, blinking or scrolling content that **starts automatically, lasts more than 5 seconds, and is presented in parallel with other content** must be pausable, stoppable or hideable by the user. The same applies to **auto-updating** content (live feeds, tickers, auto-refreshing dashboards, rotating notifications).

| Pattern | What it needs |
|---|---|
| Carousel / hero slider | A visible, keyboard-reachable **pause/stop** control. Auto-advance without one is a straight 2.2.2 failure. |
| Marquee / news ticker | Pause control, or don't auto-scroll. |
| Auto-refreshing feed or results list | Pause control, or a "load new items" button the user presses. |
| Animated background video | Pause control, or make it stop within 5 seconds. |
| Animated GIF longer than 5 seconds | Replace with a `<video>` that has controls, or a static image. |

**How to verify:** watch the page for 10 seconds without touching anything. List everything that moves or changes. For each, Tab to see whether a pause control exists and works. Console: `document.getAnimations().filter(a=>a.playState==='running')` shows CSS/WAAPI animations still running.

---

## 5. Flashing — 2.3.1 (A), 2.3.2 (AAA)

| SC | Level | Requirement |
|---|---|---|
| 2.3.1 Three Flashes or Below Threshold | A | Nothing flashes more than **3 times per second**, unless the flash is below the general flash and red flash thresholds. |
| 2.3.2 Three Flashes | AAA | Nothing flashes more than 3 times per second, full stop — no threshold exception. |

This is a **safety criterion, not an aesthetic one**. Photosensitive epilepsy seizures are a real and immediate harm. Treat any finding here as Tier 1 regardless of page traffic.

**How to verify:** watch any rapidly changing content — strobing hero animations, flashing "sale" badges, video content with lightning/camera-flash sequences, loading spinners with high-contrast blink. For video, run the **PEAT** (Photosensitive Epilepsy Analysis Tool) or the Harding test on the file. For CSS, look for `animation` on `opacity`, `background-color` or `visibility` with a `duration` under ~333 ms and `infinite` iteration:
```js
[...document.getAnimations()].map(a=>[a.effect?.target, a.effect?.getTiming().duration, a.effect?.getTiming().iterations])
  .filter(([,d,i])=>d<334 && i===Infinity)
```

---

## 6. Animation from interactions — 2.3.3 (AAA) and `prefers-reduced-motion`

**2.3.3 Animation from Interactions** (AAA, since 2.1): motion animation triggered by an interaction can be disabled, unless the animation is essential.

`prefers-reduced-motion` is an **implementation technique**, not a success criterion. Honouring it is how you satisfy 2.3.3, and it is good practice regardless — but note that:
- Satisfying 2.3.3 does **not** require the media query specifically; a site-level "reduce motion" setting also works.
- Honouring the media query does **not** by itself satisfy 2.2.2 — a carousel that stops for reduced-motion users still needs a pause control for everyone else.
- 2.3.3 is AAA, so it is not part of an AA claim. Report it as a recommendation unless the client has adopted it individually (`targets.md`).

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
  }
}
```

Parallax scrolling, large scroll-triggered transitions, and full-page transition wipes are the motion patterns that actually make people ill. They are also the ones most likely to be implemented in JS rather than CSS, where the media query does nothing unless the script checks `window.matchMedia('(prefers-reduced-motion: reduce)').matches`.

**How to verify:** DevTools → Rendering → **Emulate CSS media feature `prefers-reduced-motion: reduce`**, then reload and repeat every interaction. JS-driven animation will keep running — that is the finding. Grep the source: `grep -rn "prefers-reduced-motion" ./` should appear in **both** CSS and JS if there is any JS animation.

---

## 7. Timing — 2.2.1 (A) and relatives

**2.2.1 Timing Adjustable** (A): for each time limit set by the content, at least one of —

| Option | Detail |
|---|---|
| **Turn off** | The user can switch the limit off before encountering it. |
| **Adjust** | The user can adjust it to at least **10×** the default, before encountering it. |
| **Extend** | The user is **warned before time expires**, given at least **20 seconds** to extend with a simple action (e.g. "press the space bar"), and can extend at least **10 times**. |

Exceptions: real-time events (an auction, a live exam), where the limit is essential, or where it exceeds 20 hours.

Related: **2.2.5 Re-authenticating** (AAA) — data is preserved across re-authentication. **2.2.6 Timeouts** (AAA) — users are warned about data-loss timeouts unless the data is preserved for **20+ hours**. **2.2.3 No Timing** (AAA) — no time limits at all except real-time events. **2.2.4 Interruptions** (AAA) — interruptions can be postponed or suppressed.

Session timeouts on government and banking forms are the everyday case: a 15-minute idle timeout that silently discards a half-completed 40-field application fails 2.2.1 and is the single most-complained-about behaviour in public-sector accessibility feedback channels.

**How to verify:** open a form, fill two fields, leave it idle past the documented session timeout. Does a warning appear? Is it announced (it needs a live region — see `html-core.md` §11)? Is there ≥20 seconds to respond? Does the data survive re-login? Check the server session length in the response headers or the client-side timer: `grep -rn "setTimeout\|sessionTimeout\|idleTimeout" ./` in the front-end source.

---

## 8. Media and motion criterion map

| SC | Level | Implementation |
|---|---|---|
| 1.2.1 | A | Transcript adjacent to or linked from audio/video-only content |
| 1.2.2 | A | `<track kind="captions" srclang="en" label="English" src="…vtt">`, human-verified |
| 1.2.3 | A | Audio description track **or** full text alternative |
| 1.2.4 | AA | Live captioning service integration |
| 1.2.5 | AA | `<track kind="descriptions">` or a described-video version |
| 1.4.2 | A | No autoplay; or a pause control within the first tab stops |
| 1.4.5 | AA | Real text + web fonts, not rendered images |
| 2.2.1 | A | Session timeout warning with an extend option |
| 2.2.2 | A | Pause/stop on carousels, marquees, auto-updating feeds |
| 2.3.1 | A | No flashing >3 Hz |
| 2.5.1 | A | Single-pointer alternative for swipe/path gestures in media players and galleries |
| 4.1.3 | AA | Live region for "playing", "paused", "5 new items" status changes |

**Note:** captions, transcripts and audio description apply to **video embedded in documents too** — a video in a PowerPoint deck still needs captions. See `documents-office.md`.
