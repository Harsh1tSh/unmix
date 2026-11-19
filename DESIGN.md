# Unmix: design strategy

The brief the frontend is built to. It is written as a senior product designer would brief a UI engineer: what the product should feel like, where that comes from, the system that delivers it, and the rules that keep it fast and accessible. Read it before changing anything visual. Tokens live in `frontend/src/index.css`; the sphere lives in `frontend/src/visual/`.

## 1. What we are designing for

Unmix takes one recording and pulls it apart into its sources: voices, drums, bass, guitar, piano, other, and optionally your voice vs. the singer's. Separation runs for seconds (a GPU) to many minutes (a laptop CPU). The person spends their time on three screens:

1. **Landing**: record by holding something, or drop a file. One decision (what to give it), a few options.
2. **Unmixing**: waiting. Often for minutes. This screen has to be calm and show steady progress.
3. **Stem board**: listening. Hover a card to hear one source, solo/mute, read the meters, follow lyrics.

The product's single idea is **one thing becoming many**. Every visual decision should express that idea or stay out of its way.

## 2. What we take from the reference sites

| Site | What works | How it is built | What we take |
|---|---|---|---|
| melon.studio | Atmosphere. Huge light-weight type (172 px, weight 300, -5 % tracking), warm near-black, a single saturated colour field that bleeds through grain, liquid SVG filters, cards that bend in WebGL on scroll | Next.js + three.js, Lenis smooth scroll, canvas grain, `feTurbulence`/`feDisplacementMap` "liquid" filters, `color-dodge` blending, Geist Mono for small print | Light display weights at very large sizes; mono for metadata; grain; colour as a *field* that glows through darkness, not as flat fills |
| preymaker.com | Restraint. Black, white, one typeface; the content (video) is the only colour | Next.js, Lenis, full-bleed video, `backdrop-filter: blur(10px)` on chrome | Chrome stays monochrome so the content can carry the colour. For us the content is the sound, so **colour means a source** |
| genesisfxmarkets.com | The colour system and the spheres. Cool blue-black (`#05070d`), cool white text (`#e8ecf5`), one luminous hue used as light: glows, rims, gradients that fade to nothing. A particle globe made of thousands of points | three.js; points sampled on a mesh (`MeshSurfaceSampler`), simplex-noise displacement, fresnel rim, `AdditiveBlending`, bloom; glass cards = `linear-gradient(white 8% → 2%)` + 1 px white/8 % border + a radial glow of the hue at a corner; luminous pill buttons | Base colours; the glass-card recipe; luminous primary button; **a point-cloud sphere with a bright rim and additive light** |
| hamol.studio | Spheric animation and composition. Faceted spheres floating in space, big light headline on the left, the object on the right, floating capsule navigation, small tracked-out uppercase labels, a horizon of light at the bottom | three.js, `MeshPhysicalMaterial`, Manrope at weight 400 with -5.7 % tracking, `backdrop-filter: blur(14px)`, corner radial gradients | Hero composition (type left, object right); Manrope display; capsule nav and a bottom capsule dock; horizon glow; eyebrow labels |

What we do **not** take: scroll-jacking and smooth-scroll libraries (this is a tool, not a showcase), full-page WebGL on every screen (a 4-core laptop also runs the separation), and decorative 3D models.

## 3. Concept: a sphere of sound, pulled apart

The landing page shows the mix as a **sphere made of thousands of points of light**. Its latitude bands are the seven frequency bands of the analysis: sub-bass at the bottom pole, brilliance at the top.

- **Idle**: the sphere is silver-white, turning slowly, breathing. A faint spectrum shows only at its rim.
- **Recording** (hold the sphere): the bands light up in their colours, and the surface of each band swells with the live energy of that band from the microphone. Loudness swells the whole sphere.
- **Unmixing**: the sphere glides from the landing into the progress screen (it is one object drawn by one app-wide scene, so it really travels). As the job progresses, its seven bands bud off one after another into seven small round spheres on a ring facing the camera: one sphere becoming many, at the rate the job really progresses. (A first version split the sphere into horizontal slices; seen at an angle they read as broken ellipses, so it was replaced.)
- **Done**: the mix waveform splits into one coloured strand per stem, and the strands fly into their cards (the existing split animation).

Colour rule, everywhere: **interface chrome is monochrome; colour only ever stands for sound.** A pink element means the voices, an amber one the drums. The one exception is the *aurora*, a gradient made of the stem colours themselves, used as the brand signature (headline accent word, the progress fill, the active stage).

### Space

The page is not a flat black: it is a sky. A nebula (domain-warped noise in the aurora tints, painted once at low resolution, so it costs nothing per frame), three depth layers of stars that twinkle, drift and shift with the pointer (parallax), a few bright stars with diffraction spikes, the planet's horizon on the landing and Live screens, and an occasional shooting star on the landing. On the stem board the sky is frozen and dimmer: that screen is for listening and reading meters, and motion behind data competes with it.

## 4. Design system

### Colour tokens

| Token | Value | Use |
|---|---|---|
| `ink` | `#05060a` | page background (cool blue-black) |
| `surface` | `#0b0d13` | panels |
| `raised` | `#12151d` | inputs, wells |
| `line` / `line-strong` | white 7 % / 14 % | hairlines, borders |
| `text` | `#eef0f5` | primary text (cool white) |
| `muted` | `#8f96a6` | secondary text |
| `faint` | `#565d6d` | tertiary text, disabled |
| `danger` | `#ff5c6c` | delete, errors, live dot |
| `ok` / `warn` | `#4ade80` / `#fbbf24` | status dots only |
| stem colours | from the API (`backend/app/config.py`) | each source's identity |
| aurora | `#FF6B9A → #E879F9 → #A78BFA → #38BDF8 → #34D399` | brand gradient, sparingly |

Contrast: `text` on `ink` 17:1, `muted` on `ink` 6.6:1 (passes AA for body), `faint` is for non-essential text only.

### Type

| Role | Font | Spec |
|---|---|---|
| Display | Manrope Variable | weight 300 to 400, tracking -0.045em, line-height 0.95; `clamp(2.75rem, 7.2vw, 6.75rem)` for the hero |
| UI / body | Inter Variable | 400/500; 15 to 17 px body, 13 px small |
| Data / labels | Geist Mono Variable | times, Hz, percentages, eyebrows; eyebrows are 11 px uppercase with 0.22em tracking |

All three are bundled with the app (`@fontsource-variable/*`): the CSP allows fonts from the same origin only. Tabular figures only where numbers change in place (`.num`); never on the whole page (it puts spaces around hyphens in file names).

### Surfaces

- **Glass card** (stem cards, panels): `linear-gradient(180deg, white 5.5 %, white 1.5 %)` over `surface`, 1 px `line` border, radius 24 px, a top inner highlight, and a radial glow of the card's stem colour from the top-left corner at 12 %. Hover: the glow rises to ~26 %, the border takes the stem colour, the card lifts and tilts 2 to 4 degrees toward the cursor.
- **Floating chrome** (navigation capsule, status capsule, options dock): the same gradient with `backdrop-filter: blur(20px) saturate(1.4)` (`.glass`). Backdrop blur is used **only** on these few fixed elements, never on cards. The transport dock is the exception: it floats over meters and waveforms that change every frame, so a live blur would be recomputed every frame; it uses `.glass-solid` (the same look with a 92 % opaque fill).
- **Buttons**: primary = luminous white pill (white-to-silver gradient, a soft white outer glow, ink text). Secondary = ghost pill (1 px `line-strong`). Destructive = danger fill, only after a confirmation step. Icon buttons are 36 px circles.

### Motion

| Token | Value | Use |
|---|---|---|
| `ease-out-expo` | `cubic-bezier(0.16, 1, 0.3, 1)` | entrances, hover lift |
| `ease-in-out` | `cubic-bezier(0.65, 0, 0.35, 1)` | the split animation's flights |
| durations | 160 to 240 ms (micro), 600 to 900 ms (entrances) | |

- Entrances: headline lines rise from behind a mask, staggered 80 ms; cards rise 18 px and fade in, staggered 50 ms. Once per screen, never on scroll.
- Ambient motion is slow (sphere ~1 revolution per 2.5 minutes, breathing over 4 s) so that it reads as alive, not busy.
- Grain is static. An animated full-screen grain layer was tried and measured: together with the transport's live blur it dropped ~20 % of frames on the board (headless Chromium, software compositing). Both were removed; the board now drops almost none.
- `prefers-reduced-motion`: no entrances, no tilt, no ambient rotation; the sphere draws one still frame (and still reacts to the microphone while recording, since that is information, not decoration).

## 5. Screens

### Landing

- **Top**: wordmark left; a glass status capsule ("Ready · GPU (Tesla T4) · remote") and a "Go live" pill right.
- **Hero**: on wide screens, type on the left and the sphere on the right (hamol composition). Eyebrow ("Source separation · runs on your machine"), the headline in light Manrope with "apart by source." in the aurora, one line of supporting copy, then the actions: a luminous **Choose a file** pill and "or drop it anywhere". The sphere is the record button: "Hold to record · tap to start and stop" under it in mono.
- **Bottom**: the options sit in one glass dock (Fast/Quality segmented control + three switches) with the hints on hover; a horizon glow (a wide ellipse of aurora light) rises from the bottom edge.
- **Phone (375 px)**: sphere first (it is the main action), headline and actions below it, options dock wraps and stays reachable without overlapping anything.

### Unmixing

- The sphere sits on the right (above the content on phones) in "separating" mode, its rings pulling apart with real progress.
- Left: eyebrow with the elapsed time, the file name in light display type, the mix waveform filling with the aurora as the job advances, the message and a large mono percentage, then the stages as a vertical timeline (done = solid node with a check, active = glowing pulsing node, to do = hollow).
- The footnote about speed reflects the actual machine ("on this GPU", "on a CPU this takes several times the clip length").

### Stem board

- A floating glass navigation capsule: wordmark, file name, Delete (with confirm), New.
- Header: eyebrow, "7 sources" in light display type with mono facts beside it, and a **mix composition bar**: one segment per played source, sized by its share of the mix's energy and coloured by the source (hover shows the name and share).
- Cards: the glass card recipe; label in display type, the energy share as a mono figure; waveform, spectrogram strip and the 7-band meter (thinner bars with a gradient and a peak cap). Hover = listen stays the core interaction.
- "Inside Music": the children branch from their parent with a line in the parent's colour.
- Transport: a floating glass dock at the bottom, centred, with a luminous play button.

### Live

Same atmosphere and components: eyebrow with a pulsing red dot, light display headline, a glass panel with the mic meter and the two lanes, luminous / ghost / danger pills.

## 6. Engineering rules for the visuals

1. **One WebGL context for the whole app, raw WebGL2**, no three.js. One full-screen canvas behind the interface (`SpaceBackdrop`) draws the stars and the sphere; screens only place an anchor (`<SoundSphere>`) and the sphere eases to it. Creating a WebGL canvas when the screen changed showed a white box for a few frames on Intel/ANGLE; with one long-lived context nothing is created on a screen change. ~9,000 sphere points (each with a second target on its band's small sphere, blended in the vertex shader) plus a dust shell and ~1,000 stars; additive blending; the canvas is kept under ~5 megapixels. Measured on an Intel UHD 620: 60 fps on every screen.
2. **Fallback**: without WebGL2 the same canvas draws a static sky and the ringed 2D orb. A lost context is restored and rebuilt. The app must work identically.
3. **Pause when not visible**: the render loop stops when the tab is hidden or the canvas is scrolled away.
4. **Per-frame values never go through React state** (as in the rest of the app: `onFrame` and CSS variables).
5. **Keep every `data-testid`** and the structure the Playwright tests rely on; the redesign changes looks, not behaviour.
6. Contrast AA for text; visible focus (white ring); every icon button has an accessible name; status changes go to `aria-live` regions; the active stage carries `aria-current="step"`.

## 7. Flaws found in the previous UI (fixed in this pass)

1. Tabular figures were switched on for the whole page, so Inter put wide gaps around hyphens and punctuation ("talk - over - song.wav").
2. Every stem card recomputed its "strongest band" summary over all analysis frames on every re-render (each hover, play or pause re-renders all cards).
3. The transport said "Listening to speaker 1 only" (the internal stem name), ignoring the labels and the "That's me" choice.
4. The progress screen always said "on a CPU this takes several times the clip length", even with a GPU or the Colab backend.
5. A job restored from the URL showed "your recording" instead of its file name, which the API returns.
6. On a 375 px screen the options were below the fold and the orb's rings ran into the text above them.
7. The landing page ran two independent animation loops (particles and orb) and kept running in a background tab.
8. The page's drag-and-drop listeners were removed and re-added on every render.
9. Switches used the voice colour (pink) for "on", which reads as "voices" under the colour rule.
10. The active stage and the progress bar had no accessible name or state.
11. "Clean up this voice" wrapped onto two lines on phones.

Second round, after using it: the plain black background became space; the progress sphere splits into round spheres instead of slices; Live has a visible Back button (and Esc); a white box that flashed when the screen changed after recording (a new WebGL canvas on Intel/ANGLE) is gone with the single scene; the focus ring no longer forces an 8 px radius (it drew a square around the round orb); large screens (2560, ultrawide) get a centred frame and larger type, short windows size the sphere by height; `body` no longer paints over the fixed background layers.

And one found while building: the first version of this redesign dropped frames on the board (animated grain + a live blur over the meters). Measured, then fixed (see Motion and Surfaces).
