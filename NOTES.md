# How source separation works

Notes to learn from. Everything here maps onto code in this repo; file references are at the end of each section.

## 1. The problem

A recording is one waveform: air pressure over time. If you talk while a song plays, the microphone captures the sum

```
mix(t) = voice(t) + drums(t) + bass(t) + guitar(t) + ...
```

Separation means recovering the terms from only their sum. That is under-determined (infinitely many sets of signals add up to the same mix), so it can only be done with prior knowledge of what a voice or a drum kit sounds like. Neural networks learn that prior from thousands of songs for which the individual tracks are available.

## 2. Why not just use filters (an equaliser)?

A filter splits by **frequency**: "everything below 250 Hz", "everything from 300 to 3400 Hz". Sources do not respect those borders:

- A voice has a fundamental around 100–250 Hz and harmonics up to 8 kHz and beyond.
- A guitar, a piano and a snare drum all have energy across that same range.
- A bass guitar's string noise reaches several kHz; a kick drum's click too.

So a "voice band" filter keeps the guitar that lives in that band and throws away the voice's breath and consonants above it. This repo contains exactly such a filter bank as `FakeBackend` (used only for tests): listen to its output once and the difference from the real models is obvious.

What Unmix does instead: separate by **source** with neural models, and only afterwards describe each separated stem by **frequency** (the 7-band meters and spectrograms on each card). The bands are a description, not the separation method.

## 3. The spectrogram: time and frequency at once (STFT)

A waveform shows *when* things are loud but not *which pitches* are sounding. The Fourier transform shows which frequencies are in a signal but not when. The **short-time Fourier transform (STFT)** gives both:

1. Cut the signal into short overlapping frames (e.g. 4096 samples ≈ 93 ms, a new frame every 1024 samples).
2. Multiply each frame by a window (Hann) that fades it in and out, so the cuts do not create artificial clicks.
3. Take the FFT of each frame.

The result is a 2-D grid of complex numbers `X[f, t]`: one column per frame (time), one row per frequency bin. Each number has

- a **magnitude** `|X|`: how much of frequency `f` is present at time `t`,
- a **phase**: where in its cycle that sinusoid is.

Plotting the magnitude in dB gives the familiar spectrogram image. In it, sources look different: a sung note is a stack of thin horizontal lines (harmonics) that wobble with vibrato; a drum hit is a vertical stripe (all frequencies at once, very briefly); a bass note is a few strong lines at the bottom. That visual difference is what a network learns to exploit.

Trade-off to remember: a longer frame gives finer frequency resolution but blurrier timing, and vice versa. Frequency bin width = `sample_rate / n_fft` (44100 / 4096 ≈ 10.8 Hz here, which is why the analysis can resolve a 20–60 Hz sub-bass band at all).

*Code: `backend/app/analysis.py` (`_power_spectrogram`).*

## 4. Masks

Most cells of the grid are dominated by one source at a time: music is **sparse** in time-frequency. So separation can be posed as: for every cell, what fraction belongs to the source I want?

```
V̂[f, t] = M[f, t] · X[f, t]
```

`M` is a **mask**, the same shape as the spectrogram.

- **Binary mask**: 0 or 1 per cell. Simple, but produces "musical noise" (watery artefacts).
- **Ratio (soft) mask**: a value between 0 and 1. If the true magnitudes were known, the ideal ratio mask would be `|V| / (|V| + |rest|)`. Much smoother.
- **Complex mask**: a complex number per cell, so it can also rotate the *phase*. This matters where two sources overlap in the same cell: the mix's phase there is wrong for both of them, and a magnitude-only mask leaves that error in.

The network's job is **mask prediction**: input the mix spectrogram, output a mask per source. It is trained on songs with known stems by comparing its estimate with the true stem and adjusting weights to shrink the difference.

The models used here:

| Model | Idea |
|---|---|
| **BS-RoFormer** (Band-Split RoFormer, Stage 1 on GPU / quality) | Splits the frequency axis into ~60 bands of musically chosen widths, embeds each band, then alternates Transformer layers along *time* and along *bands* (with rotary position embeddings). Predicts a **complex** mask. Among the best published vocal separators (the checkpoint's file name records 12.98 dB vocal SDR on its authors' validation set; it measured 12.5 dB on two MUSDB clips here). |
| **MDX-Net** (Stage 1 on CPU in fast mode) | A U-Net (convolutional encoder/decoder) over the spectrogram. Older and less accurate than BS-RoFormer (about 3 dB lower SDR in `backend/eval/vocal_model_bench.md`), but about 6× cheaper to run. |
| **HT-Demucs** (Stage 2) | A **hybrid**: one U-Net branch works on the spectrogram, a second works directly on the raw waveform, and a Transformer in the middle lets them exchange information. It outputs waveforms, so it is not limited by masking. `htdemucs_6s` is the 6-source variant (adds guitar and piano; piano is its weakest stem). `htdemucs_ft` is a "bag" of four models, each fine-tuned for one source. |

*Code: `backend/app/separation.py` chooses and runs the models.*

## 5. Back to audio: the inverse STFT

Apply the mask, then invert the STFT: inverse-FFT each column back into a short frame, apply the window again and **overlap-add** the frames. With a suitable window and hop (the "COLA" condition, e.g. Hann with 75 % overlap) the overlapping windows sum to a constant, so an unmodified spectrogram reconstructs the original signal exactly.

Long songs do not fit through the network in one piece. They are cut into chunks of a few seconds with overlap, each chunk is separated, and the chunk outputs are cross-faded together (another overlap-add, one level up). More overlap = smoother joins = proportionally more compute. That is the `overlap` setting in `separation.py`, and the reason progress arrives chunk by chunk.

## 6. The residual trick, and why the stems here sum to the mix

Unmix does not ask the model for "the music". It asks only for the voices and computes

```
music = mix − voices
other = music − (drums + bass + guitar + piano)
```

Consequences:

- **Nothing is lost.** Every sample of the input ends up in exactly one leaf stem, so `voices + drums + bass + guitar + piano + other = mix` to within rounding. Playing all stems together *is* the original; muting a stem removes exactly that stem.
- Whatever the models fail to classify lands in `other` rather than disappearing.
- This is what lets the player keep everything in sync with plain gain changes (see `frontend/src/audio/engine.ts`).

The cost: any error in the voice estimate appears with opposite sign in the music ("bleed").

*Code: `backend/app/pipeline.py` (`VoicesStep`, `MusicStep`); checked by `test_stems_sum_back_to_mix`.*

## 7. Measuring quality: SDR

To score a separator you need the true stem `s` (which is why evaluation uses datasets like MUSDB18, where the multitrack is available) and the estimate `ŝ`. The **signal-to-distortion ratio** is

```
SDR = 10 · log10( ‖s‖² / ‖s − ŝ‖² )   [dB]
```

i.e. the energy of the true signal relative to the energy of everything that is wrong in the estimate (leftover bleed from other sources, artefacts, missing parts).

- It is logarithmic: +3 dB means the error energy halved; +10 dB means it fell tenfold.
- Rough feel for music: below ~3 dB is poor, 5–7 dB is usable, above ~8 dB sounds clean, and the best vocal models reach 11–13 dB on MUSDB18.
- 0 dB means the error is as loud as the signal. Outputting silence scores 0 dB, not minus infinity, so low scores can be worse than nothing.

Variants you will see: BSSEval additionally decomposes the error into **SIR** (interference from the other sources) and **SAR** (artefacts the algorithm introduced). `museval` computes these per 1-second window and reports the **median** over windows, which is the MUSDB convention; a plain whole-clip SDR (as in the formula above) gives somewhat different numbers. SI-SDR is a scale-invariant version common in speech separation.

Limits: SDR is not perception. A stem with slightly more bleed but fewer artefacts can sound better at the same SDR. Always listen as well.

*Code: `backend/eval/run_eval.py` (museval) and `backend/eval/bench_vocal_models.py` (plain SDR). Results: `backend/eval/results.md`.*

## 8. Loudness normalisation

Inputs arrive at wildly different levels (a quiet phone recording, a loud master). Before separation the mix is scaled by **one constant gain** so its integrated loudness is −14 LUFS (ITU-R BS.1770, a perceptually weighted average), with the peak capped at −1 dBFS. A constant gain changes nothing about the content, so it cannot hurt the models; it just puts every input in the level range they were trained on and makes analysis values comparable between jobs. Dynamic processing (compression) is deliberately avoided, because it would change the mix in ways the models never saw.

*Code: `backend/app/audio_io.py` (`loudness_normalize`).*

## 9. What the cards show

| Display | What it is |
|---|---|
| Waveform | Max absolute sample in each of 2000 time buckets. All cards share one vertical scale, so quiet stems look quiet. |
| Spectrogram strip | Mel spectrogram: the STFT's frequency axis warped to the mel scale (roughly logarithmic, like hearing), in dB. |
| 7-band meter | STFT power summed inside each band (sub-bass 20–60 Hz … brilliance 6–20 kHz) at 20 frames per second, in dB relative to the loudest band-frame of any stem in the job. |
| Energy share | `Σ stem² / Σ mix²`. Sibling stems add up to roughly their parent's share (exactly, if they were uncorrelated). |

## 10. Speech over music: what to expect

The vocal models were trained on singing. They transfer well to speech, with typical failure modes worth knowing: backing vocals and the lead singer land in `voices` together with your speech (splitting *your* voice from the singer is Stage 3, a different kind of model: speaker separation); reverb tails of the voice partly stay in `music`; very quiet speech under loud music loses consonants.

## 11. Voice vs. voice: speaker separation

Splitting voices from music works because the two look different in a spectrogram. Two voices do not: both are harmonic stacks in the same range. So the speaker model (MossFormer2) learns a different cue, which is *continuity*. A voice's pitch, timbre and formants change smoothly, so the model groups time-frequency cells that belong to the same continuing voice. It is trained on thousands of artificial two-person mixtures ("permutation-invariant training": it is scored on whichever pairing of its two outputs with the two true speakers is best, because it cannot know which is "first").

Two consequences you can see in the app:

- **The outputs have no names.** `speaker_1` is simply the louder of the two; the model cannot know which one is you.
- **It runs at 16 kHz.** It was trained on 16 kHz speech, so its outputs contain nothing above 8 kHz. Instead of using them directly, Unmix uses them as a *mask* (section 4) on the real 44.1 kHz voices stem: below 7.5 kHz each cell is split in the ratio of the two estimates, and above that the per-frame ratio is extended upward. The speaker stems therefore keep their full brightness, and `speaker_1 + speaker_2 = voices` exactly.

Measured on 20 s of speech over a song: the "you" stem matches the clean speech at 7.2 dB SI-SDR, up from 3.1 dB in the voices stem before splitting (where the singer was still mixed in). The other stem contained no measurable speech.

*Code: `backend/app/speakers.py`, `backend/speakers/run_split.py`.*

## 12. Lyrics with word timing

Whisper is a sequence-to-sequence model: an encoder turns 30-second windows of log-mel spectrogram into features, and a decoder writes text token by token, attending to those features. Word timestamps come from that attention. For each output token, the cross-attention weights say which audio frames it was looking at. Aligning tokens to frames with dynamic time warping gives each word a start and end time.

Choices made here, and why:

- **Transcribe the separated voices, not the mix.** Whisper was trained on speech; a backing track mostly adds errors. With split voices, each speaker is transcribed on its own, because Whisper follows only the dominant voice in a mixture: on the voices stem of "you talking over a song" it wrote down every word of the speech and none of the singing.
- **No voice-activity filter.** Silero VAD is tuned for speech and clips sung phrases.
- **`condition_on_previous_text=False`.** Otherwise one misheard line can be fed back as context and repeat for the rest of a song.
- **Bleed removal.** A loud voice leaks a little into the other speaker's stem and can be transcribed twice. Lines from different speakers that overlap in time and read alike are deduplicated, keeping the more confident copy.

Sung lyrics remain harder than speech: stretched vowels, melisma and effects all hurt. Expect speech to be nearly word-perfect and singing to be approximately right.

*Code: `backend/app/lyrics.py`; display in `frontend/src/components/LyricsPanel.tsx`.*

## 13. Streaming and in-browser separation

Both phase 4 modes reuse one idea from the offline MDX code: the model always sees a fixed window of 261 120 samples (256 STFT frames), but only its **middle** 253 440 samples are kept. The 3 840 samples on each side are context: the network needs to hear a little before and after a moment to judge it, and the spectrogram's edge frames are less reliable. Consecutive windows overlap by exactly that context, so the kept parts tile the signal with no seams.

- **Live:** the server cannot output a block until it has heard its right-hand context, so the minimum delay is one block (5.75 s) plus the time to compute it. With compute ≈ block length on a laptop CPU, that is ~12 s.
- **In the browser:** the same windows, but the STFT has to be rebuilt in JavaScript. FFT size 7680 is not a power of two (7680 = 2⁹·3·5), so it uses a mixed-radix FFT (radix 4, 2, 3, 5). "The model gets the same numbers" is checked directly: the JS spectrogram matches `torch.stft` to better than 1 part in 10 000.

*Code: `backend/app/live.py`, `frontend/src/local/mdx.ts`, `frontend/src/local/fft.ts`.*

## 14. Running on a GPU: measure, don't assume

A separation model is mostly large matrix multiplications and convolutions, which a GPU does in parallel. Three different runtimes run Unmix's models, and each reaches the GPU differently:

| Runtime | Models | GPU route |
|---|---|---|
| PyTorch | BS-RoFormer, HT-Demucs, MossFormer2 | CUDA (NVIDIA) or MPS (Apple) |
| ONNX Runtime | Kim Vocal 2 (fast voices on CPU machines, live mode) | CUDA, or DirectML: any DirectX 12 GPU on Windows |
| CTranslate2 | Whisper | CUDA only |

"Is there a GPU?" is the wrong question; "does this model run correctly and faster on it?" is the right one. A laptop can list a GPU that its PyTorch build cannot use (too old), that has too little memory, or whose driver crashes. So the server *measures*: it runs one real window of the voice model on each available back end in a throwaway process (if the driver takes that process down, the server survives), checks the output matches the CPU's, and keeps the fastest. On the development laptop this picks the GeForce MX150 through DirectML (1.1 s per window against 5.3–8.2 s on the CPU) and skips the Intel iGPU, whose DirectML driver fails.

Two practical lessons from that laptop:

- **One session per model.** A second DirectML session in the same process made the MX150 stop responding (`887A0006`). Fast mode and live mode therefore share one session, and calls to it are serialised.
- **Fall back, then stay back.** When a GPU fails mid-job (out of memory, driver reset), that step reruns on the CPU and the server stays on the CPU until restarted: a driver that failed once tends to fail again, and correct-but-slower beats fast-but-flaky.

Why the voices step on the MX150 only got 2.5–3× faster when the model itself got 5–7× faster: the STFT, the overlap-add and the file I/O around the model stay on the CPU. Live mode benefits most, because there the model call dominates: 0.27× real time instead of ~1×, so the delay falls to its floor of one block plus compute (~7.5 s).

*Code: `backend/app/accel.py`, `backend/scripts/pick_accel.py`.*
