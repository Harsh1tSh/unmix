# Security and privacy

Unmix is a single-user app that runs on your own computer and listens only on `127.0.0.1`. It handles private material: recordings of your voice, separated stems and transcripts of what you said. This file records how it is protected, what a security review on 2026-11-17 found, and how each fix is tested.

## Threat model

The realistic attackers for a local app are not on the network. They are:

1. **Other web pages open in the same browser.** Any site can make the browser send requests to `127.0.0.1`.
2. **Untrusted audio files.** A file you were sent can be crafted to exploit the decoder.
3. **Tampered model downloads.** Weights come from third-party hosts.
4. **Anything that copies your data off the machine:** telemetry, logs, caches, error messages.
5. **Remote GPU mode only** (`colab/unmix_gpu.ipynb`): anyone on the internet who finds the tunnel address.

## Findings and fixes

| # | Finding | Severity | Fix | Test |
|---|---|---|---|---|
| 1 | **Cross-site request forgery.** Any web page could make the browser upload files and start jobs on the local API (a multipart POST is a "simple" request with no CORS preflight). Measured: `201 Created` with `Origin: https://evil.example`. | High | Requests that change state are refused when `Origin` is not localhost, or `Sec-Fetch-Site` is `cross-site` (403). | `test_foreign_origin_cannot_create_or_delete_jobs` |
| 2 | **DNS rebinding.** A page on `attacker.example` that re-points its domain to `127.0.0.1` could read the API, including your stems and lyrics, as same-origin. Measured: `200 OK` with `Host: attacker.example`. | High | `Host` header allowlist: localhost, 127.0.0.1, [::1] (421 otherwise). | `test_dns_rebinding_host_is_refused` |
| 3 | **Decoder input not constrained.** ffmpeg auto-detected the input format, so playlist / concat / reference formats could be steered into reading other local files and serving them back as "your" audio. ffmpeg 9's defaults happened to block the variants tried, but nothing in Unmix did. | High (latent) | The container is identified from its magic bytes and only wav, aiff, flac, ogg, Matroska/WebM, MP4/M4A, MP3 and ADTS-AAC are accepted. That demuxer is then forced (`-f`), with `-protocol_whitelist file`. Anything else is refused before ffmpeg runs. | `test_playlists_and_references_are_refused`, `test_unknown_container_with_audio_extension_is_refused`, `test_sniffing` |
| 4 | **Disk exhaustion.** The multipart parser stores the whole request body in a temp file before the endpoint can check its size, so the 200 MB limit could be bypassed by size. | Medium | Body size is capped in middleware: refused up front from `Content-Length`, and counted while streaming for chunked uploads (413). | `test_oversized_body_is_refused_before_it_is_stored` |
| 5 | **Unbounded queue.** Any number of jobs could be queued (each holds a decoded file on disk). | Medium | At most `UNMIX_MAX_WAITING_JOBS` (3) queued or running jobs; then 429. | `test_waiting_queue_is_capped` |
| 6 | **Internal details in error messages.** Failed jobs and failed model loads sent raw exception text (which can contain local paths) to the browser. | Low | Clients get a short message plus an error id; details go to the server log only. ffmpeg messages are scrubbed of paths. | `test_failed_job_does_not_leak_internals`, `test_health_reports_model_failure`, `test_scrub_removes_paths` |
| 7 | **Recordings left in the browser cache.** Stems, analysis and lyrics were served with `max-age=3600`, so copies outlived deletion. | Low | `Cache-Control: no-store` on all job data. A **Delete** button removes a job's files immediately (otherwise after 24 h). | `test_security_headers_and_private_caching`, e2e "Delete removes the job" |
| 8 | **No security headers.** | Low | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`, a deny-all CSP on API responses. The built frontend ships a strict CSP (same-origin only, no inline scripts). | `test_security_headers_and_private_caching` |
| 9 | **Telemetry.** The Hugging Face client (used to download Whisper and MossFormer2) sends usage telemetry and any stored token unless told not to. | Low | `HF_HUB_DISABLE_TELEMETRY`, `HF_HUB_DISABLE_IMPLICIT_TOKEN` and `DO_NOT_TRACK` are set at import, and in the voice-split subprocess. | n/a |
| 10 | **Unpinned model downloads.** Weights were trusted on first download; the Hugging Face models followed the latest revision. | Low (defence in depth) | SHA-256 pins for every weight file (`backend/app/model_hashes.json`), checked before loading (cached by size+mtime); Whisper and MossFormer2 pinned to exact revisions. `UNMIX_STRICT_MODELS=1` also refuses unpinned files. | `test_tampered_model_file_is_refused` |
| 11 | **API docs pulled from a CDN.** `/docs` loads Swagger UI from jsDelivr, so opening it contacts a third party. | Info | Off by default; `UNMIX_API_DOCS=1` enables it. | `test_api_docs_are_off_by_default` |
| 12 | **Untrusted file name displayed and logged.** | Info | Reduced to a display name: no directories (either separator), no control characters, at most 120 characters. | `test_uploaded_filename_is_reduced_to_a_display_name` |
| 14 | **Cross-site WebSocket hijacking** (found while building live mode). Browsers do not apply CORS to WebSockets, so any page could open `/api/live` and stream audio in or read results out. | High | The guard middleware now covers WebSocket handshakes: a local `Origin` is *required* and the `Host` must be local (closed with 1008 otherwise). Frames are size-capped, must be finite, and sessions are time-limited. | `test_cross_site_websockets_are_refused`, `test_bad_start_and_malformed_frames` |
| 15 | **Model file served to the browser** (browser-only mode). | Info | One allowlisted file, hash-verified before serving; no directory access. | `test_model_route_serves_only_the_allowlisted_verified_model` |
| 16 | **Remote GPU mode puts the API on a public address** (a Cloudflare quick tunnel to Colab). The API has no accounts by design. | High (by construction) | The server refuses to start in remote mode without an access key of at least 24 characters (the notebook generates 43 random URL-safe characters). Every request and WebSocket must carry it in `X-Unmix-Key` (constant-time comparison; 401 / close 1008 otherwise). Only the local frontend proxy holds the key and adds it, so no page script can read it. The tunnel's host name is allowlisted exactly; the Origin and Host checks still apply end to end, because the proxy forwards the browser's `Origin` (another site in your browser still gets 403). | `test_requests_without_the_key_are_refused`, `test_tunnel_host_is_allowed_but_other_protections_stay`, `test_websocket_needs_the_key`, `test_short_keys_are_rejected` |
| 17 | **What the notebook downloads and runs.** | Medium | cloudflared is pinned to a release older than two weeks and checked against its published SHA-256 before it runs; uv is pinned; `uv sync --frozen` installs exactly `uv.lock`. Model files cached in Google Drive are verified against their pins before use, like local ones. The GitHub token for a private repo comes from Colab Secrets and is sent as a one-off header: it is never in the clone URL, `.git/config` or the cell output. | review |
| 13 | **Progress could freeze** when the live progress stream died (the client polled once and stopped). Not a vulnerability, but an API failure mode. | Bug | Falls back to polling with backoff until the job ends. | e2e suite |

## Checked and found fine

- **Path traversal:** job ids must be 32 hex characters; stem and spectrogram names must be in the job's own list; files are never addressed by user input.
- **SQL:** all values are parameterised.
- **Subprocesses** (ffmpeg, the voice-split runner): argument lists, never a shell.
- **Model loading:** torch loads weights with `weights_only=True` (or Demucs' restricted unpickler), so a model file cannot run code. ClearerVoice relies on torch's default, which is safe from torch 2.6 on (2.15 is installed).
- **Network calls by libraries:** audio-separator only contacts its model hosts (GitHub releases, Meta's CDN). No analytics anywhere.
- **Browser-only mode:** the audio stays in a Web Worker; only the model is fetched (from the local server). The production CSP allows exactly what it needs: `'wasm-unsafe-eval'` (WebAssembly compilation, not JavaScript `eval`) and `blob:` for the stems it creates. Verified with zero CSP violations on the production build.
- **Live mode:** nothing from a session is written to disk; the recording stays in the browser unless you choose "Unmix fully".
- **Frontend:** React escapes all text (no `dangerouslySetInnerHTML`); no third-party requests (the font is bundled); the mic is opened only while recording; localStorage holds only UI preferences.
- **Known vulnerabilities in dependencies:** `pip-audit` (both Python environments) and `npm audit` report none (2026-11-17).

## Things to know

- **Data at rest.** Uploads, stems and transcripts live unencrypted under `backend/data/jobs/` for 24 h (or until you press Delete). Anyone with access to your user account can read them.
- **Downloads contact third parties.** First use of a model downloads it from GitHub, Meta's CDN or Hugging Face, which reveals your IP address to them. After that Unmix works offline.
- **Do not expose the server.** It has no accounts by design. Do not bind it to `0.0.0.0` or forward its port. The only supported way to reach it remotely is the Colab notebook, which turns on the access key.
- **Remote GPU mode sends your audio to Google and through Cloudflare.** The tunnel is encrypted (HTTPS) and Colab deletes everything when the session ends, but the audio does leave your computer, and those companies' terms apply. Local mode stays the default. Treat the printed access key like a password for that session; a new session gets a new key and address.
- **GPU drivers.** Unmix measures GPU back ends in a separate process before using them, and moves to the CPU if a GPU fails during a job, so a faulty driver costs speed, not results.

## Re-running the checks

```bash
cd backend && uv run pytest tests/test_security.py -v
cd backend && uv export --no-hashes --all-groups --format requirements-txt > req.txt && uvx pip-audit -r req.txt
cd frontend && npm audit
```
