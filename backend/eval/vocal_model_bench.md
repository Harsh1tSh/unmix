# Stage 1 vocal model benchmark

x realtime = inference seconds per second of audio (lower is faster). SDR in dB, higher is better.

### 2026-11-17 09:24 - 2 clips (14s), 4 threads, device cpu

| model | load s | infer s | x realtime | vocals SDR | accompaniment SDR |
|---|---|---|---|---|---|
| model_bs_roformer_ep_317_sdr_12.9755.ckpt | 15.0 | 266.2 | 19.57 | 12.48 | 14.90 |
| Kim_Vocal_2.onnx | 68.3 | 45.1 | 3.32 | 9.43 | 11.85 |
