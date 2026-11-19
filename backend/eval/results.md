# Separation quality (SDR)

- Date: 2026-11-17 09:51
- Dataset: MUSDB18 7-second preview set (test subset, lossy AAC stems) - not MUSDB18-HQ
- Clips: 5 (34 s of audio), mode `fast`, device `cpu`
- Models: Stage 1 `Kim_Vocal_2.onnx`, Stage 2 `htdemucs_6s.yaml`
- Metric: SDR in dB (BSSEval v4 via `museval`, median over 1 s frames). Higher is better;
  roughly: < 3 dB poor, 5-7 dB usable, > 8 dB good. `other` = everything that is not vocals/drums/bass.
- Wall time: 344 s including scoring

| clip | vocals | drums | bass | other | separation s |
|---|---|---|---|---|---|
| AM Contra - Heart Peripheral | 11.38 | 8.07 | 4.07 | 0.46 | 50 |
| Al James - Schoolboy Facination | 8.28 | 6.29 | 13.24 | 4.52 | 48 |
| Angels In Amplifiers - I'm Alright | 11.82 | 6.78 | 10.14 | 7.71 | 49 |
| Arise - Run Run Run | 10.92 | 9.96 | 0.63 | 1.29 | 52 |
| BKS - Bulldozer | 7.74 | 10.46 | 7.85 | 5.27 | 45 |
| **median** | **10.92** | **8.07** | **7.85** | **4.52** | |
| **mean** | **10.03** | **8.31** | **7.19** | **3.85** | |

Reproduce: see the docstring of `eval/run_eval.py`.
