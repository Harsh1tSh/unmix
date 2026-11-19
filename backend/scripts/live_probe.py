"""Stream a file to the live endpoint at real-time pace (like a microphone)
and report the separation speed and the delay you would hear.

Usage:  uv run python scripts/live_probe.py <audio.wav> [--url ws://127.0.0.1:8000/api/live]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets

SR = 44100
FRAME = 4096  # what the browser's AudioWorklet sends


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/api/live")
    args = ap.parse_args()
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    assert sr == SR, "resample to 44.1 kHz first"
    audio = audio[:, :2] if audio.shape[1] >= 2 else np.repeat(audio, 2, axis=1)

    async with websockets.connect(args.url, origin="http://localhost:5173", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": SR, "channels": 2}))
        print(await ws.recv())
        t_start = time.perf_counter()
        arrivals = []

        async def sender() -> None:
            for i in range(0, len(audio), FRAME):
                await ws.send(np.ascontiguousarray(audio[i : i + FRAME], dtype="<f4").tobytes())
                # real-time pace: wait until this audio "has been spoken"
                await asyncio.sleep(max(0.0, t_start + (i + FRAME) / SR - time.perf_counter()))
            await ws.send(json.dumps({"type": "stop"}))

        send_task = asyncio.create_task(sender())
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data["type"] == "block":
                now = time.perf_counter() - t_start
                end_of_audio = (data["start"] + data["frames"]) / SR
                arrivals.append((data["index"], now, end_of_audio, data["compute_s"], data["frames"] / SR))
                print(f"block {data['index']}: audio up to {end_of_audio:5.1f}s arrived at {now:5.1f}s "
                      f"(compute {data['compute_s']:.2f}s for {data['frames'] / SR:.2f}s -> {data['compute_s'] / (data['frames'] / SR):.2f}x)")
            elif data["type"] in ("behind", "error"):
                print(data)
            elif data["type"] == "end":
                break
        await send_task
    full = [a for a in arrivals if a[4] > 5]
    if full:
        speed = np.mean([a[3] / a[4] for a in full])
        # delay for the first sample of each block = arrival time - capture time of that sample
        delays = [a[1] - (a[2] - a[4]) for a in full]
        print(f"\nmean speed {speed:.2f}x real time; delay from speaking to hearing: {min(delays):.1f}-{max(delays):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
