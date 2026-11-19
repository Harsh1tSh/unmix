/**
 * AudioWorklet that forwards microphone audio to the main thread as
 * interleaved float32 frames, batched to ~4096 frames per message (93 ms at
 * 44.1 kHz) so the WebSocket is not flooded with 128-frame packets.
 */

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor();
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor): void;

const BATCH = 4096;

class CaptureProcessor extends AudioWorkletProcessor {
  private buf = new Float32Array(BATCH * 2);
  private filled = 0;

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const left = input[0]!;
    const right = input[1] ?? left; // mono mic: duplicate
    for (let i = 0; i < left.length; i++) {
      this.buf[this.filled * 2] = left[i]!;
      this.buf[this.filled * 2 + 1] = right[i]!;
      this.filled++;
      if (this.filled === BATCH) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(BATCH * 2);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("unmix-capture", CaptureProcessor);
export {};
