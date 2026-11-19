/**
 * The space behind the app: a nebula and a star field.
 *
 * Nebula: domain-warped value-noise clouds in the aurora tints, computed ONCE on
 * the CPU at low resolution and scaled up by the browser. It costs nothing per
 * frame (full-screen noise shaders are the expensive part of most "space"
 * backgrounds and weak laptop GPUs feel it).
 *
 * Stars: three depth layers of points in WebGL2. Twinkle, a slow drift and the
 * pointer parallax are computed in the vertex shader, so a frame is one draw call.
 * A few bright stars get small diffraction spikes.
 */

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ nebula

function valueNoise(seed: number): (x: number, y: number) => number {
  const rand = mulberry32(seed);
  const values = Float32Array.from({ length: 256 }, rand);
  const perm = Uint8Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [perm[i], perm[j]] = [perm[j]!, perm[i]!];
  }
  const at = (x: number, y: number) => values[perm[(perm[x & 255]! + y) & 255]!]!;
  return (x, y) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const u = xf * xf * (3 - 2 * xf);
    const v = yf * yf * (3 - 2 * yf);
    const a = at(xi, yi);
    const b = at(xi + 1, yi);
    const c = at(xi, yi + 1);
    const d = at(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Paints the nebula into `canvas` (a few hundred pixels wide; CSS scales it up). */
export function paintNebula(canvas: HTMLCanvasElement, aspect: number, seed = 11): void {
  const w = 320;
  const h = Math.max(120, Math.round(w / Math.max(0.4, aspect)));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const n1 = valueNoise(seed);
  const n2 = valueNoise(seed + 1);
  const fbm = (noise: (x: number, y: number) => number, x: number, y: number) => {
    let sum = 0;
    let amp = 0.5;
    let f = 1;
    for (let o = 0; o < 5; o++) {
      sum += amp * noise(x * f, y * f);
      f *= 2.03;
      amp *= 0.5;
    }
    return sum;
  };
  const img = ctx.createImageData(w, h);
  // true purples only (hue ~280-290): over the blue-black base, blue-violets and cyan read as a blue patch
  const violet = [180, 90, 236];
  const magenta = [232, 121, 249];
  const lavender = [222, 182, 248];
  const scale = 3.2 / w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = x * scale;
      const sy = y * scale;
      // domain warp: wispy, filament-like clouds instead of round blobs
      const qx = fbm(n1, sx, sy);
      const qy = fbm(n1, sx + 5.2, sy + 1.3);
      const n = fbm(n2, sx + 2.4 * qx, sy + 2.4 * qy);
      const tint = fbm(n1, sx * 0.6 + 9.1, sy * 0.6 + 2.7);
      // keep the clouds toward the edges and corners; calm where the text sits
      const ux = x / w;
      const uy = y / h;
      const edge = Math.max(smooth(0.35, 1.0, Math.hypot(ux - 0.3, uy - 0.45) * 1.25), 0.15);
      const density = smooth(0.42, 0.82, n) * edge;
      const c = tint < 0.5 ? mix3(violet, magenta, smooth(0.3, 0.5, tint)) : mix3(violet, lavender, smooth(0.5, 0.7, tint));
      const i = (y * w + x) * 4;
      img.data[i] = c[0]!;
      img.data[i + 1] = c[1]!;
      img.data[i + 2] = c[2]!;
      img.data[i + 3] = Math.round(density * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
}

function mix3(a: number[], b: number[], t: number): number[] {
  return a.map((v, k) => v + (b[k]! - v) * t);
}

// ------------------------------------------------------------------- stars

const STAR_VERT = /* glsl */ `#version 300 es
precision highp float;
in vec4 aStar; // x, y (0..1), depth (0 far .. 1 near), seed
uniform float uTime;
uniform vec2 uParallax; // pointer, -1..1
uniform float uDpr;
uniform float uDim;
out vec4 vColor;
out float vSpike;
void main(){
  float d = aStar.z;
  float seed = aStar.w;
  // slow drift and pointer parallax; near stars move more
  vec2 p = aStar.xy + vec2(uTime * 0.0018 * (0.25 + d), 0.0) - uParallax * (0.003 + d * 0.016);
  p = fract(p);
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);

  float twinkle = 0.62 + 0.38 * sin(uTime * (0.5 + fract(seed * 13.7) * 2.4) + seed * 61.0);
  bool bright = fract(seed * 97.3) > 0.985 && d > 0.6;
  float size = mix(0.8, 2.3, d * d) * (bright ? 2.6 : 1.0);
  gl_PointSize = size * uDpr * (bright ? 3.0 : 1.6);
  // colour temperature: blue-white to warm white, a few in the aurora hues
  vec3 cool = vec3(0.72, 0.80, 1.0);
  vec3 warm = vec3(1.0, 0.88, 0.78);
  vec3 col = mix(cool, warm, fract(seed * 7.31));
  float hue = fract(seed * 3.91);
  if (hue > 0.93) col = mix(col, vec3(1.0, 0.42, 0.6), 0.45);
  else if (hue > 0.86) col = mix(col, vec3(0.65, 0.55, 1.0), 0.45);
  float alpha = mix(0.22, 0.95, d) * twinkle * uDim;
  // thin out toward the bottom edge, where the planet's horizon rises on some screens
  alpha *= 1.0 - smoothstep(0.8, 0.97, p.y) * 0.9;
  vColor = vec4(col * alpha, alpha);
  vSpike = bright ? 1.0 : 0.0;
}
`;

const STAR_FRAG = /* glsl */ `#version 300 es
precision mediump float;
in vec4 vColor;
in float vSpike;
out vec4 outColor;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r = length(c);
  if (r > 1.0) discard;
  float core = vSpike > 0.5 ? exp(-r * 9.0) : (1.0 - r) * (1.0 - r);
  float spikes = vSpike * (max(0.0, 1.0 - abs(c.y) * 14.0) + max(0.0, 1.0 - abs(c.x) * 14.0)) * (1.0 - r) * 0.55;
  outColor = vColor * min(1.0, core + spikes);
}
`;

export class StarRenderer {
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly buffer: WebGLBuffer;
  private readonly count: number;
  private readonly u: Record<string, WebGLUniformLocation | null>;
  /** Builds the star field in an existing WebGL2 context (the app's one scene canvas). */
  static create(gl: WebGL2RenderingContext, count: number): StarRenderer | null {
    try {
      return new StarRenderer(gl, count);
    } catch (err) {
      console.warn(err);
      return null;
    }
  }

  private constructor(
    private readonly gl: WebGL2RenderingContext,
    count: number,
  ) {
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`stars shader: ${gl.getShaderInfoLog(sh)}`);
      return sh;
    };
    const program = gl.createProgram()!;
    const vs = compile(gl.VERTEX_SHADER, STAR_VERT);
    const fs = compile(gl.FRAGMENT_SHADER, STAR_FRAG);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`stars program: ${gl.getProgramInfoLog(program)}`);
    this.program = program;

    // Far stars are many and faint, near ones few and bright.
    const rand = mulberry32(3);
    const data = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) data.set([rand(), rand(), Math.pow(rand(), 2.4), rand()], i * 4);
    this.count = count;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    this.buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, "aStar");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.u = Object.fromEntries(["uTime", "uParallax", "uDpr", "uDim"].map((n) => [n, gl.getUniformLocation(program, n)]));
  }

  /** Draws into the current framebuffer (the caller sets the viewport, clears and blends). */
  render(time: number, parallaxX: number, parallaxY: number, dim: number, dpr: number): void {
    const { gl, u } = this;
    gl.useProgram(this.program);
    gl.uniform1f(u.uTime!, time);
    gl.uniform2f(u.uParallax!, parallaxX, parallaxY);
    gl.uniform1f(u.uDpr!, dpr);
    gl.uniform1f(u.uDim!, dim);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.program);
  }
}
