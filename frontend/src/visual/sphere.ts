/**
 * The sound sphere: a point cloud on a sphere, drawn with raw WebGL2.
 *
 * Latitude bands of the sphere are the seven analysis bands (sub-bass at the
 * bottom pole ... brilliance at the top). Fibonacci sampling gives equal-area
 * spacing, so every band holds the same number of points.
 *
 * Inputs per frame (all eased on the CPU, never through React state):
 *   levels[7]  band energy 0..1 (microphone while recording; a slow idle swell otherwise)
 *   loudness   0..1, swells the whole sphere
 *   color      0..1, how much of the band colours shows (idle: only at the rim)
 *   split      0..1, the bands detach into rings that drift apart and spin (progress screen)
 *
 * Look: soft round sprites with additive blending, so the limb (where points
 * pile up in projection) glows on its own, like a fresnel rim; front points
 * are brighter than the back ones; a shell of dust drifts around it.
 */

export const BAND_COLORS = ["#EF4444", "#F59E0B", "#FF6B9A", "#E879F9", "#A78BFA", "#38BDF8", "#34D399"] as const;
const BANDS = BAND_COLORS.length;
const SPHERE_POINTS = 9000;
const DUST_POINTS = 650;
const CAM_Z = 3.6;
const ORBIT = 1.02; // radius of the ring the seven band spheres sit on
const SAT_R = 0.31; // radius of each band sphere

export interface SphereFrame {
  time: number;
  levels: Float32Array;
  loudness: number;
  color: number;
  split: number;
  /** per-band spin angle (radians), integrated by the caller while splitting */
  phases: Float32Array;
  /** angle of the ring of band spheres (radians), integrated by the caller */
  orbit: number;
  /** yaw / pitch of the whole sphere (radians) */
  yaw: number;
  pitch: number;
  /** extra scale, e.g. on hover */
  scale: number;
}

const VERT = /* glsl */ `#version 300 es
precision highp float;
in vec3 aPos;
in vec3 aData; // seed, band, kind (0 sphere, 1 dust)
in vec3 aSat;  // where the point sits on its band's own small sphere (unit vector)
uniform float uTime;
uniform float uLevels[${BANDS}];
uniform float uPhases[${BANDS}];
uniform vec3 uBandColor[${BANDS}];
uniform float uLoud;
uniform float uColor;
uniform float uSplit;
uniform float uOrbit;
uniform float uScale;
uniform float uRadiusPx;
uniform float uPointPx;
uniform vec2 uHalf;
uniform vec2 uCenter; // where the sphere sits, in buffer pixels from the canvas centre (y up)
uniform float uAlpha;
uniform mat3 uRot;
out vec4 vColor;

vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+10.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
// 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT licence)
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);
  const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));
  vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);
  vec3 l=1.0-g;
  vec3 i1=min(g.xyz,l.zxy);
  vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;
  vec3 x2=x0-i2+C.yyy;
  vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;
  vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z);
  vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;
  vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);
  vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;
  vec4 s1=floor(b1)*2.0+1.0;
  vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
  vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);
  vec3 p1=vec3(a0.zw,h.y);
  vec3 p2=vec3(a1.xy,h.z);
  vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.5-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
  m=m*m;
  return 105.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

vec3 rotY(vec3 p, float a){ float c=cos(a), s=sin(a); return vec3(c*p.x+s*p.z, p.y, -s*p.x+c*p.z); }

void main(){
  float seed = aData.x;
  int band = int(aData.y + 0.5);
  bool dust = aData.z > 0.5;
  float lvl = uLevels[band];
  vec3 v;      // view-space position
  vec3 nrm;    // view-space surface normal (for the rim light)
  float size;
  float alpha;
  vec3 col;
  float e = 0.0; // how far this point has moved into its band's own sphere
  vec3 silver = vec3(0.80, 0.84, 0.93);
  vec3 bandCol = uBandColor[band];

  if (!dust) {
    // the whole sphere: slow organic drift + band energy pushing the surface out
    float n1 = snoise(aPos * 1.6 + vec3(0.0, uTime * 0.22, uTime * 0.07));
    float n2 = snoise(aPos * 4.2 + vec3(uTime * 0.55, 0.0, -uTime * 0.31));
    float r = 1.0 + n1 * 0.03 + (n2 * 0.5 + 0.5) * lvl * 0.28 + uLoud * 0.08;
    vec3 whole = uRot * (aPos * r);

    // separation: each band streams out into a small sphere of its own; the seven
    // sit on a ring facing the camera (so it always reads round), budding off one
    // band after another as the split progresses
    float b = float(band);
    e = smoothstep(b * 0.075, b * 0.075 + 0.48, uSplit);
    float ang = uOrbit + b * 0.8975979 + 1.5707963; // 2*pi/7 apart, first one on top
    vec3 center = vec3(cos(ang) * ${ORBIT.toFixed(2)}, sin(ang) * ${ORBIT.toFixed(2)}, sin(ang * 2.0 + uOrbit) * 0.12);
    float sn = snoise(aSat * 2.4 + vec3(uTime * 0.4, b, 0.0));
    vec3 own = uRot * rotY(aSat, uPhases[band]) * (${SAT_R.toFixed(2)} * (1.0 + sn * 0.05)) + center;
    // on the way out the points swirl a little instead of moving in straight lines
    vec3 swirl = vec3(sn, snoise(aPos * 3.0 + uTime * 0.3), 0.0) * 0.18 * sin(3.14159 * e);
    v = mix(whole, own, e) + swirl;
    nrm = normalize(mix(uRot * aPos, uRot * aSat, e));

    size = uPointPx * (0.9 + seed * 0.5 + lvl * 1.6);
    float tint = clamp(uColor * (0.22 + lvl * 1.5) + e * 0.9, 0.0, 1.0);
    col = mix(silver, bandCol, tint);
    col = mix(col, vec3(1.0), lvl * 0.22);
    alpha = 0.34 + lvl * 0.6 + e * 0.1;
  } else {
    // dust shell: drifts slowly around the sphere and twinkles
    vec3 p = rotY(aPos, uTime * (0.015 + seed * 0.02));
    p.y += sin(uTime * 0.25 + seed * 6.2831) * 0.04;
    v = uRot * p;
    nrm = normalize(v);
    size = uPointPx * (0.5 + seed * 0.9);
    col = mix(silver, bandCol, 0.35 + uColor * 0.4);
    alpha = (0.10 + 0.22 * (0.5 + 0.5 * sin(uTime * (0.6 + seed) + seed * 40.0))) * (0.6 + uColor * 0.6);
  }

  float z = ${CAM_Z.toFixed(1)} - v.z;
  float persp = ${CAM_Z.toFixed(1)} / z;
  vec2 px = v.xy * persp * uRadiusPx * uScale;
  gl_Position = vec4((px + uCenter) / uHalf, 0.0, 1.0);

  // front points brighter; the limb glows (rim light from the surface normal)
  float front = smoothstep(-1.0, 1.0, nrm.z);
  float rim = 1.0 - abs(nrm.z);
  if (!dust) {
    alpha *= (0.28 + 0.72 * front);
    alpha += rim * rim * 0.4 * (0.6 + uColor * 0.4);
    // rim picks up the band colour even at idle (the faint spectrum at the edge)
    col = mix(col, bandCol, rim * rim * (0.35 + uColor * 0.4));
  }
  gl_PointSize = max(1.0, size * persp);
  alpha *= uAlpha;
  vColor = vec4(col * alpha, alpha);
}
`;

const FRAG = /* glsl */ `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 outColor;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d = dot(c, c);
  if (d > 1.0) discard;
  float a = 1.0 - d;
  outColor = vColor * a * a;
}
`;

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Deterministic pseudo-random numbers so the sphere looks the same every load. */
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

function fibonacci(k: number, n: number): [number, number, number] {
  const y = 1 - (2 * (k + 0.5)) / n;
  const r = Math.sqrt(1 - y * y);
  const th = Math.PI * (3 - Math.sqrt(5)) * k;
  return [Math.cos(th) * r, y, Math.sin(th) * r];
}

/**
 * Positions on the big sphere (aPos), seed/band/kind (aData), and each point's
 * place on its band's own small sphere (aSat). Fibonacci points come out sorted
 * by height, so every band is one contiguous run of indices.
 */
function buildPoints(): { pos: Float32Array; data: Float32Array; sat: Float32Array; count: number } {
  const count = SPHERE_POINTS + DUST_POINTS;
  const pos = new Float32Array(count * 3);
  const data = new Float32Array(count * 3);
  const sat = new Float32Array(count * 3);
  const rand = mulberry32(7);
  const bandOf = (i: number) => Math.min(BANDS - 1, Math.floor(((1 - (2 * (i + 0.5)) / SPHERE_POINTS + 1) / 2) * BANDS));
  const firstOf = new Array<number>(BANDS).fill(-1);
  const countOf = new Array<number>(BANDS).fill(0);
  for (let i = 0; i < SPHERE_POINTS; i++) {
    const b = bandOf(i);
    if (firstOf[b] === -1) firstOf[b] = i;
    countOf[b]! += 1;
  }
  for (let i = 0; i < SPHERE_POINTS; i++) {
    const band = bandOf(i);
    pos.set(fibonacci(i, SPHERE_POINTS), i * 3);
    sat.set(fibonacci(i - firstOf[band]!, countOf[band]!), i * 3);
    data.set([rand(), band, 0], i * 3);
  }
  for (let k = 0; k < DUST_POINTS; k++) {
    const i = SPHERE_POINTS + k;
    const u = rand() * 2 - 1;
    const th = rand() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    const dist = 1.3 + Math.pow(rand(), 1.6) * 1.25;
    pos.set([Math.cos(th) * r * dist, u * dist * 0.8, Math.sin(th) * r * dist], i * 3);
    data.set([rand(), Math.floor(rand() * BANDS), 1], i * 3);
  }
  return { pos, data, sat, count };
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`sphere shader: ${log}`);
  }
  return shader;
}

export class SphereRenderer {
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly buffers: WebGLBuffer[];
  private readonly count: number;
  private readonly u: Record<string, WebGLUniformLocation | null>;
  /** Builds the sphere in an existing WebGL2 context (the app's one scene canvas). */
  static create(gl: WebGL2RenderingContext): SphereRenderer | null {
    try {
      return new SphereRenderer(gl);
    } catch (err) {
      console.warn(err);
      return null;
    }
  }

  private constructor(private readonly gl: WebGL2RenderingContext) {
    const program = gl.createProgram()!;
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`sphere program: ${gl.getProgramInfoLog(program)}`);
    this.program = program;

    const { pos, data, sat, count } = buildPoints();
    this.count = count;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const attributes: [string, Float32Array][] = [
      ["aPos", pos],
      ["aData", data],
      ["aSat", sat],
    ];
    this.buffers = attributes.map(([name, array]) => {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      return buf;
    });
    gl.bindVertexArray(null);

    const names = ["uTime", "uLevels", "uPhases", "uBandColor", "uLoud", "uColor", "uSplit", "uOrbit", "uScale", "uRadiusPx", "uPointPx", "uHalf", "uCenter", "uAlpha", "uRot"];
    this.u = Object.fromEntries(names.map((n) => [n, gl.getUniformLocation(program, n)]));
    gl.useProgram(program);
    gl.uniform3fv(this.u.uBandColor!, new Float32Array(BAND_COLORS.flatMap(hexToRgb)));
  }

  /**
   * Draws into the current framebuffer (the caller sets the viewport, clears and
   * enables additive blending). `view` places the sphere: centre in CSS pixels
   * from the canvas' top-left, radius in CSS pixels, opacity 0..1.
   */
  render(f: SphereFrame, view: { x: number; y: number; radius: number; alpha: number; dpr: number }): void {
    const { gl, u } = this;
    const cw = gl.drawingBufferWidth;
    const ch = gl.drawingBufferHeight;
    gl.useProgram(this.program);

    // the whole group must still fit in the same circle when the bands spread out
    const radius = (view.radius * view.dpr) / (1 + f.split * 0.45);
    const cy = Math.cos(f.yaw);
    const sy = Math.sin(f.yaw);
    const cp = Math.cos(f.pitch);
    const sp = Math.sin(f.pitch);
    // R = Rx(pitch) * Ry(yaw), column-major for uniformMatrix3fv
    const rot = new Float32Array([cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy]);

    gl.uniform1f(u.uTime!, f.time);
    gl.uniform1fv(u.uLevels!, f.levels);
    gl.uniform1fv(u.uPhases!, f.phases);
    gl.uniform1f(u.uLoud!, f.loudness);
    gl.uniform1f(u.uColor!, f.color);
    gl.uniform1f(u.uSplit!, f.split);
    gl.uniform1f(u.uOrbit!, f.orbit);
    gl.uniform1f(u.uScale!, f.scale);
    gl.uniform1f(u.uRadiusPx!, radius);
    // points scale a little with the sphere's size, within limits
    gl.uniform1f(u.uPointPx!, 2.2 * view.dpr * Math.max(0.75, Math.min(1.25, view.radius / 150)));
    gl.uniform2f(u.uHalf!, cw / 2, ch / 2);
    gl.uniform2f(u.uCenter!, view.x * view.dpr - cw / 2, ch / 2 - view.y * view.dpr);
    gl.uniform1f(u.uAlpha!, view.alpha);
    gl.uniformMatrix3fv(u.uRot!, false, rot);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const { gl } = this;
    this.buffers.forEach((b) => gl.deleteBuffer(b));
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
    // No loseContext() here: React's dev double mount reuses the same canvas,
    // and a canvas whose context was lost on purpose can never draw again.
  }
}
