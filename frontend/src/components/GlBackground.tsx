import { useEffect, useRef } from 'react'
import { Mesh, Program, Renderer, Triangle } from 'ogl'

const VERTEX = /* glsl */ `
attribute vec2 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}`

// Slow silk: noise warped by noise, in the page's colours, kept pale so text on it stays easy to read
const FRAGMENT = /* glsl */ `
precision highp float;
uniform float uTime;
uniform vec2 uSize;
uniform vec2 uPointer;
varying vec2 vUv;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float total = 0.0;
  float amount = 0.5;
  for (int k = 0; k < 4; k++) {
    total += amount * noise(p);
    p = p * 2.03 + vec2(17.0, 9.0);
    amount *= 0.5;
  }
  return total;
}
void main() {
  vec2 p = vUv * vec2(uSize.x / uSize.y, 1.0) * 1.5;
  float t = uTime * 0.035;
  p += (uPointer - 0.5) * 0.08;
  vec2 q = vec2(fbm(p + vec2(t, -t * 0.6)), fbm(p + vec2(5.2, 1.3) - t));
  vec2 r = vec2(fbm(p + 2.2 * q + vec2(1.7, 9.2) + 0.7 * t), fbm(p + 2.2 * q + vec2(8.3, 2.8) - 0.5 * t));
  float f = fbm(p + 2.4 * r);

  // SightSeen's palette, washed out: paper, a breath of steel blue, a breath of tape yellow
  vec3 base = vec3(0.957, 0.965, 0.976);
  vec3 steel = vec3(0.84, 0.875, 0.93);
  vec3 tape = vec3(1.0, 0.93, 0.70);
  vec3 ink = vec3(0.80, 0.83, 0.89);
  vec3 colour = mix(base, steel, smoothstep(0.28, 0.72, q.x) * 0.7);
  colour = mix(colour, tape, smoothstep(0.36, 0.78, r.y) * 0.45);
  colour = mix(colour, ink, smoothstep(0.4, 0.84, f) * 0.35);
  colour = mix(colour, base, 0.3);
  colour += (hash(gl_FragCoord.xy + fract(uTime)) - 0.5) * 0.008; // a little grain, so the soft bands don't show
  gl_FragColor = vec4(colour, 1.0);
}`

const RESOLUTION = 0.5 // of the screen: it's all soft, and this keeps a laptop's fans quiet
const FRAME = 1000 / 30 // milliseconds: 30 frames a second is plenty for something this slow

/** The landing page's and the workspace's background: slow silk drawn in WebGL (ogl). Without WebGL, the page's CSS
 * gradient shows instead. It stops when the tab's hidden, and holds still for anyone who's asked for less motion. */
export default function GlBackground() {
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = host.current
    if (!el) return
    let renderer: Renderer
    try {
      renderer = new Renderer({ dpr: 1, antialias: false, alpha: false, powerPreference: 'low-power' })
    } catch {
      return
    }
    const gl = renderer.gl
    if (!gl) return
    const canvas = gl.canvas as HTMLCanvasElement
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    canvas.style.display = 'block'
    el.appendChild(canvas)

    const uniforms = {
      uTime: { value: Math.random() * 500 }, // each visit starts on its own pattern
      uSize: { value: [1, 1] },
      uPointer: { value: [0.5, 0.5] },
    }
    const mesh = new Mesh(gl, { geometry: new Triangle(gl), program: new Program(gl, { vertex: VERTEX, fragment: FRAGMENT, uniforms }) })
    const fit = () => {
      const width = Math.max(1, Math.round(window.innerWidth * RESOLUTION))
      const height = Math.max(1, Math.round(window.innerHeight * RESOLUTION))
      renderer.setSize(width, height)
      canvas.style.width = '100%'
      canvas.style.height = '100%'
      uniforms.uSize.value = [width, height]
      renderer.render({ scene: mesh })
    }
    fit()
    window.addEventListener('resize', fit)

    const still = matchMedia('(prefers-reduced-motion: reduce)').matches
    const pointer = [0.5, 0.5]
    const onPointer = (event: PointerEvent) => {
      pointer[0] = event.clientX / window.innerWidth
      pointer[1] = 1 - event.clientY / window.innerHeight
    }
    window.addEventListener('pointermove', onPointer, { passive: true })

    let frame = 0
    let last = 0
    const draw = (now: number) => {
      frame = requestAnimationFrame(draw)
      if (document.hidden || now - last < FRAME) return
      uniforms.uTime.value += (now - last) / 1000
      last = now
      uniforms.uPointer.value = [
        uniforms.uPointer.value[0] + (pointer[0] - uniforms.uPointer.value[0]) * 0.03,
        uniforms.uPointer.value[1] + (pointer[1] - uniforms.uPointer.value[1]) * 0.03,
      ]
      renderer.render({ scene: mesh })
    }
    if (!still) frame = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', fit)
      window.removeEventListener('pointermove', onPointer)
      canvas.remove()
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }, [])

  return <div ref={host} aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10" />
}
