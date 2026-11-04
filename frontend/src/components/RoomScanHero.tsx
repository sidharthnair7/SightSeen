import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import ShinyText from './ShinyText'

/**
 * The landing page's live 3D: a real room rebuilt by SightSeen's own pipeline (the stock hotel tour standing in for a
 * stolen listing), as its point cloud in the room's measured frame with the ceiling cut away. On load a scan sweeps
 * across it and the points fall into place, then it sways slowly; drag turns it. The tape tags are the room's
 * measured size, drawn as dimension lines on its floor.
 *
 * The file (worker-made, public/hero/room-scan.bin): "SSN1", point count, the box's size (x, y, z in metres), the
 * positions' extent, then int16 positions over +-extent, then uint8 colours.
 */
export default function RoomScanHero({ src, className = '' }: { src: string; className?: string }) {
  const mount = useRef<HTMLDivElement>(null)
  const tags = useRef<(HTMLSpanElement | null)[]>([])
  const [state, setState] = useState<'loading' | 'scanning' | 'ready' | 'failed'>('loading')
  const [size, setSize] = useState<[number, number, number] | null>(null)

  useEffect(() => {
    const host = mount.current
    if (!host) return
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
    } catch {
      setState('failed')
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x000000, 0)
    const canvas = renderer.domElement
    canvas.style.cssText = 'display:block;width:100%;height:100%;cursor:grab;touch-action:pan-y'
    host.prepend(canvas)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100)
    const target = new THREE.Vector3()
    const anchors: THREE.Vector3[] = []
    let box = new THREE.Vector3(5, 2.6, 4)
    let material: THREE.ShaderMaterial | null = null
    let geometry: THREE.BufferGeometry | null = null
    let lines: THREE.LineSegments | null = null
    let disposed = false
    let raf = 0
    let visible = false
    let scanStart = Infinity
    const startScan = () => {
      if (material && visible && scanStart === Infinity) scanStart = performance.now()
    }
    let radius = 10
    // drag turns it; a pointer resting over it leans it towards the pointer (both eased)
    const view = { drag: 0, dragTo: 0, down: false, lastX: 0, hx: 0, hy: 0, hxTo: 0, hyTo: 0, near: 0, nearTo: 0 }

    const fit = () => {
      const width = Math.max(1, host.clientWidth)
      const height = Math.max(1, host.clientHeight)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      // far enough that the whole room fits whichever way the panel is shaped
      const span = Math.hypot(box.x, box.z) * 0.56
      const half = THREE.MathUtils.degToRad(camera.fov / 2)
      radius = span / Math.tan(half) / Math.min(1, camera.aspect * 1.05)
      if (material) material.uniforms.uPixel.value = renderer.getPixelRatio() * height / (2 * Math.tan(half))
      wake()
    }

    const place = (now: number) => {
      view.hx += (view.hxTo - view.hx) * 0.06
      view.hy += (view.hyTo - view.hy) * 0.06
      view.near += (view.nearTo - view.near) * 0.05
      const sway = still ? 0 : Math.sin(now / 1000 * 0.16) * 0.42 * (1 - view.near)
      const yaw = THREE.MathUtils.degToRad(318) + sway + view.drag + view.hx * 0.6
      const pitch = THREE.MathUtils.degToRad(31) + view.hy * 0.14
      camera.position.set(
        target.x + radius * Math.sin(yaw) * Math.cos(pitch),
        target.y + radius * Math.sin(pitch),
        target.z + radius * Math.cos(yaw) * Math.cos(pitch),
      )
      camera.lookAt(target)
    }

    const frame = (now: number) => {
      raf = 0
      if (disposed) return
      view.drag += (view.dragTo - view.drag) * 0.12
      place(now)
      if (material) {
        const scan = still ? 1 : Math.max(0, Math.min(1, (now - scanStart) / 2600))
        material.uniforms.uScan.value = 1 - Math.pow(1 - scan, 2.2)
        if (scan >= 1 && state !== 'ready') setState('ready')
      }
      renderer.render(scene, camera)
      // the tape tags follow their dimension lines
      const width = host.clientWidth
      const height = host.clientHeight
      anchors.forEach((anchor, i) => {
        const tag = tags.current[i]
        if (!tag) return
        const p = anchor.clone().project(camera)
        tag.style.transform = `translate(-50%, -50%) translate(${((p.x + 1) / 2) * width}px, ${((1 - p.y) / 2) * height}px)`
      })
      const settling = Math.abs(view.dragTo - view.drag) + Math.abs(view.hxTo - view.hx) + Math.abs(view.hyTo - view.hy) > 0.001
      if (visible && !document.hidden && (!still || settling)) raf = requestAnimationFrame(frame)
    }
    const wake = () => {
      if (!raf && !disposed) raf = requestAnimationFrame(frame)
    }

    const onDown = (event: PointerEvent) => {
      view.down = true
      view.lastX = event.clientX
      canvas.style.cursor = 'grabbing'
      canvas.setPointerCapture(event.pointerId)
    }
    const onMove = (event: PointerEvent) => {
      if (event.pointerType === 'mouse' && !still) {
        const rect = canvas.getBoundingClientRect()
        view.hxTo = ((event.clientX - rect.left) / rect.width) * 2 - 1
        view.hyTo = ((event.clientY - rect.top) / rect.height) * 2 - 1
        view.nearTo = 1
      }
      if (view.down) {
        view.dragTo = THREE.MathUtils.clamp(view.dragTo - (event.clientX - view.lastX) * 0.006, -1.1, 1.1)
        view.lastX = event.clientX
      }
      wake()
    }
    const onUp = () => {
      view.down = false
      canvas.style.cursor = 'grab'
    }
    const onLeave = () => {
      view.hxTo = 0
      view.hyTo = 0
      view.nearTo = 0
      wake()
    }
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointercancel', onUp)
    canvas.addEventListener('pointerleave', onLeave)

    const resize = new ResizeObserver(fit)
    resize.observe(host)
    const seen = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (visible) {
        startScan()
        wake()
      }
    })
    seen.observe(host)
    const onVisibility = () => !document.hidden && wake()
    document.addEventListener('visibilitychange', onVisibility)

    const load = () => fetch(src)
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.arrayBuffer()
      })
      .then(buffer => {
        if (disposed) return
        const head = new DataView(buffer)
        if (String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== 'SSN1') throw new Error('not a room scan')
        const count = head.getUint32(4, true)
        box = new THREE.Vector3(head.getFloat32(8, true), head.getFloat32(12, true), head.getFloat32(16, true))
        const extent = new THREE.Vector3(head.getFloat32(20, true), head.getFloat32(24, true), head.getFloat32(28, true))
        const packed = new Int16Array(buffer, 32, count * 3)
        const colours = new Uint8Array(buffer, 32 + count * 6, count * 3)
        const positions = new Float32Array(count * 3)
        for (let i = 0; i < count; i++) {
          positions[i * 3] = (packed[i * 3] / 32767) * extent.x
          positions[i * 3 + 1] = (packed[i * 3 + 1] / 32767) * extent.y
          positions[i * 3 + 2] = (packed[i * 3 + 2] / 32767) * extent.z
        }
        geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        geometry.setAttribute('tint', new THREE.BufferAttribute(colours, 3, true))
        material = new THREE.ShaderMaterial({
          transparent: true,
          uniforms: {
            uScan: { value: 0 },
            uSize: { value: 0.03 },
            uPixel: { value: 600 },
            uBox: { value: box },
            uTape: { value: new THREE.Color('#ffd23f') },
          },
          vertexShader: /* glsl */ `
            attribute vec3 tint;
            uniform float uScan, uSize, uPixel;
            uniform vec3 uBox, uTape;
            varying vec3 vColour;
            varying float vShown;
            float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
            void main() {
              float sweep = mix(-uBox.x * 0.5 - 0.7, uBox.x * 0.5 + 0.7, uScan);
              float behind = sweep - position.x;
              float shown = smoothstep(-0.02, 0.22, behind);
              float glow = exp(-abs(behind) * 8.0) * (1.0 - step(0.999, uScan));
              vec3 p = position;
              p.y += (1.0 - shown) * (0.5 + hash(position) * 0.9);
              vec4 eye = modelViewMatrix * vec4(p, 1.0);
              gl_Position = projectionMatrix * eye;
              gl_PointSize = clamp(uSize * uPixel / -eye.z * (1.0 + glow * 1.6), 1.0, 12.0);
              vColour = mix(tint, uTape, clamp(glow * 1.3, 0.0, 1.0));
              vShown = shown;
            }`,
          fragmentShader: /* glsl */ `
            varying vec3 vColour;
            varying float vShown;
            void main() {
              vec2 c = gl_PointCoord - 0.5;
              if (dot(c, c) > 0.25 || vShown < 0.02) discard;
              gl_FragColor = vec4(vColour, vShown);
            }`,
        })
        scene.add(new THREE.Points(geometry, material))
        target.set(0, box.y * 0.16, 0)

        // dimension lines along the floor's two near edges, a little outside the room
        const x = box.x / 2
        const z = box.z / 2
        const off = 0.38
        const tick = 0.12
        const v = (a: number, b: number, c: number) => new THREE.Vector3(a, b, c)
        const runs: [THREE.Vector3, THREE.Vector3, THREE.Vector3][] = [
          [v(-x, 0, z + off), v(x, 0, z + off), v(0, 0, 1)], // width, along the front
          [v(-x - off, 0, -z), v(-x - off, 0, z), v(1, 0, 0)], // depth, along the left
        ]
        const segments: number[] = []
        runs.forEach(([a, b, across]) => {
          segments.push(...a.toArray(), ...b.toArray())
          for (const end of [a, b]) {
            segments.push(...end.clone().addScaledVector(across, -tick).toArray(), ...end.clone().addScaledVector(across, tick).toArray())
          }
          anchors.push(a.clone().lerp(b, 0.5))
        })
        const lineGeometry = new THREE.BufferGeometry()
        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3))
        lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ color: '#0f1f38', transparent: true, opacity: 0.75 }))
        scene.add(lines)

        setSize([box.x, box.y, box.z])
        setState(still ? 'ready' : 'scanning')
        startScan()
        fit()
      })
      .catch(() => !disposed && setState('failed'))
    // the file is a few MB: fetch it as the section comes near, not with the page
    const near = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return
      near.disconnect()
      load()
    }, { rootMargin: '700px 0px' })
    near.observe(host)

    fit()
    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      resize.disconnect()
      seen.disconnect()
      near.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      canvas.removeEventListener('pointerleave', onLeave)
      geometry?.dispose()
      material?.dispose()
      lines?.geometry.dispose()
      ;(lines?.material as THREE.Material | undefined)?.dispose()
      renderer.dispose()
      canvas.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src])

  const labels = size ? [`${size[0].toFixed(2)} m`, `${size[2].toFixed(2)} m`] : []
  return (
    <div ref={mount} className={`relative select-none ${className}`}>
      {labels.map((text, i) => (
        <span key={text + i} ref={element => { tags.current[i] = element }} aria-hidden="true"
          className={`pointer-events-none absolute left-0 top-0 rounded-[3px] bg-tape px-1.5 py-0.5 text-[13px] font-semibold tabular-nums text-ink shadow-[0_1px_0_rgb(15_31_56/0.25)] transition-opacity duration-500 ${state === 'ready' ? 'opacity-100' : 'opacity-0'}`}>
          {text}
        </span>
      ))}
      {(state === 'loading' || state === 'scanning') && (
        <p className="pointer-events-none absolute bottom-3 left-0 text-sm">
          <ShinyText color="#56657a" shineColor="#0f1f38" speed={2.2} shineWidth={40} glow={0} trigger="loop">
            Rebuilding the room from the video
          </ShinyText>
        </p>
      )}
      {state === 'failed' && (
        <p className="absolute inset-0 grid place-items-center text-sm text-steel">This 3D view needs WebGL, which this browser has turned off.</p>
      )}
    </div>
  )
}
