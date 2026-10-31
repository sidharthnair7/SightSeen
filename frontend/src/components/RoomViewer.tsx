import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { Ruler, X } from 'lucide-react'

const MARK = '#f59e0b'

type Look = 'solid' | 'dots'

/**
 * The 3D room, as a solid surface when the room has one, or as coloured points. Drag to turn, scroll to zoom,
 * right-drag to move. A click (not a drag) on the room drops a marker; two markers give the straight-line distance
 * between them. In the dots look, the dot-size slider makes the points bigger, so the room looks less see-through.
 */
export default function RoomViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null)
  const clear = useRef<() => void>(() => {})
  const material = useRef<THREE.PointsMaterial | null>(null)
  const baseSize = useRef(0.01)
  const surfaceMesh = useRef<THREE.Mesh | null>(null)
  const cloudPoints = useRef<THREE.Points | null>(null)
  const shown = useRef<Look>('dots')
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [picked, setPicked] = useState(0)
  const [distance, setDistance] = useState<number | null>(null)
  const [dots, setDots] = useState(1)
  const [look, setLook] = useState<Look>('solid')
  const [hasSurface, setHasSurface] = useState(false)
  const showing: Look = hasSurface && look === 'solid' ? 'solid' : 'dots'

  useEffect(() => {
    if (material.current) material.current.size = baseSize.current * dots
  }, [dots, state])

  useEffect(() => {
    shown.current = showing
    if (surfaceMesh.current) surfaceMesh.current.visible = showing === 'solid'
    if (cloudPoints.current) cloudPoints.current.visible = showing === 'dots'
  }, [showing, state])

  useEffect(() => {
    const el = host.current
    if (!el) return
    let disposed = false
    setState('loading')
    setPicked(0)
    setDistance(null)
    setHasSurface(false)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(el.clientWidth, el.clientHeight)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#10161a')
    const camera = new THREE.PerspectiveCamera(60, el.clientWidth / el.clientHeight, 0.01, 1000)
    camera.position.set(0, 0.5, 2)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true

    const raycaster = new THREE.Raycaster()
    raycaster.params.Points = { threshold: 0.02 }
    const marks = new THREE.Group()
    scene.add(marks)
    const points: THREE.Vector3[] = []
    let cloud: THREE.Points | null = null
    let surface: THREE.Mesh | null = null
    let markerRadius = 0.02

    // The solid surface sits next to the points; older rooms don't have one, and then the dots are all there is
    new PLYLoader().load(url.replace('/room.ply', '/room-surface.ply'), geometry => {
      if (disposed) return
      surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
      surface.visible = shown.current === 'solid'
      surfaceMesh.current = surface
      scene.add(surface)
      setHasSurface(true)
    }, undefined, () => {})

    new PLYLoader().load(url, geometry => {
      if (disposed) return
      geometry.computeBoundingSphere()
      const sphere = geometry.boundingSphere!
      // About 2 cm dots in a room-sized cloud; the slider scales this
      baseSize.current = Math.max(0.006, sphere.radius / 120)
      material.current = new THREE.PointsMaterial({ size: baseSize.current, vertexColors: true })
      cloud = new THREE.Points(geometry, material.current)
      cloud.visible = shown.current === 'dots'
      cloudPoints.current = cloud
      scene.add(cloud)
      // Start a step back from the middle, roughly where the video began, looking into the room
      controls.target.copy(sphere.center)
      camera.position.copy(sphere.center).add(new THREE.Vector3(0, sphere.radius * 0.25, sphere.radius * 0.9))
      camera.near = Math.max(0.001, sphere.radius / 1000)
      camera.far = sphere.radius * 50
      camera.updateProjectionMatrix()
      raycaster.params.Points = { threshold: sphere.radius / 300 }
      markerRadius = sphere.radius / 150
      setState('ready')
    }, undefined, () => {
      if (!disposed) setState('error')
    })

    let down: { x: number; y: number } | null = null
    const onDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY }
    }
    const onUp = (e: PointerEvent) => {
      const start = down
      down = null
      // A drag turns the view; only a still click measures
      if (!start || !cloud || Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) return
      const rect = renderer.domElement.getBoundingClientRect()
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(ndc, camera)
      const hit = raycaster.intersectObject(shown.current === 'solid' && surface ? surface : cloud)[0]
      if (!hit) return
      if (points.length === 2) {
        points.length = 0
        marks.clear()
      }
      points.push(hit.point.clone())
      const marker = new THREE.Mesh(new THREE.SphereGeometry(markerRadius, 16, 12), new THREE.MeshBasicMaterial({ color: MARK }))
      marker.position.copy(hit.point)
      marks.add(marker)
      if (points.length === 2) {
        marks.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: MARK })))
        setDistance(points[0].distanceTo(points[1]))
      } else {
        setDistance(null)
      }
      setPicked(points.length)
    }
    renderer.domElement.addEventListener('pointerdown', onDown)
    renderer.domElement.addEventListener('pointerup', onUp)

    clear.current = () => {
      points.length = 0
      marks.clear()
      setPicked(0)
      setDistance(null)
    }

    const resize = new ResizeObserver(() => {
      renderer.setSize(el.clientWidth, el.clientHeight)
      camera.aspect = el.clientWidth / el.clientHeight
      camera.updateProjectionMatrix()
    })
    resize.observe(el)

    let frame = 0
    const loop = () => {
      frame = requestAnimationFrame(loop)
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onDown)
      renderer.domElement.removeEventListener('pointerup', onUp)
      controls.dispose()
      cloud?.geometry.dispose()
      material.current?.dispose()
      material.current = null
      cloudPoints.current = null
      if (surface) {
        surface.geometry.dispose()
        ;(surface.material as THREE.Material).dispose()
      }
      surfaceMesh.current = null
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [url])

  return (
    <div className="relative h-[70vh] min-h-[420px] overflow-hidden rounded-xl bg-[#10161a]">
      <div ref={host} className="absolute inset-0" />
      <div className="pointer-events-none absolute left-3 top-3 max-w-xs rounded-lg bg-black/60 px-3 py-2 text-sm text-white">
        {state === 'loading' && 'Loading the 3D room...'}
        {state === 'error' && 'The 3D room could not be loaded.'}
        {state === 'ready' && (
          <span className="inline-flex items-center gap-1.5">
            <Ruler size={15} />
            {picked === 0 && 'Click two points to measure between them.'}
            {picked === 1 && 'Now click the second point.'}
            {picked === 2 && distance !== null && `${distance.toFixed(2)} m (${Math.round(distance * 100)} cm)`}
          </span>
        )}
      </div>
      {picked > 0 && (
        <button
          type="button"
          onClick={() => clear.current()}
          className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-lg bg-black/60 px-3 py-2 text-sm text-white hover:bg-black/80"
        >
          <X size={15} /> Clear
        </button>
      )}
      <p className="pointer-events-none absolute bottom-3 left-3 text-xs text-white/70">
        Drag to turn, scroll to zoom, right-drag to move.
      </p>
      {state === 'ready' && (
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          {hasSurface && (
            <div className="flex overflow-hidden rounded-lg bg-black/60 text-xs text-white">
              {(['solid', 'dots'] as const).map(option => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setLook(option)}
                  className={`px-3 py-2 ${look === option ? 'bg-amber-500 text-black' : 'hover:bg-black/40'}`}
                >
                  {option === 'solid' ? 'Solid' : 'Dots'}
                </button>
              ))}
            </div>
          )}
          {showing === 'dots' && (
            <label className="flex items-center gap-2 rounded-lg bg-black/60 px-3 py-2 text-xs text-white">
              Dot size
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.25}
                value={dots}
                onChange={e => setDots(Number(e.target.value))}
                className="w-28 accent-amber-500"
              />
            </label>
          )}
        </div>
      )}
    </div>
  )
}
