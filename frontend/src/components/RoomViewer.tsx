import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { Ruler, X } from 'lucide-react'

const MARK = '#f59e0b'

/**
 * The 3D room as coloured points. Drag to turn, scroll to zoom, right-drag to move.
 * A click (not a drag) on the room drops a marker; two markers give the straight-line distance between them.
 */
export default function RoomViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null)
  const clear = useRef<() => void>(() => {})
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [picked, setPicked] = useState(0)
  const [distance, setDistance] = useState<number | null>(null)

  useEffect(() => {
    const el = host.current
    if (!el) return
    let disposed = false
    setState('loading')
    setPicked(0)
    setDistance(null)

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
    let markerRadius = 0.02

    new PLYLoader().load(url, geometry => {
      if (disposed) return
      geometry.computeBoundingSphere()
      const sphere = geometry.boundingSphere!
      cloud = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: Math.max(0.004, sphere.radius / 250),
        vertexColors: true,
      }))
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
      const hit = raycaster.intersectObject(cloud)[0]
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
    </div>
  )
}
