import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark'
import { Ruler, X } from 'lucide-react'

const MARK = '#f59e0b'
const WALK_STEP = 0.1 // metres per scroll notch

type Look = 'real' | 'solid' | 'dots'

const LABELS: Record<Look, string> = { real: 'Real', solid: 'Solid', dots: 'Dots' }

// One WebGL renderer and one Spark renderer for the whole app. Spark keeps shared GPU data tied to the renderer it
// first draws with, so a viewer that made its own renderer (React builds effects twice in development, and every
// room change rebuilds the viewer) would leave Spark drawing into a thrown-away one, and the splats never show.
let shared: { renderer: THREE.WebGLRenderer; spark: SparkRenderer } | null = null

function sharedRenderer() {
  if (!shared) {
    // Spark asks for anti-aliasing off: it doesn't improve splats and costs a lot of speed (VR needs the speed)
    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    shared = { renderer, spark: new SparkRenderer({ renderer }) }
  }
  return shared
}

/** The look asked for, or the next best one this room has: splats, then the solid surface, then dots. */
function pick(look: Look, hasSplats: boolean, hasSurface: boolean): Look {
  if (look === 'real' && hasSplats) return 'real'
  if (look !== 'dots' && hasSurface) return 'solid'
  return 'dots'
}

/**
 * The 3D room: "Real" is Gaussian splats trained to look like the video, "Solid" the surface, "Dots" the points.
 * Older rooms may only have some of them. Drag to look around (in place), scroll to walk, right-drag to step sideways. A click (not a drag) on
 * the room drops a marker; two markers give the straight-line distance between them. Measuring always hits the solid
 * surface when there is one, even in the Real look, because a surface gives a steadier point than a blurry splat.
 * In the dots look, the dot-size slider makes the points bigger, so the room looks less see-through.
 */
export default function RoomViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null)
  const clear = useRef<() => void>(() => {})
  const material = useRef<THREE.PointsMaterial | null>(null)
  const baseSize = useRef(0.01)
  const splatMesh = useRef<SplatMesh | null>(null)
  const surfaceMesh = useRef<THREE.Mesh | null>(null)
  const cloudPoints = useRef<THREE.Points | null>(null)
  const shown = useRef<Look>('dots')
  const refreshSplats = useRef<() => void>(() => {})
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [picked, setPicked] = useState(0)
  const [distance, setDistance] = useState<number | null>(null)
  const [dots, setDots] = useState(1)
  const [look, setLook] = useState<Look>('real')
  const [hasSplats, setHasSplats] = useState(false)
  const [hasSurface, setHasSurface] = useState(false)
  const showing = pick(look, hasSplats, hasSurface)
  const looks = (['real', 'solid', 'dots'] as const).filter(
    option => option === 'dots' || (option === 'real' ? hasSplats : hasSurface),
  )

  useEffect(() => {
    if (material.current) material.current.size = baseSize.current * dots
  }, [dots, state])

  useEffect(() => {
    shown.current = showing
    if (splatMesh.current) splatMesh.current.visible = showing === 'real'
    if (surfaceMesh.current) surfaceMesh.current.visible = showing === 'solid'
    if (cloudPoints.current) cloudPoints.current.visible = showing === 'dots'
    // Spark only rebuilds its splats when the camera moves more than 1 mm or its set of splats changes, and
    // showing splats that were hidden is neither: without this they stay invisible until the first drag
    if (showing === 'real') refreshSplats.current()
  }, [showing, state])

  useEffect(() => {
    const el = host.current
    if (!el) return
    let disposed = false
    setState('loading')
    setPicked(0)
    setDistance(null)
    setHasSplats(false)
    setHasSurface(false)

    const { renderer, spark } = sharedRenderer()
    renderer.setSize(el.clientWidth, el.clientHeight)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#10161a')
    const camera = new THREE.PerspectiveCamera(60, el.clientWidth / el.clientHeight, 0.01, 1000)
    camera.position.set(0, 0.5, 2)
    // Turning happens in place, like standing in the room: the controls orbit a point 1 cm ahead, so the camera stays
    // where it is instead of swinging round into places no frame was filmed from. The wheel walks forward and back.
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enableZoom = false
    controls.rotateSpeed = -0.35 // drag the room the way you'd turn your head
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const ahead = new THREE.Vector3()
      camera.getWorldDirection(ahead)
      ahead.multiplyScalar(-Math.sign(e.deltaY) * WALK_STEP)
      camera.position.add(ahead)
      controls.target.add(ahead)
    }
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false })

    const raycaster = new THREE.Raycaster()
    raycaster.params.Points = { threshold: 0.02 }
    const marks = new THREE.Group()
    scene.add(marks)
    const points: THREE.Vector3[] = []
    let cloud: THREE.Points | null = null
    let surface: THREE.Mesh | null = null
    let markerRadius = 0.02

    // Splats come out of training in OpenCV axes (y down, z away from the camera); a half turn about x puts them
    // in three.js axes, the same flip the surface and points already had
    scene.add(spark)
    refreshSplats.current = () => {
      spark.update({ scene, camera }).catch(() => {})
    }
    const splats = new SplatMesh({ url: url.replace('/room.ply', '/room-splats.ply') })
    splats.quaternion.set(1, 0, 0, 0)
    splats.visible = false
    splats.initialized.then(() => {
      if (disposed) return
      splatMesh.current = splats
      splats.visible = shown.current === 'real'
      scene.add(splats)
      setHasSplats(true)
    }, () => {})

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
      // Start exactly where the video began (the room's origin is the first frame's camera, looking down -z): splats
      // only look like the video from where it was filmed, and it's the view the student already knows
      controls.target.set(0, 0, -0.01)
      camera.position.set(0, 0, 0)
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
      const hit = raycaster.intersectObject(shown.current !== 'dots' && surface ? surface : cloud)[0]
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
      renderer.domElement.removeEventListener('wheel', onWheel)
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
      splats.dispose()
      splatMesh.current = null
      refreshSplats.current = () => {}
      scene.remove(spark)  // the shared renderers stay alive for the next room
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
        Drag to look around, scroll to walk, right-drag to step sideways.
      </p>
      {state === 'ready' && (
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          {looks.length > 1 && (
            <div className="flex overflow-hidden rounded-lg bg-black/60 text-xs text-white">
              {looks.map(option => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setLook(option)}
                  className={`px-3 py-2 ${showing === option ? 'bg-amber-500 text-black' : 'hover:bg-black/40'}`}
                >
                  {LABELS[option]}
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
