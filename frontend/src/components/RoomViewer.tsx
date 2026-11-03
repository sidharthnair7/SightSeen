import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark'
import { Camera, Eye, Hand, Ruler, RotateCcw, X } from 'lucide-react'
import { SHELL_FACES, type RoomShell } from './walk/types'
import { texturedShellMaterials } from './walk/shellTextures'
import { materialShellMaterials } from './walk/shellMaterials'
import { loadDecals } from './walk/decals'
import { loadFurniture } from './walk/furniture'
import { movableThings, type Movable } from './walk/moveThings'
import { loadRebuilt } from './walk/rebuilt'

const MARK = '#f59e0b'
const WALK_STEP = 0.1 // metres per scroll notch
const WALK_FOV = 70 // degrees top to bottom in the Walk look: about what a person sees without turning their head
// The clean shell sits this far behind the found walls. The worker trims the splats on and beyond the walls, so the
// shell draws the walls, floor and ceiling, and the splats only the things in the room.
const SHELL_GAP = 0.01

type Look = 'rebuilt' | 'real' | 'walk' | 'solid' | 'dots'

/** One frame's camera from cameras.json: three.js axes, quaternion as w, x, y, z, widths of view in degrees. */
type CameraSpot = { frame: string; position: number[]; quaternion: number[]; fovX: number; fovY: number }

/** room-frame.json: the turn that stands the room upright (w, x, y, z) and the floor's height before it's moved to 0. */
type RoomFrame = { quaternion: number[]; floorHeight: number; eyeHeight: number; ceilingHeight: number | null }

/** room-objects.json: the furniture SAM 3.1 found, each as a block in the upright room (metres). */
type RoomObjects = {
  things: { name: string; centre: number[]; size: number[]; yawDegrees: number; colour: number[] }[]
}

/** A small JSON file next to the room, or null when this room doesn't have it (older rooms). */
function fetchJson<T>(url: string): Promise<T | null> {
  return fetch(url)
    .then(response => (response.ok ? (response.json() as Promise<T>) : null))
    .catch(() => null)
}

const LABELS: Record<Look, string> = { rebuilt: 'Rebuilt', real: 'Real', walk: 'Walk', solid: 'Solid', dots: 'Dots' }

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
function pick(look: Look, hasSplats: boolean, hasSurface: boolean, hasWalk: boolean, hasRebuilt: boolean): Look {
  if (look === 'rebuilt' && hasRebuilt) return 'rebuilt'
  if (look === 'walk' && hasWalk) return 'walk'
  if (look === 'real' && hasSplats) return 'real'
  if (look !== 'dots' && hasSurface) return 'solid'
  return 'dots'
}

/**
 * The 3D room: "Real" is Gaussian splats trained to look like the video, "Walk" the room recreated clean (the shell's
 * walls, floor and ceiling, and the furniture SAM 3.1 found as blocks at their real size) for moving around freely,
 * "Solid" the surface, "Dots" the points. Older rooms may only have some of them. Drag to look around (in place), scroll to walk, right-drag to step sideways. A click (not a drag) on
 * the room drops a marker; two markers give the straight-line distance between them. Measuring always hits the solid
 * surface when there is one, even in the Real look, because a surface gives a steadier point than a blurry splat.
 * In the dots look, the dot-size slider makes the points bigger, so the room looks less see-through.
 */
export default function RoomViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null)
  const clear = useRef<() => void>(() => {})
  const material = useRef<THREE.PointsMaterial | null>(null)
  const baseSize = useRef(0.01)
  const splatMesh = useRef<THREE.Object3D | null>(null) // the room's splats: one SplatMesh, or its movable pieces
  const movable = useRef<Movable | null>(null)
  const moveOn = useRef(false)
  const surfaceMesh = useRef<THREE.Mesh | null>(null)
  const shellMesh = useRef<THREE.Mesh | null>(null)
  const photoShell = useRef<THREE.Material[] | null>(null) // the walls as filmed (or flat colours until they load)
  const materialShell = useRef<THREE.Material[] | null>(null) // the Rebuilt look's clean materials
  const wallDecals = useRef<THREE.Group | null>(null) // and the window, door and the rest cut out onto them
  const blocks = useRef<THREE.Group | null>(null)
  const rebuiltGroup = useRef<THREE.Group | null>(null) // the Rebuilt look's things, whole on every side
  const rebuiltMovable = useRef<Movable | null>(null)
  const rebuiltTint = useRef<((on: boolean) => void) | null>(null)
  const refitView = useRef<() => void>(() => {})
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
  const [hasWalk, setHasWalk] = useState(false)
  const [hasMovable, setHasMovable] = useState(false)
  const [hasRebuilt, setHasRebuilt] = useState(false)
  const [notFilmed, setNotFilmed] = useState(false) // the Rebuilt look's never-filmed parts tinted
  const [hasMaterials, setHasMaterials] = useState(false)
  const [hasDecals, setHasDecals] = useState(false)
  const [photoWalls, setPhotoWalls] = useState(false) // the Rebuilt look showing the filmed walls instead
  const [moving, setMoving] = useState(false)
  const [held, setHeld] = useState<string | null>(null)
  const showing = pick(look, hasSplats, hasSurface, hasWalk, hasRebuilt)
  const available: Record<Look, boolean> = { rebuilt: hasRebuilt, real: hasSplats, walk: hasWalk, solid: hasSurface, dots: true }
  const looks = (['rebuilt', 'real', 'walk', 'solid', 'dots'] as const).filter(option => available[option])

  useEffect(() => {
    if (material.current) material.current.size = baseSize.current * dots
  }, [dots, state])

  // The Rebuilt look wears the clean materials, unless "Photo walls" is on; every other look wears the filmed walls
  // (with the window, door and the rest cut out onto them)
  useEffect(() => {
    const clean = showing === 'rebuilt' && !photoWalls && !!materialShell.current
    if (wallDecals.current) wallDecals.current.visible = clean
    const shell = shellMesh.current
    if (!shell) return
    const wanted = clean ? materialShell.current : photoShell.current
    if (wanted) shell.material = wanted
  }, [showing, photoWalls, hasMaterials, hasDecals, state])

  useEffect(() => {
    shown.current = showing
    if (splatMesh.current) splatMesh.current.visible = showing === 'real'
    if (surfaceMesh.current) surfaceMesh.current.visible = showing === 'solid'
    if (shellMesh.current) shellMesh.current.visible = showing !== 'dots'
    if (blocks.current) blocks.current.visible = showing === 'walk'
    if (rebuiltGroup.current) rebuiltGroup.current.visible = showing === 'rebuilt'
    // Each look moves its own things: changing look puts the hand down
    moveOn.current = false
    movable.current?.setOn(false)
    rebuiltMovable.current?.setOn(false)
    setMoving(false)
    setHeld(null)
    refitView.current()
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
    setHasWalk(false)
    setHasRebuilt(false)
    setNotFilmed(false)
    setHasMaterials(false)
    setHasDecals(false)
    setPhotoWalls(false)

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
    // In development only: the camera and scene on window, so a browser console can check where the viewer stands
    if ((import.meta as ImportMeta & { env?: { DEV?: boolean } }).env?.DEV) {
      Object.assign(window, { sightseen: { camera, controls, scene } })
    }
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
    let shell: THREE.Mesh | null = null
    let things: THREE.Group | null = null
    let markerRadius = 0.02

    // The room as filmed is tilted: its space is the first frame's camera. room-frame.json says which way is up and
    // where the floor is, so everything goes in one group that stands the room upright with the floor at height 0
    // (walking and VR both need that). Marks stay outside it: they're placed where the clicks hit, already upright.
    // The room's other generated files, by name (the room's URL may carry a ?v= version after room.ply; drop it here)
    const assets = url.split('?')[0].replace('/room.ply', '/assets/')
    const room = new THREE.Group()
    scene.add(room)

    // Splats come out of training in OpenCV axes (y down, z away from the camera); a half turn about x puts them
    // in three.js axes, the same flip the surface and points already had
    scene.add(spark)
    refreshSplats.current = () => {
      spark.update({ scene, camera }).catch(() => {})
    }
    const splats = new SplatMesh({ url: url.replace('/room.ply', '/room-splats.ply') })
    splats.quaternion.set(1, 0, 0, 0)
    splats.visible = false
    let replaced = false // true once the room's movable pieces stand in for the single splat file
    splats.initialized.then(() => {
      if (disposed || replaced) return
      splatMesh.current = splats
      splats.visible = shown.current === 'real'
      room.add(splats)
      setHasSplats(true)
      requestAnimationFrame(() => refreshSplats.current())
    }, () => {})

    // Start where the video began, at the phone's own width of view: splats only look like the video from near where
    // it was filmed, and nothing on screen is then space the phone never saw from there
    // In the Walk look nothing smears, so it uses a normal wide view instead
    let fovX: number | null = null
    const fitView = () => {
      if (shown.current === 'walk' || shown.current === 'rebuilt') {
        camera.fov = WALK_FOV
      } else if (fovX !== null) {
        const half = THREE.MathUtils.degToRad(fovX / 2)
        camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(half) / camera.aspect))
      }
      camera.updateProjectionMatrix()
    }
    refitView.current = fitView

    // Light for the Walk look's walls and blocks, so their faces shade apart (splats and the surface ignore it)
    scene.add(new THREE.HemisphereLight('#ffffff', '#8a8f98', 2.2))
    const sun = new THREE.DirectionalLight('#ffffff', 1.2)
    sun.position.set(1, 3, 2)
    scene.add(sun)
    Promise.all([
      fetchJson<RoomFrame>(url.replace('/room.ply', '/room-frame.json')),
      fetchJson<CameraSpot[]>(url.replace('/room.ply', '/cameras.json')),
      fetchJson<RoomShell>(url.replace('/room.ply', '/room-shell.json')),
      fetchJson<RoomObjects>(url.replace('/room.ply', '/room-objects.json')),
    ]).then(([frame, path, found, objects]) => {
      if (disposed) return
      if (frame) {
        const [w, x, y, z] = frame.quaternion
        room.quaternion.set(x, y, z, w)
        room.position.set(0, -frame.floorHeight, 0)
      }
      if (frame && found) {
        // The shell: one box seen from inside, its faces in BoxGeometry's order (+x, -x, +y, -y, +z, -z). It's
        // opaque, so anything the splats smeared beyond the walls is hidden behind it.
        const [width, height, length] = found.size
        const colour = (rgb: number[]) =>
          new THREE.MeshLambertMaterial({ color: new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255), side: THREE.BackSide })
        shell = new THREE.Mesh(
          new THREE.BoxGeometry(width + 2 * SHELL_GAP, height + 2 * SHELL_GAP, length + 2 * SHELL_GAP),
          SHELL_FACES.map(face => colour(found.faces[face])),
        )
        // The real walls (photos from the frames) replace the flat colours when this room has them
        const walls = shell
        photoShell.current = walls.material as THREE.Material[]
        texturedShellMaterials(assets, found).then(materials => {
          if (disposed || !materials) return
          const flat = photoShell.current
          photoShell.current = materials
          if (walls.material === flat) walls.material = materials
          flat?.forEach(m => m.dispose())
        }, () => {})
        // The clean materials for the Rebuilt look, when this room has them
        materialShellMaterials(assets, found).then(materials => {
          if (disposed || !materials) return
          materialShell.current = materials
          setHasMaterials(true)
        }, () => {})
        loadDecals(assets).then(decals => {
          if (!decals) return
          if (disposed) {
            disposeTree(decals)
            return
          }
          decals.visible = false
          wallDecals.current = decals
          scene.add(decals)
          setHasDecals(true)
        }, () => {})
        shell.position.set(found.centre[0], found.centre[1], found.centre[2])
        shell.rotation.y = -THREE.MathUtils.degToRad(found.yawDegrees)
        shell.visible = shown.current !== 'dots'
        shellMesh.current = shell
        scene.add(shell)

        // The furniture as clean blocks, each in the colour SAM 3.1 outlined, with dark edges so they read as solid
        things = new THREE.Group()
        for (const thing of objects?.things ?? []) {
          const [r, g, b] = thing.colour
          const box = new THREE.BoxGeometry(thing.size[0], thing.size[1], thing.size[2])
          // Pushed back a touch in depth so the edge lines draw cleanly on top instead of flickering through
          const face = new THREE.MeshLambertMaterial({
            color: new THREE.Color(r / 255, g / 255, b / 255),
            polygonOffset: true,
            polygonOffsetFactor: 1,
            polygonOffsetUnits: 1,
          })
          const block = new THREE.Mesh(box, face)
          block.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: '#1f2933' })))
          block.position.set(thing.centre[0], thing.centre[1], thing.centre[2])
          block.rotation.y = -THREE.MathUtils.degToRad(thing.yawDegrees)
          block.name = thing.name
          things.add(block)
        }
        things.visible = shown.current === 'walk'
        blocks.current = things
        scene.add(things)
        setHasWalk(true)
        // The furniture looking like the real thing replaces the plain blocks when this room has it
        loadFurniture(assets).then(furniture => {
          if (disposed || !furniture || !things) return
          things.clear()
          things.add(furniture)
        }, () => {})
        // The room's things rebuilt whole by SAM 3D, when this room has them: the Rebuilt look, shown first
        loadRebuilt(assets, scene, camera, renderer.domElement, found, name => setHeld(name)).then(rebuilt => {
          if (!rebuilt) return
          if (disposed) {
            rebuilt.movable.dispose()
            return
          }
          rebuiltGroup.current = rebuilt.group
          rebuiltMovable.current = rebuilt.movable
          rebuiltTint.current = rebuilt.showNotFilmed
          rebuilt.group.visible = shown.current === 'rebuilt'
          scene.add(rebuilt.group)
          setHasRebuilt(true)
          setLook('rebuilt')
        }, () => {})
      }
      // The room's things as movable pieces (the same splats, split by SAM 3.1), when the room has them
      movableThings(assets, room, scene, camera, renderer.domElement, frame && found ? found : null,
        name => setHeld(name), () => refreshSplats.current()).then(pieces => {
        if (!pieces) return
        if (disposed) {
          pieces.dispose()
          return
        }
        movable.current = pieces
        replaced = true
        room.remove(splats)
        splats.dispose()
        pieces.parts.visible = shown.current === 'real'
        room.add(pieces.parts)
        splatMesh.current = pieces.parts
        setHasSplats(true)
        setHasMovable(true)
        requestAnimationFrame(() => refreshSplats.current())
      }, () => {})
      room.updateMatrixWorld(true)
      const first = path?.[0]
      const facing = room.quaternion.clone()
      if (first) {
        const [w, x, y, z] = first.quaternion
        facing.multiply(new THREE.Quaternion(x, y, z, w))
        camera.position.set(first.position[0], first.position[1], first.position[2])
        fovX = first.fovX
        fitView()
      } else {
        camera.position.set(0, 0, 0) // older rooms: their space starts at the first frame's camera
      }
      camera.position.applyMatrix4(room.matrixWorld)
      controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -0.01).applyQuaternion(facing))
      controls.update()
      requestAnimationFrame(() => refreshSplats.current())
    })

    // The solid surface sits next to the points; older rooms don't have one, and then the dots are all there is
    new PLYLoader().load(url.replace('/room.ply', '/room-surface.ply'), geometry => {
      if (disposed) return
      surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
      surface.visible = shown.current === 'solid'
      surfaceMesh.current = surface
      room.add(surface)
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
      room.add(cloud)
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
      if (moveOn.current) return // moving things: clicks pick things up, they don't measure
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
    // Grabbing a thing comes first (capture), and holds the view still until the thing is let go
    const onGrab = (e: PointerEvent) => {
      const active = shown.current === 'rebuilt' ? rebuiltMovable.current : movable.current
      if (!active?.grab(e)) return
      controls.enabled = false
      const letGo = () => {
        controls.enabled = true
        window.removeEventListener('pointerup', letGo)
      }
      window.addEventListener('pointerup', letGo)
    }
    renderer.domElement.addEventListener('pointerdown', onGrab, { capture: true })

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
      fitView()
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
      renderer.domElement.removeEventListener('pointerdown', onGrab, { capture: true })
      renderer.domElement.removeEventListener('wheel', onWheel)
      movable.current?.dispose()
      movable.current = null
      rebuiltMovable.current?.dispose()
      rebuiltMovable.current = null
      if (rebuiltGroup.current) scene.remove(rebuiltGroup.current)
      rebuiltGroup.current = null
      rebuiltTint.current = null
      moveOn.current = false
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
      if (shell) {
        shell.geometry.dispose()
        photoShell.current?.forEach(m => m.dispose())
        materialShell.current?.forEach(m => m.dispose())
      }
      photoShell.current = null
      materialShell.current = null
      if (wallDecals.current) {
        scene.remove(wallDecals.current)
        disposeTree(wallDecals.current)
      }
      wallDecals.current = null
      shellMesh.current = null
      things?.traverse(part => {
        if (part instanceof THREE.Mesh || part instanceof THREE.LineSegments) {
          part.geometry.dispose()
          ;(part.material as THREE.Material).dispose()
        }
      })
      blocks.current = null
      if (!replaced) splats.dispose()
      splatMesh.current = null
      refreshSplats.current = () => {}
      refitView.current = () => {}
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
        {state === 'ready' && moving && (
          <span className="inline-flex items-center gap-1.5">
            <Hand size={15} />
            {held ? `Moving the ${held}. Drag it across the floor.` : 'Click a thing and drag it across the floor.'}
          </span>
        )}
        {state === 'ready' && !moving && notFilmed && showing === 'rebuilt' && (
          <span className="inline-flex items-center gap-1.5">
            <Eye size={15} />
            Purple: what the video never saw, filled in by SAM 3D.
          </span>
        )}
        {state === 'ready' && !moving && !(notFilmed && showing === 'rebuilt') && (
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
          {hasRebuilt && showing === 'rebuilt' && hasMaterials && (
            <button
              type="button"
              onClick={() => setPhotoWalls(!photoWalls)}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs ${photoWalls ? 'bg-sky-500 text-white' : 'bg-black/60 text-white hover:bg-black/80'}`}
            >
              <Camera size={14} /> Photo walls
            </button>
          )}
          {hasRebuilt && showing === 'rebuilt' && (
            <button
              type="button"
              onClick={() => {
                const next = !notFilmed
                rebuiltTint.current?.(next)
                setNotFilmed(next)
              }}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs ${notFilmed ? 'bg-purple-500 text-white' : 'bg-black/60 text-white hover:bg-black/80'}`}
            >
              <Eye size={14} /> Not filmed
            </button>
          )}
          {((hasMovable && showing === 'real') || (hasRebuilt && showing === 'rebuilt')) && (
            <div className="flex overflow-hidden rounded-lg bg-black/60 text-xs text-white">
              <button
                type="button"
                onClick={() => {
                  const next = !moving
                  moveOn.current = next
                  ;(showing === 'rebuilt' ? rebuiltMovable.current : movable.current)?.setOn(next)
                  setMoving(next)
                  if (!next) setHeld(null)
                }}
                className={`inline-flex items-center gap-1.5 px-3 py-2 ${moving ? 'bg-amber-500 text-black' : 'hover:bg-black/40'}`}
              >
                <Hand size={14} /> Move things
              </button>
              {moving && (
                <button
                  type="button"
                  onClick={() => (showing === 'rebuilt' ? rebuiltMovable.current : movable.current)?.reset()}
                  className="inline-flex items-center gap-1.5 px-3 py-2 hover:bg-black/40"
                >
                  <RotateCcw size={14} /> Put back
                </button>
              )}
            </div>
          )}
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

/** Frees every mesh's geometry and material under `root`. */
function disposeTree(root: THREE.Object3D) {
  root.traverse(part => {
    if (part instanceof THREE.Mesh) {
      part.geometry.dispose()
      ;(part.material as THREE.Material).dispose()
    }
  })
}
