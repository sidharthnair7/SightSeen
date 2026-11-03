import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark'
import { AlertTriangle, Camera, ChevronLeft, ChevronRight, Eye, Glasses, Hand, Package, Ruler, RotateCcw, RotateCw, Undo2, X } from 'lucide-react'
import { STATIC } from '../api'
import YourThings, { type Item } from './YourThings'
import Issues, { type IssuesReport, type MeasuredRoom } from './Issues'
import { SHELL_FACES, type RoomShell } from './walk/types'
import { texturedShellMaterials } from './walk/shellTextures'
import { materialShellMaterials } from './walk/shellMaterials'
import { loadDecals } from './walk/decals'
import { loadFurniture } from './walk/furniture'
import { movableThings, type Fit, type Movable } from './walk/moveThings'
import { fetchModel, loadRebuilt } from './walk/rebuilt'
import { makeVr, markPoint, onAssumedCeiling, vrSupported, type Vr } from './walk/vr'
import { filmedSpots, type Spots } from './walk/filmedSpots'

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
type RoomFrame = {
  quaternion: number[]
  floorHeight: number
  eyeHeight: number
  ceilingHeight: number | null
  /** Real metres per unit of the rebuilt room (worker/room_scale.py), and what set it */
  scale?: number | null
  scaleEvidence?: { source?: string; plates?: number; what?: string; spread?: number[]; ratiosKept?: number } | null
}

/** How sure a floor area is: twice a length's share. */
function areaTolerance(frame: RoomFrame): number {
  return 2 * lengthTolerance(frame)
}

/** Where a room's real size comes from, in a few words, with how sure it is. */
function scaleSource(frame: RoomFrame): string {
  const evidence = frame.scaleEvidence
  if (!frame.scale || !evidence) return "MapAnything's own scale, not yet checked (typically off by about 15%)"
  if (evidence.source === 'tape measure') return `checked with a tape measure (${evidence.what?.replace('-', ' ')})`
  return `rough, from ${evidence.plates ?? 'standard'} wall plates (±${Math.round(lengthTolerance(frame) * 100)}%)`
}

/** How sure a length is, as a share either way (about 95%): a tape measure to about 2%; wall plates by how much they
 * disagree (the median of n readings: 1.96 standard errors, from their middle half's spread); MapAnything's own
 * scale is off 13 to 16% on average (its paper). */
function lengthTolerance(frame: RoomFrame): number {
  const evidence = frame.scaleEvidence
  if (!frame.scale || !evidence) return 0.15
  if (evidence.source === 'tape measure') return 0.02
  if (evidence.spread?.length === 2) {
    const n = evidence.ratiosKept ?? 2 * (evidence.plates ?? 1)
    const sigma = (evidence.spread[1] - evidence.spread[0]) / 1.349
    return Math.max(0.02, (1.96 * 1.2533 * sigma) / Math.sqrt(n) / frame.scale)
  }
  return 0.15
}

/** "the bed", or "your chair" for one of the student's own things. */
const the = (name: string) => (name.startsWith('your ') ? name : `the ${name}`)

/** How a held thing fits, in a sentence: what it runs into, or how much room it has round it. */
function fitText(fit: Fit | null): string {
  if (!fit) return 'Drag it across the floor; the arrows turn it.'
  const cm = (metres: number) => `${Math.round(metres * 100)} cm`
  if (fit.into) return `It runs ${cm(fit.into.depth)} into ${the(fit.into.name)}.`
  const near = fit.nearest ? `${cm(fit.nearest.gap)} to ${the(fit.nearest.name)}, ` : ''
  return `It fits: ${near}${cm(fit.wall)} to the nearest wall.`
}

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
// The looks a visitor picks between are the room rebuilt and the room as filmed (the owner, Nov 2: too many modes).
// Walk, Solid and Dots were steps on the way there: they show with ?debug in the address, or for a room that has
// neither of the two
const DEBUG = new URLSearchParams(window.location.search).has('debug')

// One WebGL renderer and one Spark renderer for the whole app. Spark keeps shared GPU data tied to the renderer it
// first draws with, so a viewer that made its own renderer (React builds effects twice in development, and every
// room change rebuilds the viewer) would leave Spark drawing into a thrown-away one, and the splats never show.
let shared: { renderer: THREE.WebGLRenderer; spark: SparkRenderer } | null = null

function sharedRenderer() {
  if (!shared) {
    // Spark asks for anti-aliasing off: it doesn't improve splats and costs a lot of speed (VR needs the speed)
    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.xr.enabled = true // a headset's browser can step into the room (walk/vr.ts)
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
export default function RoomViewer({ url, canResize = !STATIC }: { url: string; canResize?: boolean }) {
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
  const [toCeiling, setToCeiling] = useState(false) // the measurement reaches a ceiling whose height was assumed
  const measuredUnits = useRef(0) // the last measurement, in the room's own units
  const [realCm, setRealCm] = useState('') // what a tape measure says that measurement really is
  const [scaleBusy, setScaleBusy] = useState(false)
  const [scaleError, setScaleError] = useState<string | null>(null)
  const scaleRef = useRef(1) // real metres per room unit
  const [roomSize, setRoomSize] = useState<MeasuredRoom | null>(null)
  const [issuesOpen, setIssuesOpen] = useState(false)
  const issueMarks = useRef<THREE.Group | null>(null) // red boxes (damage) and yellow outlines (never filmed)
  const [dots, setDots] = useState(1)
  const spots = useRef<Spots | null>(null) // the Real look's spots the video was filmed from
  const spotIndex = useRef(0)
  const [spot, setSpot] = useState(0)
  const [spotCount, setSpotCount] = useState(0)
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
  const [fit, setFit] = useState<Fit | null>(null) // how the held thing sits among the others
  const [itemsOpen, setItemsOpen] = useState(false) // the Rebuilt look's "Your things" panel
  const [canVr, setCanVr] = useState(false) // this browser can start VR (a headset's browser)
  const [inVr, setInVr] = useState(false)
  const [vrError, setVrError] = useState<string | null>(null)
  const vrRef = useRef<Vr | null>(null)
  const lightModels = useRef(false) // a headset's browser: the lighter copies of the models
  // The room's API address (/api/requests/{id}), from its room.ply URL: the student's things live under it
  const itemsBase = url.split('?')[0].replace('/room.ply', '')
  const showing = pick(look, hasSplats, hasSurface, hasWalk, hasRebuilt)
  const available: Record<Look, boolean> = { rebuilt: hasRebuilt, real: hasSplats, walk: hasWalk, solid: hasSurface, dots: true }
  const main = (['rebuilt', 'real'] as const).filter(option => available[option])
  const looks: Look[] = main.length && !DEBUG
    ? main
    : (['rebuilt', 'real', 'walk', 'solid', 'dots'] as const).filter(option => available[option])

  useEffect(() => {
    if (material.current) material.current.size = baseSize.current * dots
  }, [dots, state])

  useEffect(() => {
    let gone = false
    const check = () => {
      vrSupported().then(yes => {
        if (!gone) setCanVr(yes)
      })
    }
    check()
    navigator.xr?.addEventListener('devicechange', check) // a headset plugged in after the page loaded
    return () => {
      gone = true
      navigator.xr?.removeEventListener('devicechange', check)
    }
  }, [])

  // The issues in 3D: a red box round each damage flag (at least 10 cm a side, so a small stain still shows) and a
  // yellow outline over each patch never filmed, drawn over everything so they show through the furniture
  const showIssues = (report: IssuesReport) => {
    const host = rebuiltGroup.current?.parent
    if (!host || issueMarks.current) return
    const marks = new THREE.Group()
    for (const flag of report.flags) {
      if (flag.kind === 'damage' && flag.position) {
        const size = (flag.size ?? [0.1, 0.1, 0.1]).map(side => Math.max(side, 0.1))
        const box = new THREE.BoxGeometry(size[0], size[1], size[2])
        const fill = new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: '#ef4444', transparent: true, opacity: 0.25, depthTest: false }))
        fill.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: '#dc2626', depthTest: false })))
        fill.position.fromArray(flag.position)
        fill.renderOrder = 10
        marks.add(fill)
      } else if (flag.kind === 'not filmed' && (flag.fill || flag.corners?.length === 4)) {
        // Its own shape when the report has it (cells on a grid over the wall, and its edge), else its bounding box
        const v = (array: number[]) => new THREE.Vector3().fromArray(array)
        const quads: THREE.Vector3[][] = flag.fill
          ? flag.fill.cells.map(([row, col]) => {
            const { origin, across, down } = flag.fill!
            const at = v(origin).addScaledVector(v(across), col).addScaledVector(v(down), row)
            return [at, at.clone().add(v(across)), at.clone().add(v(across)).add(v(down)), at.clone().add(v(down))]
          })
          : [flag.corners!.map(v)]
        const fill = new THREE.BufferGeometry().setFromPoints(quads.flatMap(q => [q[0], q[1], q[2], q[0], q[2], q[3]]))
        const patch = new THREE.Mesh(fill, new THREE.MeshBasicMaterial({ color: '#facc15', transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthTest: false }))
        const edge = (flag.outline?.length ? flag.outline : flag.corners ?? []).map(v)
        patch.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(edge), new THREE.LineBasicMaterial({ color: '#eab308', depthTest: false })))
        patch.renderOrder = 10
        marks.add(patch)
      }
    }
    marks.visible = false
    host.add(marks)
    issueMarks.current = marks
  }
  useEffect(() => {
    if (issueMarks.current) issueMarks.current.visible = issuesOpen && showing === 'rebuilt'
  }, [issuesOpen, showing])

  // A real length for the last measurement: the server sets the room's scale from it and makes again everything the
  // scale sizes (about a minute), then the room loads afresh at its real size
  const applyRealLength = async () => {
    const cm = Number(realCm)
    if (toCeiling) {
      setScaleError("That reaches the ceiling, whose height was assumed. Measure between two things the video filmed.")
      return
    }
    if (!(cm >= 5 && cm <= 2000)) {
      setScaleError('Type the real length in centimetres (5 to 2000).')
      return
    }
    setScaleBusy(true)
    setScaleError(null)
    try {
      const response = await fetch(`${itemsBase}/scale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ units: measuredUnits.current, metres: cm / 100 }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message || `The size couldn't be set (${response.status}).`)
      }
      window.location.reload()
    } catch (problem) {
      setScaleError(problem instanceof Error ? problem.message : String(problem))
      setScaleBusy(false)
    }
  }

  // The Real look: one filmed spot to the next (or back), by the arrows, the slider or the arrow keys
  const goSpot = (index: number) => {
    const next = Math.max(0, Math.min(spotCount - 1, index))
    spotIndex.current = next
    setSpot(next)
    spots.current?.go(next)
    refreshSplats.current()
  }
  useEffect(() => {
    if (showing !== 'real' || spotCount < 2) return
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)
      if (typing || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return
      e.preventDefault()
      goSpot(spotIndex.current + (e.key === 'ArrowRight' ? 1 : -1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [showing, spotCount])

  // One of the student's things, built (its own middle at x = z = 0 and its bottom at y = 0, in the room's units):
  // square to the walls, on the free floor nearest the room's middle (the middle itself when nowhere is free), shown
  // with the rebuilt things and picked up like them
  const placeItem = (item: Item) => {
    const group = rebuiltGroup.current
    const shell = shellMesh.current
    if (!group || !shell) return
    fetchModel(`${itemsBase}/items/${item.id}/`, 'item.ply', lightModels.current).then(async response => {
      if (!response) return
      const geometry = new PLYLoader().parse(await response.arrayBuffer())
      geometry.computeVertexNormals()
      geometry.computeBoundingBox()
      const size = geometry.boundingBox!.getSize(new THREE.Vector3())
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
      mesh.name = `your ${item.name}`
      const yawDegrees = -THREE.MathUtils.radToDeg(shell.rotation.y)
      const spot = rebuiltMovable.current?.freeSpot([size.x, size.y, size.z], yawDegrees)
        ?? [shell.position.x, size.y / 2, shell.position.z]
      mesh.position.set(spot[0], 0, spot[2])
      mesh.rotation.y = shell.rotation.y
      group.add(mesh)
      rebuiltMovable.current?.add({
        name: mesh.name, object: mesh, centre: [spot[0], size.y / 2, spot[2]], size: [size.x, size.y, size.z], yawDegrees,
      })
    })
  }

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
    if (showing === 'real') spots.current?.go(spotIndex.current)
    else spots.current?.release()
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
    const dev = (import.meta as ImportMeta & { env?: { DEV?: boolean } }).env?.DEV
    if (dev) Object.assign(window, { sightseen: { camera, controls, scene } })
    controls.enableDamping = true
    controls.enableZoom = false
    controls.rotateSpeed = -0.35 // drag the room the way you'd turn your head
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (shown.current === 'real' && spots.current) return // the Real look steps between filmed spots instead
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
    const onCeiling: boolean[] = [] // each point: on a ceiling whose height was assumed
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

    // VR: the Rebuilt look at its real size, standing where the video began and looking where it first looked (the
    // room's middle when there's no camera path); the start card says how big the room is and how sure that is
    let vrSpot: { stand: THREE.Vector3; ahead: THREE.Vector3 } | null = null
    let vrAbout: string[] = ['This room, at its real size']
    const vr = makeVr(renderer, scene, {
      start: () => {
        const spot = vrSpot ?? (shell ? { stand: shell.position.clone(), ahead: new THREE.Vector3(0, 0, -1) } : null)
        return spot && { ...spot, scale: scaleRef.current, about: vrAbout }
      },
      targets: () => [rebuiltGroup.current, shell].filter(part => part !== null) as THREE.Object3D[],
      walkable: point => {
        if (!shell) return true
        const local = shell.worldToLocal(point.clone())
        const { width, depth } = (shell.geometry as THREE.BoxGeometry).parameters
        const margin = 0.15 / scaleRef.current // the head stays 15 cm off the walls
        return Math.abs(local.x) <= width / 2 - margin && Math.abs(local.z) <= depth / 2 - margin
      },
      onChange: on => {
        spark.visible = !on // VR shows the Rebuilt look only: the splats' renderer sits out, and the headset keeps its frames
        setInVr(on)
        if (on) setVrError(null)
      },
    })
    vrRef.current = vr
    if (dev) Object.assign(window, { sightseen: { camera, controls, scene, vr } })

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
      vrSupported(),
      fetchJson<{ placed: string[] }>(`${assets}room-spots.json`),
    ]).then(([frame, path, found, objects, headset, placedSpots]) => {
      if (disposed) return
      lightModels.current = headset
      if (frame) {
        const [w, x, y, z] = frame.quaternion
        room.quaternion.set(x, y, z, w)
        room.position.set(0, -frame.floorHeight, 0)
      }
      if (frame) scaleRef.current = frame.scale || 1
      if (frame && found) {
        const scale = frame.scale || 1
        setRoomSize({ width: found.size[0] * scale, length: found.size[2] * scale, height: found.size[1] * scale,
          heightAssumed: found.ceilingAssumed === true, source: scaleSource(frame), tolerance: areaTolerance(frame) })
        const floorSize = `${(found.size[0] * scale).toFixed(2)} × ${(found.size[2] * scale).toFixed(2)} m`
        vrAbout = [
          'This room, at its real size',
          found.ceilingAssumed
            ? `About ${floorSize}; the ceiling wasn't filmed (${(found.size[1] * scale).toFixed(2)} m assumed)`
            : `About ${floorSize}, ${(found.size[1] * scale).toFixed(2)} m tall`,
          `Size: ${scaleSource(frame)}`,
        ]
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
        shell.userData.gap = SHELL_GAP // VR's measuring puts its marks back on the found walls
        shell.userData.ceilingAssumed = found.ceilingAssumed === true
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
        loadRebuilt(assets, scene, camera, renderer.domElement, found, (name, fits) => {
          setHeld(name)
          setFit(fits ?? null)
        }, () => scaleRef.current, headset).then(rebuilt => {
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
      movableThings(assets, room, scene, camera, renderer.domElement, frame && found ? found : null, (name, fits) => {
        setHeld(name)
        setFit(fits ?? null)
      }, () => refreshSplats.current(), () => scaleRef.current).then(pieces => {
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
      if (first) vrSpot = { stand: camera.position.clone(), ahead: new THREE.Vector3(0, 0, -1).applyQuaternion(facing) }
      controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -0.01).applyQuaternion(facing))
      controls.update()
      // The spots the video was filmed from, upright, for the Real look: the ones COLMAP placed itself when the room
      // says which (the splats look like the video from there), else every one
      const placed = path?.filter(spot => placedSpots?.placed?.includes(spot.frame))
      const steps = placed && placed.length > 1 ? placed : path
      if (steps && steps.length > 1) {
        spots.current = filmedSpots(steps.map(spot => {
          const [w, x, y, z] = spot.quaternion
          return {
            position: new THREE.Vector3(spot.position[0], spot.position[1], spot.position[2]).applyMatrix4(room.matrixWorld),
            quaternion: room.quaternion.clone().multiply(new THREE.Quaternion(x, y, z, w)),
          }
        }), camera, controls)
        spotIndex.current = 0
        setSpot(0)
        setSpotCount(steps.length)
        if (shown.current === 'real') spots.current.go(0)
      }
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
      // The Rebuilt look measures what it draws (its things and walls, meshes only: a grey box's edge lines would
      // catch a click a unit away); the other looks measure the filmed surface, steadier than a blurry splat
      const drawn = shown.current === 'rebuilt'
        ? [rebuiltGroup.current, shell].filter(part => part !== null) as THREE.Object3D[]
        : null
      const hit = drawn
        ? raycaster.intersectObjects(drawn, true).find(found => found.object instanceof THREE.Mesh)
        : raycaster.intersectObject(shown.current !== 'dots' && surface ? surface : cloud)[0]
      if (!hit) return
      const point = markPoint(hit)
      if (points.length === 2) {
        points.length = 0
        onCeiling.length = 0
        marks.clear()
      }
      points.push(point)
      onCeiling.push(onAssumedCeiling(hit))
      const marker = new THREE.Mesh(new THREE.SphereGeometry(markerRadius, 16, 12), new THREE.MeshBasicMaterial({ color: MARK }))
      marker.position.copy(point)
      marks.add(marker)
      if (points.length === 2) {
        marks.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: MARK })))
        measuredUnits.current = points[0].distanceTo(points[1])
        setDistance(points[0].distanceTo(points[1]) * scaleRef.current)
        setToCeiling(onCeiling.some(Boolean))
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
      onCeiling.length = 0
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

    // One loop for the screen and the headset: three.js runs it at the headset's rate while VR is on
    let last = -1
    renderer.setAnimationLoop(time => {
      const seconds = last < 0 ? 0 : Math.min(0.1, (time - last) / 1000)
      last = time
      if (renderer.xr.isPresenting) {
        vr.render(seconds)
      } else {
        spots.current?.tick(time)
        controls.update()
        renderer.render(scene, camera)
      }
    })

    return () => {
      disposed = true
      spots.current = null
      setSpotCount(0)
      renderer.setAnimationLoop(null)
      vr.dispose()
      vrRef.current = null
      resize.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onDown)
      renderer.domElement.removeEventListener('pointerup', onUp)
      renderer.domElement.removeEventListener('pointerdown', onGrab, { capture: true })
      renderer.domElement.removeEventListener('wheel', onWheel)
      movable.current?.dispose()
      movable.current = null
      rebuiltMovable.current?.dispose()
      rebuiltMovable.current = null
      if (issueMarks.current) {
        issueMarks.current.parent?.remove(issueMarks.current)
        disposeTree(issueMarks.current)
        issueMarks.current = null
      }
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
      <div className="absolute left-3 top-3 z-10 flex max-w-xs flex-col gap-2">
      <div className="pointer-events-none rounded-lg bg-black/60 px-3 py-2 text-sm text-white">
        {state === 'loading' && 'Loading the 3D room...'}
        {state === 'error' && 'The 3D room could not be loaded.'}
        {state === 'ready' && moving && (
          <span className="inline-flex items-center gap-1.5">
            <Hand size={15} />
            {held ? `Moving ${the(held)}. ${fitText(fit)}` : 'Click a thing and drag it across the floor.'}
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
            {picked === 2 && distance !== null && `${distance.toFixed(2)} m (${Math.round(distance * 100)} cm)${toCeiling ? "; the ceiling's height is assumed" : ''}`}
          </span>
        )}
        {vrError && <span className="mt-1 block text-xs text-red-300">{vrError}</span>}
        {state === 'ready' && roomSize && picked === 0 && !moving && (
          <span className="mt-1 block text-xs text-white/80">
            Room about {roomSize.width.toFixed(2)} × {roomSize.length.toFixed(2)} m
            {roomSize.heightAssumed
              ? ` (the ceiling wasn't filmed: ${roomSize.height.toFixed(2)} m assumed)`
              : `, ${roomSize.height.toFixed(2)} m tall`}; {roomSize.source}.
          </span>
        )}
      </div>
      {state === 'ready' && picked === 2 && distance !== null && canResize && !moving && (
        <form className="rounded-lg bg-black/60 px-3 py-2 text-xs text-white"
          onSubmit={event => {
            event.preventDefault()
            applyRealLength()
          }}>
          <p className="text-white/80">Measured it with a tape? Its real length sets the whole room's size:</p>
          <div className="mt-1.5 flex items-center gap-1.5">
            <input value={realCm} onChange={event => setRealCm(event.target.value)} inputMode="decimal" placeholder="74"
              className="w-16 rounded bg-white/90 px-2 py-1 text-slate-900" aria-label="Its real length in centimetres" />
            <span>cm</span>
            <button type="submit" disabled={scaleBusy}
              className="rounded bg-amber-500 px-2 py-1 font-medium text-black hover:bg-amber-400 disabled:opacity-60">
              {scaleBusy ? 'Resizing the room (about a minute)...' : 'Set the size'}
            </button>
          </div>
          {scaleError && <p className="mt-1 text-red-300">{scaleError}</p>}
        </form>
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
      {state === 'ready' && hasRebuilt && (
        <YourThings base={itemsBase} open={itemsOpen && showing === 'rebuilt'} onClose={() => setItemsOpen(false)}
          onReady={placeItem} />
      )}
      {state === 'ready' && hasRebuilt && (
        <Issues base={itemsBase} open={issuesOpen && showing === 'rebuilt'} onClose={() => setIssuesOpen(false)}
          onReport={showIssues} room={roomSize} />
      )}
      <p className="pointer-events-none absolute bottom-3 left-3 text-xs text-white/70">
        {showing === 'real' && spotCount > 1
          ? 'As filmed: step between the spots the video was filmed from (the arrows), and drag to look around.'
          : 'Drag to look around, scroll to walk, right-drag to step sideways.'}
      </p>
      {state === 'ready' && (
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          {hasRebuilt && showing === 'rebuilt' && (
            <button
              type="button"
              onClick={() => {
                setItemsOpen(!itemsOpen)
                setIssuesOpen(false)
              }}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs ${itemsOpen ? 'bg-emerald-500 text-white' : 'bg-black/60 text-white hover:bg-black/80'}`}
            >
              <Package size={14} /> Your things
            </button>
          )}
          {hasRebuilt && showing === 'rebuilt' && (
            <button
              type="button"
              onClick={() => {
                setIssuesOpen(!issuesOpen)
                setItemsOpen(false)
              }}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs ${issuesOpen ? 'bg-red-500 text-white' : 'bg-black/60 text-white hover:bg-black/80'}`}
            >
              <AlertTriangle size={14} /> Issues
            </button>
          )}
          {canVr && hasRebuilt && showing === 'rebuilt' && (
            <button
              type="button"
              disabled={inVr}
              onClick={() => {
                setVrError(null)
                vrRef.current?.enter().catch(() => setVrError("VR didn't start. Try again, or reload the page."))
              }}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3 py-2 text-xs text-white hover:bg-indigo-600 disabled:opacity-60"
            >
              <Glasses size={14} /> {inVr ? 'In VR' : 'Enter VR'}
            </button>
          )}
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
              {moving && held && showing === 'rebuilt' && (
                <>
                  <button type="button" onClick={() => rebuiltMovable.current?.turn(15)}
                    className="px-2 py-2 hover:bg-black/40" aria-label="Turn anticlockwise" title="Turn 15° anticlockwise">
                    <RotateCcw size={14} />
                  </button>
                  <button type="button" onClick={() => rebuiltMovable.current?.turn(-15)}
                    className="px-2 py-2 hover:bg-black/40" aria-label="Turn clockwise" title="Turn 15° clockwise">
                    <RotateCw size={14} />
                  </button>
                </>
              )}
              {moving && (
                <button
                  type="button"
                  onClick={() => (showing === 'rebuilt' ? rebuiltMovable.current : movable.current)?.reset()}
                  className="inline-flex items-center gap-1.5 px-3 py-2 hover:bg-black/40"
                >
                  <Undo2 size={14} /> Put back
                </button>
              )}
            </div>
          )}
          {showing === 'real' && spotCount > 1 && (
            <div className="flex items-center overflow-hidden rounded-lg bg-black/60 text-xs text-white">
              <button type="button" onClick={() => goSpot(spot - 1)} disabled={spot === 0}
                className="px-2 py-2 hover:bg-black/40 disabled:opacity-40" aria-label="The spot before" title="The spot before (←)">
                <ChevronLeft size={14} />
              </button>
              <input type="range" min={0} max={spotCount - 1} value={spot} onChange={e => goSpot(Number(e.target.value))}
                className="w-24 accent-amber-500" aria-label="Filmed spot" />
              <span className="px-2 tabular-nums">{spot + 1} / {spotCount}</span>
              <button type="button" onClick={() => goSpot(spot + 1)} disabled={spot === spotCount - 1}
                className="px-2 py-2 hover:bg-black/40 disabled:opacity-40" aria-label="The next spot" title="The next spot (→)">
                <ChevronRight size={14} />
              </button>
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
