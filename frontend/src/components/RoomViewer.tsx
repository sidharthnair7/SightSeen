import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark'
import { ArrowDown, ArrowUp, Camera, ChevronLeft, ChevronRight, ChevronsLeftRight, Eye, Glasses, Hand, Info, Loader2, Package, Ruler, RotateCcw, RotateCw, ShieldCheck, Sparkles, Undo2, X } from 'lucide-react'
import type { ReactNode } from 'react'
import { API_BASE, LIVE, STATIC } from '../api'
import Dock, { type DockItem } from './Dock'
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
// One look for every control on the room: on is amber, and a press gives a little
const TOOL = 'inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-[scale,background-color] duration-150 ease-out enabled:active:scale-[0.96] motion-reduce:transition-none motion-reduce:enabled:active:scale-100'
const OFF = 'glass-dark hover:bg-black/70'
const ON = 'bg-tape text-ink'
const KEY_WALK = 1.3 // metres a second with W A S D: a walking pace
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
  if (!frame.scale || !evidence) return "MapAnything's own scale, a rough guess until one length is measured with a tape"
  if (evidence.source === 'tape measure') return `checked with a tape measure (${evidence.what?.replace('-', ' ')})`
  return `rough, from ${evidence.plates ?? 'standard'} wall plates: measure one length with a tape`
}

/** How sure a length is, as a share either way. A tape measure, carried across the whole room: about 5%. Wall plates: 35%,
 * the one error measured so far (on the sample room they said 1.03 where a tape said 1.40), since how much plates agree with
 * each other says nothing about how far off they all are. MapAnything alone: its paper says 13 to 16% on average, and the
 * sample room was 28% off, so also 35%, which keeps the size check from judging an ad on a guess. */
function lengthTolerance(frame: RoomFrame): number {
  return frame.scale && frame.scaleEvidence?.source === 'tape measure' ? 0.05 : 0.35
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
export default function RoomViewer({ url, canResize = !STATIC, startOnChecks = false, startOnAbout = false, title, about, notice, codeAsked = null }: {
  url: string
  canResize?: boolean
  /** Open on the scam checks, or on what this room is */
  startOnChecks?: boolean
  startOnAbout?: boolean
  title: string
  /** What the About panel says */
  about: ReactNode
  /** A line under the room's name while its things are still being built */
  notice?: string | null
  /** The one-time code this room's link asked for, if any */
  codeAsked?: string | null
}) {
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
  const [issuesOpen, setIssuesOpen] = useState(startOnChecks)
  // On a phone the panel would cover the room, so it waits to be asked for
  const [aboutOpen, setAboutOpen] = useState(startOnAbout && typeof window !== 'undefined' && window.innerWidth >= 900)
  const [vrHint, setVrHint] = useState(false) // a browser with no headset: where VR is
  // See it real: Muse Image's picture of this view, beside the frozen 3D view it came from
  const [picture, setPicture] = useState<{ real: string; rough: string } | null>(null)
  const [painting, setPainting] = useState(false)
  const [pictureError, setPictureError] = useState<string | null>(null)
  const [split, setSplit] = useState(62)
  const grabView = useRef<() => Promise<{ blob: Blob; width: number; height: number } | null>>(async () => null)
  const showFrame = useRef<(frame: string) => void>(() => {})
  const issueMarks = useRef<THREE.Group | null>(null) // red boxes (damage) and yellow outlines (never filmed)
  const lastReport = useRef<IssuesReport | null>(null) // kept for when the things arrive after the report does
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
  const startSpot = useRef<THREE.Vector3 | null>(null) // where the room's opening view stands
  const itemMeshes = useRef(new Map<string, THREE.Mesh>()) // the student's things in the room
  const lookTowards = useRef<(point: THREE.Vector3) => void>(() => {})
  const overview = useRef<() => void>(() => {}) // the Rebuilt look's opening view: a corner, the whole room in sight
  const walkBy = useRef<(metres: number) => void>(() => {})
  const stepBy = useRef<(metres: number) => void>(() => {})
  const [realAspect, setRealAspect] = useState<number | null>(null) // the phone's picture, width over height
  // The room's API address (/api/requests/{id}), from its room.ply URL: the student's things live under it
  const itemsBase = url.split('?')[0].replace('/room.ply', '')
  const showing = pick(look, hasSplats, hasSurface, hasWalk, hasRebuilt)
  const available: Record<Look, boolean> = { rebuilt: hasRebuilt, real: hasSplats, walk: hasWalk, solid: hasSurface, dots: true }
  const main = (['rebuilt', 'real'] as const).filter(option => available[option])
  const looks: Look[] = DEBUG ? (['rebuilt', 'real', 'walk', 'solid', 'dots'] as const).filter(option => available[option]) : main

  useEffect(() => {
    if (material.current) material.current.size = baseSize.current * dots
  }, [dots, state])

  useEffect(() => {
    if (hasRebuilt && lastReport.current) showIssues(lastReport.current)
  }, [hasRebuilt])

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
      const body = await response.json().catch(() => null)
      for (let wait = 0; body?.pending; wait++) {
        if (wait >= 100) throw new Error('It is taking a while. Reload the page in a few minutes to see the new size.')
        await new Promise(done => window.setTimeout(done, 3000))
        const state = await fetch(itemsBase).then(r => r.json()).then(r => r.scaleState).catch(() => 'running')
        if (state === 'done') break
        if (state && state !== 'running') throw new Error(state)
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

  // W A S D or the arrow keys walk you round the room, at a walking pace in real metres
  useEffect(() => {
    if (showing !== 'rebuilt' && showing !== 'walk') return
    const KEYS: Record<string, [number, number]> = {
      w: [1, 0], arrowup: [1, 0], s: [-1, 0], arrowdown: [-1, 0], a: [0, -1], arrowleft: [0, -1], d: [0, 1], arrowright: [0, 1],
    }
    const pressed = new Set<string>()
    const onDown = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)
      const key = e.key.toLowerCase()
      if (typing || !(key in KEYS) || e.metaKey || e.ctrlKey || e.altKey) return
      e.preventDefault()
      pressed.add(key)
    }
    const onUp = (e: KeyboardEvent) => pressed.delete(e.key.toLowerCase())
    const onBlur = () => pressed.clear()
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const seconds = Math.min(0.1, (now - last) / 1000)
      last = now
      if (pressed.size) {
        const pace = (KEY_WALK * seconds) / scaleRef.current
        let ahead = 0
        let side = 0
        pressed.forEach(key => {
          ahead += KEYS[key][0]
          side += KEYS[key][1]
        })
        if (ahead) walkBy.current(Math.sign(ahead) * pace)
        if (side) stepBy.current(Math.sign(side) * pace)
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    window.addEventListener('blur', onBlur)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [showing])

  // One of the student's things, built (its own middle at x = z = 0 and its bottom at y = 0, in the room's units):
  // square to the walls, on the free floor nearest the room's middle (the middle itself when nowhere is free), shown
  // with the rebuilt things and picked up like them
  const placeItem = (item: Item, fresh = false, tries = 5) => {
    const group = rebuiltGroup.current
    const shell = shellMesh.current
    if (!group || !shell) return
    fetchModel(item.folder ?? `${itemsBase}/items/${item.id}/`, 'item.ply', lightModels.current).then(async response => {
      if (!response) {
        if (tries > 0) window.setTimeout(() => placeItem(item, fresh, tries - 1), 3000)
        return
      }
      const geometry = new PLYLoader().parse(await response.arrayBuffer())
      geometry.computeVertexNormals()
      geometry.computeBoundingBox()
      const size = geometry.boundingBox!.getSize(new THREE.Vector3())
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
      mesh.name = `your ${item.name}`
      // Built at the room's scale then; the room's may have changed since (a tape length): the same real size
      const resize = item.roomScale ? item.roomScale / scaleRef.current : 1
      mesh.scale.setScalar(resize)
      size.multiplyScalar(resize)
      const yawDegrees = -THREE.MathUtils.radToDeg(shell.rotation.y)
      const opening = startSpot.current && { x: startSpot.current.x, z: startSpot.current.z, reach: 1.0 / scaleRef.current }
      const spot = rebuiltMovable.current?.freeSpot([size.x, size.y, size.z], yawDegrees, opening)
        ?? rebuiltMovable.current?.freeSpot([size.x, size.y, size.z], yawDegrees)
        ?? [shell.position.x, size.y / 2, shell.position.z]
      mesh.position.set(spot[0], 0, spot[2])
      mesh.rotation.y = shell.rotation.y
      group.add(mesh)
      itemMeshes.current.set(item.id, mesh)
      if (fresh) {
        setItemsOpen(false)
        lookTowards.current(mesh.position.clone().setY(0.4 / scaleRef.current))
      }
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
    const before = shown.current
    shown.current = showing
    showSplats(splatMesh.current, showing === 'real')
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
    if (showing === 'rebuilt' && before !== 'rebuilt') overview.current()
    // Spark only rebuilds its splats when the camera moves more than 1 mm or its set of splats changes, and showing or
    // hiding them is neither: without this they stay invisible until the first drag, or ghost over the Rebuilt look
    refreshSplats.current()
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
      walkBy.current(THREE.MathUtils.clamp(-e.deltaY * 0.002, -WALK_STEP, WALK_STEP))
    }
    grabView.current = () => {
      renderer.render(scene, camera) // straight before reading it: the canvas is only readable in the same moment
      const source = renderer.domElement
      const shrink = Math.min(1, 1024 / Math.max(source.width, source.height))
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(source.width * shrink)
      canvas.height = Math.round(source.height * shrink)
      canvas.getContext('2d')?.drawImage(source, 0, 0, canvas.width, canvas.height)
      return new Promise(done => canvas.toBlob(blob => done(blob ? { blob, width: canvas.width, height: canvas.height } : null), 'image/jpeg', 0.92))
    }
    walkBy.current = amount => {
      const ahead = camera.getWorldDirection(new THREE.Vector3()).setY(0)
      if (ahead.lengthSq() < 1e-6) return
      ahead.normalize().multiplyScalar(amount)
      camera.position.add(ahead)
      controls.target.add(ahead)
    }
    stepBy.current = amount => {
      const ahead = camera.getWorldDirection(new THREE.Vector3()).setY(0)
      if (ahead.lengthSq() < 1e-6) return
      const side = new THREE.Vector3(-ahead.z, 0, ahead.x).normalize().multiplyScalar(amount)
      camera.position.add(side)
      controls.target.add(side)
    }
    // Wherever the wheel, a key or a right-drag leads, the view stays in the room
    const keepInside = () => {
      if (!shell || shown.current === 'real') return
      const local = shell.worldToLocal(camera.position.clone())
      const { width, height, depth } = (shell.geometry as THREE.BoxGeometry).parameters
      const inside = local.clone()
      inside.x = THREE.MathUtils.clamp(inside.x, -width / 2 + 0.1, width / 2 - 0.1)
      inside.y = THREE.MathUtils.clamp(inside.y, -height / 2 + 0.1, height / 2 - 0.1)
      inside.z = THREE.MathUtils.clamp(inside.z, -depth / 2 + 0.1, depth / 2 - 0.1)
      if (inside.equals(local)) return
      const back = shell.localToWorld(inside).sub(camera.position)
      camera.position.add(back)
      controls.target.add(back)
    }
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false })

    lookTowards.current = point => {
      controls.target.copy(camera.position).addScaledVector(point.clone().sub(camera.position).normalize(), 0.01)
    }
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
    const loadSurface = () => new PLYLoader().load(url.replace('/room.ply', '/room-surface.ply'), geometry => {
      if (disposed || surface) return
      surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
      surface.visible = shown.current === 'solid'
      surfaceMesh.current = surface
      room.add(surface)
      setHasSurface(true)
    }, undefined, () => {})

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
    let splats: SplatMesh | null = null
    let replaced = false // true once the room's movable pieces stand in for the single splat file
    // A headset's browser never loads them: it shows the Rebuilt look only, and they're 20 MB it has no memory to spare for
    vrSupported().then(headset => {
      if (disposed || headset) return
      const loaded = new SplatMesh({ url: url.replace('/room.ply', '/room-splats.ply') })
      splats = loaded
      loaded.quaternion.set(1, 0, 0, 0)
      loaded.visible = false
      loaded.initialized.then(() => {
        if (disposed || replaced) return
        splatMesh.current = loaded
        loaded.visible = shown.current === 'real'
        room.add(loaded)
        setHasSplats(true)
        requestAnimationFrame(() => refreshSplats.current())
      }, () => {})
    })

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
          if (!rebuilt) {
            if (!disposed) loadSurface() // its things aren't built yet: the solid surface is the best look there is
            return
          }
          if (disposed) {
            rebuilt.movable.dispose()
            return
          }
          rebuiltGroup.current = rebuilt.group
          rebuiltMovable.current = rebuilt.movable
          rebuiltTint.current = rebuilt.showNotFilmed
          rebuilt.group.visible = shown.current === 'rebuilt'
          scene.add(rebuilt.group)
          // The opening view: the corner clearest of the room's things, at eye height, looking at the middle
          overview.current = () => {
            const walls = shell
            if (!walls) return
            walls.updateMatrixWorld(true)
            const { width, height, depth } = (walls.geometry as THREE.BoxGeometry).parameters
            const boxes = rebuilt.group.children.map(part => new THREE.Box3().setFromObject(part))
            const eye = Math.min(1.6 / scaleRef.current, height - 0.15)
            let best: THREE.Vector3 | null = null
            let clearest = -Infinity
            for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
              const spot = walls.localToWorld(new THREE.Vector3(sx * (width / 2 - 0.3), 0, sz * (depth / 2 - 0.3))).setY(eye)
              const clear = Math.min(...boxes.map(box => box.distanceToPoint(spot)))
              if (clear > clearest) {
                clearest = clear
                best = spot
              }
            }
            if (!best) return
            camera.position.copy(best)
            const middle = walls.position.clone().setY(0.95 / scaleRef.current) // about desk height: the room, more than the floor
            controls.target.copy(best).addScaledVector(middle.sub(best).normalize(), 0.01)
            startSpot.current = best.clone()
            controls.update()
          }
          setHasRebuilt(true)
          setLook('rebuilt')
        }, () => {})
      }
      // The room's things as movable pieces (the same splats, split by SAM 3.1), when the room has them
      if (!headset) movableThings(assets, room, scene, camera, renderer.domElement, frame && found ? found : null, (name, fits) => {
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
        if (splats) {
          room.remove(splats)
          splats.dispose()
        }
        showSplats(pieces.parts, shown.current === 'real')
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
      startSpot.current = camera.position.clone()
      controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -0.01).applyQuaternion(facing))
      controls.update()
      // The spots the video was filmed from, upright, for the Real look: the ones COLMAP placed itself when the room
      // says which (the splats look like the video from there), else every one
      if (first?.fovY) setRealAspect(Math.tan(THREE.MathUtils.degToRad(first.fovX / 2)) / Math.tan(THREE.MathUtils.degToRad(first.fovY / 2)))
      showFrame.current = frame => {
        const spot = path?.find(item => item.frame === frame)
        if (!spot) return
        const [w, x, y, z] = spot.quaternion
        const turn = room.quaternion.clone().multiply(new THREE.Quaternion(x, y, z, w))
        const at = new THREE.Vector3(spot.position[0], spot.position[1], spot.position[2]).applyMatrix4(room.matrixWorld)
        camera.position.copy(at)
        controls.target.copy(at).add(new THREE.Vector3(0, 0, -0.01).applyQuaternion(turn))
        startSpot.current = at.clone()
        controls.update()
      }
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
    if (DEBUG) loadSurface()

    // A hiccup on the way (a host's cold start, a dropped connection) shouldn't end the page, so it tries again first
    const loadRoom = (triesLeft: number) => new PLYLoader().load(url, geometry => {
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
      if (disposed) return
      if (triesLeft > 0) window.setTimeout(() => !disposed && loadRoom(triesLeft - 1), 2000)
      else setState('error')
    })
    loadRoom(2)

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
        keepInside()
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
      if (!replaced) splats?.dispose()
      splatMesh.current = null
      refreshSplats.current = () => {}
      refitView.current = () => {}
      scene.remove(spark)  // the shared renderers stay alive for the next room
      el.removeChild(renderer.domElement)
    }
  }, [url])

  const seeItReal = async () => {
    setPictureError(null)
    setPainting(true)
    try {
      const view = await grabView.current()
      if (!view) throw new Error('That view couldn’t be read. Try again.')
      const form = new FormData()
      form.append('view', view.blob, 'view.jpg')
      form.append('width', String(view.width))
      form.append('height', String(view.height))
      const response = await fetch(`${API_BASE}/api/imagine`, { method: 'POST', body: form })
      const body = await response.json().catch(() => null)
      if (!response.ok) throw new Error(body?.message || `Something went wrong (${response.status}).`)
      setSplit(62)
      setPicture({ real: `data:image/png;base64,${body.image}`, rough: URL.createObjectURL(view.blob) })
    } catch (problem) {
      setPictureError(problem instanceof Error ? problem.message : String(problem))
    } finally {
      setPainting(false)
    }
  }

  const panel = (which: 'things' | 'issues' | 'about') => {
    setItemsOpen(which === 'things' && !itemsOpen)
    setIssuesOpen(which === 'issues' && !issuesOpen)
    setAboutOpen(which === 'about' && !aboutOpen)
  }
  const rebuiltLook = hasRebuilt && showing === 'rebuilt'
  const canMove = (hasMovable && showing === 'real') || rebuiltLook
  const dockItems: (DockItem | false)[] = [
    { label: 'Before you sign', icon: <ShieldCheck />, active: issuesOpen, onClick: () => panel('issues') },
    rebuiltLook && { label: 'Your things', icon: <Package />, active: itemsOpen, onClick: () => panel('things') },
    LIVE && rebuiltLook && { label: painting ? 'Painting...' : 'See it real', icon: painting ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : <Sparkles className="text-tape" />, disabled: painting, onClick: seeItReal },
    {
      label: inVr ? 'In VR' : canVr ? 'Enter VR' : 'VR on a Meta Quest', icon: <Glasses />, active: inVr || vrHint, disabled: inVr || (canVr && !hasRebuilt),
      onClick: () => {
        if (!canVr) return setVrHint(!vrHint)
        setVrError(null)
        vrRef.current?.enter().catch(() => setVrError("VR didn't start. Try again, or reload the page."))
      },
    },
    canMove && {
      label: 'Move things', icon: <Hand />, active: moving, separator: true,
      onClick: () => {
        const next = !moving
        moveOn.current = next
        ;(showing === 'rebuilt' ? rebuiltMovable.current : movable.current)?.setOn(next)
        setMoving(next)
        if (!next) setHeld(null)
      },
    },
    rebuiltLook && hasMaterials && { label: 'Photo walls', icon: <Camera />, active: photoWalls, onClick: () => setPhotoWalls(!photoWalls) },
    rebuiltLook && {
      label: 'Not filmed', icon: <Eye />, active: notFilmed,
      onClick: () => {
        const next = !notFilmed
        rebuiltTint.current?.(next)
        setNotFilmed(next)
      },
    },
    { label: 'About this room', icon: <Info />, active: aboutOpen, separator: true, onClick: () => panel('about') },
  ]

  return (
    <div className="relative h-dvh w-full overflow-hidden bg-[#10161a]">
      <div ref={host} className="absolute inset-0"
        style={showing === 'real' && realAspect && realAspect < 1
          ? { inset: '0 auto 0 50%', translate: '-50% 0', height: '100%', aspectRatio: realAspect, maxWidth: '100%', overflow: 'hidden' }
          : undefined} />

      {/* The room's name and what's going on, top left */}
      <div className="pointer-events-none absolute left-3 top-[4.5rem] z-10 flex max-w-[min(21rem,calc(100%-1.5rem))] flex-col gap-2 sm:top-20 [&>button]:pointer-events-auto [&>form]:pointer-events-auto">
        <div className="glass-dark rounded-xl px-3 py-2 text-sm">
          <p className="mb-1 truncate text-[10px] font-semibold uppercase tracking-[0.2em] text-[#fdf2ed]/90">{title}</p>
          {state === 'loading' && (
            <span className="inline-flex items-center gap-1.5">
              <Loader2 size={15} className="animate-spin motion-reduce:animate-none" /> Loading the 3D room...
            </span>
          )}
          {state === 'error' && 'The 3D room could not be loaded. Reload the page to try again.'}
          {state === 'ready' && moving && (
            <span className="inline-flex items-center gap-1.5">
              <Hand size={15} className="shrink-0" />
              {held ? `Moving ${the(held)}. ${fitText(fit)}` : 'Tap or click a thing and drag it across the floor.'}
            </span>
          )}
          {state === 'ready' && !moving && notFilmed && showing === 'rebuilt' && (
            <span className="inline-flex items-center gap-1.5">
              <Eye size={15} className="shrink-0" />
              Purple: what the video never saw, filled in by AI.
            </span>
          )}
          {state === 'ready' && !moving && !(notFilmed && showing === 'rebuilt') && (
            <span className="inline-flex items-center gap-1.5">
              <Ruler size={15} className="shrink-0" />
              {picked === 0 && 'Click or tap two points to measure.'}
              {picked === 1 && 'Now the second point.'}
              {picked === 2 && distance !== null && `${distance.toFixed(2)} m${roomSize ? `, give or take ${Math.max(1, Math.round(distance * 100 * roomSize.tolerance / 2))} cm` : ''}${toCeiling ? "; the ceiling's height is assumed" : ''}`}
            </span>
          )}
          {vrError && <span className="mt-1 block text-xs text-red-300">{vrError}</span>}
          {painting && (
            <span className="mt-1 flex items-center gap-1.5 text-xs text-tape">
              <Loader2 size={13} className="animate-spin motion-reduce:animate-none" /> Muse Image is painting this view (about 30 seconds).
            </span>
          )}
          {pictureError && <span className="mt-1 block text-xs text-red-300">{pictureError}</span>}
          {state === 'ready' && roomSize && picked === 0 && !moving && (
            <span className="mt-1 block text-xs text-[#fdf2ed]/75">
              Room about {roomSize.width.toFixed(2)} × {roomSize.length.toFixed(2)} m
              {roomSize.heightAssumed
                ? ` (the ceiling wasn't filmed: ${roomSize.height.toFixed(2)} m assumed)`
                : `, ${roomSize.height.toFixed(2)} m tall`}; {roomSize.source}.
            </span>
          )}
        </div>
        {notice && (
          <p className="glass-dark flex items-center gap-2 rounded-xl px-3 py-2 text-xs">
            <Loader2 size={14} className="shrink-0 animate-spin motion-reduce:animate-none" /> {notice}
          </p>
        )}
        {state === 'ready' && picked === 2 && distance !== null && canResize && !moving && (
          <form className="glass-dark rounded-xl px-3 py-2 text-xs"
            onSubmit={event => {
              event.preventDefault()
              applyRealLength()
            }}>
            <p className="text-[#fdf2ed]/80">Measured it with a tape? Its real length sets the whole room's size:</p>
            <div className="mt-1.5 flex items-center gap-1.5">
              <input value={realCm} onChange={event => setRealCm(event.target.value)} inputMode="decimal" placeholder="74"
                className="w-16 rounded bg-white/90 px-2 py-1 text-base text-slate-900 sm:text-xs" aria-label="Its real length in centimetres" />
              <span>cm</span>
              <button type="submit" disabled={scaleBusy}
                className="rounded bg-tape px-2 py-1 font-medium text-ink hover:brightness-105 disabled:opacity-60">
                {scaleBusy ? 'Resizing the room (about a minute)...' : 'Set the size'}
              </button>
            </div>
            {scaleError && <p className="mt-1 text-red-300">{scaleError}</p>}
          </form>
        )}
        {picked > 0 && (
          <button type="button" onClick={() => clear.current()} className={`${TOOL} ${OFF} self-start`}>
            <X size={14} /> Clear the measurement
          </button>
        )}
      </div>

      {/* Which look, top right */}
      {state === 'ready' && looks.length > 1 && (
        <div className="absolute right-3 top-[4.5rem] z-10 flex gap-1 sm:top-20" role="group" aria-label="How the room looks">
          {looks.map(option => (
            <button key={option} type="button" onClick={() => setLook(option)} aria-pressed={showing === option} className="glass-chip rounded-lg">
              {LABELS[option]}
            </button>
          ))}
          {showing === 'dots' && (
            <label className="glass-chip flex items-center gap-2 rounded-lg">
              Dot size
              <input type="range" min={0.5} max={3} step={0.25} value={dots} onChange={e => setDots(Number(e.target.value))} className="w-20 accent-tape" />
            </label>
          )}
        </div>
      )}

      {state === 'ready' && hasRebuilt && (
        <YourThings base={itemsBase} room={itemsBase.split('/').pop() ?? ''} scale={() => scaleRef.current}
          open={itemsOpen && showing === 'rebuilt'} onClose={() => setItemsOpen(false)} onReady={placeItem}
          onFind={id => {
            const mesh = itemMeshes.current.get(id)
            if (mesh) lookTowards.current(mesh.position.clone().setY(0.4 / scaleRef.current))
          }} />
      )}
      {state === 'ready' && (
        <Issues base={itemsBase} roomId={itemsBase.split('/').pop() ?? ''} open={issuesOpen} stage={notice ?? null} codeAsked={codeAsked}
          onClose={() => setIssuesOpen(false)} room={roomSize}
          onReport={report => {
            lastReport.current = report
            showIssues(report)
          }}
          onShowFrame={frame => {
            showFrame.current(frame)
            if (window.innerWidth < 640) setIssuesOpen(false) // the panel covers the view on a phone
          }} />
      )}
      {aboutOpen && (
        <div className="stage-panel">
          <div className="mb-3 flex items-center justify-between">
            <p className="inline-flex items-center gap-1.5 font-medium text-slate-900"><Info size={16} /> About this room</p>
            <button type="button" onClick={() => setAboutOpen(false)} className="rounded p-1 hover:bg-slate-100" aria-label="Close">
              <X size={16} />
            </button>
          </div>
          {about}
        </div>
      )}

      {/* What's in reach, along the bottom */}
      <p className="pointer-events-none absolute bottom-5 left-4 z-10 hidden max-w-[14rem] text-xs text-white/70 [text-shadow:0_1px_2px_rgb(0_0_0/0.6)] xl:block">
        {showing === 'real' && spotCount > 1
          ? 'As filmed: step between the spots the video was filmed from (the arrows), and drag to look around.'
          : 'Drag to look around. W A S D or scroll to walk.'}
      </p>
      {state === 'ready' && (
        <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex flex-col items-center gap-2 [&>*]:pointer-events-auto">
          {vrHint && !canVr && (
            <p className="glass-dark max-w-[calc(100vw-1.5rem)] rounded-xl px-3 py-2 text-center text-xs sm:max-w-md">
              Stand in this room at its real size: open this page in a Meta Quest’s browser and press Enter VR. Left stick walks,
              right stick turns, and the trigger measures between two spots.
            </p>
          )}
          {moving && (held || showing === 'rebuilt') && (
            <div className="glass-dark flex overflow-hidden rounded-xl text-xs">
              {held && showing === 'rebuilt' && (
                <>
                  <button type="button" onClick={() => rebuiltMovable.current?.turn(15)} className="px-2.5 py-2 hover:bg-white/10" aria-label="Turn anticlockwise" title="Turn 15° anticlockwise">
                    <RotateCcw size={14} />
                  </button>
                  <button type="button" onClick={() => rebuiltMovable.current?.turn(-15)} className="px-2.5 py-2 hover:bg-white/10" aria-label="Turn clockwise" title="Turn 15° clockwise">
                    <RotateCw size={14} />
                  </button>
                </>
              )}
              <button type="button" onClick={() => (showing === 'rebuilt' ? rebuiltMovable.current : movable.current)?.reset()} className="inline-flex items-center gap-1.5 px-3 py-2 hover:bg-white/10">
                <Undo2 size={14} /> Put back
              </button>
            </div>
          )}
          {showing === 'real' && spotCount > 1 && (
            <div className="glass-dark flex items-center overflow-hidden rounded-xl text-xs">
              <button type="button" onClick={() => goSpot(spot - 1)} disabled={spot === 0} className="px-2.5 py-2 hover:bg-white/10 disabled:opacity-40" aria-label="The spot before" title="The spot before (←)">
                <ChevronLeft size={14} />
              </button>
              <input type="range" min={0} max={spotCount - 1} value={spot} onChange={e => goSpot(Number(e.target.value))} className="w-28 accent-tape" aria-label="Filmed spot" />
              <span className="px-2 tabular-nums">{spot + 1} / {spotCount}</span>
              <button type="button" onClick={() => goSpot(spot + 1)} disabled={spot === spotCount - 1} className="px-2.5 py-2 hover:bg-white/10 disabled:opacity-40" aria-label="The next spot" title="The next spot (→)">
                <ChevronRight size={14} />
              </button>
            </div>
          )}
          {showing === 'rebuilt' && !issuesOpen && !aboutOpen && !itemsOpen && (
            <div className="glass-dark flex overflow-hidden rounded-xl [@media(hover:hover)]:hidden">
              <button type="button" onClick={() => walkBy.current(0.3 / scaleRef.current)} className="px-3 py-2" aria-label="Walk forward"><ArrowUp size={14} /></button>
              <button type="button" onClick={() => walkBy.current(-0.3 / scaleRef.current)} className="px-3 py-2" aria-label="Walk back"><ArrowDown size={14} /></button>
            </div>
          )}
          <Dock items={dockItems.filter((item): item is DockItem => Boolean(item))} />
        </div>
      )}

      {picture && (
        <div className="absolute inset-0 z-30 select-none overflow-hidden bg-black">
          <img src={picture.rough} alt="The 3D room from this view" className="absolute inset-0 h-full w-full" />
          <img src={picture.real} alt="Muse Image's picture of this view" className="absolute inset-0 h-full w-full"
            style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
          <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_8px_rgb(0_0_0/0.5)]" style={{ left: `${split}%` }}>
            <span className="absolute left-1/2 top-1/2 grid h-8 w-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-slate-900 shadow-lg">
              <ChevronsLeftRight size={16} />
            </span>
          </div>
          <input type="range" min={0} max={100} value={split} onChange={event => setSplit(Number(event.target.value))}
            aria-label="Slide between Muse Image's picture and the 3D room" className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
          <span className="glass-dark pointer-events-none absolute bottom-3 left-3 rounded-lg px-2 py-1 text-xs font-medium">AI picture</span>
          <span className="glass-dark pointer-events-none absolute bottom-3 right-3 rounded-lg px-2 py-1 text-xs font-medium">The 3D room</span>
          <div className="pointer-events-none absolute inset-x-3 top-[4.5rem] flex items-start justify-between gap-3 sm:top-20">
            <p className="glass-dark max-w-md rounded-lg px-3 py-2 text-xs">
              Muse Image’s picture of this view, with everything left where it stands. It’s for the feel of the room, so no
              check or measurement ever uses it. Those all come from the 3D room.
            </p>
            <button type="button" onClick={() => {
              URL.revokeObjectURL(picture.rough)
              setPicture(null)
            }} className={`${TOOL} ${OFF} pointer-events-auto relative shrink-0`}>
              <X size={14} /> Back to the 3D room
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/** Shows or hides the room's splats: the single file, or every one of its movable pieces. Each piece gets the flag
 *  itself, not just their group, so a desktop browser can't draw pieces over the Rebuilt look while the group is hidden. */
function showSplats(root: THREE.Object3D | null, show: boolean) {
  root?.traverse(part => {
    part.visible = show
  })
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
