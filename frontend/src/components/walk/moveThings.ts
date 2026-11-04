import * as THREE from 'three'
import { SplatMesh } from '@sparkjsdev/spark'
import type { RoomShell } from './types'

/** room-splat-objects.json (worker/splat_objects.py): the room's splats split by SAM 3.1 into its things and the rest. */
type SplatParts = {
  things: { name: string; file: string; centre: number[]; size: number[]; yawDegrees: number }[]
  rest: { file: string }
}

/** One thing that can be picked up: what moves, and its box in the upright room (metres) that the pointer grabs. */
export type Piece = {
  name: string
  object: THREE.Object3D
  centre: number[]
  size: number[]
  yawDegrees: number
  /** A move across the upright room in the object's own parent's frame (the splats sit in the turned room group). */
  toLocal?: (move: THREE.Vector3) => THREE.Vector3
  /** A grey box standing in for a thing SAM 3D couldn't build, at its measured size: a small one is clutter and never
   * in the way (take 4's "sock" measured 51 cm tall); a big one is furniture all the same */
  placeholder?: boolean
}

export type Movable = {
  /** Turns picking things up on or off. */
  setOn: (on: boolean) => void
  /** Puts every thing back where the video found it. */
  reset: () => void
  /** True when a pointer-down started dragging a thing (the viewer then skips its own turning and measuring). */
  grab: (event: PointerEvent) => boolean
  /** One more thing to pick up (one of the student's own, built after the room loaded). */
  add: (piece: Piece) => void
  /** Turns the thing last picked up about its own middle, in degrees (things moved within the turned room group, the
   * splats, don't turn). */
  turn: (degrees: number) => void
  /** Where a new thing of this size and turn can stand: the free floor nearest the room's middle, clear of every
   * thing standing on the floor and inside the walls (its middle, in the room), or null when nowhere is free. */
  freeSpot: (size: number[], yawDegrees: number, avoid?: { x: number; z: number; reach: number } | null) => number[] | null
  dispose: () => void
}

/** How a held thing sits among the others, in real metres: to the nearest wall, to the nearest thing standing on the
 * floor, or how far it runs into one (footprints on the floor, in the walls' directions). */
export type Fit = { wall: number; nearest: { name: string; gap: number } | null; into: { name: string; depth: number } | null }

const FLOOR_REACH = 0.15 // metres: a thing whose bottom is this close to the floor stands on it
const CLUTTER = 0.1 // metres: anything lower is clutter, not in the way
const PLACEHOLDER_MIN = 0.4 // metres: a grey box narrower than this both ways is clutter too
const SPOT_STEP = 0.05 // room units between the spots tried for a new thing
const SPOT_MARGIN = 0.03 // room units kept clear round a new thing

/**
 * Picking up things and moving them across the floor: an invisible box round each thing is what the pointer grabs,
 * and the thing moves with it. Things stay inside the walls, and `changed` hears how the held one fits. `fixed` things
 * (the window, the door, the heater) can't be picked up but still take up room.
 */
export function makeMovable(
  pieces: Piece[],
  scene: THREE.Scene,
  camera: THREE.Camera,
  dom: HTMLElement,
  shell: RoomShell | null,
  changed: (name: string | null, fit?: Fit | null) => void,
  refresh: () => void = () => {},
  options: { fixed?: Piece[]; scale?: () => number } = {},
): Movable {
  const boxes = new THREE.Group()
  const fixed = new THREE.Group() // in the way, not movable
  scene.add(boxes, fixed)
  type Hold = { piece: Piece; home: THREE.Vector3; boxHome: THREE.Vector3; turnHome: number; boxTurnHome: number }
  const held = new Map<THREE.Object3D, Hold>()
  const boxFor = (piece: Piece) => {
    const box = new THREE.Mesh(new THREE.BoxGeometry(piece.size[0], piece.size[1], piece.size[2]),
      new THREE.MeshBasicMaterial({ visible: false }))
    box.position.set(piece.centre[0], piece.centre[1], piece.centre[2])
    box.rotation.y = -THREE.MathUtils.degToRad(piece.yawDegrees)
    box.name = piece.name
    box.userData.placeholder = piece.placeholder === true
    return box
  }
  const addPiece = (piece: Piece) => {
    const box = boxFor(piece)
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: '#f59e0b' }))
    outline.visible = false
    box.add(outline)
    boxes.add(box)
    held.set(box, { piece, home: piece.object.position.clone(), boxHome: box.position.clone(),
      turnHome: piece.object.rotation.y, boxTurnHome: box.rotation.y })
  }
  pieces.forEach(addPiece)
  options.fixed?.forEach(piece => fixed.add(boxFor(piece)))
  boxes.visible = false
  fixed.visible = false
  const scale = options.scale ?? (() => 1)

  // The walls' own directions: across (the shell's x) and along (its z), from its middle
  const wallYaw = shell ? THREE.MathUtils.degToRad(shell.yawDegrees) : 0
  const across = new THREE.Vector3(Math.cos(wallYaw), 0, Math.sin(wallYaw))
  const along = new THREE.Vector3(-Math.sin(wallYaw), 0, Math.cos(wallYaw))
  const middle = shell ? new THREE.Vector3(...shell.centre).setY(0) : new THREE.Vector3()
  /** Half a footprint's size in the walls' directions, for a thing `width` by `depth` turned `rotationY`. */
  const halves = (width: number, depth: number, rotationY: number) => {
    const turn = rotationY + wallYaw // its turn against the walls
    const [c, s] = [Math.abs(Math.cos(turn)), Math.abs(Math.sin(turn))]
    return { hx: (c * width + s * depth) / 2, hz: (s * width + c * depth) / 2 }
  }
  /** A box's footprint in the walls' directions, at `position`: its middle (x across, z along) and half-sizes. */
  const footprint = (box: THREE.Mesh, position = box.position) => {
    const { width, depth } = (box.geometry as THREE.BoxGeometry).parameters
    const offset = position.clone().sub(middle)
    return { x: offset.dot(across), z: offset.dot(along), ...halves(width, depth, box.rotation.y) }
  }
  // In the way: standing on the floor, and taller than clutter you'd pick up (a sock, a bottle on its side)
  const standing = (box: THREE.Mesh) => {
    const { height } = (box.geometry as THREE.BoxGeometry).parameters
    return box.position.y - height / 2 < FLOOR_REACH / scale() && height >= CLUTTER / scale()
  }
  // A grey box under PLACEHOLDER_MIN across is clutter whose measured size is a guess
  const clutter = (box: THREE.Mesh) => {
    const { width, depth } = (box.geometry as THREE.BoxGeometry).parameters
    return box.userData.placeholder && Math.max(width, depth) * scale() < PLACEHOLDER_MIN
  }
  const others = (box: THREE.Mesh | null) => [...boxes.children, ...fixed.children]
    .filter(other => other !== box && !clutter(other as THREE.Mesh) && standing(other as THREE.Mesh)) as THREE.Mesh[]

  const fitOf = (box: THREE.Mesh): Fit | null => {
    if (!shell) return null
    const metres = scale()
    const f = footprint(box)
    const wall = Math.min(shell.size[0] / 2 - f.hx - Math.abs(f.x), shell.size[2] / 2 - f.hz - Math.abs(f.z))
    let nearest: Fit['nearest'] = null
    let into: Fit['into'] = null
    for (const other of others(box)) {
      const g = footprint(other)
      const gapX = Math.abs(f.x - g.x) - (f.hx + g.hx)
      const gapZ = Math.abs(f.z - g.z) - (f.hz + g.hz)
      if (gapX < 0 && gapZ < 0) {
        const depth = Math.min(-gapX, -gapZ) * metres
        if (!into || depth > into.depth) into = { name: other.name, depth }
      } else {
        const gap = Math.hypot(Math.max(gapX, 0), Math.max(gapZ, 0)) * metres
        if (!nearest || gap < nearest.gap) nearest = { name: other.name, gap }
      }
    }
    return { wall: Math.max(0, wall) * metres, nearest, into }
  }

  const raycaster = new THREE.Raycaster()
  const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
  let on = false
  let holding: THREE.Mesh | null = null
  const last = new THREE.Vector3()

  const pointerRay = (event: PointerEvent) => {
    const rect = dom.getBoundingClientRect()
    raycaster.setFromCamera(new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    ), camera)
  }
  const onFloor = (event: PointerEvent) => {
    pointerRay(event)
    return raycaster.ray.intersectPlane(floor, new THREE.Vector3())
  }

  // Keeps a moved thing's box inside the walls, in the walls' own directions, as it's turned now
  const inside = (box: THREE.Mesh, move: THREE.Vector3) => {
    if (!shell) return move
    const f = footprint(box, box.position.clone().add(move))
    const limitX = Math.max(0, shell.size[0] / 2 - f.hx)
    const limitZ = Math.max(0, shell.size[2] / 2 - f.hz)
    const x = THREE.MathUtils.clamp(f.x, -limitX, limitX)
    const z = THREE.MathUtils.clamp(f.z, -limitZ, limitZ)
    const clamped = middle.clone().addScaledVector(across, x).addScaledVector(along, z)
    clamped.y = box.position.y
    return clamped.sub(box.position)
  }
  // Moves a box and its thing together
  const shift = (box: THREE.Mesh, move: THREE.Vector3) => {
    box.position.add(move)
    const { piece } = held.get(box)!
    piece.object.position.add(piece.toLocal ? piece.toLocal(move.clone()) : move)
  }

  const onMove = (event: PointerEvent) => {
    if (!holding) return
    const point = onFloor(event)
    if (!point) return
    shift(holding, inside(holding, point.clone().sub(last).setY(0)))
    last.copy(point)
    changed(holding.name, fitOf(holding))
    refresh()
  }
  const onUp = () => {
    if (!holding) return
    holding = null
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
  }

  let selected: THREE.Mesh | null = null
  const select = (box: THREE.Mesh | null) => {
    selected = box
    boxes.children.forEach(child => ((child.children[0] as THREE.LineSegments).visible = child === box))
    changed(box ? box.name : null, box ? fitOf(box) : null)
  }

  return {
    setOn: value => {
      on = value
      boxes.visible = value
      if (!value) select(null)
    },
    reset: () => {
      for (const [box, { piece, home, boxHome, turnHome, boxTurnHome }] of held) {
        box.position.copy(boxHome)
        box.rotation.y = boxTurnHome
        piece.object.position.copy(home)
        piece.object.rotation.y = turnHome
      }
      select(null)
      refresh()
    },
    add: addPiece,
    turn: degrees => {
      const hold = selected && held.get(selected)
      if (!selected || !hold || hold.piece.toLocal) return
      // about the box's middle: the box turns in place, and the thing turns round the same spot
      const angle = THREE.MathUtils.degToRad(degrees)
      const object = hold.piece.object
      const pivot = selected.position.clone().setY(0)
      const offset = object.position.clone().setY(0).sub(pivot).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle)
      object.position.set(pivot.x + offset.x, object.position.y, pivot.z + offset.z)
      object.rotation.y += angle
      selected.rotation.y += angle
      shift(selected, inside(selected, new THREE.Vector3())) // turned, it may reach a wall: back inside
      changed(selected.name, fitOf(selected))
      refresh()
    },
    freeSpot: (size, yawDegrees, avoid) => {
      if (!shell) return null
      const f = halves(size[0], size[2], -THREE.MathUtils.degToRad(yawDegrees))
      const blocks = others(null).map(other => footprint(other))
      const limitX = shell.size[0] / 2 - f.hx - SPOT_MARGIN
      const limitZ = shell.size[2] / 2 - f.hz - SPOT_MARGIN
      let best: { x: number; z: number; far: number } | null = null
      for (let x = -limitX; x <= limitX; x += SPOT_STEP) {
        for (let z = -limitZ; z <= limitZ; z += SPOT_STEP) {
          const far = Math.hypot(x, z)
          if (best && far >= best.far) continue
          if (avoid) {  // not in the face of whoever stands there (the room's opening view)
            const at = middle.clone().addScaledVector(across, x).addScaledVector(along, z)
            if (Math.hypot(at.x - avoid.x, at.z - avoid.z) < avoid.reach) continue
          }
          const clear = blocks.every(g => Math.abs(x - g.x) >= f.hx + g.hx + SPOT_MARGIN || Math.abs(z - g.z) >= f.hz + g.hz + SPOT_MARGIN)
          if (clear) best = { x, z, far }
        }
      }
      if (!best) return null
      const spot = middle.clone().addScaledVector(across, best.x).addScaledVector(along, best.z)
      return [spot.x, size[1] / 2, spot.z]
    },
    grab: event => {
      if (!on) return false
      pointerRay(event)
      const hit = raycaster.intersectObjects(boxes.children, false)[0]
      if (!hit) {
        select(null)
        return false
      }
      holding = hit.object as THREE.Mesh
      select(holding)
      const point = onFloor(event)
      if (point) last.copy(point)
      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
      return true
    },
    dispose: () => {
      onUp()
      for (const group of [boxes, fixed]) {
        group.traverse(child => {
          if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
            child.geometry.dispose()
            ;(child.material as THREE.Material).dispose()
          }
        })
      }
      scene.remove(boxes, fixed)
    },
  }
}

/**
 * The room's things as movable splats, still looking like the video: each thing's own splats (split by SAM 3.1's
 * outlines) move together. Null when this room hasn't been split into things.
 */
export async function movableThings(
  assets: string,
  room: THREE.Group,
  scene: THREE.Scene,
  camera: THREE.Camera,
  dom: HTMLElement,
  shell: RoomShell | null,
  changed: (name: string | null, fit?: Fit | null) => void,
  refresh: () => void,
  scale: () => number = () => 1,
): Promise<(Movable & { parts: THREE.Group }) | null> {
  let found: SplatParts
  try {
    const response = await fetch(`${assets}room-splat-objects.json`)
    if (!response.ok) return null
    found = await response.json()
  } catch {
    return null
  }

  const parts = new THREE.Group()
  const pieces: Piece[] = []
  // The splats live in the room group, which is turned and lifted: a move across the room in its own frame
  const toLocal = (move: THREE.Vector3) => move.applyQuaternion(room.quaternion.clone().invert())
  for (const part of [...found.things, { ...found.rest, name: 'rest' }]) {
    const splats = new SplatMesh({ url: `${assets}${part.file}` })
    splats.quaternion.set(1, 0, 0, 0) // OpenCV axes to three.js axes, like room-splats.ply
    splats.name = part.name
    parts.add(splats)
    if ('centre' in part) pieces.push({ ...part, object: splats, toLocal })
  }
  const movable = makeMovable(pieces, scene, camera, dom, shell, changed, refresh, { scale })
  return {
    ...movable,
    parts,
    dispose: () => {
      movable.dispose()
      parts.children.forEach(child => (child as SplatMesh).dispose())
    },
  }
}
