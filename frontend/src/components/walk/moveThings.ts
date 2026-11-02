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
}

export type Movable = {
  /** Turns picking things up on or off. */
  setOn: (on: boolean) => void
  /** Puts every thing back where the video found it. */
  reset: () => void
  /** True when a pointer-down started dragging a thing (the viewer then skips its own turning and measuring). */
  grab: (event: PointerEvent) => boolean
  dispose: () => void
}

/**
 * Picking up things and moving them across the floor: an invisible box round each thing is what the pointer grabs,
 * and the thing moves with it. Things stay inside the walls.
 */
export function makeMovable(
  pieces: Piece[],
  scene: THREE.Scene,
  camera: THREE.Camera,
  dom: HTMLElement,
  shell: RoomShell | null,
  changed: (name: string | null) => void,
  refresh: () => void = () => {},
): Movable {
  const boxes = new THREE.Group()
  scene.add(boxes)
  const held = new Map<THREE.Object3D, { piece: Piece; home: THREE.Vector3; boxHome: THREE.Vector3 }>()
  for (const piece of pieces) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(piece.size[0], piece.size[1], piece.size[2]),
      new THREE.MeshBasicMaterial({ visible: false }))
    box.position.set(piece.centre[0], piece.centre[1], piece.centre[2])
    box.rotation.y = -THREE.MathUtils.degToRad(piece.yawDegrees)
    box.name = piece.name
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: '#f59e0b' }))
    outline.visible = false
    box.add(outline)
    boxes.add(box)
    held.set(box, { piece, home: piece.object.position.clone(), boxHome: box.position.clone() })
  }
  boxes.visible = false

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

  // Keeps a moved thing's box inside the walls, in the walls' own directions
  const inside = (box: THREE.Mesh, move: THREE.Vector3) => {
    if (!shell) return move
    const yaw = THREE.MathUtils.degToRad(shell.yawDegrees)
    const across = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw))
    const along = new THREE.Vector3(-Math.sin(yaw), 0, Math.cos(yaw))
    const size = (box.geometry as THREE.BoxGeometry).parameters
    const next = box.position.clone().add(move).sub(new THREE.Vector3(...shell.centre))
    const limitX = Math.max(0, shell.size[0] / 2 - size.width / 2)
    const limitZ = Math.max(0, shell.size[2] / 2 - size.depth / 2)
    const x = THREE.MathUtils.clamp(next.dot(across), -limitX, limitX)
    const z = THREE.MathUtils.clamp(next.dot(along), -limitZ, limitZ)
    const clamped = across.clone().multiplyScalar(x).add(along.clone().multiplyScalar(z))
    clamped.y = next.y
    return clamped.add(new THREE.Vector3(...shell.centre)).sub(box.position)
  }

  const onMove = (event: PointerEvent) => {
    if (!holding) return
    const point = onFloor(event)
    if (!point) return
    const move = inside(holding, point.clone().sub(last).setY(0))
    last.copy(point)
    holding.position.add(move)
    const { piece } = held.get(holding)!
    piece.object.position.add(piece.toLocal ? piece.toLocal(move.clone()) : move)
    refresh()
  }
  const onUp = () => {
    if (!holding) return
    holding = null
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
  }

  const select = (box: THREE.Mesh | null) => {
    boxes.children.forEach(child => ((child.children[0] as THREE.LineSegments).visible = child === box))
    changed(box ? box.name : null)
  }

  return {
    setOn: value => {
      on = value
      boxes.visible = value
      if (!value) select(null)
    },
    reset: () => {
      for (const [box, { piece, home, boxHome }] of held) {
        box.position.copy(boxHome)
        piece.object.position.copy(home)
      }
      refresh()
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
      boxes.traverse(child => {
        if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
          child.geometry.dispose()
          ;(child.material as THREE.Material).dispose()
        }
      })
      scene.remove(boxes)
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
  changed: (name: string | null) => void,
  refresh: () => void,
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
  const movable = makeMovable(pieces, scene, camera, dom, shell, changed, refresh)
  return {
    ...movable,
    parts,
    dispose: () => {
      movable.dispose()
      parts.children.forEach(child => (child as SplatMesh).dispose())
    },
  }
}
