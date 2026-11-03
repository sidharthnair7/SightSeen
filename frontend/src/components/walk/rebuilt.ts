import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { makeMovable, type Fit, type Movable, type Piece } from './moveThings'
import type { RoomShell } from './types'

/** room-things.json (worker/rebuild_things.py): the room's things rebuilt whole by SAM 3D, in the upright room. */
type RebuiltThings = {
  things: {
    name: string
    file: string
    centre: number[]
    size: number[]
    yawDegrees: number
    seenShare: number
    /** Built into the room (a window, a door): drawn, but not one of the things you can move */
    fixed?: boolean
  }[]
  failed: { name: string; box?: { centre: number[]; size: number[]; yawDegrees: number } | null }[]
}

/** A model's PLY: in a headset's browser (`light`) its lighter copy, vr-<file>, when there is one (a Quest draws about
 * a million triangles a frame, and the hosted copy has these), else the full one. A static host answers a missing
 * file with its page, so an HTML answer counts as missing. */
export async function fetchModel(folder: string, file: string, light = false): Promise<Response | null> {
  const ok = (response: Response | null) =>
    !!response?.ok && !(response.headers.get('content-type') ?? '').includes('text/html')
  if (light) {
    const lighter = await fetch(`${folder}vr-${file}`).catch(() => null)
    if (ok(lighter)) return lighter
  }
  const full = await fetch(`${folder}${file}`).catch(() => null)
  return ok(full) ? full : null
}

// What the video never saw, when it's shown: tinted towards this colour, half and half
const NOT_FILMED = new THREE.Color('#a855f7')

export type Rebuilt = {
  group: THREE.Group
  movable: Movable
  count: number
  /** Tints every part the video never saw (SAM 3D's own guess), or puts the colours back. */
  showNotFilmed: (on: boolean) => void
}

/**
 * The Rebuilt look's things: each one whole on every side (SAM 3D Objects built it from its best photo, the sides
 * the video saw painted from that photo), at its measured size and spot, and movable. A thing SAM 3D couldn't build
 * stands in as a grey box of its measured size (the owner's rule 3). Null when the room hasn't been rebuilt.
 */
export async function loadRebuilt(
  assets: string,
  scene: THREE.Scene,
  camera: THREE.Camera,
  dom: HTMLElement,
  shell: RoomShell | null,
  changed: (name: string | null, fit?: Fit | null) => void,
  scale: () => number = () => 1,
  light = false,
): Promise<Rebuilt | null> {
  let found: RebuiltThings
  try {
    const response = await fetch(`${assets}room-things.json`)
    if (!response.ok) return null
    found = await response.json()
  } catch {
    return null
  }
  if (!found.things?.length) return null

  const group = new THREE.Group()
  const pieces: Piece[] = []
  const fixed: Piece[] = [] // drawn and in the way, not movable
  const geometries: THREE.BufferGeometry[] = []
  const loader = new PLYLoader()
  loader.setCustomPropertyNameMapping({ seen: ['seen'] }) // 1 where the colour is the photo's, 0 where it's SAM 3D's
  await Promise.all(found.things.map(async thing => {
    const response = await fetchModel(assets, thing.file, light)
    if (!response) return
    const geometry = loader.parse(await response.arrayBuffer())
    geometry.computeVertexNormals()
    geometries.push(geometry)
    // Unlit like the walls: the colours are the photo's own light and shade (both sides, for any open seam)
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
    mesh.name = thing.name
    group.add(mesh)
    ;(thing.fixed ? fixed : pieces).push({ ...thing, object: mesh })
  }))
  for (const thing of found.failed ?? []) {
    if (!thing.box) continue
    const { centre, yawDegrees } = thing.box
    // (a box measured below the floor came out with a negative height: a flat thing lying on it)
    const size = thing.box.size.map(side => Math.max(Math.abs(side), 0.01))
    const geometry = new THREE.BoxGeometry(size[0], size[1], size[2])
    const block = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: '#9aa0a6' }))
    block.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: '#1f2933' })))
    block.position.set(centre[0], centre[1], centre[2])
    block.rotation.y = -THREE.MathUtils.degToRad(yawDegrees)
    block.name = `${thing.name} (not built)`
    group.add(block)
    // The box itself is what moves, so it moves from its own spot
    pieces.push({ name: block.name, object: block, centre, size, yawDegrees, placeholder: true })
  }

  // The photo's colours and the tinted ones, swapped by showNotFilmed
  const colours = new Map<THREE.BufferGeometry, { real: THREE.BufferAttribute; tinted: THREE.BufferAttribute }>()
  const showNotFilmed = (on: boolean) => {
    for (const geometry of geometries) {
      const seen = geometry.getAttribute('seen')
      const colour = geometry.getAttribute('color') as THREE.BufferAttribute | undefined
      if (!seen || !colour) continue
      let pair = colours.get(geometry)
      if (!pair) {
        const tinted = colour.clone()
        for (let i = 0; i < tinted.count; i++) {
          if (seen.getX(i) > 0.5) continue
          tinted.setXYZ(i, (tinted.getX(i) + NOT_FILMED.r) / 2, (tinted.getY(i) + NOT_FILMED.g) / 2,
            (tinted.getZ(i) + NOT_FILMED.b) / 2)
        }
        pair = { real: colour, tinted }
        colours.set(geometry, pair)
      }
      geometry.setAttribute('color', on ? pair.tinted : pair.real)
    }
  }

  const movable = makeMovable(pieces, scene, camera, dom, shell, changed, undefined, { fixed, scale })
  return {
    group,
    count: found.things.length,
    showNotFilmed,
    movable: {
      ...movable,
      dispose: () => {
        movable.dispose()
        group.traverse(part => {
          if (part instanceof THREE.Mesh || part instanceof THREE.LineSegments) {
            part.geometry.dispose()
            ;(part.material as THREE.Material).dispose()
          }
        })
      },
    },
  }
}
