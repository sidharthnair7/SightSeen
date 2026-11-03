import * as THREE from 'three'

/** room-decals.json (worker/room_decals.py): the things built into the walls (the window and its curtains, the door,
 * the heater, switches, outlets) as cut-out pictures, each where it was filmed on its wall, in the upright room. */
type RoomDecals = {
  decals: {
    name: string
    file: string
    centre: number[]
    right: number[]
    up: number[]
    normal: number[]
    width: number
    height: number
  }[]
}

// Out from the wall by this much (and a touch more for each later one), so no picture flickers into the wall or into
// another picture
const OFF_WALL = 0.004
const APART = 0.0005

/**
 * The Rebuilt look's wall cut-outs on its clean walls: each a flat picture with see-through edges, lit like the
 * walls. Null when this room has none.
 */
export async function loadDecals(assets: string): Promise<THREE.Group | null> {
  let found: RoomDecals
  try {
    const response = await fetch(`${assets}room-decals.json`)
    if (!response.ok) return null
    found = await response.json()
  } catch {
    return null
  }
  if (!found.decals?.length) return null
  const group = new THREE.Group()
  const loader = new THREE.TextureLoader()
  await Promise.all(found.decals.map(async (decal, i) => {
    const texture = await loader.loadAsync(`${assets}${decal.file}`)
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = 8
    const material = new THREE.MeshLambertMaterial({ map: texture, transparent: true, depthWrite: false })
    material.addEventListener('dispose', () => texture.dispose())
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(decal.width, decal.height), material)
    const normal = new THREE.Vector3().fromArray(decal.normal)
    // The picture's across, up and out of the wall (into the room) are the plane's x, y and z
    mesh.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(new THREE.Vector3().fromArray(decal.right), new THREE.Vector3().fromArray(decal.up), normal),
    )
    mesh.position.fromArray(decal.centre).addScaledVector(normal, OFF_WALL + i * APART)
    mesh.name = decal.name
    group.add(mesh)
  }))
  return group
}
