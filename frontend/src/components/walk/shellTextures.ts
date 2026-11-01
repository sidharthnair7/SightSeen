import * as THREE from 'three'
import { SHELL_FACES, type RoomShell, type ShellFace } from './types'

/** room-shell-textures.json, written by worker/room_textures.py: one picture per face. `columns` and `rows` say which
 * of the box's own axes (x along the walls' x, y up, z along their z) the picture runs along, left to right and top to
 * bottom, e.g. '+z' and '-y'. */
type ShellTextures = {
  faces: Record<ShellFace, { file: string; columns: string; rows: string; coverage: number }>
}

/** How BoxGeometry lays a texture on each face (flipY on, the default): which way the picture's columns and rows run
 * in the box's own axes. From BoxGeometry's buildPlane: columns run along udir on axis u, rows along vdir on axis v
 * (uv.y = 1 - iy / gridY, and with flipY the picture's top row sits at uv.y = 1). That reads the right way round from
 * outside the box, so from inside every wall is mirrored. */
const BOX_LAYOUT: Record<ShellFace, [string, string]> = {
  right: ['-z', '-y'],
  left: ['+z', '-y'],
  ceiling: ['+x', '+z'],
  floor: ['+x', '-z'],
  back: ['+x', '-y'],
  front: ['-x', '-y'],
}

const opposite = (axis: string) => (axis[0] === '+' ? '-' : '+') + axis[1]

/**
 * The shell's six faces with the room's real look on them (photos from the frames), in BoxGeometry's material order
 * (+x, -x, +y, -y, +z, -z: right, left, ceiling, floor, back, front), seen from inside. `assets` is the room's assets
 * URL ending in a slash (GET /api/requests/{id}/assets/{name}). Null when this room has no wall textures yet: the
 * viewer then keeps the flat colours.
 */
export async function texturedShellMaterials(assets: string, shell: RoomShell): Promise<THREE.Material[] | null> {
  void shell // the pictures were made for this shell; what they don't cover already fades into its face colours
  let textures: ShellTextures
  try {
    const response = await fetch(`${assets}room-shell-textures.json`)
    if (!response.ok) return null
    textures = await response.json()
  } catch {
    return null
  }
  if (!SHELL_FACES.every(face => textures.faces?.[face])) return null

  const loader = new THREE.TextureLoader()
  try {
    return await Promise.all(SHELL_FACES.map(async face => {
      const { file, columns, rows } = textures.faces[face]
      const [boxColumns, boxRows] = BOX_LAYOUT[face]
      // The worker's pictures read the right way round from inside, so each one is flipped where its direction is
      // the opposite of BoxGeometry's: the walls left to right, the floor and ceiling top to bottom.
      const flipX = columns === opposite(boxColumns)
      const flipY = rows === opposite(boxRows)
      if ((!flipX && columns !== boxColumns) || (!flipY && rows !== boxRows)) {
        throw new Error(`room-shell-textures.json: ${face} runs ${columns}, ${rows}; the box needs ${boxColumns}, ${boxRows}`)
      }
      const texture = await loader.loadAsync(`${assets}${file}`)
      texture.colorSpace = THREE.SRGBColorSpace
      texture.anisotropy = 8
      texture.repeat.set(flipX ? -1 : 1, flipY ? -1 : 1)
      texture.offset.set(flipX ? 1 : 0, flipY ? 1 : 0)
      // Unlit: the photos already carry the room's real light, and the viewer's lights would tint and darken them
      const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.BackSide })
      material.addEventListener('dispose', () => texture.dispose()) // the viewer disposes materials, not their maps
      return material
    }))
  } catch (error) {
    console.warn('Wall pictures failed to load; keeping the flat colours', error)
    return null
  }
}
