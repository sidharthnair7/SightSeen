import * as THREE from 'three'
import { SHELL_FACES, type RoomShell, type ShellFace } from './types'

/** room-materials.json (worker/room_materials.py): per face, a seamless tile of its own surface and how many metres
 * of real surface it covers, or only a colour when no clean view of it was filmed (the ceiling). */
type RoomMaterials = {
  faces: Record<ShellFace, { file: string | null; tileMetres: number | null; colour: number[] }>
}

/** Each face's size in metres, across and down, in BoxGeometry's material order (right, left, ceiling, floor, back,
 * front): the side faces run along z or x, and all of them but the floor and ceiling are as tall as the room. */
function faceSizes(shell: RoomShell): Record<ShellFace, [number, number]> {
  const [width, height, length] = shell.size
  return {
    right: [length, height],
    left: [length, height],
    ceiling: [width, length],
    floor: [width, length],
    back: [width, height],
    front: [width, height],
  }
}

/**
 * The Rebuilt look's walls, floor and ceiling as clean materials: each face's own surface tile (cut sharp from the
 * video, mirrored so it repeats without seams) repeated at its real size, or a plain colour. Lit by the scene's lights
 * (Lambert), so the faces shade apart like a modelled room; the photo walls stay in the Walk look and behind the
 * "Photo walls" switch. Null when this room has no materials yet.
 */
export async function materialShellMaterials(assets: string, shell: RoomShell): Promise<THREE.Material[] | null> {
  let materials: RoomMaterials
  try {
    const response = await fetch(`${assets}room-materials.json`)
    if (!response.ok) return null
    materials = await response.json()
  } catch {
    return null
  }
  if (!SHELL_FACES.every(face => materials.faces?.[face])) return null
  const sizes = faceSizes(shell)
  const loader = new THREE.TextureLoader()
  const textures = new Map<string, THREE.Texture>()
  try {
    return await Promise.all(SHELL_FACES.map(async face => {
      const { file, tileMetres, colour } = materials.faces[face]
      if (!file || !tileMetres) {
        return new THREE.MeshLambertMaterial({
          color: new THREE.Color(colour[0] / 255, colour[1] / 255, colour[2] / 255), side: THREE.BackSide,
        })
      }
      // A texture per face (its repeat is the face's own), loaded once per file
      if (!textures.has(file)) textures.set(file, await loader.loadAsync(`${assets}${file}`))
      const texture = textures.get(file)!.clone()
      texture.colorSpace = THREE.SRGBColorSpace
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping
      texture.anisotropy = 8
      texture.repeat.set(sizes[face][0] / tileMetres, sizes[face][1] / tileMetres)
      texture.needsUpdate = true
      const material = new THREE.MeshLambertMaterial({ map: texture, side: THREE.BackSide })
      material.addEventListener('dispose', () => texture.dispose())
      return material
    }))
  } catch (error) {
    console.warn('Wall materials failed to load; keeping the photo walls', error)
    return null
  }
}
