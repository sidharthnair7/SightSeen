import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

/**
 * The room's furniture looking like the real thing, for the Walk look, in the upright room (metres, floor at y = 0,
 * three.js axes). `assets` is the room's assets URL ending in a slash (GET /api/requests/{id}/assets/{name}). Null
 * when this room has none yet: the viewer then shows the plain blocks from room-objects.json.
 */
export async function loadFurniture(assets: string): Promise<THREE.Object3D | null> {
  // room-furniture.ply (worker/room_furniture.py): the fused surface inside the shell, already upright, each vertex
  // coloured from the frame that saw it best
  const response = await fetch(`${assets}room-furniture.ply`)
  if (!response.ok) return null
  const geometry = new PLYLoader().parse(await response.arrayBuffer())
  if (!geometry.getAttribute('position')?.count) return null
  geometry.computeVertexNormals()
  geometry.computeBoundingSphere()
  // Unlit: the colours are photos, so they already carry the room's real light and shade; lighting them again would
  // darken every side face. Both sides, because the backs nobody filmed are open and show the inside instead of a hole.
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }))
  mesh.name = 'furniture'
  return mesh
}
