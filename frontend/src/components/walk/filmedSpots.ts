import * as THREE from 'three'
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js'

/** One spot the video was filmed from, in the upright room: where the phone was and which way it looked. */
export type Spot = { position: THREE.Vector3; quaternion: THREE.Quaternion }

const LOOK_ACROSS = THREE.MathUtils.degToRad(30) // how far the view turns from where the phone looked, either way
const LOOK_UP = THREE.MathUtils.degToRad(20) // and up or down
// A glide from one spot to the next; a straight cut for anyone who's asked for less motion
const STEP_MS = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 450

export type Spots = {
  count: number
  /** Glides to spot `index` and keeps the view near the way the phone looked from there */
  go: (index: number, now?: number) => void
  /** Each frame, before the controls: moves the camera along a glide in progress */
  tick: (now: number) => void
  /** Lets the view turn and step sideways freely again (the other looks) */
  release: () => void
}

/**
 * The Real look's way round: the splats only look like the video from close to where it was filmed, and smear
 * anywhere else, so in the Real look you step between the spots the phone was at (gliding from one to the next) and
 * look around a little at each. Walking anywhere is the Rebuilt look's job.
 */
export function filmedSpots(spots: Spot[], camera: THREE.PerspectiveCamera, controls: OrbitControls): Spots {
  let glide: { from: Spot; to: Spot; start: number } | null = null
  // The controls turn the camera in place round a point 1 cm ahead of it
  const ahead = (quaternion: THREE.Quaternion) => new THREE.Vector3(0, 0, -0.01).applyQuaternion(quaternion)
  const place = (position: THREE.Vector3, quaternion: THREE.Quaternion) => {
    camera.position.copy(position)
    controls.target.copy(position).add(ahead(quaternion))
  }
  const free = () => {
    controls.minAzimuthAngle = -Infinity
    controls.maxAzimuthAngle = Infinity
    controls.minPolarAngle = 0
    controls.maxPolarAngle = Math.PI
  }
  const hold = (spot: Spot) => {
    // The controls' angles are of the camera as seen from that point: behind the way it looks
    const back = new THREE.Spherical().setFromVector3(ahead(spot.quaternion).negate())
    controls.minAzimuthAngle = back.theta - LOOK_ACROSS
    controls.maxAzimuthAngle = back.theta + LOOK_ACROSS
    controls.minPolarAngle = Math.max(0.05, back.phi - LOOK_UP)
    controls.maxPolarAngle = Math.min(Math.PI - 0.05, back.phi + LOOK_UP)
  }

  return {
    count: spots.length,
    go: (index, now = performance.now()) => {
      const to = spots[Math.max(0, Math.min(spots.length - 1, index))]
      if (!to) return
      controls.enablePan = false // stepping sideways would leave the spot
      free() // a glide mustn't be held to the last spot's view
      glide = { from: { position: camera.position.clone(), quaternion: camera.quaternion.clone() }, to, start: now }
    },
    tick: now => {
      if (!glide) return
      const t = Math.min(1, (now - glide.start) / STEP_MS)
      const eased = t * t * (3 - 2 * t)
      place(glide.from.position.clone().lerp(glide.to.position, eased),
        glide.from.quaternion.clone().slerp(glide.to.quaternion, eased))
      if (t < 1) return
      hold(glide.to)
      glide = null
    },
    release: () => {
      glide = null
      free()
      controls.enablePan = true
    },
  }
}
