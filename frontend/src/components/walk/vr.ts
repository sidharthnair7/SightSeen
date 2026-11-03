import * as THREE from 'three'

/** Where VR starts: where the video began (on the floor) and the way it first looked, in the room's units; real
 * metres per room unit; and a few lines on the room's real size, shown when VR starts. */
export type VrStart = { stand: THREE.Vector3; ahead: THREE.Vector3; scale: number; about: string[] }

/** A thumbstick, as WebXR gives it: axes 2 and 3 are the stick's x (right +) and y (down +). */
type Stick = { handedness: string; gamepad?: { axes: readonly number[] } | null }

export type Vr = {
  /** Starts VR (a Meta Quest's browser, or a desktop one with a headset), at the room's real size */
  enter: () => Promise<void>
  /** One frame in VR: the sticks move you, then the headset's two eyes are drawn */
  render: (seconds: number) => void
  /** Stands you in the room (enter does this; on its own, it lets the rig be checked without a headset) */
  place: (start: VrStart, floor?: boolean) => void
  /** Takes you out again: the room's marks and the start card go */
  leave: () => void
  /** The sticks' work for `seconds`: the left one walks where you look, the right one turns you 30 degrees a flick */
  step: (sticks: readonly Stick[], seconds: number) => void
  /** The trigger's work: a mark where a ray from `from` (room units) toward `toward` meets the room; the second mark
   * gives the distance between them in metres, and the third starts again. Null until there are two. */
  measure: (from: THREE.Vector3, toward: THREE.Vector3) => number | null
  /** You: where you stand in the room and which way you face. The headset and controllers move inside it. */
  rig: THREE.Group
  /** The headset, inside the rig (three.js sets its pose each frame) */
  eyes: THREE.PerspectiveCamera
  dispose: () => void
}

const WALK = 1.2 // metres a second at full stick
const TURN = Math.PI / 6 // a snap turn: 30 degrees
const DEAD = 0.25 // stick travel that's ignored
const EYE = 1.6 // metres: where the eyes are taken to be when the headset doesn't know where the floor is
const MARK = '#f59e0b'
const START_CARD = 20_000 // milliseconds the start card stays, unless you measure first

/**
 * VR at true scale. The rig is you: the headset and controllers move inside it in real metres, and it scales them into
 * the room's units (1 / scale), so the room shows at its real size whatever units it was rebuilt in: a 3.25 m wall is
 * 3.25 m in the headset. It starts where the video began, facing where it first looked, on the room's floor. The left
 * stick walks, the right stick turns, and the trigger measures between two spots, in metres.
 */
export function makeVr(renderer: THREE.WebGLRenderer, scene: THREE.Scene, options: {
  start: () => VrStart | null
  /** What a measuring ray can hit: what's drawn (the rebuilt things and the walls) */
  targets: () => THREE.Object3D[]
  /** Whether the head may be at this point (room units): inside the walls */
  walkable?: (point: THREE.Vector3) => boolean
  onChange: (inVr: boolean) => void
}): Vr {
  const rig = new THREE.Group()
  const eyes = new THREE.PerspectiveCamera(70, 1, 0.02, 100) // near and far in metres: three.js hands them to the headset
  rig.add(eyes)
  const marks = new THREE.Group() // the measuring marks, in the room
  const points: THREE.Vector3[] = []
  const raycaster = new THREE.Raycaster()
  const up = new THREE.Vector3(0, 1, 0)
  let session: XRSession | null = null
  let starting = false // a second press while the headset is still starting does nothing
  let scale = 1
  const assumed: boolean[] = [] // each mark: on a ceiling whose height was assumed
  let turned = false // the right stick has to come back to the middle between turns
  let card: THREE.Mesh | null = null
  let cardUntil = 0

  const hideCard = () => {
    if (!card) return
    rig.remove(card)
    disposeMesh(card)
    card = null
  }
  const clearMarks = () => {
    points.length = 0
    assumed.length = 0
    for (const mark of [...marks.children]) {
      marks.remove(mark)
      disposeMesh(mark as THREE.Mesh)
    }
  }

  const measure = (from: THREE.Vector3, toward: THREE.Vector3) => {
    raycaster.set(from, toward.clone().normalize())
    // Meshes only: a box's edge lines would catch a ray passing a metre away
    const hit = raycaster.intersectObjects(options.targets(), true).find(found => found.object instanceof THREE.Mesh)
    if (!hit) return null
    hideCard()
    if (points.length === 2) clearMarks()
    const point = markPoint(hit)
    points.push(point)
    assumed.push(onAssumedCeiling(hit))
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.015 / scale, 16, 12), new THREE.MeshBasicMaterial({ color: MARK, depthTest: false }))
    dot.position.copy(point)
    dot.renderOrder = 20
    marks.add(dot)
    if (points.length < 2) return null
    const metres = points[0].distanceTo(points[1]) * scale
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: MARK, depthTest: false }))
    line.renderOrder = 20
    marks.add(line)
    const { map, aspect } = textCard([`${metres.toFixed(2)} m`,
      assumed.some(Boolean) ? "the ceiling's height is assumed" : `${Math.round(metres * 100)} cm`])
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map, depthTest: false }))
    const tall = 0.12 / scale // 12 cm tall, whatever the room's units
    label.scale.set(tall * aspect, tall, 1)
    label.position.copy(points[0]).add(points[1]).multiplyScalar(0.5).addScaledVector(up, 0.08 / scale)
    label.renderOrder = 21
    marks.add(label)
    return metres
  }

  // The controllers: a short ray from each, and the trigger measures where it points
  const hands = [0, 1].map(index => {
    const hand = renderer.xr.getController(index)
    const ray = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1.5)]),
      new THREE.LineBasicMaterial({ color: MARK }),
    )
    hand.add(ray)
    const onSelect = () => {
      hand.updateWorldMatrix(true, false) // the rig may have moved since the last frame
      measure(new THREE.Vector3().setFromMatrixPosition(hand.matrixWorld), new THREE.Vector3(0, 0, -1).transformDirection(hand.matrixWorld))
    }
    hand.addEventListener('select', onSelect)
    rig.add(hand)
    return { hand, ray, onSelect }
  })

  const place = (start: VrStart, floor = true) => {
    scale = start.scale || 1
    rig.scale.setScalar(1 / scale)
    // Standing on the room's floor ("local-floor": the headset's floor is at 0, as the room's is); a headset that
    // doesn't know its floor puts its origin at the eyes, so those go at eye height
    rig.position.set(start.stand.x, floor ? 0 : EYE / scale, start.stand.z)
    const ahead = start.ahead.clone().setY(0)
    rig.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), ahead.lengthSq() > 1e-8 ? ahead.normalize() : new THREE.Vector3(0, 0, -1))
    scene.add(rig, marks)
    // The start card: the room's real size and how to move, a little below the eyes, 1.3 m ahead
    hideCard()
    const { map, aspect } = textCard([...start.about, 'Left stick: walk. Right stick: turn.',
      'Trigger: point at two spots to measure between them.', 'To leave VR: press the Meta button.'])
    card = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.9 / aspect), new THREE.MeshBasicMaterial({ map, transparent: true, depthTest: false }))
    card.position.set(0, floor ? 1.3 : 1.3 - EYE, -1.3)
    card.renderOrder = 20
    rig.add(card)
    cardUntil = performance.now() + START_CARD
  }
  const leave = () => {
    hideCard()
    clearMarks()
    scene.remove(rig, marks)
  }
  const end = () => {
    session = null
    leave()
    options.onChange(false)
  }

  const step = (sticks: readonly Stick[], seconds: number) => {
    for (const stick of sticks) {
      const axes = stick.gamepad?.axes
      if (!axes || axes.length < 4) continue
      const [x, y] = [axes[2], axes[3]]
      if (stick.handedness === 'left') {
        if (Math.hypot(x, y) < DEAD) continue
        // Across the floor, where the head looks: the room's units are 1 / scale of a metre
        const ahead = eyes.getWorldDirection(new THREE.Vector3()).setY(0)
        if (ahead.lengthSq() < 1e-8) continue
        ahead.normalize()
        const right = new THREE.Vector3().crossVectors(ahead, up)
        const metres = (WALK * seconds) / scale
        const move = new THREE.Vector3().addScaledVector(ahead, -y * metres).addScaledVector(right, x * metres)
        // Not through walls: the move, or as much of it as slides along the wall, or none (from outside, any move,
        // so a head that's out can come back)
        const walkable = options.walkable ?? (() => true)
        const head = new THREE.Vector3().setFromMatrixPosition(eyes.matrixWorld)
        const tries = [move, new THREE.Vector3(move.x, 0, 0), new THREE.Vector3(0, 0, move.z)]
        const allowed = walkable(head) ? tries.find(next => walkable(head.clone().add(next))) : move
        if (allowed) rig.position.add(allowed)
      } else if (stick.handedness === 'right') {
        if (!turned && Math.abs(x) > 0.6) {
          // Turned round where the head is, so you stay where you stand: right turns you clockwise, seen from above
          const angle = -Math.sign(x) * TURN
          const pivot = new THREE.Vector3().setFromMatrixPosition(eyes.matrixWorld)
          rig.position.sub(pivot).applyAxisAngle(up, angle).add(pivot)
          rig.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(up, angle))
          turned = true
        } else if (Math.abs(x) < 0.3) {
          turned = false
        }
      }
    }
  }

  return {
    rig,
    eyes,
    place,
    leave,
    step,
    measure,
    enter: async () => {
      const start = options.start()
      if (!navigator.xr || session || starting || !start) return
      starting = true
      const next = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor'] })
        .finally(() => (starting = false))
      const enabled = (next as XRSession & { enabledFeatures?: readonly string[] }).enabledFeatures
      const floor = !enabled || enabled.includes('local-floor')
      session = next
      next.addEventListener('end', end)
      place(start, floor)
      renderer.xr.setReferenceSpaceType(floor ? 'local-floor' : 'local')
      try {
        await renderer.xr.setSession(next)
      } catch (error) {
        next.end().catch(() => {})
        throw error
      }
      options.onChange(true)
    },
    render: seconds => {
      if (!session) return
      step(session.inputSources as unknown as readonly Stick[], seconds)
      if (card && performance.now() > cardUntil) hideCard()
      renderer.render(scene, eyes)
    },
    dispose: () => {
      session?.end().catch(() => {})
      session = null
      leave()
      for (const { hand, ray, onSelect } of hands) {
        hand.removeEventListener('select', onSelect)
        hand.remove(ray)
        disposeMesh(ray)
        rig.remove(hand)
      }
    },
  }
}

/** Where a measuring ray's mark goes. The walls are drawn a hair behind where they were found (userData.gap, in room
 * units), so a mark on one goes back onto the found wall, and wall to wall reads the room's measured size. */
export function markPoint(hit: THREE.Intersection): THREE.Vector3 {
  const point = hit.point.clone()
  const gap = hit.object.userData.gap as number | undefined
  if (gap && hit.face) point.addScaledVector(hit.face.normal.clone().transformDirection(hit.object.matrixWorld), -gap)
  return point
}

/** Whether a measuring ray hit a ceiling whose height was assumed (userData.ceilingAssumed on the walls): a
 * distance to it is only as good as that guess. */
export function onAssumedCeiling(hit: THREE.Intersection): boolean {
  if (!hit.object.userData.ceilingAssumed || !hit.face) return false
  return hit.face.normal.clone().transformDirection(hit.object.matrixWorld).y > 0.5 // the box's top face, from inside
}

/** Whether this browser can start VR (a headset's browser, or a desktop one with a headset plugged in). */
export async function vrSupported(): Promise<boolean> {
  try {
    return Boolean(navigator.xr && (await navigator.xr.isSessionSupported('immersive-vr')))
  } catch {
    return false
  }
}

/** Lines of text on a dark rounded card, as a texture, with its width over its height. The first line is bigger. */
function textCard(lines: string[]): { map: THREE.CanvasTexture; aspect: number } {
  const canvas = document.createElement('canvas')
  const draw = canvas.getContext('2d')!
  const pad = 32
  const sizes = lines.map((_, i) => (i === 0 ? 52 : 36))
  const font = (i: number) => `${i === 0 ? 600 : 400} ${sizes[i]}px system-ui, sans-serif`
  let widest = 0
  lines.forEach((line, i) => {
    draw.font = font(i)
    widest = Math.max(widest, draw.measureText(line).width)
  })
  canvas.width = Math.ceil(widest + 2 * pad)
  canvas.height = Math.ceil(sizes.reduce((sum, size) => sum + size * 1.4, 0) + 2 * pad)
  // Sizing the canvas clears its settings, so the font is set again for each line
  draw.fillStyle = 'rgba(16, 22, 26, 0.88)'
  draw.beginPath()
  draw.roundRect(0, 0, canvas.width, canvas.height, 28)
  draw.fill()
  draw.textBaseline = 'top'
  let y = pad
  lines.forEach((line, i) => {
    draw.font = font(i)
    draw.fillStyle = i === 0 ? '#ffffff' : '#d1d5db'
    draw.fillText(line, pad, y)
    y += sizes[i] * 1.4
  })
  const map = new THREE.CanvasTexture(canvas)
  map.colorSpace = THREE.SRGBColorSpace
  return { map, aspect: canvas.width / canvas.height }
}

/** Frees a mark's geometry, material and texture. */
function disposeMesh(mark: THREE.Object3D) {
  const part = mark as THREE.Mesh
  part.geometry?.dispose()
  const material = part.material as THREE.MeshBasicMaterial | undefined
  material?.map?.dispose()
  material?.dispose()
}
