/** room-shell.json: the walls, floor and ceiling as a box in the upright room (metres, floor at y = 0), with each
 * face's median colour (RGB 0-255). Size is along the walls' own x, up, and along their z; the box is turned about
 * up by -yawDegrees. */
export type RoomShell = {
  centre: number[]
  size: number[]
  yawDegrees: number
  faces: Record<ShellFace, number[]>
  /** The video never showed the ceiling: its height is a standard one (2.44 real metres), not measured */
  ceilingAssumed?: boolean
}

/** The shell's faces, in BoxGeometry's material order: +x, -x, +y, -y, +z, -z. */
export const SHELL_FACES = ['right', 'left', 'ceiling', 'floor', 'back', 'front'] as const
export type ShellFace = (typeof SHELL_FACES)[number]
