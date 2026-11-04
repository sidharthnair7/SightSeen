export type RequestStatus = 'WAITING_FOR_VIDEO' | 'VIDEO_RECEIVED' | 'RECONSTRUCTING' | 'READY' | 'FAILED'

/** The last finished 3D room (RequestView.Room on the server). */
export interface Room {
  frames: number
  seconds: number
  peakVramGb: number | null
  scaleFactor: number | null
  points: number
  sizeX: number
  sizeY: number
  sizeZ: number
  plyUrl: string
}

export interface RoomRequest {
  id: string
  label: string
  status: RequestStatus
  stage: string | null
  landlordPath: string
  roomPath: string
  createdAt: string
  videoSource: 'LANDLORD_LINK' | 'STUDENT_UPLOAD' | null
  model: string | null
  room: Room | null
  failureMessage: string | null
}

/** All the landlord's page gets. */
export interface FilmView {
  label: string
  videoReceived: boolean
  /** The hosted pipeline: a video is being built into a room right now */
  building?: boolean
}
