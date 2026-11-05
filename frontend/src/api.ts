import type { FilmView, RoomRequest } from './types'

async function read<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(await reason(response))
  return response.json() as Promise<T>
}

/** Spring Boot puts the reason in "message" (server.error.include-message=always). */
async function reason(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (body && typeof body.message === 'string' && body.message) return body.message
  } catch {
    // not JSON: fall through to the plain message
  }
  return `Something went wrong (${response.status}).`
}

/** A static copy of one room for a free host (scripts/export_static.py): no server, so a room's details come from a
 * file next to its other files, and nothing can be uploaded or rebuilt. */
export const STATIC = import.meta.env.VITE_STATIC === '1'
export const STATIC_ROOM: string | undefined = import.meta.env.VITE_STATIC_ROOM
/** A second room in the static copy: a fake tour, to show the scam checks catching one. */
export const FAKE_ROOM: string | undefined = import.meta.env.VITE_FAKE_ROOM
/** The hosted pipeline's address (worker/modal_app.py on Modal), when this copy was built with one: every room but the
 * sample is built and served there, and anyone can start their own. Empty: the server this page came from. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')
export const LIVE = STATIC && API_BASE !== ''
/** The static copy's own room, whose files sit next to the page. */
export const isSample = (id: string) => STATIC && (id === STATIC_ROOM || id === FAKE_ROOM)

/** The rooms this browser has started or opened, newest first: the workspace's list (there are no accounts). */
export function myRooms(): string[] {
  try {
    return JSON.parse(localStorage.getItem('sightseen-rooms') ?? '[]')
  } catch {
    return []
  }
}
export function rememberRoom(id: string) {
  try {
    localStorage.setItem('sightseen-rooms', JSON.stringify([id, ...myRooms().filter(known => known !== id)].slice(0, 30)))
  } catch {
    // private window: the room's own link still works
  }
}

/** When the server can't be reached at all, a sentence a visitor can act on instead of the browser's "Failed to fetch". */
function reachable<T>(call: Promise<T>): Promise<T> {
  return call.catch(problem => {
    if (problem instanceof TypeError) {
      throw new Error(LIVE
        ? 'The live part isn’t answering right now. Try again in a minute. The sample room still works.'
        : 'The server isn’t answering. Is it running?')
    }
    throw problem
  })
}

export const api = {
  create: (label: string) =>
    reachable(fetch(`${API_BASE}/api/requests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    }).then(r => read<RoomRequest>(r))),
  request: (id: string) =>
    reachable(fetch(isSample(id) ? `/api/requests/${id}/request.json` : `${API_BASE}/api/requests/${id}`).then(r => read<RoomRequest>(r))),
  rerun: (id: string, model: string) =>
    reachable(fetch(`${API_BASE}/api/requests/${id}/rerun?model=${encodeURIComponent(model)}`, { method: 'POST' })
      .then(r => read<RoomRequest>(r))),
  film: (token: string) => reachable(fetch(`${API_BASE}/api/film/${token}`).then(r => read<FilmView>(r))),
}

const ARRIVAL_WAIT = 120_000 // milliseconds a fully sent video gets to show up, before it counts as lost

/** Sends a video and reports how much has gone (fetch can't report upload progress; XHR can). `arrived` says whether the
 * server has it: past 150 s the hosted pipeline's gateway answers with a redirect a browser can't follow across sites,
 * though the video got there and the room is being built, so a fully sent upload that "fails" gets checked first. */
export function uploadVideo(url: string, file: File, onProgress: (fraction: number) => void, arrived?: () => Promise<boolean>): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let sent = false
    xhr.open('POST', url)
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.upload.onload = () => {
      sent = true
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve()
      let message = `The upload didn't go through (${xhr.status}).`
      try {
        const body = JSON.parse(xhr.responseText)
        if (body?.message) message = body.message
      } catch {
        // keep the plain message
      }
      reject(new Error(message))
    }
    const stopped = () => reject(new Error('The upload stopped. Check the connection and try again.'))
    xhr.onerror = () => {
      if (!sent || !arrived) return stopped()
      const giveUp = Date.now() + ARRIVAL_WAIT
      const check = () => {
        arrived().catch(() => false).then(yes => {
          if (yes) resolve()
          else if (Date.now() > giveUp) stopped()
          else window.setTimeout(check, 3000)
        })
      }
      check()
    }
    const form = new FormData()
    form.append('video', file)
    xhr.send(form)
  })
}
