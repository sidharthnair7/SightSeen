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
/** The hosted pipeline's address (worker/modal_app.py on Modal), when this copy was built with one: every room but the
 * sample is built and served there, and anyone can start their own. Empty: the server this page came from. */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''
export const LIVE = STATIC && API_BASE !== ''
/** The static copy's own room, whose files sit next to the page. */
export const isSample = (id: string) => STATIC && id === STATIC_ROOM

export const api = {
  create: (label: string) =>
    fetch(`${API_BASE}/api/requests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    }).then(r => read<RoomRequest>(r)),
  request: (id: string) =>
    fetch(isSample(id) ? `/api/requests/${id}/request.json` : `${API_BASE}/api/requests/${id}`).then(r => read<RoomRequest>(r)),
  rerun: (id: string, model: string) =>
    fetch(`${API_BASE}/api/requests/${id}/rerun?model=${encodeURIComponent(model)}`, { method: 'POST' })
      .then(r => read<RoomRequest>(r)),
  film: (token: string) => fetch(`${API_BASE}/api/film/${token}`).then(r => read<FilmView>(r)),
}

/** Sends a video and reports how much has gone (fetch can't report upload progress; XHR can). */
export function uploadVideo(url: string, file: File, onProgress: (fraction: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
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
    xhr.onerror = () => reject(new Error('The upload stopped. Check the connection and try again.'))
    const form = new FormData()
    form.append('video', file)
    xhr.send(form)
  })
}
