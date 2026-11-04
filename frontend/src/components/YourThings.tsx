import { useEffect, useRef, useState } from 'react'
import { Camera, Crosshair, ImageUp, Loader2, Package, RotateCcw, X } from 'lucide-react'
import { API_BASE, LIVE, STATIC } from '../api'

/** items/{id}/item.json (worker/items.py): one of the student's own things, built from a photo at its real size. */
export type Item = {
  id: string
  name: string
  widthCm: number
  status: 'queued' | 'outlining' | 'building' | 'done' | 'failed'
  why?: string | null
  sizeCm?: number[]
  builtBy?: string
  roomScale?: number
  createdAt?: string
  /** Where its model's folder is, when that isn't this room's items folder (a hosted visitor's own thing) */
  folder?: string
}

const STEPS: Record<Item['status'], string> = {
  queued: 'Waiting for the GPU',
  outlining: 'SAM 3.1 is finding it in the photo',
  building: 'SAM 3D is building it (about 3 minutes)',
  done: 'In the room',
  failed: 'Couldn’t build it',
}

const PRESS = 'transition-[scale,background-color] duration-150 ease-out enabled:active:scale-[0.96] motion-reduce:transition-none motion-reduce:enabled:active:scale-100'

/** A hosted visitor's own things: which ones are theirs, per room, in this browser (it's only a convenience). */
function remembered(room: string): string[] {
  try {
    return JSON.parse(localStorage.getItem(`sightseen-things-${room}`) ?? '[]')
  } catch {
    return []
  }
}
function remember(room: string, ids: string[]) {
  try {
    localStorage.setItem(`sightseen-things-${room}`, JSON.stringify(ids))
  } catch {
    // private window: the thing still shows until the page closes
  }
}

/** The camera, inside the page, so a laptop's works as well as a phone's. `onFail` when it can't open. */
function CameraView({ onTake, onCancel, onFail }: {
  onTake: (file: File) => void
  onCancel: () => void
  onFail: (why: string) => void
}) {
  const video = useRef<HTMLVideoElement>(null)
  const stream = useRef<MediaStream | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let gone = false
    navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 } }, audio: false })
      .then(found => {
        if (gone) {
          found.getTracks().forEach(track => track.stop())
          return
        }
        stream.current = found
        if (video.current) video.current.srcObject = found
      })
      .catch(() => {
        if (!gone) onFail('The camera didn’t open. Choose a photo instead.')
      })
    return () => {
      gone = true
      stream.current?.getTracks().forEach(track => track.stop())
    }
  }, [])

  const take = () => {
    const element = video.current
    if (!element?.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = element.videoWidth
    canvas.height = element.videoHeight
    canvas.getContext('2d')!.drawImage(element, 0, 0)
    canvas.toBlob(blob => {
      if (blob) onTake(new File([blob], 'photo.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.92)
  }

  return (
    <div className="space-y-2">
      <video ref={video} autoPlay playsInline muted onLoadedData={() => setReady(true)}
        className="aspect-[4/3] w-full rounded-lg bg-slate-900 object-cover" />
      <p className="text-xs text-slate-500">Get the whole thing in the picture, with some floor round it.</p>
      <div className="flex gap-2">
        <button type="button" onClick={take} disabled={!ready}
          className={`inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-slate-900 px-3 py-2 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50 ${PRESS}`}>
          <Camera size={14} /> Take the picture
        </button>
        <button type="button" onClick={onCancel}
          className={`rounded-lg bg-slate-100 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-200 ${PRESS}`}>
          Cancel
        </button>
      </div>
    </div>
  )
}

/**
 * The student's own things: the list (checked while any is being built) and a form: a picture of it (from the camera,
 * or a photo they have), what it is, and its longest side. Each one that's done is handed to `onReady` once, which
 * puts it in the room, and `onFind` turns the view to it. Stays mounted while closed, so things built earlier appear
 * when the room opens. A hosted copy builds a visitor's things on Modal, and only they see them.
 */
export default function YourThings({ base, room, scale, open, onClose, onReady, onFind }: {
  base: string
  /** The room's id, to remember a hosted visitor's own things by */
  room: string
  /** Real metres per room unit */
  scale: () => number
  open: boolean
  onClose: () => void
  onReady: (item: Item, fresh: boolean) => void
  onFind: (id: string) => void
}) {
  const canAdd = !STATIC || LIVE
  const [items, setItems] = useState<Item[]>([])
  const [photo, setPhoto] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [name, setName] = useState('')
  const [width, setWidth] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const placed = useRef(new Set<string>())
  const waited = useRef(new Set<string>()) // seen unfinished: when it's done, it's new news
  const mine = useRef<string[]>(remembered(room)) // a hosted visitor's, even where this browser can't keep them
  const ready = useRef(onReady)
  ready.current = onReady

  useEffect(() => {
    if (!photo) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(photo)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [photo])

  const refresh = async () => {
    try {
      const response = await fetch(STATIC ? `${base}/items.json` : `${base}/items`)
      const here: Item[] = response.ok ? await response.json().catch(() => []) : []
      const theirs = LIVE
        ? (await Promise.all(mine.current.map(id => fetch(`${API_BASE}/api/items/${id}`)
          .then(r => (r.ok ? r.json() as Promise<Item> : null)).catch(() => null))))
          .filter((item): item is Item => item !== null)
          .map(item => ({ ...item, folder: `${API_BASE}/api/items/${item.id}/` }))
        : []
      const found = [...here, ...theirs].map(item => {
        // A build that never reported back (its machine restarted): say so instead of spinning for ever
        const stale = item.status !== 'done' && item.status !== 'failed' && item.createdAt
          && Date.now() - Date.parse(item.createdAt) > 20 * 60 * 1000
        return stale ? { ...item, status: 'failed' as const, why: 'It took too long and stopped. Try it again.' } : item
      })
      setItems(found)
      for (const item of found) {
        if (item.status === 'done' && !placed.current.has(item.id)) {
          placed.current.add(item.id)
          ready.current(item, waited.current.has(item.id))
        } else if (item.status !== 'done') {
          waited.current.add(item.id)
        }
      }
    } catch {
      // the server's away: keep the list as it is
    }
  }

  useEffect(() => {
    refresh()
  }, [base])
  const working = items.some(item => item.status !== 'done' && item.status !== 'failed')
  useEffect(() => {
    if (!working) return
    const timer = setInterval(refresh, 3000)
    return () => clearInterval(timer)
  }, [working, base])

  const send = async () => {
    const cm = Number(width)
    if (!photo) return setError('Take a picture of it first, or choose a photo.')
    if (!name.trim()) return setError('Say what it is: desk, chair, monitor.')
    if (!(cm >= 5 && cm <= 500)) return setError('Give its longest side in cm (5 to 500). A tape measure makes the fit check true.')
    setSending(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('photo', photo)
      form.append('name', name.trim())
      form.append('widthCm', String(cm))
      if (LIVE) form.append('scale', String(scale()))
      const response = await fetch(LIVE ? `${API_BASE}/api/items` : `${base}/items`, { method: 'POST', body: form })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message || `Something went wrong (${response.status}).`)
      }
      if (LIVE) {
        const added: Item = await response.json()
        mine.current = [...mine.current, added.id]
        remember(room, mine.current)
      }
      setPhoto(null)
      setName('')
      setWidth('')
      await refresh()
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem))
    } finally {
      setSending(false)
    }
  }

  if (!open) return null
  const canFilm = typeof window !== 'undefined' && window.isSecureContext && !!navigator.mediaDevices?.getUserMedia
  return (
    <div className="stage-panel">
      <div className="flex items-center justify-between">
        <p className="inline-flex items-center gap-1.5 font-medium text-slate-900">
          <Package size={16} /> Your things
        </p>
        <button type="button" onClick={onClose} className="rounded p-1 hover:bg-slate-100" aria-label="Close">
          <X size={16} />
        </button>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        Take a picture of something you own. SAM 3D builds it at its real size and stands it on free floor here. Move
        it with Move things: it says how much room it has, or what it runs into.
      </p>

      {items.length > 0 && (
        <ul className="mt-3 space-y-2">
          {items.map(item => (
            <li key={item.id} className="flex items-start justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2">
              <div className="min-w-0">
                <p className="font-medium capitalize">{item.name}</p>
                <p className="text-xs text-slate-600">
                  {item.status === 'done' && item.sizeCm
                    ? `${Math.round(item.sizeCm[0])} × ${Math.round(item.sizeCm[2])} cm, ${Math.round(item.sizeCm[1])} cm tall`
                    : `${item.widthCm} cm on its longest side`}
                </p>
                <p className={`mt-0.5 inline-flex items-center gap-1 text-xs ${item.status === 'failed' ? 'text-red-700' : 'text-slate-500'}`}>
                  {item.status !== 'done' && item.status !== 'failed' && <Loader2 size={12} className="shrink-0 animate-spin motion-reduce:animate-none" />}
                  {item.status === 'failed' && item.why ? item.why : STEPS[item.status]}
                </p>
              </div>
              {item.status === 'done' && (
                <button type="button" onClick={() => onFind(item.id)} title="Turn the view to it" aria-label={`Show the ${item.name}`}
                  className={`shrink-0 rounded-lg p-1.5 text-slate-600 hover:bg-slate-200 ${PRESS}`}>
                  <Crosshair size={15} />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      {canAdd ? (
        <div className="mt-3 space-y-2 border-t border-slate-200 pt-3">
          {cameraOn ? (
            <CameraView
              onTake={file => {
                setPhoto(file)
                setCameraOn(false)
              }}
              onCancel={() => setCameraOn(false)}
              onFail={why => {
                setError(why)
                setCameraOn(false)
              }}
            />
          ) : preview ? (
            <div className="relative">
              <img src={preview} alt="Your picture" className="max-h-32 w-full rounded-lg object-cover outline outline-1 -outline-offset-1 outline-black/10" />
              <button type="button" onClick={() => setPhoto(null)}
                className={`absolute right-2 top-2 inline-flex items-center gap-1 rounded-lg bg-black/70 px-2 py-1 text-xs text-white hover:bg-black/90 ${PRESS}`}>
                <RotateCcw size={12} /> Retake
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {canFilm ? (
                <button type="button" onClick={() => {
                  setError(null)
                  setCameraOn(true)
                }} className={`inline-flex items-center justify-center gap-1.5 rounded-lg bg-slate-900 px-2 py-2.5 text-xs font-medium text-white hover:bg-slate-700 ${PRESS}`}>
                  <Camera size={14} /> Take a picture
                </button>
              ) : (
                <label className="inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-slate-900 px-2 py-2.5 text-xs font-medium text-white hover:bg-slate-700">
                  <Camera size={14} /> Take a picture
                  <input type="file" accept="image/*" capture="environment" className="hidden"
                    onChange={event => setPhoto(event.target.files?.[0] ?? null)} />
                </label>
              )}
              <label className="inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-slate-100 px-2 py-2.5 text-xs font-medium text-slate-800 hover:bg-slate-200">
                <ImageUp size={14} /> Choose a photo
                <input type="file" accept="image/jpeg,image/png,image/webp" className="hidden"
                  onChange={event => setPhoto(event.target.files?.[0] ?? null)} />
              </label>
            </div>
          )}
          <input value={name} onChange={event => setName(event.target.value)} placeholder="What is it? (desk, chair, monitor)"
            aria-label="What it is" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-base sm:text-xs" maxLength={40} />
          <input value={width} onChange={event => setWidth(event.target.value)} placeholder="Its longest side across the floor, in cm"
            aria-label="Its longest side in centimetres" inputMode="decimal" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-base sm:text-xs" />
          {error && <p className="text-xs text-red-700" role="alert">{error}</p>}
          <button type="button" onClick={send} disabled={sending}
            className={`w-full rounded-full bg-tape text-ink px-3 py-2 text-xs font-semibold hover:brightness-105 disabled:opacity-50 ${PRESS}`}>
            {sending ? 'Sending...' : 'Build it in 3D'}
          </button>
        </div>
      ) : (
        <p className="mt-3 border-t border-slate-200 pt-3 text-xs text-slate-500">
          Building a new thing from a picture runs on a GPU; the demo video shows it, and the chair above was made that way.
        </p>
      )}
    </div>
  )
}
