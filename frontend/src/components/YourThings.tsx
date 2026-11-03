import { useEffect, useRef, useState } from 'react'
import { Loader2, Package, Upload, X } from 'lucide-react'
import { STATIC } from '../api'

/** items/{id}/item.json (worker/items.py): one of the student's own things, built from a photo at its real width. */
export type Item = {
  id: string
  name: string
  widthCm: number
  status: 'queued' | 'outlining' | 'building' | 'done' | 'failed'
  why?: string | null
  sizeCm?: number[]
  builtBy?: string
}

const STEPS: Record<Item['status'], string> = {
  queued: 'Waiting for the GPU',
  outlining: 'SAM 3.1 is finding it in the photo',
  building: 'Building it in 3D (a few minutes)',
  done: 'In the room',
  failed: 'Couldn’t build it',
}

/**
 * The student's own things: the list (polled while any is being built) and, unless this is the static copy, a form to
 * add one from a photo, what it is, and its real width. Each one that's done is handed to `onReady` once, which puts
 * it in the room. Stays mounted while closed, so things built earlier appear when the room opens.
 */
export default function YourThings({ base, open, onClose, onReady }: {
  base: string
  open: boolean
  onClose: () => void
  onReady: (item: Item) => void
}) {
  const [items, setItems] = useState<Item[]>([])
  const [photo, setPhoto] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [width, setWidth] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const placed = useRef(new Set<string>())
  const ready = useRef(onReady)
  ready.current = onReady

  const refresh = async () => {
    try {
      const response = await fetch(STATIC ? `${base}/items.json` : `${base}/items`)
      if (!response.ok) return
      const found: Item[] = await response.json()
      setItems(found)
      for (const item of found) {
        if (item.status === 'done' && !placed.current.has(item.id)) {
          placed.current.add(item.id)
          ready.current(item)
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
    if (!photo || !name.trim() || !(cm >= 5 && cm <= 500)) {
      setError('Pick a photo, say what it is, and give its longest side in cm (5 to 500).')
      return
    }
    setSending(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('photo', photo)
      form.append('name', name.trim())
      form.append('widthCm', String(cm))
      const response = await fetch(`${base}/items`, { method: 'POST', body: form })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message || `Something went wrong (${response.status}).`)
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
  return (
    <div className="absolute right-3 top-3 z-10 max-h-[calc(100%-4.5rem)] w-80 overflow-y-auto rounded-xl bg-white/95 p-4 text-sm text-slate-800 shadow-lg">
      <div className="flex items-center justify-between">
        <p className="inline-flex items-center gap-1.5 font-medium text-slate-900">
          <Package size={16} /> Your things
        </p>
        <button type="button" onClick={onClose} className="rounded p-1 hover:bg-slate-100" aria-label="Close">
          <X size={16} />
        </button>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        A photo of something you own becomes a 3D model at its real size, on free floor in the room. Move it with Move
        things: it says how much room it has, or what it runs into.
      </p>

      {items.length > 0 && (
        <ul className="mt-3 space-y-2">
          {items.map(item => (
            <li key={item.id} className="rounded-lg bg-slate-50 px-3 py-2">
              <p className="font-medium capitalize">{item.name}</p>
              <p className="text-xs text-slate-600">
                {item.status === 'done' && item.sizeCm
                  ? `${Math.round(item.sizeCm[0])} × ${Math.round(item.sizeCm[2])} cm, ${Math.round(item.sizeCm[1])} cm tall · ${item.builtBy ?? ''}`
                  : `${item.widthCm} cm on its longest side`}
              </p>
              <p className={`mt-0.5 inline-flex items-center gap-1 text-xs ${item.status === 'failed' ? 'text-red-700' : 'text-slate-500'}`}>
                {working && item.status !== 'done' && item.status !== 'failed' && <Loader2 size={12} className="animate-spin" />}
                {STEPS[item.status]}{item.status === 'failed' && item.why ? `: ${item.why}` : ''}
              </p>
            </li>
          ))}
        </ul>
      )}

      {!STATIC && (
        <div className="mt-3 space-y-2 border-t border-slate-200 pt-3">
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-slate-300 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50">
            <Upload size={14} />
            {photo ? photo.name : 'Pick a photo of it (JPG or PNG)'}
            <input type="file" accept="image/jpeg,image/png,image/webp" className="hidden"
              onChange={event => setPhoto(event.target.files?.[0] ?? null)} />
          </label>
          <input value={name} onChange={event => setName(event.target.value)} placeholder="What is it? (desk, chair, monitor)"
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs" maxLength={40} />
          <input value={width} onChange={event => setWidth(event.target.value)} placeholder="Its longest side across the floor, in cm"
            inputMode="decimal" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs" />
          <button type="button" onClick={send} disabled={sending}
            className="w-full rounded-lg bg-slate-900 px-3 py-2 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50">
            {sending ? 'Sending...' : 'Build it in 3D'}
          </button>
          {error && <p className="text-xs text-red-700">{error}</p>}
        </div>
      )}
    </div>
  )
}
