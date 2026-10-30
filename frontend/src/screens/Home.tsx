import { useState, type FormEvent } from 'react'
import { api } from '../api'
import CopyLink from '../components/CopyLink'
import type { RoomRequest } from '../types'

/** The student starts here: name the place, get one link for the landlord and a private page for themselves. */
export default function Home() {
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [request, setRequest] = useState<RoomRequest | null>(null)

  async function start(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      setRequest(await api.create(label))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto max-w-xl px-4 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">SightSeen</h1>
      <p className="mt-2 text-slate-600">
        See the room before you sign. Send the landlord one link, and their video becomes a 3D room you can look
        around and measure.
      </p>

      {!request ? (
        <form onSubmit={start} className="mt-8 space-y-3">
          <label htmlFor="label" className="block text-sm font-medium">Which place is it?</label>
          <input
            id="label"
            value={label}
            onChange={e => setLabel(e.target.value)}
            placeholder="Room on Champlain Cres"
            maxLength={120}
            required
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2"
          />
          <button
            disabled={busy}
            className="rounded-lg bg-teal-700 px-4 py-2.5 font-medium text-white hover:bg-teal-600 disabled:opacity-60"
          >
            {busy ? 'Starting...' : 'Start a request'}
          </button>
          {error && <p className="text-sm text-red-700">{error}</p>}
        </form>
      ) : (
        <section className="mt-8 space-y-8">
          <div>
            <h2 className="font-medium">1. Send this link to the landlord</h2>
            <p className="mt-1 text-sm text-slate-600">It shows them how to film the room and takes the video.</p>
            <div className="mt-3"><CopyLink url={window.location.origin + request.landlordPath} /></div>
          </div>
          <div>
            <h2 className="font-medium">2. Keep your own page</h2>
            <p className="mt-1 text-sm text-slate-600">
              The 3D room shows up there once the video is in. It's your private link, so bookmark it.
            </p>
            <a href={request.roomPath} className="mt-3 inline-block font-medium text-teal-700 underline">
              Open your room page
            </a>
          </div>
        </section>
      )}
    </main>
  )
}
