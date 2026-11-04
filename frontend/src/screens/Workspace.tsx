import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { ArrowRight, Box, Check, FileWarning, KeyRound, Loader2, MapPin, RotateCcw, Video } from 'lucide-react'
import { FAKE_ROOM, STATIC_ROOM, api, myRooms, rememberRoom } from '../api'
import ClickSpark from '../components/ClickSpark'
import CopyLink from '../components/CopyLink'
import GlBackground from '../components/GlBackground'
import SiteNav from '../components/SiteNav'
import type { RequestStatus, RoomRequest } from '../types'

const STATUS: Record<RequestStatus, { text: string; dot: string }> = {
  WAITING_FOR_VIDEO: { text: 'Waiting for the landlord', dot: 'bg-amber-400' },
  VIDEO_RECEIVED: { text: 'Video in', dot: 'bg-sky-400' },
  RECONSTRUCTING: { text: 'Building the 3D room', dot: 'bg-sky-400' },
  READY: { text: 'Ready', dot: 'bg-emerald-500' },
  FAILED: { text: 'Stopped', dot: 'bg-red-500' },
}

const BUTTON = 'inline-flex min-h-10 items-center justify-center gap-1.5 rounded-xl px-3.5 text-sm font-semibold transition-[scale,filter,background-color] duration-150 ease-out enabled:active:scale-[0.96] motion-reduce:transition-none'

/** One of the student's rooms: where it's got to, the link for the landlord while it's waiting, and the way in. */
function RoomCard({ room }: { room: RoomRequest }) {
  const status = STATUS[room.status]
  const working = room.status === 'VIDEO_RECEIVED' || room.status === 'RECONSTRUCTING' || (room.status === 'READY' && !!room.stage)
  return (
    <article className="liquid-glass rounded-2xl p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-semibold text-slate-950">{room.label}</h3>
          <p className="mt-1 inline-flex items-center gap-1.5 text-xs text-slate-600">
            {working
              ? <Loader2 size={12} className="animate-spin text-sky-600 motion-reduce:animate-none" />
              : <span className={`inline-block h-2 w-2 rounded-full ${status.dot}`} />}
            {status.text}{room.stage ? `: ${room.stage}` : ''}
          </p>
        </div>
        {(room.status === 'READY' || working) && (
          <a href={room.roomPath} className={`${BUTTON} shrink-0 bg-slate-900 text-white hover:bg-slate-700`}>
            Open <ArrowRight size={15} />
          </a>
        )}
        {room.status === 'FAILED' && (
          <a href={room.roomPath} className={`${BUTTON} shrink-0 bg-white/80 text-slate-900 hover:bg-white`}>
            <RotateCcw size={15} /> See why
          </a>
        )}
      </div>
      {room.status === 'WAITING_FOR_VIDEO' && (
        <div className="mt-3 space-y-2">
          <p className="text-xs font-bold tracking-wide text-sky-700">SEND THIS TO THE LANDLORD</p>
          <CopyLink url={window.location.origin + room.landlordPath} />
          {room.code && (
            <p className="inline-flex items-center gap-1.5 text-xs text-slate-600">
              <KeyRound size={13} className="text-sky-600" /> They’ll hold up the code <span className="font-semibold tabular-nums text-slate-900">{room.code}</span> as they film.
            </p>
          )}
        </div>
      )}
    </article>
  )
}

/**
 * The workspace: where a student actually starts a request and keeps track of their rooms. There are no accounts, so the
 * list lives in this browser; a room's own link works anywhere.
 */
export default function Workspace() {
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rooms, setRooms] = useState<RoomRequest[]>([])
  const [loaded, setLoaded] = useState(false)

  const refresh = useCallback(async () => {
    const found = await Promise.all(myRooms().map(id => api.request(id).catch(() => null)))
    setRooms(found.filter((room): room is RoomRequest => room !== null))
    setLoaded(true)
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(() => {
      if (!document.hidden) refresh()
    }, 6000)
    return () => clearInterval(timer)
  }, [refresh])

  async function start(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const made = await api.create(label)
      rememberRoom(made.id)
      setRooms(now => [made, ...now.filter(room => room.id !== made.id)])
      setLabel('')
    } catch (problem) {
      setError((problem as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <GlBackground />
      <SiteNav
        activeHref="/workspace"
        items={[
          { label: 'Home', href: '/' },
          { label: 'Workspace', href: '/workspace' },
          ...(STATIC_ROOM ? [{ label: 'Sample room', href: `/room/${STATIC_ROOM}` }] : []),
          ...(FAKE_ROOM ? [{ label: 'Catch a fake', href: `/room/${FAKE_ROOM}` }] : []),
        ]}
      />
      <main className="mx-auto min-h-screen max-w-6xl px-5 pb-16 pt-24 sm:px-8">
        <header className="max-w-2xl">
          <p className="text-sm font-bold tracking-[0.16em] text-pink-600">WORKSPACE</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">Your rooms</h1>
          <p className="mt-2 leading-7 text-slate-700">
            Ask a landlord for a video, watch it get built, and check it before you sign. There’s no account: your list is kept
            in this browser.
          </p>
        </header>

        <div className="mt-8 grid items-start gap-6 lg:grid-cols-[23rem_1fr]">
          <section className="liquid-glass rounded-3xl p-5">
            <h2 className="font-semibold text-slate-950">Start a request</h2>
            <p className="mt-1 text-sm text-slate-600">Name the place. You’ll get one link to text the landlord.</p>
            <form onSubmit={start} className="mt-4 space-y-3">
              <label htmlFor="label" className="sr-only">Which place is it?</label>
              <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/90 px-4 focus-within:border-sky-400">
                <MapPin className="shrink-0 text-sky-600" size={18} />
                <input
                  id="label"
                  value={label}
                  onChange={event => setLabel(event.target.value)}
                  placeholder="Room on Champlain Cres"
                  maxLength={120}
                  required
                  className="min-h-12 w-full min-w-0 bg-transparent text-base text-slate-900 outline-none placeholder:text-slate-500 sm:text-sm"
                />
              </div>
              <ClickSpark sparkColor={['#22d3ee', '#f472b6', '#0ea5e9']} sparkCount={18} sparkRadius={68} sparkSize={12} sparkWidth={2} duration={650} shape="star" glow={0.4} ring className="h-auto!">
                <button disabled={busy}
                  className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-300 via-sky-400 to-pink-400 px-5 text-sm font-black text-slate-950 shadow-lg shadow-sky-500/25 transition-[scale,box-shadow] duration-200 enabled:hover:scale-[1.02] enabled:hover:shadow-pink-500/25 enabled:active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-60 motion-reduce:transition-none motion-reduce:enabled:hover:scale-100 motion-reduce:enabled:active:scale-100">
                  {busy ? 'Starting...' : 'Get the landlord’s link'}
                  {!busy && <ArrowRight size={18} />}
                </button>
              </ClickSpark>
              {error && <p className="text-sm text-red-700" role="alert">{error}</p>}
            </form>
            <ol className="mt-5 space-y-2.5 border-t border-slate-900/10 pt-4 text-sm text-slate-700">
              <li className="flex gap-2.5"><Video size={16} className="mt-0.5 shrink-0 text-sky-600" /> The landlord films in the browser, holding up a one-time code. No app, no account.</li>
              <li className="flex gap-2.5"><Box size={16} className="mt-0.5 shrink-0 text-sky-600" /> The room builds in 3D: its walls in about 5 minutes, its things in about 20.</li>
              <li className="flex gap-2.5"><Check size={16} className="mt-0.5 shrink-0 text-sky-600" /> You walk it, measure it, fit your things, and check it for scams.</li>
            </ol>
          </section>

          <div className="min-w-0 space-y-6">
            <section>
              <h2 className="sr-only">Your rooms</h2>
              {rooms.length > 0 ? (
                <div className="grid gap-3">
                  {rooms.map(room => <RoomCard key={room.id} room={room} />)}
                </div>
              ) : (
                <div className="liquid-glass rounded-2xl p-6 text-sm text-slate-600">
                  {loaded ? 'No rooms yet. Start a request, or try one of the examples below.' : 'Loading your rooms...'}
                </div>
              )}
            </section>

            {(STATIC_ROOM || FAKE_ROOM) && (
              <section>
                <h2 className="text-sm font-bold tracking-[0.16em] text-slate-600">EXAMPLES</h2>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {STATIC_ROOM && (
                    <a href={`/room/${STATIC_ROOM}`} className="liquid-glass group block overflow-hidden rounded-2xl">
                      <div className="overflow-hidden">
                        <img src="/images/example-sample.jpg" alt="The sample room in 3D: a desk with two monitors, a blue suitcase and the bed" loading="lazy"
                          className="aspect-video w-full object-cover transition-[scale] duration-300 ease-out group-hover:scale-[1.03] motion-reduce:transition-none motion-reduce:group-hover:scale-100" />
                      </div>
                      <div className="p-4">
                        <Box size={20} className="text-sky-600" />
                        <h3 className="mt-2 font-semibold text-slate-950">Walk a sample room</h3>
                        <p className="mt-1 text-sm text-slate-600">A real student’s room, built from one phone video. Measure it, move things, fit your own.</p>
                      </div>
                    </a>
                  )}
                  {FAKE_ROOM && (
                    <a href={`/room/${FAKE_ROOM}`} className="liquid-glass group block overflow-hidden rounded-2xl">
                      <div className="relative overflow-hidden">
                        <img src="/images/example-fake.jpg" alt="The suspicious listing in 3D: a hotel bed, a dresser and two terrace chairs" loading="lazy"
                          className="aspect-video w-full object-cover transition-[scale] duration-300 ease-out group-hover:scale-[1.03] motion-reduce:transition-none motion-reduce:group-hover:scale-100" />
                        <span className="absolute left-3 top-3 rounded-full bg-red-600 px-2.5 py-1 text-xs font-bold text-white shadow-sm">Don’t pay yet</span>
                      </div>
                      <div className="p-4">
                        <FileWarning size={20} className="text-pink-600" />
                        <h3 className="mt-2 font-semibold text-slate-950">See it catch a fake</h3>
                        <p className="mt-1 text-sm text-slate-600">A tour from somewhere else, passed off as the room in an ad. See what the checks say.</p>
                      </div>
                    </a>
                  )}
                </div>
              </section>
            )}
          </div>
        </div>
      </main>
    </>
  )
}
