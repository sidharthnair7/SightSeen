import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Box, Check, KeyRound, Loader2, MapPin, RotateCcw, Video } from 'lucide-react'
import { FAKE_ROOM, STATIC_ROOM, api, myRooms, rememberRoom } from '../api'
import ClickSpark from '../components/ClickSpark'
import CopyLink from '../components/CopyLink'
import SiteNav from '../components/SiteNav'
import type { RequestStatus, RoomRequest } from '../types'

const STATUS: Record<RequestStatus, { text: string; dot: string }> = {
  WAITING_FOR_VIDEO: { text: 'Waiting for the landlord', dot: 'bg-amber-400' },
  VIDEO_RECEIVED: { text: 'Video in', dot: 'bg-steel' },
  RECONSTRUCTING: { text: 'Building the 3D room', dot: 'bg-steel' },
  READY: { text: 'Ready', dot: 'bg-emerald-500' },
  FAILED: { text: 'Stopped', dot: 'bg-red-500' },
}

const BUTTON = 'inline-flex min-h-10 items-center justify-center gap-1.5 rounded-xl px-3.5 text-sm font-semibold transition-[scale,filter,background-color] duration-150 ease-out enabled:active:scale-[0.96] motion-reduce:transition-none'

/** One of the student's rooms: where it's got to, the link for the landlord while it's waiting, and the way in. */
function RoomCard({ room }: { room: RoomRequest }) {
  const status = STATUS[room.status]
  const working = room.status === 'VIDEO_RECEIVED' || room.status === 'RECONSTRUCTING' || (room.status === 'READY' && !!room.stage)
  return (
    <article className="rounded-2xl bg-white p-4 ring-1 ring-hair">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-semibold text-ink">{room.label}</h3>
          <p className="mt-1 inline-flex items-center gap-1.5 text-xs text-steel">
            {working
              ? <Loader2 size={12} className="animate-spin text-ink motion-reduce:animate-none" />
              : <span className={`inline-block h-2 w-2 rounded-full ${status.dot}`} />}
            {status.text}{room.stage ? `: ${room.stage}` : ''}
          </p>
        </div>
        {(room.status === 'READY' || working) && (
          <a href={room.roomPath} className={`${BUTTON} shrink-0 bg-ink text-white hover:bg-ink/85`}>
            Open the room
          </a>
        )}
        {room.status === 'FAILED' && (
          <a href={room.roomPath} className={`${BUTTON} shrink-0 bg-white text-ink ring-1 ring-hair hover:ring-ink/40`}>
            <RotateCcw size={15} /> See why
          </a>
        )}
      </div>
      {room.status === 'WAITING_FOR_VIDEO' && (
        <div className="mt-3 space-y-2">
          <p className="text-sm font-semibold text-ink">Send the landlord this link</p>
          <CopyLink url={window.location.origin + room.landlordPath} />
          {room.code && (
            <p className="inline-flex items-center gap-1.5 text-xs text-steel">
              <KeyRound size={13} className="text-ink" /> They’ll hold up the code <span className="rounded-[3px] bg-tape px-1 font-semibold tabular-nums text-ink">{room.code}</span> as they film.
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
                    <h1 className="font-display text-[clamp(2.4rem,4.6vw,3.6rem)] font-[760] leading-none text-ink">Your rooms</h1>
          <p className="mt-4 text-lg leading-8 text-steel">
            Ask a landlord for a video, watch it get built, and check it before you sign. There’s no account: your list is kept
            in this browser.
          </p>
        </header>

        <div className="mt-8 grid items-start gap-6 lg:grid-cols-[23rem_1fr]">
          <section className="rounded-[22px] bg-white p-5 ring-1 ring-hair">
            <h2 className="text-lg font-semibold text-ink">Start a request</h2>
            <p className="mt-1 text-sm text-steel">Name the place. You’ll get one link to text the landlord.</p>
            <form onSubmit={start} className="mt-4 space-y-3">
              <label htmlFor="label" className="sr-only">Which place is it?</label>
              <div className="flex items-center gap-3 rounded-2xl border border-hair bg-paper px-4 focus-within:border-ink">
                <MapPin className="shrink-0 text-steel" size={18} />
                <input
                  id="label"
                  value={label}
                  onChange={event => setLabel(event.target.value)}
                  placeholder="Room on Champlain Cres"
                  maxLength={120}
                  required
                  className="min-h-12 w-full min-w-0 bg-transparent text-base text-ink outline-none placeholder:text-steel sm:text-sm"
                />
              </div>
              <ClickSpark sparkColor={['#0f1f38', '#ffd23f']} sparkCount={14} sparkRadius={52} sparkSize={10} sparkWidth={2} duration={550} shape="line" className="h-auto!">
                <button disabled={busy}
                  className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-tape px-5 text-[15px] font-semibold text-ink shadow-[inset_0_-2px_0_rgb(15_31_56/0.16)] transition-[scale] duration-150 ease-out enabled:active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-60 motion-reduce:transition-none motion-reduce:enabled:active:scale-100">
                  {busy ? 'Starting...' : 'Get the landlord’s link'}
                </button>
              </ClickSpark>
              {error && <p className="text-sm text-red-700" role="alert">{error}</p>}
            </form>
            <ol className="mt-5 space-y-2.5 border-t border-hair pt-4 text-sm text-steel">
              <li className="flex gap-2.5"><Video size={16} className="mt-0.5 shrink-0 text-ink" /> The landlord films in the browser, holding up a one-time code. No app, no account.</li>
              <li className="flex gap-2.5"><Box size={16} className="mt-0.5 shrink-0 text-ink" /> The room builds in 3D: its walls in about 5 minutes, its things in about 20.</li>
              <li className="flex gap-2.5"><Check size={16} className="mt-0.5 shrink-0 text-ink" /> You walk it, measure it, fit your things, and check it for scams.</li>
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
                <div className="rounded-2xl border border-dashed border-hair p-6 text-sm text-steel">
                  {loaded ? 'No rooms yet. Start a request, or try one of the examples below.' : 'Loading your rooms...'}
                </div>
              )}
            </section>

            {(STATIC_ROOM || FAKE_ROOM) && (
              <section>
                <h2 className="text-sm font-semibold text-steel">Examples to try</h2>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {STATIC_ROOM && (
                    <a href={`/room/${STATIC_ROOM}`} className="group block overflow-hidden rounded-2xl bg-white ring-1 ring-hair transition-[box-shadow] duration-200 hover:ring-ink/30">
                      <div className="overflow-hidden">
                        <img src="/images/example-sample.jpg" alt="The sample room in 3D: a desk with two monitors, a blue suitcase and the bed" loading="lazy"
                          className="aspect-video w-full object-cover transition-[scale] duration-300 ease-out group-hover:scale-[1.03] motion-reduce:transition-none motion-reduce:group-hover:scale-100" />
                      </div>
                      <div className="p-4">
                                                <h3 className="font-semibold text-ink">Walk a sample room</h3>
                        <p className="mt-1 text-sm text-steel">A real student’s room, built from one phone video. Measure it, move things, fit your own.</p>
                      </div>
                    </a>
                  )}
                  {FAKE_ROOM && (
                    <a href={`/room/${FAKE_ROOM}`} className="group block overflow-hidden rounded-2xl bg-white ring-1 ring-hair transition-[box-shadow] duration-200 hover:ring-ink/30">
                      <div className="relative overflow-hidden">
                        <img src="/images/example-fake.jpg" alt="The suspicious listing in 3D: a hotel bed, a dresser and two terrace chairs" loading="lazy"
                          className="aspect-video w-full object-cover transition-[scale] duration-300 ease-out group-hover:scale-[1.03] motion-reduce:transition-none motion-reduce:group-hover:scale-100" />
                        <span className="absolute left-3 top-3 rounded-full bg-alert px-2.5 py-1 text-xs font-semibold text-white shadow-sm">Don’t pay yet</span>
                      </div>
                      <div className="p-4">
                                                <h3 className="font-semibold text-ink">See it catch a fake</h3>
                        <p className="mt-1 text-sm text-steel">A tour from somewhere else, passed off as the room in an ad. See what the checks say.</p>
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
