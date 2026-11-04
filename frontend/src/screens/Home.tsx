import { useState, type FormEvent } from 'react'
import {
  ArrowRight,
  Box,
  Check,
  MapPin,
  ScanLine,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { STATIC_ROOM, api } from '../api'
import CopyLink from '../components/CopyLink'
import type { RoomRequest } from '../types'

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
    <main className="relative min-h-screen overflow-hidden px-5 py-5 sm:px-8 sm:py-8">
      <div className="pointer-events-none absolute inset-0 opacity-30">
        <div className="absolute left-[8%] top-[18%] h-64 w-64 rounded-full border border-cyan-300/20" />
        <div className="absolute left-[12%] top-[22%] h-48 w-48 rounded-full border border-cyan-300/10" />
        <div className="absolute bottom-[-6rem] right-[-4rem] h-80 w-80 rounded-full border border-pink-300/20" />
      </div>

      <div className="relative mx-auto flex min-h-[calc(100vh-2.5rem)] max-w-7xl flex-col">
        <header className="flex items-center justify-between">
          <a
            href="/"
            className="flex items-center gap-2 text-lg font-semibold tracking-tight text-white"
          >
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20">
              <ScanLine size={20} strokeWidth={2.5} />
            </span>
            SightSeen
          </a>

          {STATIC_ROOM && (
            <a
              href="/"
              className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cyan-300/45 hover:bg-white/10"
            >
              Explore demo
            </a>
          )}
        </header>

        <section className="grid flex-1 items-center gap-12 py-16 lg:grid-cols-[1.05fr_0.95fr] lg:py-20">
          <div className="max-w-2xl">
            <div className="animate-rise-in inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1.5 text-xs font-semibold tracking-wide text-cyan-100">
              <Sparkles size={14} />
              RENT WITH A CLEARER VIEW
            </div>

            <h1 className="animate-rise-in delay-1 mt-6 text-5xl font-semibold leading-[0.98] tracking-[-0.055em] text-white sm:text-6xl lg:text-7xl">
              See the room
              <span className="block bg-gradient-to-r from-cyan-300 via-sky-200 to-pink-300 bg-clip-text text-transparent">
                before you sign.
              </span>
            </h1>

            <p className="animate-rise-in delay-2 mt-6 max-w-xl text-base leading-7 text-slate-300 sm:text-lg">
              Turn a landlord’s phone video into a room you can explore at real size.
              Walk through it, measure it, inspect it, and find out if your things fit.
            </p>

            {!request ? (
              <form
                onSubmit={start}
                className="animate-rise-in delay-2 mt-9 rounded-3xl border border-white/12 bg-white/[0.07] p-3 shadow-2xl shadow-black/20 backdrop-blur-xl"
              >
                <label htmlFor="label" className="sr-only">
                  Which place is it?
                </label>

                <div className="flex flex-col gap-3 sm:flex-row">
                  <div className="flex flex-1 items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/35 px-4">
                    <MapPin className="shrink-0 text-cyan-300" size={19} />
                    <input
                      id="label"
                      value={label}
                      onChange={e => setLabel(e.target.value)}
                      placeholder="Name the place — e.g. Room on Champlain Cres"
                      maxLength={120}
                      required
                      className="min-h-14 w-full bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
                    />
                  </div>

                  <button
                    disabled={busy}
                    className="inline-flex min-h-14 items-center justify-center gap-2 rounded-2xl bg-cyan-400 px-5 text-sm font-bold text-slate-950 transition duration-200 hover:bg-cyan-300 hover:shadow-lg hover:shadow-cyan-400/20 enabled:active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {busy ? 'Starting...' : 'Start exploring'}
                    {!busy && <ArrowRight size={18} />}
                  </button>
                </div>

                {error && (
                  <p className="px-2 pb-1 pt-3 text-sm text-red-300" role="alert">
                    {error}
                  </p>
                )}
              </form>
            ) : (
              <section className="mt-9 rounded-3xl border border-cyan-300/20 bg-cyan-300/[0.08] p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
                <div className="flex items-start gap-3">
                  <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-emerald-400 text-slate-950">
                    <Check size={20} strokeWidth={3} />
                  </div>
                  <div>
                    <p className="font-semibold text-white">Your private room link is ready.</p>
                    <p className="mt-1 text-sm leading-6 text-slate-300">
                      Send the filming link to the landlord, then keep your room page bookmarked.
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-5 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-xs font-bold tracking-wide text-cyan-200">
                      1. SEND TO THE LANDLORD
                    </p>
                    <CopyLink url={window.location.origin + request.landlordPath} />
                  </div>
                  <div>
                    <p className="mb-2 text-xs font-bold tracking-wide text-cyan-200">
                      2. KEEP YOUR PRIVATE ROOM
                    </p>
                    <a
                      href={request.roomPath}
                      className="flex min-h-11 items-center justify-between rounded-xl border border-white/15 bg-white/10 px-4 text-sm font-semibold text-white transition hover:border-cyan-300/50 hover:bg-white/15"
                    >
                      Open room page <ArrowRight size={17} />
                    </a>
                  </div>
                </div>
              </section>
            )}

            <div className="mt-9 flex flex-wrap gap-x-6 gap-y-3 text-sm text-slate-400">
              <span className="inline-flex items-center gap-2">
                <ShieldCheck size={16} className="text-cyan-300" />
                No account for landlords
              </span>
              <span className="inline-flex items-center gap-2">
                <Box size={16} className="text-cyan-300" />
                Real-scale 3D space
              </span>
            </div>
          </div>

          <div className="animate-float relative mx-auto w-full max-w-xl">
            <div className="absolute -inset-5 rounded-[2.5rem] bg-gradient-to-br from-cyan-400/20 via-transparent to-pink-400/20 blur-2xl" />

            <div className="relative overflow-hidden rounded-[2rem] border border-white/15 bg-slate-900/60 p-4 shadow-2xl shadow-black/40 backdrop-blur-xl">
              <div className="relative aspect-[4/5] overflow-hidden rounded-[1.4rem] bg-[linear-gradient(145deg,#123e4e_0%,#0d202b_43%,#451a42_100%)]">
                <div className="absolute inset-x-[12%] bottom-[8%] h-[38%] rounded-t-[2rem] border border-cyan-100/20 bg-cyan-200/10 backdrop-blur-sm" />
                <div className="absolute left-[22%] top-[17%] h-[38%] w-[56%] rounded-2xl border border-cyan-100/20 bg-gradient-to-b from-cyan-200/15 to-transparent" />
                <div className="absolute left-[28%] top-[24%] h-[20%] w-[18%] rounded-lg bg-cyan-200/20 shadow-[0_0_40px_rgba(92,221,255,0.25)]" />
                <div className="absolute right-[28%] top-[24%] h-[20%] w-[18%] rounded-lg bg-pink-200/15 shadow-[0_0_40px_rgba(255,105,180,0.2)]" />

                <div className="absolute left-5 top-5 rounded-2xl border border-white/15 bg-slate-950/55 px-3 py-2.5 backdrop-blur-md">
                  <p className="text-[10px] font-bold tracking-[0.16em] text-cyan-200">ROOM SCAN</p>
                  <p className="mt-1 text-sm font-semibold text-white">Ready to explore</p>
                </div>

                <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-white/15 bg-slate-950/60 p-4 backdrop-blur-md">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-slate-400">Measured space</p>
                      <p className="mt-1 text-xl font-semibold text-white">12.8 m²</p>
                    </div>
                    <div className="rounded-xl bg-cyan-400 px-3 py-2 text-xs font-bold text-slate-950">
                      3D READY
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}