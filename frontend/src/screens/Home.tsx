import { useState, type FormEvent } from 'react'
import {
  ArrowRight,
  Box,
  Check,
  Compass,
  MapPin,
  Ruler,
  ScanLine,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { STATIC_ROOM, api } from '../api'
import ClickSpark from '../components/ClickSpark'
import CopyLink from '../components/CopyLink'
import DitherVeil from '../components/DitherVeil'
import PillNav from '../components/PillNav'
import ShinyText from '../components/ShinyText'
import type { RoomRequest } from '../types'

const navItems = [
  { label: 'Home', href: '#top' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Trust', href: '#trust' },
  { label: 'Start', href: '#start' },
]

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
    <main id="top" className="relative min-h-screen overflow-hidden px-5 py-5 sm:px-8 sm:py-8">
      <div className="grid-lines pointer-events-none absolute inset-x-0 top-0 h-[44rem]" />
      <div className="pointer-events-none absolute -left-24 top-20 h-72 w-72 rounded-full bg-cyan-400/25 blur-[110px]" />
      <div className="pointer-events-none absolute -right-24 top-32 h-80 w-80 rounded-full bg-pink-500/25 blur-[120px]" />
      <div className="pointer-events-none absolute left-[42%] top-[40%] h-64 w-64 rounded-full bg-violet-500/20 blur-[120px]" />

      <div className="relative mx-auto max-w-7xl">
        <header className="flex items-center justify-between gap-4">
          <a href="#top" className="flex shrink-0 items-center gap-2 text-lg font-bold tracking-tight text-white">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-cyan-300 via-sky-400 to-pink-500 text-slate-950 shadow-lg shadow-cyan-500/30">
              <ScanLine size={21} strokeWidth={2.7} />
            </span>
            <span className="hidden sm:block">SightSeen</span>
          </a>

          <PillNav
            items={navItems}
            defaultActiveIndex={0}
            theme="dark"
            layout="joined"
            size="sm"
            surface="glass"
            itemStyle="plain"
            hoverEffect="glide"
            indicator="pill"
            highlightColor="linear-gradient(135deg, #00d7ff 0%, #d83995 100%)"
            highlightTextColor="#04101b"
            barColor="rgba(5, 13, 32, 0.72)"
            textColor="#e9fbff"
            collapse="auto"
            className="max-w-[calc(100vw-6.5rem)]"
          />
        </header>

        <section className="grid min-h-[calc(100vh-5.5rem)] items-center gap-12 py-16 lg:grid-cols-[1.05fr_0.95fr] lg:py-20">
          <div className="max-w-2xl">
            <div className="animate-rise-in inline-flex items-center gap-2 rounded-full border border-cyan-200/25 bg-cyan-300/10 px-3 py-1.5 text-xs font-bold tracking-wide text-cyan-100 shadow-lg shadow-cyan-500/10">
              <Sparkles size={14} />
              RENT WITH A CLEARER VIEW
            </div>

            <h1 className="animate-rise-in delay-1 mt-6 text-5xl font-semibold leading-[0.96] tracking-[-0.06em] text-white sm:text-6xl lg:text-7xl">
              Don’t just view
              <span className="block bg-gradient-to-r from-cyan-300 via-sky-200 to-pink-400 bg-clip-text text-transparent">
                the listing.
              </span>
              <span className="block">Step inside it.</span>
            </h1>

            <div className="animate-rise-in delay-2 mt-6 max-w-xl text-base leading-7 sm:text-lg">
              <ShinyText
                color="#a7c9db"
                shineColor="#ffffff"
                speed={3}
                shineWidth={55}
                glow={0.18}
                trigger="loop"
              >
                Turn a landlord’s phone video into a room you can explore, measure,
                inspect, and fill with your own things.
              </ShinyText>
            </div>

            <div id="start">
              {!request ? (
                <form
                  onSubmit={start}
                  className="animate-rise-in delay-2 mt-9 rounded-[2rem] border border-cyan-200/20 bg-slate-950/40 p-3 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl"
                >
                  <label htmlFor="label" className="sr-only">
                    Which place is it?
                  </label>

                  <div className="flex flex-col gap-3 sm:flex-row">
                    <div className="flex flex-1 items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/55 px-4">
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

                    <ClickSpark
                      sparkColor={['#00d7ff', '#d83995', '#ffffff']}
                      sparkCount={18}
                      sparkRadius={68}
                      sparkSize={12}
                      sparkWidth={2}
                      duration={650}
                      shape="star"
                      glow={0.75}
                      ring
                      className="h-auto w-auto shrink-0"
                    >
                      <button
                        disabled={busy}
                        className="inline-flex min-h-14 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-300 via-sky-400 to-pink-500 px-5 text-sm font-black text-slate-950 shadow-lg shadow-cyan-500/30 transition duration-200 hover:scale-[1.03] hover:shadow-pink-500/30 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {busy ? 'Starting...' : 'Start exploring'}
                        {!busy && <ArrowRight size={18} />}
                      </button>
                    </ClickSpark>
                  </div>

                  {error && (
                    <p className="px-2 pb-1 pt-3 text-sm text-red-300" role="alert">
                      {error}
                    </p>
                  )}
                </form>
              ) : (
                <section className="mt-9 rounded-[2rem] border border-cyan-300/30 bg-cyan-300/[0.09] p-6 shadow-2xl shadow-cyan-950/30 backdrop-blur-xl">
                  <div className="flex items-start gap-3">
                    <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-emerald-300 to-cyan-300 text-slate-950">
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
                        className="flex min-h-11 items-center justify-between rounded-xl border border-white/15 bg-white/10 px-4 text-sm font-semibold text-white transition hover:border-pink-300/60 hover:bg-white/15"
                      >
                        Open room page <ArrowRight size={17} />
                      </a>
                    </div>
                  </div>
                </section>
              )}
            </div>

            <div className="mt-9 flex flex-wrap gap-x-6 gap-y-3 text-sm text-slate-300">
              <span className="inline-flex items-center gap-2">
                <ShieldCheck size={16} className="text-cyan-300" />
                No account for landlords
              </span>
              <span className="inline-flex items-center gap-2">
                <Box size={16} className="text-pink-300" />
                Real-scale 3D space
              </span>
            </div>
          </div>

          <div className="animate-float relative mx-auto w-full max-w-xl">
            <div className="animate-pulse-glow absolute -inset-7 rounded-[3rem] bg-gradient-to-br from-cyan-400/35 via-violet-500/15 to-pink-500/35 blur-3xl" />

            <div className="relative aspect-[4/5] overflow-hidden rounded-[2.2rem] border border-cyan-100/25 bg-slate-950 shadow-2xl shadow-black/60">
              <DitherVeil
                src="/images/room-hero.jpg"
                fit="cover"
                pattern="floyd"
                pixelSize={3}
                levels={2}
                palette="duotone"
                inkColor="#080316"
                paperColor="#bcf6ff"
                contrast={1.18}
                brightness={-0.02}
                revealRadius={165}
                softness={0.68}
                linger={0.8}
                rimColor="#ff4cb5"
                rim={0.14}
                clickBurst
                className="absolute inset-0"
              />

              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#080316]/95 via-transparent to-[#080316]/15" />

              <div className="pointer-events-none absolute left-5 top-5 rounded-2xl border border-cyan-100/20 bg-slate-950/65 px-3 py-2.5 backdrop-blur-md">
                <p className="text-[10px] font-bold tracking-[0.16em] text-cyan-200">LIVE ROOM SCAN</p>
                <p className="mt-1 text-sm font-semibold text-white">Reveal the hidden details</p>
              </div>

              <div className="pointer-events-none absolute bottom-5 left-5 right-5 rounded-2xl border border-white/15 bg-slate-950/75 p-4 backdrop-blur-md">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-cyan-100/70">Move your cursor across the room</p>
                    <p className="mt-1 text-lg font-semibold text-white">See what’s really there.</p>
                  </div>
                  <div className="rounded-xl bg-gradient-to-br from-cyan-300 to-pink-400 px-3 py-2 text-xs font-black text-slate-950">
                    SCAN
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="scroll-mt-8 border-t border-white/10 py-20">
          <div className="max-w-2xl">
            <p className="text-sm font-bold tracking-[0.16em] text-pink-300">HOW IT WORKS</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              From a phone video to a room you can trust.
            </h2>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {[
              ['01', 'Send one link', 'The landlord opens a private browser link and films the room.'],
              ['02', 'Build the space', 'SightSeen turns the walkthrough into a measured 3D room.'],
              ['03', 'Explore before signing', 'Walk around, measure, inspect, and see whether your furniture fits.'],
            ].map(([number, title, description]) => (
              <article
                key={number}
                className="group rounded-[1.75rem] border border-white/10 bg-white/[0.055] p-6 backdrop-blur-md transition duration-300 hover:-translate-y-1 hover:border-cyan-300/50 hover:bg-cyan-300/[0.09]"
              >
                <span className="text-sm font-black text-cyan-300">{number}</span>
                <h3 className="mt-8 text-xl font-semibold text-white">{title}</h3>
                <p className="mt-3 leading-6 text-slate-300">{description}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="trust" className="scroll-mt-8 pb-20">
          <div className="rounded-[2rem] border border-pink-300/20 bg-gradient-to-r from-cyan-400/10 via-violet-500/10 to-pink-500/15 p-7 shadow-2xl shadow-pink-950/20 backdrop-blur-xl sm:p-10">
            <Compass className="text-pink-300" size={28} />
            <h2 className="mt-5 text-3xl font-semibold tracking-tight text-white">
              A better way to decide from far away.
            </h2>
            <p className="mt-3 max-w-2xl leading-7 text-slate-200">
              SightSeen helps students see more than flattering listing photos:
              real dimensions, missed areas, possible damage, and space for the things they already own.
            </p>

            <div className="mt-7 flex flex-wrap gap-4">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-slate-950/35 px-4 py-2 text-sm text-cyan-100">
                <Ruler size={16} className="text-cyan-300" />
                Measure the room
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-slate-950/35 px-4 py-2 text-sm text-pink-100">
                <ShieldCheck size={16} className="text-pink-300" />
                Check red flags
              </span>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}