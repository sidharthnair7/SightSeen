import { useEffect, useState } from 'react'
import {
  ArrowRight,
  Box,
  FileWarning,
  Glasses,
  KeyRound,
  Layers,
  ShieldCheck,
  ScanSearch,
  Sparkles,
} from 'lucide-react'
import { FAKE_ROOM, STATIC_ROOM } from '../api'
import ClickSpark from '../components/ClickSpark'
import DitherVeil from '../components/DitherVeil'
import GlBackground from '../components/GlBackground'
import ShinyText from '../components/ShinyText'
import SiteNav from '../components/SiteNav'

const sampleHref = STATIC_ROOM ? `/room/${STATIC_ROOM}` : null
const fakeHref = FAKE_ROOM ? `/room/${FAKE_ROOM}` : null
const REPO = 'https://github.com/sidharthnair7/SightSeen'

const navItems = [
  { label: 'Home', href: '#top' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Scam checks', href: '#scam-checks' },
  { label: 'Workspace', href: '/workspace' },
]

const STEPS = [
  ['01', 'Send one link', 'The landlord opens a private link and films the room in the browser, with no app and no account, holding up a one-time code as they start.'],
  ['02', 'Build the space', 'MapAnything rebuilds the room in metres from the video. Muse Spark and SAM 3.1 find each thing in it, and SAM 3D Objects rebuilds it whole.'],
  ['03', 'Explore before signing', 'Walk it, measure it, fit your own things by taking a picture of them, then step inside at real size on a Meta Quest.'],
]

const CHECKS = [
  {
    icon: KeyRound,
    title: 'Filmed now, in that room',
    text: 'Muse Spark reads every number held up in the video without being told the code. Plain code compares them. An old tour can’t have today’s code in it.',
  },
  {
    icon: ScanSearch,
    title: 'The room in the ad',
    text: 'Add the ad’s photos. If their details line up with the video in 3D, it’s the same place. If not, Muse Spark names what can’t have changed: windows, doors, the floor.',
  },
  {
    icon: Layers,
    title: 'Real depth',
    text: 'A “tour” panned across a photo has no depth: one flat warp explains every frame. A walk through a real room can’t look like that.',
  },
  {
    icon: FileWarning,
    title: 'The size, and what’s missing',
    text: 'The floor measured against the ad’s number, the walls nobody filmed, and damage only when two different views agree on it.',
  },
]

export default function Home() {
  const [section, setSection] = useState('#top')

  // The nav follows the page: the section nearest the top of the view is the lit one
  useEffect(() => {
    const parts = ['how-it-works', 'scam-checks'].map(id => document.getElementById(id)).filter(Boolean) as HTMLElement[]
    const update = () => {
      const passed = parts.filter(part => part.getBoundingClientRect().top < window.innerHeight * 0.4)
      setSection(passed.length ? `#${passed[passed.length - 1].id}` : '#top')
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    return () => window.removeEventListener('scroll', update)
  }, [])

  return (
    <>
      <GlBackground />
      <SiteNav items={navItems} activeHref={section} />
      <main id="top" className="relative min-h-screen overflow-hidden px-5 pb-5 pt-20 sm:px-8 sm:pt-24">
        <div className="relative mx-auto max-w-7xl">
          <section className="grid min-h-[calc(100vh-7rem)] items-center gap-12 pb-16 pt-4 lg:grid-cols-[1.05fr_0.95fr] lg:pb-20">
            <div className="max-w-2xl">
              <div className="liquid-glass animate-rise-in inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-bold tracking-wide text-sky-800">
                <Sparkles size={14} />
                RENT WITH A CLEARER VIEW
              </div>

              <h1 className="animate-rise-in delay-1 mt-6 text-5xl font-semibold leading-[1.02] tracking-[-0.05em] text-slate-950 sm:text-6xl lg:text-7xl">
                Don’t just view
                <span className="block w-fit bg-gradient-to-r from-cyan-600 via-sky-600 to-pink-500 bg-clip-text pb-[0.12em] text-transparent">
                  the listing.
                </span>
                <span className="-mt-[0.12em] block">Step inside it.</span>
              </h1>

              <div className="animate-rise-in delay-2 mt-6 max-w-xl text-base leading-7 sm:text-lg">
                <ShinyText color="#475569" shineColor="#0f172a" speed={3} shineWidth={55} glow={0} trigger="loop">
                  Turn a landlord’s phone video into a 3D room at real size. Walk it, measure it, fit your own
                  things in it, and check it isn’t a scam before you sign.
                </ShinyText>
              </div>

              <div className="animate-rise-in delay-2 mt-9 flex flex-wrap items-center gap-3">
                <ClickSpark sparkColor={['#22d3ee', '#f472b6', '#0ea5e9']} sparkCount={18} sparkRadius={68} sparkSize={12} sparkWidth={2} duration={650} shape="star" glow={0.4} ring className="h-auto! w-auto!">
                  <a href="/workspace" className="inline-flex min-h-14 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-300 via-sky-400 to-pink-400 px-6 text-sm font-black whitespace-nowrap text-slate-950 shadow-lg shadow-sky-500/25 transition-[scale,box-shadow] duration-200 hover:scale-[1.03] hover:shadow-pink-500/25 active:scale-[0.96] motion-reduce:transition-none motion-reduce:hover:scale-100 motion-reduce:active:scale-100">
                    Open the workspace <ArrowRight size={18} />
                  </a>
                </ClickSpark>
                {sampleHref && (
                  <a href={sampleHref} className="liquid-glass inline-flex min-h-14 items-center gap-2 rounded-2xl px-5 text-sm font-semibold text-slate-900 transition-[background-color] hover:bg-white/80">
                    <Box size={17} className="text-sky-600" /> Walk a sample room
                  </a>
                )}
                {fakeHref && (
                  <a href={fakeHref} className="liquid-glass inline-flex min-h-14 items-center gap-2 rounded-2xl px-5 text-sm font-semibold text-slate-900 transition-[background-color] hover:bg-white/80">
                    <FileWarning size={17} className="text-pink-600" /> See it catch a fake
                  </a>
                )}
              </div>

              <div className="mt-8 flex flex-wrap gap-x-6 gap-y-3 text-sm text-slate-600">
                <span className="inline-flex items-center gap-2">
                  <ShieldCheck size={16} className="text-sky-600" />
                  No account for landlords
                </span>
                <span className="inline-flex items-center gap-2">
                  <Box size={16} className="text-pink-600" />
                  Real-scale 3D space
                </span>
                <span className="inline-flex items-center gap-2">
                  <Glasses size={16} className="text-violet-600" />
                  Step inside on a Meta Quest
                </span>
              </div>
            </div>

            <div className="animate-float relative mx-auto w-full max-w-xl">
              <div className="animate-pulse-glow absolute -inset-7 rounded-[3rem] bg-gradient-to-br from-cyan-300/40 via-violet-300/20 to-pink-300/40 blur-3xl" />

              <div className="relative aspect-[4/5] overflow-hidden rounded-[2.2rem] border border-slate-900/10 bg-white shadow-2xl shadow-slate-900/15">
                <DitherVeil
                  src="/images/room-hero.jpg"
                  fit="cover"
                  pattern="floyd"
                  pixelSize={3}
                  levels={2}
                  palette="duotone"
                  inkColor="#0b1f33"
                  paperColor="#f1fbff"
                  contrast={1.18}
                  brightness={0.04}
                  revealRadius={165}
                  softness={0.68}
                  linger={0.8}
                  rimColor="#f472b6"
                  rim={0.14}
                  clickBurst
                  className="absolute inset-0"
                />

                <div className="pointer-events-none absolute left-5 top-5 rounded-2xl border border-slate-900/10 bg-white/85 px-3 py-2.5 shadow-sm backdrop-blur-md">
                  <p className="text-[10px] font-bold tracking-[0.16em] text-sky-700">THE ROOM IN THE AD</p>
                  <p className="mt-1 text-sm font-semibold text-slate-900">Is it really like this?</p>
                </div>

                <div className="pointer-events-none absolute bottom-5 left-5 right-5 rounded-2xl border border-slate-900/10 bg-white/90 p-4 shadow-sm backdrop-blur-md">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs text-slate-600">
                        <span className="[@media(hover:none)]:hidden">Move your cursor across the room</span>
                        <span className="hidden [@media(hover:none)]:inline">Touch the room to see through</span>
                      </p>
                      <p className="mt-1 text-lg font-semibold text-slate-950">See what’s really there.</p>
                      <p className="mt-0.5 text-[10px] text-slate-500">Picture drawn by Muse Image</p>
                    </div>
                    {sampleHref && (
                      <a href={sampleHref} className="pointer-events-auto shrink-0 rounded-xl bg-gradient-to-br from-cyan-300 to-pink-400 px-3 py-2 text-xs font-black text-slate-950">
                        STEP INSIDE
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section id="how-it-works" className="border-t border-slate-900/10 py-20">
            <div className="max-w-2xl">
              <p className="text-sm font-bold tracking-[0.16em] text-pink-600">HOW IT WORKS</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                From a phone video to a room you can check.
              </h2>
            </div>

            <div className="mt-10 grid gap-4 md:grid-cols-3">
              {STEPS.map(([number, title, description]) => (
                <article
                  key={number}
                  className="liquid-glass rounded-[1.75rem] p-6 transition-[translate,background-color] duration-300 hover:-translate-y-1 hover:bg-white/75 motion-reduce:transition-none motion-reduce:hover:translate-y-0"
                >
                  <span className="text-sm font-black text-sky-600">{number}</span>
                  <h3 className="mt-8 text-xl font-semibold text-slate-950">{title}</h3>
                  <p className="mt-3 leading-6 text-slate-600">{description}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="scam-checks" className="pb-16">
            <div className="liquid-glass rounded-[2rem] p-7 sm:p-10">
              <p className="text-sm font-bold tracking-[0.16em] text-pink-600">SCAM CHECKS</p>
              <h2 className="mt-3 max-w-3xl text-3xl font-semibold tracking-tight text-slate-950">
                A rental scam shows you a real room the scammer can’t get into.
              </h2>
              <p className="mt-3 max-w-2xl leading-7 text-slate-700">
                So SightSeen asks what stolen photos and a stolen tour can’t answer. The AI reads and compares; plain rules decide.
              </p>

              <div className="mt-8 grid gap-4 md:grid-cols-2">
                {CHECKS.map(({ icon: Icon, title, text }) => (
                  <article key={title} className="rounded-2xl border border-white/70 bg-white/55 p-5">
                    <Icon size={20} className="text-sky-600" />
                    <h3 className="mt-3 font-semibold text-slate-950">{title}</h3>
                    <p className="mt-1.5 text-sm leading-6 text-slate-600">{text}</p>
                  </article>
                ))}
              </div>

              <p className="mt-6 max-w-2xl text-sm leading-6 text-slate-700">
                What this can’t do: it shows whoever filmed was standing in that room, not that they own it. Ask for the
                lease and an ID before you pay anything.
              </p>
            </div>
          </section>

          <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-900/10 pb-4 pt-6 text-sm text-slate-600">
            <p>Built for Meta’s Global AI Developer Hackathon with MapAnything, Muse Spark, Muse Image, SAM 3.1 and SAM 3D Objects.</p>
            <a href={REPO} className="font-semibold text-slate-900 underline decoration-slate-300 underline-offset-4 hover:decoration-sky-500">Code on GitHub</a>
          </footer>
        </div>
      </main>
    </>
  )
}
