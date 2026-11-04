import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, X } from 'lucide-react'
import { FAKE_ROOM, STATIC_ROOM } from '../api'
import BlurText from '../components/BlurText'
import ClickSpark from '../components/ClickSpark'
import CountUp from '../components/CountUp'
import DitherVeil from '../components/DitherVeil'
import LogoLoop from '../components/LogoLoop'
import Magnet from '../components/Magnet'
import Reveal from '../components/Reveal'
import RoomScanHero from '../components/RoomScanHero'
import RotatingText from '../components/RotatingText'
import SiteNav from '../components/SiteNav'
import Threads from '../components/Threads'
import TiltedCard from '../components/TiltedCard'

const sampleHref = STATIC_ROOM ? `/room/${STATIC_ROOM}` : null
const fakeHref = FAKE_ROOM ? `/room/${FAKE_ROOM}` : null
const REPO = 'https://github.com/sidharthnair7/SightSeen'

const navItems = [
  { label: 'Home', href: '#top' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Scam checks', href: '#scam-checks' },
  { label: 'Workspace', href: '/workspace' },
]

const MODELS = ['MapAnything', 'Muse Spark', 'Muse Image', 'SAM 3.1', 'SAM 3D Objects']

const STEPS = [
  {
    title: 'You send the landlord one link',
    text: 'They open it on their phone and film the room in the browser, holding up a one-time code as they start. No app, no account.',
  },
  {
    title: 'The video becomes the room',
    text: 'MapAnything rebuilds it in metres. Muse Spark and SAM 3.1 find each thing in it, and SAM 3D Objects rebuilds each one whole.',
  },
  {
    title: 'You check it before you sign',
    text: 'Walk it, measure it, fit your own desk in it, read the scam checks, and stand in it at real size on a Meta Quest.',
  },
]

const CHECKS = [
  {
    title: 'Filmed now, in that room',
    text: 'Muse Spark reads every number held up in the video without being told the code. An old tour can’t have today’s code in it.',
  },
  {
    title: 'The same room as the ad',
    text: 'If an ad photo’s details line up with the video in one 3D layout, it’s the same place. If not, Muse Spark names what can’t have changed.',
  },
  {
    title: 'Real depth',
    text: 'A “tour” panned across a photo has no depth: one flat warp explains every frame. A walk through a real room can’t look like that.',
  },
  {
    title: 'The size, and what’s missing',
    text: 'The floor against the ad’s number, the walls nobody filmed, and damage only when two different views agree on it.',
  },
]

// The fake room's real verdict, as its page shows it
const VERDICT: { mark: 'fail' | 'pass' | 'warn'; text: string }[] = [
  { mark: 'fail', text: 'The code 5645 never shows in the video' },
  { mark: 'pass', text: 'The video moves through real 3D space' },
  { mark: 'fail', text: 'An ad photo doesn’t look like this room' },
  { mark: 'pass', text: 'No damage found' },
  { mark: 'warn', text: '2 parts of the room never filmed' },
]

const NUMBERS = [
  { to: 31.2, decimals: 1, suffix: ' dB', text: 'How sharp the 3D is on frames it never trained on, with COLMAP lining up the cameras. MapAnything’s cameras alone: 21.4.' },
  { to: 229, decimals: 0, suffix: '', text: 'Details lining up between an ad’s photo and the video when it’s the same room. Photos of two other rooms: 11 and 12.' },
  { to: 23, decimals: 0, suffix: ' of 23', text: 'Flat frame pairs when a “tour” is a pan across one photo. A phone walk through a real room: 0 of 20.' },
  { to: 5, decimals: 0, suffix: ' min', text: 'Until a new room’s walls are ready to walk, on Modal’s GPUs. Its things follow in about 20.' },
]

function VerdictCard() {
  return (
    <div className="w-[min(25rem,calc(100vw-2.5rem))] rounded-[18px] bg-white p-5 text-left shadow-[0_1px_0_rgb(15_31_56/0.06),0_24px_60px_-20px_rgb(15_31_56/0.35)]">
      <p className="text-sm font-semibold text-ink">Before you sign</p>
      <div className="mt-3 rounded-xl bg-alert/10 px-4 py-3 ring-1 ring-alert/30">
        <p className="font-display text-2xl font-[760] text-alert">Don’t pay yet</p>
        <p className="mt-0.5 text-sm text-ink/80">At least one check failed. The details are below.</p>
      </div>
      <ul className="mt-4 space-y-2.5 text-[15px] text-ink">
        {VERDICT.map(({ mark, text }) => (
          <li key={text} className="flex items-start gap-2.5">
            {mark === 'fail' && <X size={18} className="mt-0.5 shrink-0 text-alert" strokeWidth={2.6} />}
            {mark === 'pass' && <Check size={18} className="mt-0.5 shrink-0 text-emerald-600" strokeWidth={2.6} />}
            {mark === 'warn' && <AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-600" strokeWidth={2.3} />}
            {text}
          </li>
        ))}
      </ul>
      <p className="mt-4 border-t border-hair pt-3 text-xs leading-5 text-steel">
        A stock video of a hotel room, passed off as the room in an ad. The verdict is the pipeline’s own.
      </p>
    </div>
  )
}

export default function Home() {
  const [section, setSection] = useState('#top')
  const hero = useRef<HTMLElement>(null)
  const [heroOn, setHeroOn] = useState(true)

  // The threads only draw while the hero is on screen: the rest of the page scrolls without a full-screen WebGL pass
  useEffect(() => {
    if (!hero.current) return
    const watch = new IntersectionObserver(([entry]) => setHeroOn(entry.isIntersecting))
    watch.observe(hero.current)
    return () => watch.disconnect()
  }, [])

  // A link to a section (/#scam-checks) lands on it: the page draws after the browser has looked for the anchor
  useEffect(() => {
    if (!window.location.hash) return
    const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)))
    const timer = setTimeout(() => target?.scrollIntoView({ behavior: 'instant' as ScrollBehavior }), 50)
    return () => clearTimeout(timer)
  }, [])

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
      <SiteNav items={navItems} activeHref={section} />
      <main id="top" className="relative overflow-x-clip text-ink">
        {/* The hero: Trini's dithered room on the right, threads that lean towards the cursor behind everything */}
        <section ref={hero} className="relative isolate">
          <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10 [mask-image:linear-gradient(to_bottom,black_55%,transparent)]">
            <Threads color="#0f1f38" accentColor="#ffd23f" amplitude={1.4} distance={0.3} enableMouseInteraction lineCount={46}
              thickness={0.7} softness={1.2} speed={0.45} brightness={1} opacity={0.42} angle={18} paused={!heroOn} />
          </div>
          <div className="mx-auto grid min-h-[100svh] max-w-7xl items-center gap-12 px-5 pb-16 pt-28 sm:px-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <div className="max-w-2xl">
              <h1 className="font-display text-[clamp(2.7rem,6.1vw,5.1rem)] font-[780] leading-[0.92] text-ink">
                <BlurText text="See the room before you sign." animateBy="words" direction="bottom" delay={90} trigger="load" />
              </h1>
              <Reveal delay={0.45}>
                <p className="mt-7 max-w-[34rem] text-lg leading-8 text-steel sm:text-xl sm:leading-9">
                  The landlord films it with one link. SightSeen rebuilds it in 3D at its real size, so you can{' '}
                  <RotatingText
                    texts={['walk through it', 'measure it', 'fit your desk in it', 'check it for a scam']}
                    highlight
                    highlightColor="#ffd23f"
                    effect="slide"
                    direction="up"
                    rotationInterval={2600}
                    staggerDuration={0.012}
                    mainClassName="inline-flex overflow-hidden rounded-[4px] px-1.5 align-baseline font-semibold text-ink"
                  />{' '}
                  before you pay anything.
                </p>
              </Reveal>
              <Reveal delay={0.6}>
                <div className="mt-10 flex flex-wrap items-center gap-x-6 gap-y-4">
                  <Magnet padding={70} magnetStrength={3} maxOffset={10}>
                    <ClickSpark sparkColor={['#0f1f38', '#ffd23f']} sparkCount={14} sparkRadius={52} sparkSize={10} sparkWidth={2} duration={550} shape="line" className="h-auto! w-auto!">
                      <a href="/workspace"
                        className="inline-flex min-h-13 items-center rounded-full bg-tape px-7 text-base font-semibold text-ink shadow-[inset_0_-2px_0_rgb(15_31_56/0.16)] transition-[scale] duration-150 ease-out active:scale-[0.96] motion-reduce:transition-none">
                        Start a request
                      </a>
                    </ClickSpark>
                  </Magnet>
                  {sampleHref && (
                    <a href={sampleHref}
                      className="inline-flex min-h-13 items-center rounded-full bg-white/70 px-6 text-base font-semibold text-ink ring-1 ring-ink/20 backdrop-blur transition-[box-shadow] duration-200 hover:ring-ink/60">
                      Walk a sample room
                    </a>
                  )}
                  {fakeHref && (
                    <a href={fakeHref}
                      className="text-base font-semibold text-ink underline decoration-alert decoration-2 underline-offset-[7px] transition-[text-decoration-thickness] hover:decoration-[3px]">
                      See it catch a fake
                    </a>
                  )}
                </div>
              </Reveal>
            </div>

            <Reveal delay={0.25} className="mx-auto w-full max-w-md">
              <div className="animate-float relative aspect-[4/5] overflow-hidden rounded-[26px] bg-white shadow-[0_30px_70px_-30px_rgb(15_31_56/0.55)] ring-1 ring-ink/10">
                <DitherVeil
                  src="/images/room-hero.jpg"
                  fit="cover"
                  pattern="floyd"
                  pixelSize={3}
                  levels={2}
                  palette="duotone"
                  inkColor="#0f1f38"
                  paperColor="#f4f6f9"
                  contrast={1.12}
                  brightness={0.12}
                  revealRadius={165}
                  softness={0.68}
                  linger={0.8}
                  rimColor="#ffd23f"
                  rim={0.18}
                  clickBurst
                  className="absolute inset-0"
                />
                <div className="pointer-events-none absolute left-4 top-4 rounded-xl bg-white/90 px-3 py-2 shadow-sm backdrop-blur-md">
                  <p className="text-xs text-steel">The room in the ad</p>
                  <p className="mt-0.5 text-sm font-semibold text-ink">Is it really like this?</p>
                </div>
                <div className="pointer-events-none absolute bottom-4 left-4 right-4 rounded-2xl bg-white/92 p-4 shadow-sm backdrop-blur-md">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs text-steel">
                        <span className="[@media(hover:none)]:hidden">Move your cursor across the room</span>
                        <span className="hidden [@media(hover:none)]:inline">Touch the room to see through</span>
                      </p>
                      <p className="mt-0.5 text-lg font-semibold text-ink">See what’s really there.</p>
                      <p className="mt-0.5 text-[11px] text-steel">An ad photo Muse Image drew for this demo</p>
                    </div>
                    {sampleHref && (
                      <a href={sampleHref} className="pointer-events-auto shrink-0 rounded-full bg-tape px-3.5 py-2 text-sm font-semibold text-ink">
                        Step inside
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </section>

        <section className="border-y border-hair bg-paper py-7" aria-label="Meta’s models in SightSeen">
          <div className="mx-auto flex max-w-7xl items-center gap-8 px-5 sm:px-8">
            <p className="shrink-0 text-sm text-steel">Built on Meta’s models</p>
            <LogoLoop
              logos={MODELS.map(name => ({ node: <span className="font-display text-xl font-[650] whitespace-nowrap text-ink sm:text-2xl">{name}</span>, title: name }))}
              speed={38}
              gap={64}
              logoHeight={32}
              fadeOut
              fadeOutColor="#f4f6f9"
              pauseOnHover
              ariaLabel="Meta’s models in SightSeen"
              className="min-w-0 flex-1"
            />
          </div>
        </section>

        {/* The room itself, live: the pipeline's own output */}
        <section id="live-room" className="mx-auto grid max-w-7xl items-center gap-10 px-5 py-24 sm:px-8 sm:py-28 lg:grid-cols-[minmax(0,0.75fr)_minmax(0,1.25fr)]">
          <Reveal>
            <h2 className="font-display text-[clamp(2.1rem,4.2vw,3.4rem)] font-[740] leading-[0.98]">One video in. The room, at its real size.</h2>
            <p className="mt-6 max-w-md text-lg leading-8 text-steel">
              This is a hotel room SightSeen rebuilt from a stock video, live in your browser. Move your cursor over it, or drag to
              turn it. The tape marks are its measured floor: 5.15 by 4.11 m, from two wall plates, so rough until someone checks one
              length with a tape.
            </p>
            {sampleHref && (
              <a href={sampleHref} className="mt-8 inline-flex min-h-12 items-center rounded-full bg-ink px-6 text-base font-semibold text-white transition-[scale] duration-150 active:scale-[0.96]">
                Walk a real student’s room
              </a>
            )}
          </Reveal>
          <Reveal delay={0.15}>
            <RoomScanHero src="/hero/room-scan.bin" className="aspect-[5/4] w-full lg:aspect-auto lg:h-[min(70vh,40rem)]" />
          </Reveal>
        </section>

        <section id="how-it-works" className="border-t border-hair">
          <div className="mx-auto max-w-7xl px-5 py-24 sm:px-8 sm:py-28">
            <Reveal>
              <h2 className="font-display max-w-3xl text-[clamp(2.1rem,4.2vw,3.4rem)] font-[740] leading-[0.98]">
                One link in. A room you can check out.
              </h2>
            </Reveal>
            <ol className="relative mt-14 grid gap-10 md:grid-cols-3 md:gap-10">
              <span aria-hidden="true" className="absolute inset-x-0 top-0 hidden h-px bg-ink/25 md:block" />
              {STEPS.map((step, index) => (
                <li key={step.title} className="relative md:pt-9">
                  <span aria-hidden="true" className="absolute left-0 top-[-7px] hidden h-[15px] w-px bg-ink md:block" />
                  <Reveal delay={0.1 + index * 0.12}>
                    <span className="inline-block rounded-[3px] bg-tape px-1.5 text-sm font-semibold tabular-nums text-ink">{index + 1}</span>
                    <h3 className="mt-3 text-xl font-semibold leading-snug">{step.title}</h3>
                    <p className="mt-2 max-w-sm leading-7 text-steel">{step.text}</p>
                  </Reveal>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section id="scam-checks" className="border-t border-hair bg-white/60">
          <div className="mx-auto grid max-w-7xl items-center gap-14 px-5 py-24 sm:px-8 sm:py-28 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
            <div>
              <Reveal>
                <h2 className="font-display text-[clamp(2.1rem,4.2vw,3.4rem)] font-[740] leading-[0.98]">
                  A scam tour shows a real room the scammer can’t get into.
                </h2>
                <p className="mt-6 max-w-xl text-lg leading-8 text-steel">
                  So SightSeen asks what stolen photos and a stolen tour can’t answer. The AI reads and compares. Plain rules decide.
                </p>
              </Reveal>
              <dl className="mt-10 divide-y divide-hair border-y border-hair">
                {CHECKS.map((check, index) => (
                  <Reveal key={check.title} delay={0.08 * index}>
                    <div className="grid gap-1 py-5 sm:grid-cols-[12.5rem_minmax(0,1fr)] sm:gap-6">
                      <dt className="font-semibold">{check.title}</dt>
                      <dd className="leading-7 text-steel">{check.text}</dd>
                    </div>
                  </Reveal>
                ))}
              </dl>
            </div>
            <Reveal delay={0.15} className="flex flex-col items-center gap-5">
              <div className="hidden lg:block">
                <TiltedCard containerWidth="25rem" containerHeight="24.25rem" imageWidth="25rem" imageHeight="24.25rem"
                  rotateAmplitude={9} scaleOnHover={1.03} glare={0.14} shadow={0} radius={18} perspective={900}>
                  <VerdictCard />
                </TiltedCard>
              </div>
              <div className="lg:hidden"><VerdictCard /></div>
              {fakeHref && (
                <a href={fakeHref} className="text-sm font-semibold underline decoration-ink/30 underline-offset-4 hover:decoration-ink">
                  Open this room and see the evidence
                </a>
              )}
            </Reveal>
          </div>
        </section>

        <section id="numbers" className="border-y border-hair" aria-label="Measured on real runs">
          <div className="mx-auto grid max-w-7xl gap-x-10 gap-y-12 px-5 py-20 sm:grid-cols-2 sm:px-8 lg:grid-cols-4">
            {NUMBERS.map((item, index) => (
              <Reveal key={item.text} delay={index * 0.1}>
                <p className="font-display text-5xl font-[760] tabular-nums leading-none">
                  <CountUp to={item.to} decimals={item.decimals} duration={1.6} suffix={item.suffix} trigger="view" />
                </p>
                <p className="mt-4 max-w-[17rem] leading-7 text-steel">{item.text}</p>
              </Reveal>
            ))}
          </div>
        </section>

        <section className="mx-auto grid max-w-7xl gap-6 px-5 py-20 sm:px-8 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
          <Reveal><h2 className="font-display text-3xl font-[740] leading-none sm:text-4xl">What it can’t do</h2></Reveal>
          <Reveal delay={0.1}>
            <p className="max-w-2xl text-lg leading-8 text-steel">
              It shows whoever filmed was standing in that room, not that they own it, so ask for the lease and an ID before you pay
              anything. And a new room’s size is rough until someone measures one length with a tape.
            </p>
          </Reveal>
        </section>

        <footer className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 border-t border-hair px-5 py-8 text-sm text-steel sm:px-8">
          <p>Sidharth Nair and Trinidad Laguardia</p>
          <div className="flex gap-6">
            <a href="/workspace" className="font-semibold text-ink hover:underline">Workspace</a>
            <a href={REPO} className="font-semibold text-ink hover:underline">The code on GitHub</a>
          </div>
        </footer>
      </main>
    </>
  )
}
