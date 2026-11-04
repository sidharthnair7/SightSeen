import { useEffect, useState } from 'react'
import { ScanLine } from 'lucide-react'
import PillNav, { type PillNavItem } from './PillNav'

const SHRINK_AT = 48 // pixels scrolled before the bar shrinks
const GROW_AT = 12 // and back: apart, so a page resting between them doesn't flicker

/**
 * The floating nav on every page. At the top of the page it's a wide, open bar; once you scroll, it shrinks into a small
 * glass pill floating over the page, with the name folded away, so it stays in reach and out of the way.
 */
export default function SiteNav({ items, activeHref, compact = false }: { items: PillNavItem[]; activeHref?: string; compact?: boolean }) {
  const [scrolled, setScrolled] = useState(false)
  const small = compact || scrolled

  useEffect(() => {
    const onScroll = () => setScrolled(was => (was ? window.scrollY > GROW_AT : window.scrollY > SHRINK_AT))
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="pointer-events-none fixed inset-x-0 top-3 z-50 flex justify-center px-3 sm:top-4">
      <header
        data-small={small ? '' : undefined}
        style={{ maxWidth: small ? '37rem' : '80rem' }}
        className={`pointer-events-auto flex w-full items-center justify-between gap-3 rounded-full border transition-[max-width,padding,background-color,border-color,box-shadow] duration-300 ease-out motion-reduce:transition-none ${
          small
            ? 'border-slate-900/10 bg-white/80 py-1 pl-1.5 pr-1 shadow-xl shadow-slate-900/10 backdrop-blur-xl'
            : 'border-transparent bg-transparent py-1.5 pl-1 pr-1'
        }`}
      >
        <a href="/" aria-label="SightSeen, home" className="flex shrink-0 items-center gap-2 rounded-full font-bold tracking-tight text-slate-900 outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500">
          <span className={`grid place-items-center rounded-full bg-gradient-to-br from-cyan-300 via-sky-400 to-pink-500 text-slate-950 shadow-md shadow-sky-500/25 transition-[width,height] duration-300 ease-out motion-reduce:transition-none ${small ? 'h-8 w-8' : 'h-10 w-10'}`}>
            <ScanLine size={small ? 17 : 21} strokeWidth={2.7} />
          </span>
          <span className={`overflow-hidden whitespace-nowrap text-lg transition-[max-width,opacity] duration-300 ease-out motion-reduce:transition-none ${small ? 'max-w-0 opacity-0' : 'max-w-32 opacity-100'}`}>
            SightSeen
          </span>
        </a>

        <PillNav
          items={items}
          activeHref={activeHref}
          defaultActiveIndex={activeHref === undefined ? 0 : undefined}
          theme="light"
          layout="joined"
          size={small ? 'sm' : 'md'}
          surface={small ? 'none' : 'glass'}
          itemStyle="plain"
          hoverEffect="glide"
          indicator="pill"
          highlightColor="linear-gradient(135deg, #22d3ee 0%, #f472b6 100%)"
          highlightTextColor="#04101b"
          textColor="#0f172a"
          collapse="auto"
          intro={false}
          className="max-w-[calc(100vw-5.5rem)] data-[collapsed]:w-auto! data-[collapsed]:justify-end"
        />
      </header>
    </div>
  )
}
