import { Fragment, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { AnimatePresence, motion, useMotionValue, useReducedMotion, useSpring, useTransform, type MotionValue } from 'framer-motion'

export type DockItem = {
  icon: ReactNode
  label: string
  onClick?: () => void
  /** A line before this one */
  separator?: boolean
  active?: boolean
  disabled?: boolean
}

const SPRING = { stiffness: 400, damping: 25, mass: 0.4 }
const GROW = 1.6 // how much an icon grows right under the pointer
const REACH = 100 // pixels: how far along the dock the growing reaches
const SIZE = 40 // pixels: an icon at rest

const BUTTON = 'flex h-full w-full items-center justify-center rounded-xl text-[#fdf2ed]/75 outline-none transition-colors duration-150 hover:bg-[#fdf2ed]/[0.1] hover:text-[#fdf2ed] focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-[52%] data-[active=true]:bg-cyan-300 data-[active=true]:text-slate-950'

function Icon({ item, mouseX, onHover, still }: {
  item: DockItem
  mouseX: MotionValue<number>
  onHover: (element: HTMLElement | null, label: string | null) => void
  still: boolean
}) {
  const wrapper = useRef<HTMLDivElement>(null)
  const distance = useTransform(mouseX, value => {
    const box = wrapper.current?.getBoundingClientRect()
    return box ? Math.abs(value - (box.left + box.width / 2)) : REACH * 100
  })
  const raw = useTransform(distance, value => SIZE * (still ? 1 : (GROW - 1) * Math.exp(-(value * value) / (2 * REACH * REACH)) + 1))
  const size = useSpring(raw, SPRING)
  return (
    <motion.div ref={wrapper} className="relative flex items-end justify-center" style={{ width: size, height: SIZE }}>
      <motion.div className="absolute bottom-0" style={{ width: size, height: size }}>
        <button
          type="button"
          onClick={item.onClick}
          disabled={item.disabled}
          aria-label={item.label}
          aria-pressed={item.active}
          data-active={item.active ? 'true' : undefined}
          onMouseEnter={event => onHover(event.currentTarget, item.label)}
          onMouseLeave={() => onHover(null, null)}
          onFocus={event => onHover(event.currentTarget, item.label)}
          onBlur={() => onHover(null, null)}
          className={BUTTON}
        >
          {item.icon}
        </button>
      </motion.div>
    </motion.div>
  )
}

/**
 * The room's tools, as a glass dock: icons that grow under the pointer, with a label above the one you're on. Adapted from
 * the dock in my earlier project, Remembrance. On a touch screen there's no pointer to follow, so each icon wears its label
 * and the row scrolls sideways.
 */
export default function Dock({ items }: { items: DockItem[] }) {
  const mouseX = useMotionValue(Infinity)
  const dock = useRef<HTMLDivElement>(null)
  const still = useReducedMotion() ?? false
  const [touch, setTouch] = useState(() => typeof matchMedia !== 'undefined' && matchMedia('(hover: none)').matches)
  const [tip, setTip] = useState<{ label: string; left: number; bottom: number } | null>(null)
  const held = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const query = matchMedia('(hover: none)')
    const change = () => setTouch(query.matches)
    query.addEventListener('change', change)
    return () => query.removeEventListener('change', change)
  }, [])

  // The label follows its icon as it grows
  useEffect(() => {
    if (!tip) return
    let frame = 0
    const follow = () => {
      const icon = held.current?.getBoundingClientRect()
      const box = dock.current?.getBoundingClientRect()
      if (icon && box) {
        const left = icon.left - box.left + icon.width / 2
        const bottom = box.bottom - icon.top
        setTip(now => (now && (now.left !== left || now.bottom !== bottom) ? { ...now, left, bottom } : now))
      }
      frame = requestAnimationFrame(follow)
    }
    frame = requestAnimationFrame(follow)
    return () => cancelAnimationFrame(frame)
  }, [tip?.label])

  const hover = useCallback((element: HTMLElement | null, label: string | null) => {
    held.current = element
    if (!element || !label) return setTip(null)
    const icon = element.getBoundingClientRect()
    const box = dock.current?.getBoundingClientRect()
    setTip({ label, left: box ? icon.left - box.left + icon.width / 2 : 0, bottom: box ? box.bottom - icon.top : 0 })
  }, [])

  if (touch) {
    return (
      <div className="glass-dark flex max-w-[calc(100vw-1.5rem)] gap-0.5 overflow-x-auto overscroll-contain rounded-2xl p-1.5">
        {items.map(item => (
          <button key={item.label} type="button" onClick={item.onClick} disabled={item.disabled} aria-pressed={item.active}
            data-active={item.active ? 'true' : undefined}
            className="flex w-[3.9rem] shrink-0 flex-col items-center gap-1 rounded-xl px-1 py-1.5 text-[9.5px] leading-tight text-[#fdf2ed]/80 outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:opacity-40 data-[active=true]:bg-cyan-300 data-[active=true]:text-slate-950 [&_svg]:size-[18px]">
            {item.icon}
            <span className="text-center">{item.label}</span>
          </button>
        ))}
      </div>
    )
  }

  return (
    <div ref={dock} className="glass-dark relative flex items-end gap-1 overflow-visible rounded-2xl p-2 transition-shadow duration-200 hover:shadow-[0_0_0_1px_rgba(253,242,237,0.06),0_8px_24px_rgba(0,0,0,0.35)]"
      onMouseMove={event => mouseX.set(event.clientX)} onMouseLeave={() => mouseX.set(Infinity)}>
      {items.map(item => (
        <Fragment key={item.label}>
          {item.separator && (
            <div className="mx-1 flex items-center self-stretch">
              <div className="h-6 w-px bg-[#fdf2ed]/15" />
            </div>
          )}
          <Icon item={item} mouseX={mouseX} onHover={hover} still={still} />
        </Fragment>
      ))}
      <AnimatePresence>
        {tip && (
          <motion.div key="tip" className="pointer-events-none absolute z-50" style={{ left: tip.left, bottom: tip.bottom + 8, x: '-50%' }}
            initial={{ opacity: 0, y: still ? 0 : 6, scale: still ? 1 : 0.94 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: still ? 0 : 6, scale: still ? 1 : 0.94 }} transition={{ duration: 0.13, ease: 'easeOut' }}>
            <span className="block whitespace-nowrap rounded-lg border border-[#fdf2ed]/15 bg-[#12110f] px-2 py-1 text-xs font-medium text-[#fdf2ed] shadow-sm">
              {tip.label}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
