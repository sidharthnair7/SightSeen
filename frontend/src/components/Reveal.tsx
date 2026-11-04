import type { ReactNode } from 'react'
import { motion, useReducedMotion } from 'framer-motion'

/** Eases a block into place the first time it scrolls into view: up a little and in. Opacity and movement only, which
 *  the browser animates without repainting (a blur filter made scrolling stutter). Still for reduced motion. */
export default function Reveal({ children, delay = 0, className = '' }: { children: ReactNode; delay?: number; className?: string }) {
  const still = useReducedMotion()
  return (
    <motion.div
      className={className}
      initial={still ? false : { opacity: 0, y: 22 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}
