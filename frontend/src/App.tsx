import { LIVE, STATIC, STATIC_ROOM } from './api'
import Film from './screens/Film'
import Home from './screens/Home'
import Room from './screens/Room'

/** Three pages, picked by the address: /film/{token} for the landlord, /room/{id} for the student, / to start. The
 * hosted copy opens on its sample room instead, and starts at /new when it has a live pipeline. */
export default function App() {
  const path = window.location.pathname
  const film = path.match(/^\/film\/([\w-]+)\/?$/)
  const room = path.match(/^\/room\/([\w-]+)\/?$/)
  if (STATIC && !LIVE) return <Room id={room?.[1] ?? STATIC_ROOM ?? ''} />  // the static copy shows its one room
  if (film) return <Film token={film[1]} />
  if (room) return <Room id={room[1]} />
  if (STATIC && path !== '/new') return <Room id={STATIC_ROOM ?? ''} />
  return <Home />
}
