import { LIVE, STATIC, STATIC_ROOM } from './api'
import Film from './screens/Film'
import Home from './screens/Home'
import Room from './screens/Room'
import Workspace from './screens/Workspace'

/** Four pages, picked by the address: / says what it is, /workspace is where a student starts requests and keeps their
 * rooms, /room/{id} is one room, and /film/{token} is the landlord's. A static copy with no live pipeline behind it has
 * nothing to start, so it opens on its sample room. */
export default function App() {
  const path = window.location.pathname
  const film = path.match(/^\/film\/([\w-]+)\/?$/)
  const room = path.match(/^\/room\/([\w-]+)\/?$/)
  if (STATIC && !LIVE) return <Room id={room?.[1] ?? STATIC_ROOM ?? ''} />  // the static copy shows its one room
  if (film) return <Film token={film[1]} />
  if (room) return <Room id={room[1]} />
  if (/^\/(workspace|new)\/?$/.test(path)) return <Workspace />
  return <Home />
}
