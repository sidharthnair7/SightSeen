import Film from './screens/Film'
import Home from './screens/Home'
import Room from './screens/Room'

/** Three pages, picked by the address: /film/{token} for the landlord, /room/{id} for the student, / to start. */
export default function App() {
  const path = window.location.pathname
  const film = path.match(/^\/film\/([\w-]+)\/?$/)
  const room = path.match(/^\/room\/([\w-]+)\/?$/)
  if (film) return <Film token={film[1]} />
  if (room) return <Room id={room[1]} />
  return <Home />
}
