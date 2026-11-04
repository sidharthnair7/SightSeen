import { useEffect, useState } from 'react'
import { CheckCircle2, KeyRound } from 'lucide-react'
import { API_BASE, api, uploadVideo } from '../api'
import Recorder from '../components/Recorder'
import SiteNav from '../components/SiteNav'
import VideoPicker from '../components/VideoPicker'
import { filmingSteps } from '../instructions'
import type { FilmView } from '../types'

/** The landlord's page, opened from the student's link: how to film, then send. No account, nothing to install. */
export default function Film({ token }: { token: string }) {
  const [view, setView] = useState<FilmView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [sent, setSent] = useState(false)

  useEffect(() => {
    api.film(token).then(setView).catch(e => setError((e as Error).message))
  }, [token])

  // live: recorded in this page (the student's issues view says so), not a file picked from the phone
  async function send(file: File, live = false) {
    setError(null)
    setProgress(0)
    try {
      await uploadVideo(`${API_BASE}/api/film/${token}/video${live ? '?live=true' : ''}`, file, setProgress, () =>
        api.film(token).then(film => film.building === true))
      setSent(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setProgress(null)
    }
  }

  if (!view) {
    return (
      <>
        <SiteNav activeHref="" items={[{ label: 'What is SightSeen?', href: '/' }]} />
        <main className="mx-auto max-w-xl px-4 pb-12 pt-24">
          <p className={error ? 'text-red-700' : 'text-slate-600'}>{error ?? 'Loading...'}</p>
        </main>
      </>
    )
  }

  return (
    <>
    <SiteNav activeHref="" items={[{ label: 'What is SightSeen?', href: '/' }]} />
    <main className="mx-auto max-w-xl px-4 pb-12 pt-24">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-950">Film the room: {view.label}</h1>
      <p className="mt-2 text-slate-600">
        A student asked to see this room before signing. Your video becomes a 3D room they can look around and
        measure. There's no account and nothing to install.
      </p>

      {view.code && (
        <section className="mt-6 flex items-start gap-3 rounded-2xl border border-tape bg-white p-4 shadow-sm">
          <KeyRound className="mt-0.5 shrink-0 text-ink" size={20} />
          <div>
            <p className="font-medium text-slate-950">Your one-time code: <span className="text-2xl font-bold tabular-nums tracking-wider">{view.code}</span></p>
            <p className="mt-1 text-sm text-slate-600">
              Write it on a piece of paper, in big numbers. When you start filming, hold it up to the camera for about 3
              seconds, with the room behind it. That's how the student knows this video was filmed now, in this room.
            </p>
          </div>
        </section>
      )}

      <h2 className="mt-8 font-medium text-slate-950">How to film it</h2>
      <ol className="mt-3 list-decimal space-y-2 pl-5 text-slate-700">
        {filmingSteps.map(step => <li key={step}>{step}</li>)}
      </ol>

      <div className="mt-8">
        {sent ? (
          <div className="flex items-start gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-emerald-900">
            <CheckCircle2 className="mt-0.5 shrink-0" size={20} />
            <p>Thanks, it's sent. The student will see the room soon. If they ask for another take, use this same link.</p>
          </div>
        ) : (
          <>
            <Recorder code={view.code} sending={progress} onSend={file => send(file, true)} />
            <div className="mt-6 border-t border-slate-200 pt-4">
              <p className="mb-2 text-sm text-slate-600">Already filmed it? Send that video instead.</p>
              <VideoPicker label="Choose a video" progress={progress} onPick={send} />
            </div>
            {view.videoReceived && (
              <p className="mt-3 text-sm text-slate-600">A video was already sent. Sending another one replaces it.</p>
            )}
          </>
        )}
        {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
      </div>
    </main>
    </>
  )
}
