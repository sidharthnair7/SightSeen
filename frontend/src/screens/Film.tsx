import { useEffect, useState } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { api, uploadVideo } from '../api'
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

  async function send(file: File) {
    setError(null)
    setProgress(0)
    try {
      await uploadVideo(`/api/film/${token}/video`, file, setProgress)
      setSent(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setProgress(null)
    }
  }

  if (!view) {
    return (
      <main className="mx-auto max-w-xl px-4 py-12">
        <p className={error ? 'text-red-700' : 'text-slate-600'}>{error ?? 'Loading...'}</p>
      </main>
    )
  }

  return (
    <main className="mx-auto max-w-xl px-4 py-12">
      <p className="text-sm font-medium text-teal-700">SightSeen</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Film the room: {view.label}</h1>
      <p className="mt-2 text-slate-600">
        A student asked to see this room before signing. Your video becomes a 3D room they can look around and
        measure. There's no account and nothing to install.
      </p>

      <h2 className="mt-8 font-medium">How to film it</h2>
      <ol className="mt-3 list-decimal space-y-2 pl-5 text-slate-700">
        {filmingSteps.map(step => <li key={step}>{step}</li>)}
      </ol>

      <div className="mt-8">
        {sent ? (
          <div className="flex items-start gap-3 rounded-lg bg-teal-50 p-4 text-teal-900">
            <CheckCircle2 className="mt-0.5 shrink-0" size={20} />
            <p>Thanks, it's sent. The student will see the room soon. If they ask for another take, use this same link.</p>
          </div>
        ) : (
          <>
            <VideoPicker label="Record or choose the video" progress={progress} onPick={send} />
            {view.videoReceived && (
              <p className="mt-3 text-sm text-slate-600">A video was already sent. Sending another one replaces it.</p>
            )}
          </>
        )}
        {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
      </div>
    </main>
  )
}
