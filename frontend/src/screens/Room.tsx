import { useCallback, useEffect, useState } from 'react'
import { Loader2, RotateCcw } from 'lucide-react'
import { API_BASE, FAKE_ROOM, LIVE, STATIC, STATIC_ROOM, api, isSample, rememberRoom, uploadVideo } from '../api'
import CopyLink from '../components/CopyLink'
import GlBackground from '../components/GlBackground'
import RoomViewer from '../components/RoomViewer'
import SiteNav from '../components/SiteNav'
import VideoPicker from '../components/VideoPicker'
import type { RequestStatus, RoomRequest } from '../types'

const STATUS_TEXT: Record<RequestStatus, string> = {
  WAITING_FOR_VIDEO: "Waiting for the landlord's video",
  VIDEO_RECEIVED: 'Video in',
  RECONSTRUCTING: 'Building the 3D room',
  READY: 'Ready',
  FAILED: 'Stopped',
}

const MODELS: Record<string, string> = { apache: 'Apache', default: 'default (non-commercial)' }

/** The student's private page: the landlord link while waiting, progress while building, then the room on a full stage. */
export default function Room({ id }: { id: string }) {
  const [request, setRequest] = useState<RoomRequest | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)

  const refresh = useCallback(() => {
    api.request(id)
      .then(r => {
        setRequest(r)
        setError(null)
      })
      .catch(e => setError((e as Error).message))
  }, [id])

  useEffect(refresh, [refresh])

  // Keep checking until there's a room or a reason it stopped, and while a hosted room's things are still being built
  // (it opens once its walls are done: when the rest is in, its address changes and the 3D view loads it again)
  const status = request?.status
  const finishing = status === 'READY' && !!request?.stage
  useEffect(() => {
    if (!status || status === 'FAILED' || (status === 'READY' && !finishing)) return
    const timer = setInterval(refresh, status === 'WAITING_FOR_VIDEO' || finishing ? 5000 : 2000)
    return () => clearInterval(timer)
  }, [status, finishing, refresh])
  const sample = isSample(id)
  const fake = STATIC && id === FAKE_ROOM

  // A room someone has opened goes on their workspace's list, wherever the link came from
  useEffect(() => {
    if (request && !sample) rememberRoom(id)
  }, [request?.id, sample, id])

  async function sendBackup(file: File) {
    setError(null)
    setProgress(0)
    try {
      await uploadVideo(`${API_BASE}/api/requests/${id}/video`, file, setProgress, () =>
        api.request(id).then(r => r.status === 'VIDEO_RECEIVED' || r.status === 'RECONSTRUCTING'))
      refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setProgress(null)
    }
  }

  async function rerun(model: string) {
    setError(null)
    try {
      setRequest(await api.rerun(id, model))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const sampleHref = STATIC_ROOM ? `/room/${STATIC_ROOM}` : null
  const fakeHref = FAKE_ROOM ? `/room/${FAKE_ROOM}` : null
  const live = !STATIC || LIVE
  const nav = (compact: boolean) => (
    <SiteNav
      compact={compact}
      activeHref={id === STATIC_ROOM ? sampleHref ?? '' : id === FAKE_ROOM ? fakeHref ?? '' : ''}
      items={[
        { label: 'Home', href: '/' },
        ...(live ? [{ label: 'Workspace', href: '/workspace' }] : []),
        ...(sampleHref ? [{ label: 'Sample room', href: sampleHref }] : []),
        ...(fakeHref ? [{ label: 'Catch a fake', href: fakeHref }] : []),
      ]}
    />
  )

  if (!request) {
    return (
      <>
        <GlBackground />
        {nav(false)}
        <main className="mx-auto max-w-5xl px-4 pb-12 pt-24">
          <p className={error ? 'text-red-700' : 'text-slate-600'}>{error ?? 'Loading...'}</p>
        </main>
      </>
    )
  }

  const room = request.room
  const building = request.status === 'VIDEO_RECEIVED' || request.status === 'RECONSTRUCTING'
  const otherModel = request.model === 'default' ? 'apache' : 'default'

  const backup = !building && !sample && (
    <section className="max-w-xl">
      <p className="text-sm text-slate-600">
        {request.status === 'WAITING_FOR_VIDEO'
          ? 'Did the landlord already send you a video some other way? Upload it here instead.'
          : 'Got a better video? Upload it here, or ask the landlord to film again with the same link.'}
      </p>
      <div className="mt-3">
        <VideoPicker label="Upload a video" progress={progress} onPick={sendBackup} />
      </div>
    </section>
  )

  if (request.status === 'READY' && room) {
    const about = (
      <div className="space-y-4 text-sm text-slate-700">
        {fake && (
          <div>
            <p className="font-medium text-slate-900">What a fake looks like</p>
            <p className="mt-1">
              A landlord who can’t get into the room sends a tour from somewhere else. Here, a free stock video of a hotel room
              (Mixkit) stands in for that stolen tour, and the ad’s photo was made with Muse Image. Nothing here is about a real
              listing. The room was built by the same pipeline as any other, and the checks ran the same way.
            </p>
          </div>
        )}
        {sample && !fake && (
          <div>
            <p className="font-medium text-slate-900">See the room before you sign</p>
            <p className="mt-1">
              A student texts the landlord one link. The landlord’s phone video becomes this: a 3D room at its real size to walk,
              measure, fit your own things in, and check for red flags. This one is a real student’s room.{' '}
              {LIVE && <a href="/workspace" className="font-medium text-sky-700 underline underline-offset-2">Try it with your own room</a>}
            </p>
          </div>
        )}
        <div>
          <p className="font-medium text-slate-900">Try this</p>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            <li><span className="font-medium">Before you sign:</span> the scam checks. Was it filmed now, is it the room in the ad, is it the size they claim, and what did they not show?</li>
            <li><span className="font-medium">Your things:</span> take a picture of something you own and see it here at its real size.</li>
            <li><span className="font-medium">Move things:</span> drag anything across the floor. It says how much room it has, or what it runs into.</li>
            {LIVE && <li><span className="font-medium">See it real:</span> Muse Image paints the view you’re looking at as a photo, with your things where you put them.</li>}
            {(!LIVE || sample) && !fake && <li><span className="font-medium">Real:</span> step between the spots the phone stood at, and look around.</li>}
            <li><span className="font-medium">On a Meta Quest:</span> open this page in its browser and press Enter VR to stand in the room at its real size.</li>
          </ul>
        </div>
        <div>
          <p className="font-medium text-slate-900">How this room was built</p>
          <p className="mt-1">
            {room.frames} frames from the video, turned into 3D in metres by MapAnything in {room.seconds} s. Muse Spark and SAM 3.1
            found the room’s things, and SAM 3D Objects rebuilt them whole.
            {sample && !fake && ' In this sample, 14 of its 27 things are hand-made models, not SAM 3D (the big built-in pieces like the bed, desk, monitors, wardrobe, door and window), to show where this is going, and Muse Image continued the blanket where the video never saw it (Not filmed tints that part). In a new room, everything is automatic.'}
          </p>
          <p className="mt-2">
            A new room’s size starts as a rough guess: on this sample, MapAnything alone came out 28% small against a tape. Wall plates
            in the video give a rough check, and one length measured with a tape brings the whole room to about 5%. The view always
            says which one you’re looking at.
          </p>
        </div>
        {!STATIC && new URLSearchParams(window.location.search).has('debug') && (
          <button type="button" onClick={() => rerun(otherModel)} className="rounded-lg bg-slate-900 px-3 py-2 font-medium text-white hover:bg-slate-700">
            Build again with the {MODELS[otherModel]} model
          </button>
        )}
        {backup}
        {error && <p className="text-sm text-red-700">{error}</p>}
      </div>
    )
    return (
      <>
        {nav(true)}
        <main>
          <h1 className="sr-only">{request.label}</h1>
          <RoomViewer url={room.plyUrl} canResize={!sample} startOnChecks={fake} startOnAbout={sample && !fake} title={request.label}
            about={about} notice={finishing ? `The walls are ready. Now: ${request.stage}.` : null} codeAsked={request.code ?? null} />
        </main>
      </>
    )
  }

  return (
    <>
      <GlBackground />
      {nav(false)}
      <main className="mx-auto max-w-3xl px-4 pb-10 pt-20 sm:pt-24">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-950">{request.label}</h1>
          <span className="liquid-glass rounded-full px-3 py-1 text-sm text-slate-700">{STATUS_TEXT[request.status]}</span>
        </div>

        {request.status === 'WAITING_FOR_VIDEO' && (
          <section className="liquid-glass mt-6 max-w-xl space-y-3 rounded-2xl p-5">
            <p className="text-slate-700">Send the landlord this link. This page updates by itself when their video comes in.</p>
            <CopyLink url={window.location.origin + request.landlordPath} />
            {request.code && (
              <p className="text-sm text-slate-600">
                The link asks them to hold up the code <span className="font-semibold tabular-nums text-slate-900">{request.code}</span> on
                camera. It shows the video was filmed after you asked, by someone standing in the room.
              </p>
            )}
          </section>
        )}

        {building && (
          <section className="liquid-glass mt-6 flex items-center gap-3 rounded-2xl p-5">
            <Loader2 className="shrink-0 animate-spin text-sky-600 motion-reduce:animate-none" />
            <div>
              <p className="font-medium text-slate-950">Building the 3D room</p>
              <p className="text-sm text-slate-600">
                {request.stage ? `Now: ${request.stage}.` : 'Starting...'} The walls take about 5 minutes, and the whole room about 20. You can leave the page open.
              </p>
            </div>
          </section>
        )}

        {request.status === 'FAILED' && (
          <section className="mt-6 max-w-xl rounded-2xl border border-red-200 bg-red-50/90 p-5 text-red-900">
            <p className="font-medium">It stopped</p>
            <p className="mt-1 text-sm">{request.failureMessage}</p>
            {request.videoSource && (
              <button
                type="button"
                onClick={() => rerun(request.model ?? '')}
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-red-900 px-3 py-2 text-sm font-medium text-white transition-[scale] duration-150 enabled:active:scale-[0.96] hover:bg-red-800 motion-reduce:transition-none"
              >
                <RotateCcw size={15} /> Try again
              </button>
            )}
          </section>
        )}

        {backup && <div className="mt-8">{backup}</div>}
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      </main>
    </>
  )
}
