import { useCallback, useEffect, useState } from 'react'
import { Loader2, RotateCcw } from 'lucide-react'
import { api, uploadVideo } from '../api'
import CopyLink from '../components/CopyLink'
import RoomViewer from '../components/RoomViewer'
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

/** The student's private page: the landlord link while waiting, progress while building, then the 3D room. */
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

  // Keep checking until there's a room or a reason it stopped
  const status = request?.status
  useEffect(() => {
    if (!status || status === 'READY' || status === 'FAILED') return
    const timer = setInterval(refresh, status === 'WAITING_FOR_VIDEO' ? 5000 : 2000)
    return () => clearInterval(timer)
  }, [status, refresh])

  async function sendBackup(file: File) {
    setError(null)
    setProgress(0)
    try {
      await uploadVideo(`/api/requests/${id}/video`, file, setProgress)
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

  if (!request) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-12">
        <p className={error ? 'text-red-700' : 'text-slate-600'}>{error ?? 'Loading...'}</p>
      </main>
    )
  }

  const room = request.room
  const building = request.status === 'VIDEO_RECEIVED' || request.status === 'RECONSTRUCTING'
  const otherModel = request.model === 'default' ? 'apache' : 'default'

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      <p className="text-sm font-medium text-teal-700">SightSeen</p>
      <div className="mt-1 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">{request.label}</h1>
        <span className="rounded-full bg-slate-200 px-3 py-1 text-sm text-slate-700">{STATUS_TEXT[request.status]}</span>
      </div>

      {request.status === 'WAITING_FOR_VIDEO' && (
        <section className="mt-6 max-w-xl space-y-3">
          <p className="text-slate-600">Send the landlord this link. This page updates by itself when their video comes in.</p>
          <CopyLink url={window.location.origin + request.landlordPath} />
        </section>
      )}

      {building && (
        <section className="mt-6 flex items-center gap-3 rounded-xl bg-white p-5 shadow-sm">
          <Loader2 className="animate-spin text-teal-700" />
          <div>
            <p className="font-medium">Building the 3D room</p>
            <p className="text-sm text-slate-600">
              {request.stage ? `Now: ${request.stage}.` : 'Starting...'} This takes a few minutes. You can leave the page open.
            </p>
          </div>
        </section>
      )}

      {request.status === 'FAILED' && (
        <section className="mt-6 max-w-xl rounded-xl bg-red-50 p-5 text-red-900">
          <p className="font-medium">It stopped</p>
          <p className="mt-1 text-sm">{request.failureMessage}</p>
          {request.videoSource && (
            <button
              type="button"
              onClick={() => rerun(request.model ?? '')}
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-red-900 px-3 py-2 text-sm font-medium text-white hover:bg-red-800"
            >
              <RotateCcw size={15} /> Try again
            </button>
          )}
        </section>
      )}

      {request.status === 'READY' && room && (
        <section className="mt-6 space-y-4">
          <RoomViewer url={room.plyUrl} />
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl bg-white p-5 text-sm text-slate-700 shadow-sm">
              <p className="font-medium text-slate-900">How this room was built</p>
              <p className="mt-2">
                {room.frames} frames from the video, rebuilt by MapAnything ({MODELS[request.model ?? ''] ?? request.model} model)
                in {room.seconds} s{room.peakVramGb !== null && `, using ${room.peakVramGb} GB of GPU memory`}.
                {' '}{room.points.toLocaleString()} points.
              </p>
              <p className="mt-2">
                Rough extent: {room.sizeX.toFixed(2)} × {room.sizeY.toFixed(2)} × {room.sizeZ.toFixed(2)} m, along the
                first frame's directions, not the walls. Measure a wall with two clicks instead.
              </p>
              <p className="mt-2 text-slate-500">
                Sizes come from MapAnything's own scale and haven't been checked against anything in the room yet.
              </p>
            </div>
            <div className="rounded-xl bg-white p-5 text-sm text-slate-700 shadow-sm">
              <p className="font-medium text-slate-900">Compare</p>
              <p className="mt-2">Build the same video again with the other MapAnything model and measure the same wall.</p>
              <button
                type="button"
                onClick={() => rerun(otherModel)}
                className="mt-3 rounded-lg bg-slate-900 px-3 py-2 font-medium text-white hover:bg-slate-700"
              >
                Build again with the {MODELS[otherModel]} model
              </button>
            </div>
          </div>
        </section>
      )}

      {!building && (
        <section className="mt-8 max-w-xl">
          <p className="text-sm text-slate-600">
            {request.status === 'WAITING_FOR_VIDEO'
              ? 'Did the landlord already send you a video some other way? Upload it here instead.'
              : 'Got a better video? Upload it here, or ask the landlord to film again with the same link.'}
          </p>
          <div className="mt-3">
            <VideoPicker label="Upload a video" progress={progress} onPick={sendBackup} />
          </div>
        </section>
      )}

      {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
    </main>
  )
}
