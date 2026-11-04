import { useEffect, useRef, useState } from 'react'
import { Circle, RotateCcw, Send, Square, TriangleAlert } from 'lucide-react'
import { beforeFilming, coachSteps } from '../instructions'

// A turn faster than this (degrees a second, from the phone's gyroscope) blurs the frames and loses the cameras:
// take 3's fast turn ran about 39 a second, against about 20 for the rest, and COLMAP couldn't place it
const TOO_FAST = 30
const LONGEST = 180 // seconds: the recording stops itself here, so a forgotten phone doesn't fill the upload
const BITRATE = 8_000_000 // about 120 MB for two minutes: plenty of detail at 1080p
const CODE_SECONDS = 6 // the coach waits this long for the landlord to hold up the one-time code

type Stage = 'ready' | 'recording' | 'review' | 'unsupported' | 'error'

/** The file type this browser records best: MP4 (H.264) where it can (iPhones), else WebM. */
function bestType() {
  for (const type of ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']) {
    if (MediaRecorder.isTypeSupported(type)) return type
  }
  return ''
}

/**
 * The landlord's recorder: the phone's back camera at full size, inside the page. (A file input's "Take Video" on an
 * iPhone records 480 x 360, which caps how real the room can look.) A coach walks them round the room once, every
 * wall from the doorway and from the middle, and warns when they turn too fast. When they stop, they see what they
 * got and send it with the same upload as before.
 */
export default function Recorder({ code, onSend, sending }: { code?: string; onSend: (file: File) => void; sending: number | null }) {
  const supported = typeof window !== 'undefined' && window.isSecureContext && !!navigator.mediaDevices?.getUserMedia
    && typeof MediaRecorder !== 'undefined'
  const [stage, setStage] = useState<Stage>(supported ? 'ready' : 'unsupported')
  const [error, setError] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(0)
  const [tooFast, setTooFast] = useState(false)
  const [size, setSize] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const preview = useRef<HTMLVideoElement>(null)
  const stream = useRef<MediaStream | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const stopLater = useRef<() => void>(() => {})

  useEffect(() => () => stopLater.current(), [])

  async function start() {
    setError(null)
    // iPhones ask before a page can read the gyroscope, and only from a tap: this is that tap
    const motion = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> }
    await motion.requestPermission?.().catch(() => 'denied')
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
        audio: false,
      })
    } catch (e) {
      setError(`The camera didn't open: ${(e as Error).message}. You can still upload a video below.`)
      setStage('error')
      return
    }
    const track = stream.current.getVideoTracks()[0]
    const settings = track.getSettings()
    setSize(settings.width && settings.height ? `${settings.width} x ${settings.height}` : null)
    if (preview.current) preview.current.srcObject = stream.current

    const type = bestType()
    const chunks: Blob[] = []
    const media = new MediaRecorder(stream.current, { mimeType: type || undefined, videoBitsPerSecond: BITRATE })
    media.ondataavailable = event => event.data.size && chunks.push(event.data)
    media.onstop = () => {
      const kind = media.mimeType || type || 'video/mp4'
      const ending = kind.includes('webm') ? 'webm' : 'mp4'
      setFile(new File([new Blob(chunks, { type: kind })], `room.${ending}`, { type: kind }))
      setStage('review')
    }
    recorder.current = media
    media.start(1000)
    setSeconds(0)
    setStage('recording')

    const began = Date.now()
    const clock = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - began) / 1000)
      setSeconds(elapsed)
      if (elapsed >= LONGEST) stop()
    }, 250)
    let calm: number | undefined
    const onMotion = (event: DeviceMotionEvent) => {
      const turn = event.rotationRate
      const speed = Math.max(Math.abs(turn?.alpha ?? 0), Math.abs(turn?.beta ?? 0), Math.abs(turn?.gamma ?? 0))
      if (speed > TOO_FAST) {
        setTooFast(true)
        window.clearTimeout(calm)
        calm = window.setTimeout(() => setTooFast(false), 1500)
      }
    }
    window.addEventListener('devicemotion', onMotion)
    stopLater.current = () => {
      window.clearInterval(clock)
      window.clearTimeout(calm)
      window.removeEventListener('devicemotion', onMotion)
      if (recorder.current?.state === 'recording') recorder.current.stop()
      stream.current?.getTracks().forEach(t => t.stop())
      stream.current = null
    }
  }

  function stop() {
    stopLater.current()
    setTooFast(false)
  }

  function again() {
    setFile(null)
    setStage('ready')
  }

  if (stage === 'unsupported') {
    return (
      <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">
        This browser can't record inside the page (it needs a secure https link and a recent phone), so use the upload
        button below.
      </p>
    )
  }

  const step = [...coachSteps].reverse().find(s => seconds >= s.at) ?? coachSteps[0]
  const minutes = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`

  return (
    <div className="space-y-3">
      {stage === 'ready' && (
        <>
          <ul className="space-y-1 text-sm text-slate-700">
            {beforeFilming.map(line => <li key={line}>{line}</li>)}
          </ul>
          <button
            type="button"
            onClick={start}
            className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-3 font-medium text-white transition-[scale,background-color] duration-150 ease-out hover:bg-red-700 enabled:active:scale-[0.96] motion-reduce:transition-none"
          >
            <Circle size={18} fill="currentColor" /> Start filming
          </button>
        </>
      )}

      <div className={stage === 'recording' ? 'relative overflow-hidden rounded-xl bg-black' : 'hidden'}>
        <video ref={preview} autoPlay muted playsInline className="aspect-video w-full object-cover" />
        <div className="absolute inset-x-0 top-0 flex items-center justify-between bg-black/50 px-3 py-2 text-sm text-white">
          <span className="inline-flex items-center gap-2"><Circle size={10} fill="#ef4444" className="text-red-500" /> {minutes}</span>
          {size && <span className="text-white/70">{size}</span>}
        </div>
        <p className="absolute inset-x-0 bottom-14 mx-3 rounded-lg bg-black/60 px-3 py-2 text-center text-white">
          {code && seconds < CODE_SECONDS ? <>Hold up your code first: <span className="font-bold tabular-nums">{code}</span></> : step.say}
        </p>
        {tooFast && (
          <p className="absolute inset-x-0 top-12 mx-auto w-fit rounded-lg bg-amber-500 px-3 py-1.5 text-sm font-medium text-black">
            <TriangleAlert size={15} className="mr-1 inline" /> Turn slower
          </p>
        )}
        <button
          type="button"
          onClick={stop}
          className="absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-slate-900"
        >
          <Square size={14} fill="currentColor" /> Stop
        </button>
      </div>

      {stage === 'review' && file && (
        <div className="space-y-3 rounded-xl border border-slate-900/10 bg-white p-4">
          <p className="text-slate-700">
            {minutes} of video{size ? ` at ${size}` : ''}, {(file.size / 1e6).toFixed(0)} MB.
            {seconds < 60 && ' That might be too short to see every wall; you can film again.'}
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={sending !== null}
              onClick={() => onSend(file)}
              className="inline-flex items-center gap-2 rounded-full bg-tape text-ink px-4 py-2 font-semibold transition-[scale,filter] duration-150 ease-out hover:brightness-105 enabled:active:scale-[0.96] motion-reduce:transition-none disabled:opacity-60"
            >
              <Send size={16} /> {sending !== null ? `Sending ${Math.round(sending * 100)}%` : 'Send this video'}
            </button>
            <button
              type="button"
              disabled={sending !== null}
              onClick={again}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-4 py-2 text-slate-800 hover:bg-white disabled:opacity-60"
            >
              <RotateCcw size={16} /> Film again
            </button>
          </div>
        </div>
      )}

      {error && <p className="text-sm text-red-700">{error}</p>}
    </div>
  )
}
