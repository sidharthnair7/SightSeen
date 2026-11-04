import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, BadgeCheck, Check, CircleHelp, ImageUp, Loader2, MapPin, ShieldAlert, ShieldCheck, X } from 'lucide-react'
import { API_BASE, LIVE, STATIC } from '../api'

/** room-issues.json (worker/room_issues.py): flags that passed the gate, and candidates that didn't, with why. */
export type IssuesReport = {
  gate: { frames: number; sameSpotCm: number; viewsApartCm: number; viewsTurnDeg: number; smallestCm: number }
  framesChecked?: number
  /** Possible damage Muse Spark listed, before the gate */
  damageListed?: number
  flags: Flag[]
  notShown: { name: string; why?: string | null; failed: string; face?: string; areaM2?: number }[]
}

export type Flag = {
  kind: 'damage' | 'not filmed'
  name: string
  why?: string | null
  ask?: string
  position?: number[]
  size?: number[]
  frames?: { frame: number; file: string }[]
  gate?: { frames: number; sameSpotCm: number; viewsApartCm: number; viewsTurnDeg: number }
  corners?: number[][]
  /** A never-filmed patch's own shape: its edge, and its inside as cells of a grid on the wall */
  outline?: number[][]
  fill?: { origin: number[]; across: number[]; down: number[]; cells: number[][] }
  areaM2?: number
}

/** The room's measured size and how sure it is, from the viewer. */
export type MeasuredRoom = {
  width: number
  length: number
  height: number
  /** The video never showed the ceiling: the height is a standard one, not measured */
  heightAssumed?: boolean
  source: string
  tolerance: number
}

/** room-trust.json (worker/room_trust.py): the checks a video gets when it's built. */
type Trust = {
  depth?: { flat: boolean; known: boolean; pairs: number; flatPairs: number } | null
  code?: { code: string; seen: boolean; atSeconds: number | null; file: string | null; inRoom: boolean | null; read: string[]; framesChecked: number; failed?: boolean; readFailed?: boolean } | null
}

/** One ad photo against the video (room_trust.check_ad). `folder` is where its pictures are. */
type AdResult = {
  id: string
  status: 'match' | 'likely' | 'mismatch' | 'unsure'
  inliers: number
  frame: string | null
  same: string[]
  different: string[]
  why: string
  photo: string | null
  evidence: string | null
  label?: string
  folder: string
}

type Tone = 'good' | 'bad' | 'warn' | 'none'
type VideoBadge = { level: 'live' | 'link' | 'upload' | 'unknown'; title: string; detail: string }

const FT2_PER_M2 = 10.7639
const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)
const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
const PRESS = 'transition-[scale,background-color] duration-150 ease-out enabled:active:scale-[0.96] motion-reduce:transition-none motion-reduce:enabled:active:scale-100'

const TONES: Record<Tone, string> = {
  good: 'text-emerald-600',
  bad: 'text-red-600',
  warn: 'text-amber-600',
  none: 'text-slate-400',
}
const VERDICTS: Record<'good' | 'bad' | 'warn' | 'none', { box: string; title: string; text: string }> = {
  none: { box: 'bg-slate-100 text-slate-900 border-slate-200', title: 'Nothing checked yet', text: 'The checks run on the video as the room is built. Add the ad’s photos and size below.' },
  bad: { box: 'bg-red-50 text-red-950 border-red-200', title: 'Don’t pay yet', text: 'At least one check failed. The details are below.' },
  warn: { box: 'bg-amber-50 text-amber-950 border-amber-200', title: 'Ask before you pay', text: 'Nothing failed outright, but some things need an answer.' },
  good: { box: 'bg-emerald-50 text-emerald-950 border-emerald-200', title: 'No red flags so far', text: 'Every check that ran came back clear.' },
}
const AD_CHIPS: Record<AdResult['status'], { text: string; box: string }> = {
  match: { text: 'Lines up with the video', box: 'bg-emerald-100 text-emerald-900' },
  likely: { text: 'Looks like the same room', box: 'bg-emerald-100 text-emerald-900' },
  mismatch: { text: 'Doesn’t look like this room', box: 'bg-red-100 text-red-900' },
  unsure: { text: 'Can’t tell from this photo', box: 'bg-amber-100 text-amber-900' },
}

function rememberedChecks(room: string): string[] {
  try {
    return JSON.parse(localStorage.getItem(`sightseen-ad-checks-${room}`) ?? '[]')
  } catch {
    return []
  }
}
function rememberChecks(room: string, ids: string[]) {
  try {
    localStorage.setItem(`sightseen-ad-checks-${room}`, JSON.stringify(ids))
  } catch {
    // private window: the results still show until the page closes
  }
}

function Row({ tone, title, children }: { tone: Tone; title: string; children?: string }) {
  const Icon = tone === 'good' ? Check : tone === 'bad' ? X : tone === 'warn' ? AlertTriangle : CircleHelp
  return (
    <li className="flex items-start gap-2">
      <Icon size={15} strokeWidth={2.6} className={`mt-0.5 shrink-0 ${TONES[tone]}`} />
      <span className="min-w-0">
        <span className="font-medium text-slate-900">{title}</span>
        {children && <span className="block text-xs text-slate-600">{children}</span>}
      </span>
    </li>
  )
}

function AdCard({ result, onShowFrame }: { result: AdResult; onShowFrame: (frame: string) => void }) {
  const chip = AD_CHIPS[result.status]
  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-slate-200 bg-white">
      {result.evidence && (
        <img src={`${result.folder}${result.evidence}`} alt="The ad's photo beside the closest frame of the video" className="w-full bg-slate-900" />
      )}
      <div className="p-3">
        {result.label && <p className="mb-1 text-[11px] text-slate-500">{result.label}</p>}
        <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${chip.box}`}>{chip.text}</span>
        {result.status === 'match' && (
          <p className="mt-1.5 text-xs text-slate-600">
            {result.inliers} details in the ad’s photo fit one 3D layout with this frame of the video. That’s plain
            geometry, and it’s the same place whatever any AI thinks.
          </p>
        )}
        {result.status !== 'match' && result.different.length > 0 && (
          <ul className="mt-1.5 list-disc pl-4 text-xs text-slate-700">
            {result.different.map(item => <li key={item}>{sentence(item)}</li>)}
          </ul>
        )}
        {result.status === 'likely' && result.same.length > 0 && (
          <ul className="mt-1.5 list-disc pl-4 text-xs text-slate-700">
            {result.same.map(item => <li key={item}>{sentence(item)}</li>)}
          </ul>
        )}
        {result.status !== 'match' && result.why && <p className="mt-1.5 text-xs text-slate-600">Muse Spark: {result.why}</p>}
        {result.status !== 'match' && <p className="mt-1 text-[11px] text-slate-500">{result.inliers} details lined up; it takes 40 to call it the same place by geometry alone.</p>}
        {result.frame && result.status !== 'mismatch' && (
          <button type="button" onClick={() => onShowFrame(result.frame!)}
            className={`mt-2 inline-flex items-center gap-1 rounded-lg bg-slate-100 px-2 py-1.5 text-xs font-medium text-slate-800 hover:bg-slate-200 ${PRESS}`}>
            <MapPin size={13} /> Go to this spot in the room
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * The student's "should I sign this?" panel. It opens with the verdict, then the evidence for each line. The AI reads and
 * compares; plain rules decide, and everything shown says why. What didn't pass says which rule it failed.
 */
export default function Issues({ base, roomId, open, stage, codeAsked, onClose, onReport, onShowFrame, room }: {
  base: string
  roomId: string
  open: boolean
  /** What the room is still being built with, if it's not finished: its checks arrive as it goes */
  stage: string | null
  /** The one-time code its link asked for, if any */
  codeAsked: string | null
  onClose: () => void
  onReport: (report: IssuesReport) => void
  onShowFrame: (frame: string) => void
  room: MeasuredRoom | null
}) {
  const [report, setReport] = useState<IssuesReport | null>(null)
  const [badge, setBadge] = useState<VideoBadge | null>(null)
  const [trust, setTrust] = useState<Trust | null>(null)
  const [examples, setExamples] = useState<AdResult[]>([])
  const [examplesCount, setExamplesCount] = useState(false)
  const [mine, setMine] = useState<AdResult[]>([])
  const [checking, setChecking] = useState<string | null>(null)
  const [adError, setAdError] = useState<string | null>(null)
  const [claim, setClaim] = useState('')
  const [unit, setUnit] = useState<'m2' | 'ft2'>('ft2')
  const checks = useRef<string[]>(rememberedChecks(roomId))

  useEffect(() => {
    let gone = false
    const json = <T,>(url: string): Promise<T | null> => fetch(url).then(r => (r.ok ? r.json() as Promise<T> : null)).catch(() => null)
    json<IssuesReport>(`${base}/assets/room-issues.json`).then(found => {
      if (gone || !found) return
      setReport(found)
      onReport(found)
    })
    json<Trust>(`${base}/assets/room-trust.json`).then(found => {
      if (!gone && found) setTrust(found)
    })
    json<{ counts?: boolean; examples: Omit<AdResult, 'folder'>[] }>(`${base}/assets/room-ad-examples.json`).then(found => {
      if (gone || !found?.examples) return
      setExamples(found.examples.map(item => ({ ...item, folder: `${base}/assets/` })))
      setExamplesCount(found.counts === true)
    })
    Promise.all([
      fetch(STATIC ? `${base}/request.json` : base).then(r => (r.ok ? r.json() : null)).catch(() => null),
      // The marker's content, not just that something came back: a static host answers a missing file with the page
      fetch(`${base}/assets/video-live.json`).then(r => (r.ok ? r.json() : null))
        .then(found => found?.recordedInPage === true).catch(() => false),
    ]).then(([request, live]) => {
      if (gone) return
      const source = request?.videoSource
      if (source === 'LANDLORD_LINK' && live) {
        setBadge({ level: 'live', title: 'Filmed live in SightSeen, through the landlord’s link',
          detail: 'Recorded in the page the landlord opened from your link. That’s much harder to fake than sending a file of another listing’s tour, though it isn’t proof.' })
      } else if (source === 'LANDLORD_LINK') {
        setBadge({ level: 'link', title: 'Sent through the landlord’s link, as a file',
          detail: 'The person you sent the link to sent this, but a file can be a copy of another listing’s tour. To be sure, ask them to film it live with the link’s recorder.' })
      } else if (source === 'STUDENT_UPLOAD') {
        setBadge({ level: 'upload', title: 'Uploaded by you',
          detail: 'It didn’t come through SightSeen’s link, so nothing shows who filmed it or when. Ask the landlord to film it live with the link.' })
      } else {
        setBadge({ level: 'unknown', title: 'Where the video came from isn’t known', detail: 'Ask the landlord to film it live with the link.' })
      }
    })
    return () => {
      gone = true
    }
  }, [base, stage])

  // The ad photos this visitor has checked here, and any still being checked
  const refreshChecks = async () => {
    const records = await Promise.all(checks.current.map(id => fetch(`${API_BASE}/api/ad-checks/${id}`)
      .then(r => (r.ok ? r.json() : null)).catch(() => null)))
    const done: AdResult[] = []
    const failed: string[] = []
    let working: string | null = null
    records.forEach((record, k) => {
      if (record?.status === 'done') {
        for (const result of record.results ?? []) done.push({ ...result, folder: `${API_BASE}/api/ad-checks/${checks.current[k]}/` })
      } else if (record?.status === 'checking' || record === null) {
        working = sentence(record?.stage ?? 'starting') // a null is a dropped connection: keep asking
      } else if (record?.status === 'failed') {
        failed.push(checks.current[k])
        setAdError(record.why ?? 'The check stopped. Try again.')
      }
    })
    if (failed.length) { // said once: it doesn't come back on every look
      checks.current = checks.current.filter(id => !failed.includes(id))
      rememberChecks(roomId, checks.current)
    }
    setMine(done)
    setChecking(working)
  }
  useEffect(() => {
    if (LIVE && checks.current.length) refreshChecks()
  }, [])
  useEffect(() => {
    if (!checking) return
    const timer = setInterval(refreshChecks, 2500)
    return () => clearInterval(timer)
  }, [checking])

  const sendPhotos = async (files: FileList | null) => {
    if (!files?.length) return
    setAdError(null)
    setChecking('Sending the photos')
    try {
      const form = new FormData()
      Array.from(files).slice(0, 5).forEach(file => form.append('photos', file))
      const response = await fetch(`${API_BASE}/api/requests/${roomId}/ad-check`, { method: 'POST', body: form })
      const body = await response.json().catch(() => null)
      if (!response.ok) throw new Error(body?.message || `Something went wrong (${response.status}).`)
      checks.current = [...checks.current, body.id]
      rememberChecks(roomId, checks.current)
      setChecking('Starting')
    } catch (problem) {
      setAdError(problem instanceof Error ? problem.message : String(problem))
      setChecking(null)
    }
  }

  if (!open) return null
  const damage = report?.flags.filter(f => f.kind === 'damage') ?? []
  const unfilmed = report?.flags.filter(f => f.kind === 'not filmed') ?? []
  const damageNotShown = report?.notShown.filter(d => !d.failed.includes('unfilmed wall patch')) ?? []
  const otherWalls = report?.notShown.filter(d => d.failed.includes('unfilmed wall patch')) ?? []

  // The size check: plain arithmetic, the ad's number against the measured floor and how sure that is
  let sizeVerdict: { tone: 'good' | 'warn'; text: string; short: string } | null = null
  const claimed = Number(claim.replace(/[^0-9.]/g, ''))
  if (room && claimed > 0) {
    const area = room.width * room.length
    const claimM2 = unit === 'ft2' ? claimed / FT2_PER_M2 : claimed
    const margin = area * room.tolerance
    const show = (m2: number) => (unit === 'ft2' ? `${Math.round(m2 * FT2_PER_M2)} ft²` : `${m2.toFixed(1)} m²`)
    if (room.tolerance > 0.2) {
      // Flag less and be right: a size this unsure can't call the ad wrong, or right
      sizeVerdict = { tone: 'warn', short: 'Too rough to judge until one length is measured', text: `Can't judge it yet: the room measures ${show(area)}, but only to ± ${Math.round(room.tolerance * 100)}% until one length is measured with a tape (the room's width, or the desk's height). Then this check is to about ± 10%.` }
    } else if (claimM2 > area + margin) {
      sizeVerdict = { tone: 'warn', short: `${Math.round((1 - area / claimM2) * 100)}% smaller than the ad says`, text: `The ad says ${show(claimM2)}; the room measures ${show(area)} (± ${show(margin)}), ${Math.round((1 - area / claimM2) * 100)}% smaller than claimed. Ask the landlord which room the size is for, or measure it together on a call. (An ad's size sometimes means the whole unit.)` }
    } else if (claimM2 < area - margin) {
      sizeVerdict = { tone: 'good', short: 'Bigger than the ad says', text: `The room measures ${show(area)} (± ${show(margin)}), more than the ${show(claimM2)} the ad says.` }
    } else {
      sizeVerdict = { tone: 'good', short: 'Matches the ad', text: `Matches the ad: the room measures ${show(area)}, within ± ${show(margin)} of the ${show(claimM2)} claimed.` }
    }
  }

  // The verdict: each check's line, the worst of them on top
  const code = trust?.code
  const counted = examplesCount ? [...examples, ...mine] : mine
  const worstAd = counted.find(r => r.status === 'mismatch') ?? counted.find(r => r.status === 'unsure') ?? counted[0]
  const uploaded = badge?.level === 'upload'
  // The code's line. Only a landlord's link asks for one, so a video the student uploaded isn't blamed for not having it,
  // and a check that broke, or a code whose frames don't sit with the rest of the room, is something to ask, not a verdict
  const codeProblem = code?.failed || code?.readFailed ? 'broke' : code?.seen && code.inRoom === false ? 'apart' : null
  const codeMissing = !!codeAsked && !code && !uploaded // asked for one, and there's no answer
  const codeTone: Tone = codeProblem || codeMissing ? 'warn' : code ? (code.seen ? 'good' : uploaded ? 'warn' : 'bad') : uploaded ? 'warn' : 'none'
  const codeTitle = codeProblem === 'broke' ? 'The code check didn’t finish'
    : codeProblem === 'apart' ? `Code ${code?.code} is in the video, but ask them to film again`
      : codeMissing ? (stage ? `Looking for the code ${codeAsked}` : 'The code check didn’t finish')
        : code?.seen ? `Filmed now: code ${code.code} is in the video`
          : code && !uploaded ? `The code ${code.code} never shows in the video`
            : 'No one-time code on this video'
  const codeNote = codeProblem === 'broke' ? 'Nothing is known either way. Ask the landlord to film again with the link, or try again later.'
    : codeProblem === 'apart' ? 'The frames around it don’t line up with the rest of the room. A paper can do that close up, and so can a clip stuck on the front of someone else’s tour.'
      : codeMissing ? (stage ? 'It’s still being built.' : 'Nothing is known either way.')
        : code?.seen ? `At ${clock(code.atSeconds ?? 0)}. An old tour can’t have it.`
          : code && !uploaded ? 'It may be an old or copied tour. Ask them to film it again with the code.'
            : uploaded ? 'It was uploaded, so nobody was asked to hold one up. Ask the landlord to film it with the link.'
              : 'This video came before the code. A new room’s link asks for one.'
  const depthTone: Tone = trust?.depth?.known ? (trust.depth.flat ? 'bad' : 'good') : 'none'
  const adTone: Tone = !worstAd ? 'none' : worstAd.status === 'mismatch' ? 'bad' : worstAd.status === 'unsure' ? 'warn' : 'good'
  const sizeTone: Tone = sizeVerdict?.tone ?? 'none'
  const damageTone: Tone = !report ? 'none' : damage.length ? 'warn' : 'good'
  const filmedTone: Tone = !report ? 'none' : unfilmed.length ? 'warn' : 'good'
  const tones = [codeTone, depthTone, adTone, sizeTone, damageTone, filmedTone]
  const verdict = VERDICTS[tones.includes('bad') ? 'bad' : tones.includes('warn') ? 'warn' : tones.includes('good') ? 'good' : 'none']
  const BadgeIcon = badge?.level === 'live' ? BadgeCheck : badge?.level === 'unknown' ? CircleHelp : ShieldAlert
  const heading = 'mt-4 font-medium text-slate-900'

  return (
    <div className="stage-panel">
      <div className="flex items-center justify-between">
        <p className="inline-flex items-center gap-1.5 font-medium text-slate-900">
          <ShieldCheck size={16} /> Before you sign
        </p>
        <button type="button" onClick={onClose} className="rounded p-1 hover:bg-slate-100" aria-label="Close">
          <X size={16} />
        </button>
      </div>

      <div className={`mt-3 rounded-lg border p-3 ${verdict.box}`}>
        <p className="text-base font-semibold">{verdict.title}</p>
        <p className="text-xs">{verdict.text}</p>
      </div>

      <ul className="mt-3 space-y-2">
        <Row tone={codeTone} title={codeTitle}>{codeNote}</Row>
        {trust?.depth?.known && (
          <Row tone={depthTone} title={trust.depth.flat ? 'No real depth in the video' : 'The video moves through real 3D space'}>
            {trust.depth.flat ? 'It looks like a photo or a screen being panned, not someone walking through a room.' : undefined}
          </Row>
        )}
        <Row tone={adTone} title={!worstAd ? 'Same room as the ad? Add its photos below' : worstAd.status === 'mismatch' ? 'An ad photo doesn’t look like this room' : worstAd.status === 'unsure' ? 'Couldn’t tell from the ad’s photos' : 'The ad’s photos match this room'} />
        <Row tone={sizeTone} title={sizeVerdict ? `Size: ${sizeVerdict.short.charAt(0).toLowerCase()}${sizeVerdict.short.slice(1)}` : 'Size against the ad? Type it below'} />
        {report && <Row tone={damageTone} title={damage.length ? `${damage.length} possible problem${damage.length > 1 ? 's' : ''} to ask about` : 'No damage found'} />}
        {report && <Row tone={filmedTone} title={unfilmed.length ? `${unfilmed.length} part${unfilmed.length > 1 ? 's' : ''} of the room never filmed` : 'Every wall and the ceiling were filmed'} />}
      </ul>

      <p className="mt-3 rounded-lg bg-slate-100 p-2 text-xs text-slate-700">
        What this can’t do: it shows whoever filmed was standing in that room, not that they own it. Ask for the lease and an
        ID before you pay anything.
      </p>

      <p className={heading}>How the video came in</p>
      {badge && (
        <div className={`mt-1 flex gap-2 rounded-lg p-3 ${badge.level === 'live' ? 'bg-emerald-50 text-emerald-900' : 'bg-amber-50 text-amber-900'}`}>
          <BadgeIcon size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">{badge.title}</p>
            <p className="mt-0.5 text-xs">{badge.detail}</p>
          </div>
        </div>
      )}

      {code && (
        <>
          <p className={heading}>The one-time code</p>
          {code.seen && code.file && (
            <img src={`${base}/assets/${code.file}`} alt={`The frame showing the code ${code.code}`} className="mt-1 w-full rounded-lg" />
          )}
          <p className="mt-1 text-xs text-slate-600">
            {code.failed || code.readFailed
              ? 'The check couldn’t read the video, so it says nothing either way.'
              : code.seen
                ? `Muse Spark read ${code.code} in the video without being told the code, and plain code compared them.${code.inRoom ? ' The frames around it line up in 3D with the rest of the room, so it was held up in this room.' : ''}`
                : `Muse Spark looked through ${code.framesChecked} frames for a number held up to the camera${code.read?.length ? ` and read ${code.read.join(', ')}, which isn’t the code.` : ' and found none.'}`}
          </p>
        </>
      )}

      <p className={heading}>Same room as the ad?</p>
      <p className="mt-1 text-xs text-slate-600">
        A scammer’s ad and tour can come from different listings, and this catches that. If details in the ad’s photo line up with
        the video in 3D, it’s the same place. If not, Muse Spark compares what can’t easily change: windows, doors, the floor. An ad
        and a tour stolen from one listing will match, so only the code can tell.
      </p>
      {examples.map(result => <AdCard key={result.id} result={result} onShowFrame={onShowFrame} />)}
      {mine.map(result => <AdCard key={result.id} result={result} onShowFrame={onShowFrame} />)}
      {LIVE && (
        <div className="mt-2">
          {checking ? (
            <p className="inline-flex items-center gap-1.5 text-xs text-slate-700">
              <Loader2 size={13} className="animate-spin motion-reduce:animate-none" /> {checking} (about a minute).
            </p>
          ) : (
            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-2 text-xs font-medium text-white hover:bg-slate-700">
              <ImageUp size={14} /> Add photos from the ad
              <input type="file" accept="image/jpeg,image/png,image/webp" multiple className="hidden"
                onChange={event => {
                  sendPhotos(event.target.files)
                  event.target.value = ''
                }} />
            </label>
          )}
          {adError && <p className="mt-1 text-xs text-red-700" role="alert">{adError}</p>}
        </div>
      )}

      <p className={heading}>Size against the ad</p>
      {room ? (
        <>
          <div className="mt-1 flex gap-2">
            <input value={claim} onChange={event => setClaim(event.target.value)} inputMode="decimal"
              placeholder="The ad's size" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-base sm:text-xs" />
            <select value={unit} onChange={event => setUnit(event.target.value as 'm2' | 'ft2')}
              className="rounded-lg border border-slate-300 px-2 text-xs">
              <option value="ft2">ft²</option>
              <option value="m2">m²</option>
            </select>
          </div>
          {sizeVerdict && (
            <p className={`mt-2 rounded-lg p-2 text-xs ${sizeVerdict.tone === 'warn' ? 'bg-amber-50 text-amber-900' : 'bg-emerald-50 text-emerald-900'}`}>
              {sizeVerdict.text}
            </p>
          )}
          <p className="mt-1 text-[11px] text-slate-500">
            Measured floor: {room.width.toFixed(2)} × {room.length.toFixed(2)} m ({(room.width * room.length).toFixed(1)} m²,
            {' '}{Math.round(room.width * room.length * FT2_PER_M2)} ft²); its size is {room.source}.
          </p>
        </>
      ) : (
        <p className="mt-1 text-xs text-slate-600">This room's size isn't known yet.</p>
      )}

      <p className={heading}>Damage</p>
      {!report && <p className="mt-1 text-xs text-slate-600">{stage ? 'This comes once the room is finished.' : 'This check didn’t finish.'}</p>}
      {damage.length === 0 && report && (
        <p className="mt-1 text-xs text-slate-600">
          Muse Spark looked through all {report.framesChecked ?? 'the'} frames, as small pictures, for damage (mould,
          water stains, cracks, holes, peeling paint, broken fixtures, stained or torn carpet, pests)
          {report.damageListed ? ` and listed ${report.damageListed}; none passed the checks below.` : ' and listed none, so there was nothing to check.'}
          {' '}Small marks can hide at that size: ask for close-ups of anything you want to be sure of.
        </p>
      )}
      {damage.map(flag => (
        <div key={flag.name} className="mt-2 rounded-lg border border-red-200 bg-red-50 p-3">
          <p className="font-medium text-red-900">{sentence(flag.name)}</p>
          {flag.why && <p className="mt-0.5 text-xs text-red-900">{flag.why}</p>}
          <div className="mt-2 flex gap-1.5 overflow-x-auto">
            {flag.frames?.map(f => (
              <img key={f.file} src={`${base}/assets/${f.file}`} alt={`Frame ${f.frame}`} className="h-24 rounded" />
            ))}
          </div>
          {flag.gate && (
            <p className="mt-1 text-[11px] text-red-800">
              Passed the checks: outlined in {flag.gate.frames} frames, {flag.gate.viewsApartCm} cm apart and{' '}
              {flag.gate.viewsTurnDeg}° round, landing within {flag.gate.sameSpotCm} cm of each other.
            </p>
          )}
          {flag.ask && <p className="mt-1 text-xs font-medium text-red-900">{flag.ask}</p>}
        </div>
      ))}
      {damageNotShown.length > 0 && (
        <details className="mt-2 text-xs text-slate-600">
          <summary className="cursor-pointer">{damageNotShown.length} possible problem(s) not shown: they failed the checks</summary>
          <ul className="mt-1 list-disc pl-4">
            {damageNotShown.map(d => <li key={d.name}>{sentence(d.name)}: {d.failed}</li>)}
          </ul>
        </details>
      )}
      {report && (
        <p className="mt-2 text-[11px] text-slate-500">
          The checks (plain rules on top of the AI): a flag shows only when SAM 3.1 outlines it in at least{' '}
          {report.gate.frames} frames, from views at least {report.gate.viewsApartCm} cm apart or{' '}
          {report.gate.viewsTurnDeg}° round, landing within {report.gate.sameSpotCm} cm of each other.
        </p>
      )}

      <p className={heading}>Never filmed</p>
      {!report && <p className="mt-1 text-xs text-slate-600">{stage ? 'This comes once the room is finished.' : 'This check didn’t finish.'}</p>}
      {report && unfilmed.length === 0 && <p className="mt-1 text-xs text-slate-600">Every wall and the ceiling were filmed.</p>}
      {unfilmed.map(flag => (
        <div key={flag.name} className="mt-2 rounded-lg border border-yellow-300 bg-yellow-50 p-3">
          <p className="font-medium text-yellow-900">{sentence(flag.name)}</p>
          {flag.why && <p className="mt-0.5 text-xs text-yellow-900">{flag.why}</p>}
          {flag.ask && <p className="mt-1 text-xs font-medium text-yellow-900">{flag.ask}</p>}
        </div>
      ))}
      {otherWalls.length > 0 && (
        <p className="mt-2 text-xs text-slate-600">
          Also never filmed, smaller: {otherWalls.map(d => `${d.name.replace(/^part of /, '').replace(/, never filmed$/, '')}${d.areaM2 ? ` (${d.areaM2} m²)` : ''}`).join(', ')}.
          A handheld video rarely looks up, so the top of most walls goes unfilmed.
        </p>
      )}
    </div>
  )
}
