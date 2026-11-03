import { useEffect, useState } from 'react'
import { AlertTriangle, BadgeCheck, CircleHelp, ShieldAlert, X } from 'lucide-react'
import { STATIC } from '../api'

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

type VideoBadge = { level: 'live' | 'link' | 'upload' | 'unknown'; title: string; detail: string }

const FT2_PER_M2 = 10.7639
const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

/**
 * The issues mode: how the video came in (a fact), damage (AI found, behind plain-rules gate), what was never filmed
 * (from the coverage maps), and the room's size against the ad's (arithmetic). Flag less and be right: everything
 * shown says why, and what didn't pass says which rule it failed.
 */
export default function Issues({ base, open, onClose, onReport, room }: {
  base: string
  open: boolean
  onClose: () => void
  onReport: (report: IssuesReport) => void
  room: MeasuredRoom | null
}) {
  const [report, setReport] = useState<IssuesReport | null>(null)
  const [badge, setBadge] = useState<VideoBadge | null>(null)
  const [claim, setClaim] = useState('')
  const [unit, setUnit] = useState<'m2' | 'ft2'>('ft2')

  useEffect(() => {
    let gone = false
    fetch(`${base}/assets/room-issues.json`).then(r => (r.ok ? r.json() : null)).then((found: IssuesReport | null) => {
      if (gone || !found) return
      setReport(found)
      onReport(found)
    }).catch(() => {})
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
  }, [base])

  if (!open) return null
  const damage = report?.flags.filter(f => f.kind === 'damage') ?? []
  const unfilmed = report?.flags.filter(f => f.kind === 'not filmed') ?? []
  const damageNotShown = report?.notShown.filter(d => !d.failed.includes('unfilmed wall patch')) ?? []
  const otherWalls = report?.notShown.filter(d => d.failed.includes('unfilmed wall patch')) ?? []

  // The size check: plain arithmetic, the ad's number against the measured floor and how sure that is
  let sizeVerdict: { tone: 'good' | 'warn'; text: string } | null = null
  const claimed = Number(claim)
  if (room && claimed > 0) {
    const area = room.width * room.length
    const claimM2 = unit === 'ft2' ? claimed / FT2_PER_M2 : claimed
    const margin = area * room.tolerance
    const show = (m2: number) => (unit === 'ft2' ? `${Math.round(m2 * FT2_PER_M2)} ft²` : `${m2.toFixed(1)} m²`)
    if (room.tolerance > 0.2) {
      // Flag less and be right: a size this unsure can't call the ad wrong, or right
      sizeVerdict = { tone: 'warn', text: `Can't judge it yet: the room measures ${show(area)}, but only to ± ${Math.round(room.tolerance * 100)}% until one length is measured with a tape (the room's width, or the desk's height). Then this check is to about ± 4%.` }
    } else if (claimM2 > area + margin) {
      sizeVerdict = { tone: 'warn', text: `The ad says ${show(claimM2)}; the room measures ${show(area)} (± ${show(margin)}), ${Math.round((1 - area / claimM2) * 100)}% smaller than claimed. Ask the landlord which room the size is for, or measure it together on a call. (An ad's size sometimes means the whole unit.)` }
    } else if (claimM2 < area - margin) {
      sizeVerdict = { tone: 'good', text: `The room measures ${show(area)} (± ${show(margin)}), more than the ${show(claimM2)} the ad says.` }
    } else {
      sizeVerdict = { tone: 'good', text: `Matches the ad: the room measures ${show(area)}, within ± ${show(margin)} of the ${show(claimM2)} claimed.` }
    }
  }

  const BadgeIcon = badge?.level === 'live' ? BadgeCheck : badge?.level === 'unknown' ? CircleHelp : ShieldAlert
  return (
    <div className="absolute right-3 top-3 z-10 max-h-[calc(100%-4.5rem)] w-96 overflow-y-auto rounded-xl bg-white/95 p-4 text-sm text-slate-800 shadow-lg">
      <div className="flex items-center justify-between">
        <p className="inline-flex items-center gap-1.5 font-medium text-slate-900">
          <AlertTriangle size={16} /> Issues
        </p>
        <button type="button" onClick={onClose} className="rounded p-1 hover:bg-slate-100" aria-label="Close">
          <X size={16} />
        </button>
      </div>

      {badge && (
        <div className={`mt-3 flex gap-2 rounded-lg p-3 ${badge.level === 'live' ? 'bg-emerald-50 text-emerald-900' : 'bg-amber-50 text-amber-900'}`}>
          <BadgeIcon size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">{badge.title}</p>
            <p className="mt-0.5 text-xs">{badge.detail}</p>
          </div>
        </div>
      )}

      <p className="mt-4 font-medium text-slate-900">Damage</p>
      {damage.length === 0 && report && (
        <p className="mt-1 text-xs text-slate-600">
          Muse Spark looked through all {report.framesChecked ?? 'the'} frames, as small pictures, for damage (mould,
          water stains, cracks, holes, peeling paint, broken fixtures, carpet, pests)
          {report.damageListed ? ` and listed ${report.damageListed}; none passed the gate.` : ' and listed none, so the gate had nothing to check.'}
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
              Passed the gate: outlined in {flag.gate.frames} frames, {flag.gate.viewsApartCm} cm apart and{' '}
              {flag.gate.viewsTurnDeg}° round, landing within {flag.gate.sameSpotCm} cm of each other.
            </p>
          )}
          {flag.ask && <p className="mt-1 text-xs font-medium text-red-900">{flag.ask}</p>}
        </div>
      ))}
      {damageNotShown.length > 0 && (
        <details className="mt-2 text-xs text-slate-600">
          <summary className="cursor-pointer">{damageNotShown.length} possible problem(s) not shown: they failed the gate</summary>
          <ul className="mt-1 list-disc pl-4">
            {damageNotShown.map(d => <li key={d.name}>{sentence(d.name)}: {d.failed}</li>)}
          </ul>
        </details>
      )}
      {report && (
        <p className="mt-2 text-[11px] text-slate-500">
          The gate (plain rules on top of the AI): a flag shows only when SAM 3.1 outlines it in at least{' '}
          {report.gate.frames} frames, from views at least {report.gate.viewsApartCm} cm apart or{' '}
          {report.gate.viewsTurnDeg}° round, landing within {report.gate.sameSpotCm} cm of each other.
        </p>
      )}

      <p className="mt-4 font-medium text-slate-900">Never filmed</p>
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

      <p className="mt-4 font-medium text-slate-900">Size against the ad</p>
      {room ? (
        <>
          <div className="mt-1 flex gap-2">
            <input value={claim} onChange={event => setClaim(event.target.value)} inputMode="decimal"
              placeholder="The ad's size" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs" />
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
            {' '}{Math.round(room.width * room.length * FT2_PER_M2)} ft²); scale {room.source}.
          </p>
        </>
      ) : (
        <p className="mt-1 text-xs text-slate-600">This room's size isn't known yet.</p>
      )}
    </div>
  )
}
