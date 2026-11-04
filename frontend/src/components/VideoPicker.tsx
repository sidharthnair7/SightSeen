import { Upload } from 'lucide-react'

/** A button that opens the camera or the phone's videos, then shows the upload's progress. */
export default function VideoPicker({ label, progress, onPick }: {
  label: string
  progress: number | null
  onPick: (file: File) => void
}) {
  if (progress !== null) {
    return (
      <div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-200">
          <div className="h-full bg-ink transition-[width] duration-150 motion-reduce:transition-none" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        <p className="mt-2 text-sm text-slate-600">Sending the video... {Math.round(progress * 100)}%</p>
      </div>
    )
  }
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-tape text-ink px-4 py-2.5 font-semibold transition-[scale,filter] duration-150 ease-out hover:brightness-105 active:scale-[0.96] motion-reduce:transition-none">
      <Upload size={18} />
      {label}
      <input
        type="file"
        accept="video/*"
        className="hidden"
        onChange={e => {
          const file = e.target.files?.[0]
          e.target.value = ''
          if (file) onPick(file)
        }}
      />
    </label>
  )
}
