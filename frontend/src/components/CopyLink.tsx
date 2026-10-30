import { useRef, useState } from 'react'
import { Check, Copy } from 'lucide-react'

/** A link in a box with a Copy button. Without a clipboard (plain http on a phone), it selects the text instead. */
export default function CopyLink({ url }: { url: string }) {
  const input = useRef<HTMLInputElement>(null)
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(url)
    } catch {
      input.current?.select()
      return
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="flex gap-2">
      <input
        ref={input}
        readOnly
        value={url}
        onFocus={e => e.currentTarget.select()}
        className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700"
      />
      <button
        type="button"
        onClick={copy}
        className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700"
      >
        {copied ? <Check size={16} /> : <Copy size={16} />}
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}
