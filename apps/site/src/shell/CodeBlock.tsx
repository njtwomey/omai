import { type ReactNode, useState } from 'react'
import { CheckIcon, CopyIcon } from 'whitebox-render/react'
import { highlight } from './highlight.ts'

/**
 * Code to read and copy: highlighted TypeScript under an optional title (the file it comes from), with a copy button.
 * `renderType` may draw the type names (capitalised words) itself, such as the docs' type links with their hover.
 */
export function CodeBlock(props: {
  readonly code: string
  readonly title?: string
  readonly className?: string
  readonly renderType?: (name: string, key: number) => ReactNode
}): ReactNode {
  const { code, title, className, renderType } = props
  const [copied, setCopied] = useState(false)
  const copy = (): void => {
    void navigator.clipboard?.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <figure className={`site-code m-0 overflow-hidden rounded-lg border border-line bg-surface ${className ?? ''}`}>
      <figcaption className="flex items-center justify-between gap-2 border-b border-line px-3 py-1.5">
        <span className="truncate font-mono text-xs text-muted">{title ?? 'code'}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted hover:bg-bg hover:text-fg"
          aria-label="Copy code"
        >
          {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </figcaption>
      <pre className="m-0 overflow-x-auto p-3 text-[12.5px] leading-relaxed">
        <code>
          {highlight(code).map((t, i) =>
            t.kind === undefined ? (
              t.text
            ) : t.kind === 'type' && renderType !== undefined ? (
              renderType(t.text, i)
            ) : (
              // biome-ignore lint/suspicious/noArrayIndexKey: tokens are a fixed sequence for this code
              <span key={i} className={`tok-${t.kind}`}>
                {t.text}
              </span>
            ),
          )}
        </code>
      </pre>
    </figure>
  )
}
