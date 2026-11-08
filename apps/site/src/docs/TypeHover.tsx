/**
 * Type names that explain themselves: a type in a signature, a parameter's type or the prose links its page and, on
 * hover or keyboard focus, shows its declaration (highlighted) and one-line summary in a small popover that never
 * clips (omai-render's `Popover`). Names in code resolve only through the shown file's `refs` (its imports,
 * resolved when the catalogue was precompiled), so a global or private type stays plain; names in prose fall back to
 * the export of that name nearest the module. Definitions come from the target module's small types file, fetched on
 * the first hover.
 */

import { Popover } from 'omai-render/react'
import { type Context, createContext, type ReactNode, useContext, useEffect, useId, useRef, useState } from 'react'
import { highlight } from '../shell/highlight.ts'
import { Link } from '../shell/Link.tsx'
import { type Docs, symbolPath, useTypes } from './data.ts'
import type { WireRefs } from './wire.ts'

/** The type references of the file a page shows (name to `[module, name]`). */
export const RefsContext: Context<WireRefs> = createContext<WireRefs>({})

const TYPE_KINDS = new Set(['interface', 'type', 'class', 'enum'])
const OPEN_MS = 200
const CLOSE_MS = 150

/**
 * Where a type name used in a module points: the shown file's resolved reference, else (unless `code`) the type of
 * that name nearest the module.
 */
export function useTypeTarget(
  docs: Docs,
  name: string,
  near: string,
  code: boolean,
): { readonly module: string; readonly name: string; readonly path: string } | undefined {
  const refs = useContext(RefsContext)
  const ref = refs[name]
  if (ref !== undefined) {
    const at = docs.symbol(ref[0], ref[1])
    if (at !== undefined) return { module: ref[0], name: ref[1], path: symbolPath(at.file.path, ref[1]) }
  }
  if (code) return undefined
  const found = docs.find(name, near)
  if (found === undefined) return undefined
  const kind = found.file.symbols.find((s) => s.name === found.name)?.kind
  if (kind === undefined || !TYPE_KINDS.has(kind)) return undefined
  return { module: found.file.module, name: found.name, path: symbolPath(found.file.path, found.name) }
}

function Code(props: { readonly code: string }): ReactNode {
  return (
    <code>
      {highlight(props.code).map((t, i) =>
        t.kind === undefined ? (
          t.text
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: tokens are a fixed sequence for this code
          <span key={i} className={`tok-${t.kind}`}>
            {t.text}
          </span>
        ),
      )}
    </code>
  )
}

/**
 * A type name: a link to its page that shows its definition on hover or focus (Escape or moving away closes it).
 * Names that are not documented types render as plain text.
 */
export function TypeRef(props: {
  readonly docs: Docs
  readonly name: string
  readonly near: string
  /** A name in code: resolved only through the file's references. */
  readonly code?: boolean
  readonly className?: string
  readonly children?: ReactNode
}): ReactNode {
  const { docs, name, near, className } = props
  const target = useTypeTarget(docs, name, near, props.code === true)
  const anchor = useRef<HTMLSpanElement>(null)
  const [wanted, setWanted] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const id = useId()
  const types = useTypes(wanted && target !== undefined ? docs.typesUrl(target.module) : undefined)
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current)
    },
    [],
  )
  if (target === undefined) return <span className={className}>{props.children ?? name}</span>
  const entry = types?.[target.name]
  const later = (open: boolean, ms: number): void => {
    if (timer.current !== null) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      timer.current = null
      setWanted(open)
    }, ms)
  }
  const now = (open: boolean): void => {
    if (timer.current !== null) clearTimeout(timer.current)
    timer.current = null
    setWanted(open)
  }
  const open = wanted && entry !== undefined
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: the hover region of the link inside it, which takes the focus
    <span
      ref={anchor}
      onMouseEnter={() => later(true, OPEN_MS)}
      onMouseLeave={() => later(false, CLOSE_MS)}
      onFocus={() => now(true)}
      onBlur={() => now(false)}
      aria-describedby={open ? id : undefined}
    >
      <Link to={target.path} className={`${className ?? ''} hover:underline`}>
        {props.children ?? name}
      </Link>
      {entry === undefined ? null : (
        <Popover anchor={anchor} open={open} onClose={() => now(false)}>
          <div
            id={id}
            role="tooltip"
            onMouseEnter={() => now(true)}
            onMouseLeave={() => later(false, CLOSE_MS)}
            className="max-w-[min(36rem,calc(100vw-2rem))] px-3 py-2 text-left text-sm font-normal"
          >
            <p className="m-0 mb-1 font-mono text-xs text-muted">
              {entry.kind} {target.name} · {entry.file}
            </p>
            {entry.summary === '' ? null : <p className="m-0 mb-1.5 text-xs text-fg-secondary">{entry.summary}</p>}
            <pre className="site-code m-0 font-mono text-[12px] leading-snug break-words whitespace-pre-wrap">
              <Code code={entry.signature} />
            </pre>
          </div>
        </Popover>
      )}
    </span>
  )
}
