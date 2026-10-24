import { Fragment, type ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { type Docs, symbolPath } from './data.ts'

/** Inline marks of a doc comment: `code`, {@link name} and **strong**. */
function inline(text: string, docs: Docs, near: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\{@link\s+[^}]+\}|\*\*[^*]+\*\*)/)
  let offset = 0
  return parts.map((part) => {
    const i = `${offset}`
    offset += part.length
    if (part.startsWith('`'))
      return (
        <code key={i} className="rounded bg-neutral-100 px-1 py-0.5 font-mono text-[0.85em] dark:bg-neutral-800">
          {part.slice(1, -1)}
        </code>
      )
    if (part.startsWith('{@link')) {
      const [target = '', ...label] = part
        .slice(6, -1)
        .trim()
        .split(/\s*\|\s*|\s+/)
      const s = docs.find(target.replace(/\(\)$/, ''), near)
      const text = label.length > 0 ? label.join(' ') : target
      return s === undefined ? (
        <code key={i} className="font-mono text-[0.85em]">
          {text}
        </code>
      ) : (
        <Link
          key={i}
          to={symbolPath(s.file.path, s.name)}
          className="font-mono text-[0.85em] text-sky-700 hover:underline dark:text-sky-400"
        >
          {text}
        </Link>
      )
    }
    if (part.startsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>
    return <Fragment key={i}>{part}</Fragment>
  })
}

type Block = { kind: 'p' | 'li' | 'code'; text: string; at: number }

function blocks(text: string): Block[] {
  const out: Block[] = []
  let fence: string[] | null = null
  let open: Block | null = null
  const lines = text.split('\n')
  lines.forEach((line, at) => {
    if (/^\s*```/.test(line)) {
      if (fence !== null) out.push({ kind: 'code', text: fence.join('\n'), at })
      fence = fence === null ? [] : null
      open = null
    } else if (fence !== null) fence.push(line)
    else if (line.trim() === '') open = null
    else if (/^\s*[-*] /.test(line)) {
      open = { kind: 'li', text: line.replace(/^\s*[-*] /, ''), at }
      out.push(open)
    } else if (open !== null) open.text += ` ${line.trim()}`
    else {
      open = { kind: 'p', text: line.trim(), at }
      out.push(open)
    }
  })
  return out
}

/** One line of doc text with its inline marks. */
export function DocInline(props: { readonly text: string; readonly docs: Docs; readonly near: string }): ReactNode {
  return <>{inline(props.text, props.docs, props.near)}</>
}

/** Doc comment prose: paragraphs, lists and code, with inline marks. */
export function DocText(props: {
  readonly text: string
  readonly docs: Docs
  readonly near: string
  readonly className?: string
}): ReactNode {
  const all = blocks(props.text)
  const out: ReactNode[] = []
  for (let i = 0; i < all.length; i++) {
    const b = all[i] as Block
    if (b.kind === 'code')
      out.push(
        <pre key={b.at} className="overflow-x-auto rounded bg-neutral-100 p-3 font-mono text-xs dark:bg-neutral-900">
          {b.text}
        </pre>,
      )
    else if (b.kind === 'p') out.push(<p key={b.at}>{inline(b.text, props.docs, props.near)}</p>)
    else {
      const items: Block[] = []
      while (all[i]?.kind === 'li') items.push(all[i++] as Block)
      i--
      out.push(
        <ul key={b.at} className="list-disc space-y-1 pl-5">
          {items.map((it) => (
            <li key={it.at}>{inline(it.text, props.docs, props.near)}</li>
          ))}
        </ul>,
      )
    }
  }
  return <div className={props.className ?? 'space-y-3'}>{out}</div>
}
