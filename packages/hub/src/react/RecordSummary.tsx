/**
 * A compact fallback card for an exposition record that has no render view yet: the record's kind in words and one line
 * per field, numbers and short text as they are, arrays by their shape and range, objects by their keys.
 */

import type { ReactNode } from 'react'
import type { Exposition } from 'whitebox-engine/exposition'

const fmt = (x: number): string =>
  Number.isInteger(x) ? String(x) : Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toPrecision(4)

function shapeOf(v: unknown): number[] {
  const shape: number[] = []
  let cur: unknown = v
  while (Array.isArray(cur) || ArrayBuffer.isView(cur)) {
    const a = cur as ArrayLike<unknown>
    shape.push(a.length)
    const rows = Array.from(a)
    const len = (r: unknown): number =>
      Array.isArray(r) || ArrayBuffer.isView(r) ? (r as ArrayLike<unknown>).length : -1
    // Ragged rows (active sets, say): stop at this axis.
    if (rows.some((r) => len(r) !== len(rows[0]))) {
      shape.push(Number.NaN)
      break
    }
    cur = a.length > 0 ? a[0] : undefined
  }
  return shape
}

function numbersIn(v: unknown, out: number[] = [], cap = 100_000): number[] {
  if (out.length >= cap) return out
  if (typeof v === 'number') out.push(v)
  else if (Array.isArray(v) || ArrayBuffer.isView(v))
    for (const x of Array.from(v as ArrayLike<unknown>)) numbersIn(x, out, cap)
  return out
}

/**
 * One line describing a record field: a number or short text as it is, an array by its shape (and the range of its
 * numbers), an object by its keys.
 *
 * @param v - The field's value.
 * @returns The description.
 */
export function describeField(v: unknown): string {
  if (typeof v === 'number') return fmt(v)
  if (typeof v === 'string') return v.length > 48 ? `${v.slice(0, 45)}…` : v
  if (typeof v === 'boolean' || v === null || v === undefined) return String(v)
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const shape = shapeOf(v)
    const nums = numbersIn(v).filter(Number.isFinite)
    const items = (v as ArrayLike<unknown>).length
    const first = items > 0 ? (v as ArrayLike<unknown>)[0] : undefined
    if (typeof first === 'string') {
      const all = Array.from(v as ArrayLike<string>)
      const text = all.slice(0, 4).join(', ')
      return items > 4 ? `${text}, … (${items})` : text
    }
    const dims = shape.map((n) => (Number.isNaN(n) ? '…' : String(n))).join(' × ')
    if (nums.length === 0) return typeof first === 'object' && first !== null ? `${items} items` : `[${dims}]`
    if (items <= 4 && shape.length === 1) return `[${nums.map(fmt).join(', ')}]`
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (const x of nums) {
      lo = Math.min(lo, x)
      hi = Math.max(hi, x)
    }
    return `[${dims}], from ${fmt(lo)} to ${fmt(hi)}`
  }
  if (typeof v === 'object') {
    const keys = Object.keys(v as object)
    return `{ ${keys.slice(0, 5).join(', ')}${keys.length > 5 ? ', …' : ''} }`
  }
  return String(v)
}

/** A record kind as words: `regularisationPath` to `regularisation path`. */
export function kindLabel(kind: string): string {
  return kind.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase()
}

/**
 * A compact card for an exposition record that has no view yet: its kind and each field in a line (shapes, ranges,
 * small values).
 *
 * @param props - The record and an optional title.
 * @returns The card.
 */
export function RecordSummary(props: { readonly record: Exposition; readonly title?: string }): ReactNode {
  const { record, title } = props
  const fields = Object.entries(record).filter(([k]) => k !== 'kind')
  return (
    <section className="rounded-lg border border-(--wb-grid) bg-(--wb-surface) text-(--wb-fg)">
      <header className="flex items-baseline justify-between gap-3 border-b border-(--wb-grid) px-3 py-2">
        <span className="truncate text-sm font-medium">{title ?? kindLabel(record.kind)}</span>
        <code className="shrink-0 font-mono text-xs text-(--wb-muted)">{record.kind}</code>
      </header>
      <dl className="m-0 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1 px-3 py-2 text-xs">
        {fields.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="font-mono text-(--wb-muted)">{k}</dt>
            <dd className="m-0 min-w-0 break-words tabular-nums">{describeField(v)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
