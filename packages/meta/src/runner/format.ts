/**
 * Display of values beside example code: numbers to twelve significant digits, tensors as nested arrays with their
 * shape when large, strings quoted, objects by field, with limits on depth and length.
 */
import { plain } from './compare.ts'

/** Limits for {@link formatValue}. */
export interface FormatOptions {
  /** Array items shown before an ellipsis (default 20). */
  readonly maxItems?: number
  /** Nesting shown before `…` (default 4). */
  readonly maxDepth?: number
  /** Significant digits of non-integers (default 12). */
  readonly digits?: number
}

function number(v: number, digits: number): string {
  if (Number.isInteger(v) || !Number.isFinite(v)) return Object.is(v, -0) ? '-0' : String(v)
  return String(Number(v.toPrecision(digits)))
}

function key(k: string): string {
  return /^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k)
}

function format(value: unknown, depth: number, o: Required<FormatOptions>): string {
  const v = plain(value)
  if (typeof v === 'number') return number(v, o.digits)
  if (typeof v === 'string') return `'${v.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')}'`
  if (typeof v === 'bigint') return `${v}n`
  if (v === null || v === undefined || typeof v === 'boolean') return String(v)
  if (typeof v === 'function') return `[function ${v.name || 'anonymous'}]`
  if (typeof v === 'symbol') return v.toString()
  if (depth >= o.maxDepth) return Array.isArray(v) ? '[…]' : '{…}'
  if (Array.isArray(v)) {
    const shown = v.slice(0, o.maxItems).map((x) => format(x, depth + 1, o))
    if (v.length > o.maxItems) shown.push(`… ${v.length - o.maxItems} more`)
    return `[${shown.join(', ')}]`
  }
  if (v instanceof Map) {
    const entries = [...v.entries()].slice(0, o.maxItems)
    return `Map(${v.size}) {${entries.map(([k, x]) => ` ${format(k, depth + 1, o)} => ${format(x, depth + 1, o)}`).join(',')} }`
  }
  if (v instanceof Set) return `Set(${v.size}) ${format([...v], depth, o)}`
  if (v instanceof Error) return `${v.name}: ${v.message}`
  const fields = Object.entries(v as object).filter(([, x]) => x !== undefined && typeof x !== 'function')
  const name = (v as object).constructor?.name
  const tag = name !== undefined && name !== 'Object' && name !== '' ? `${name} ` : ''
  if (fields.length === 0) return `${tag}{}`
  const shown = fields.slice(0, o.maxItems).map(([k, x]) => `${key(k)}: ${format(x, depth + 1, o)}`)
  if (fields.length > o.maxItems) shown.push(`… ${fields.length - o.maxItems} more`)
  return `${tag}{ ${shown.join(', ')} }`
}

/**
 * A value as one line of text, in the notation examples write expectations in.
 *
 * @param value - Any value.
 * @param options - Limits on length, depth and digits.
 * @returns The text.
 */
export function formatValue(value: unknown, options: FormatOptions = {}): string {
  return format(value, 0, { maxItems: 20, maxDepth: 4, digits: 12, ...options })
}
