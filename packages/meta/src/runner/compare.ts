/**
 * Comparing what an example gave with what its `// =>` marker says, to the precision the marker states: a number
 * written with d decimals matches any value that rounds to it at d decimals, an integer or a computed expected value
 * matches to about twelve significant digits. Tensors and typed arrays compare as nested arrays.
 */

/** A number written in an expected literal, with the precision it states. */
export interface StatedNumber {
  readonly kind: 'number'
  readonly value: number
  /** Half a unit of the last written digit (`0.05` for `1.2`), or `0` for an integer or a special value. */
  readonly halfUnit: number
}

/** An expected literal as a tree, so each number keeps the precision it was written with. */
export type Literal =
  | StatedNumber
  | { readonly kind: 'atom'; readonly value: string | boolean | null | undefined }
  | { readonly kind: 'array'; readonly items: readonly Literal[] }
  | { readonly kind: 'object'; readonly fields: Readonly<Record<string, Literal>> }

/** The outcome of a comparison. */
export interface Comparison {
  readonly pass: boolean
  /** Where and how the values differ, when they do. */
  readonly message: string
}

/** Relative tolerance for numbers without a stated precision (integers, computed expected values). */
export const DEFAULT_RTOL = 1e-12

const TYPED =
  /^(?:Float64|Float32|Int32|Int16|Int8|Uint32|Uint16|Uint8|Uint8Clamped|BigInt64|BigUint64)Array(?:\(\d+\))?\s*\[/

class LiteralReader {
  private i = 0
  private readonly text: string

  constructor(text: string) {
    this.text = text
  }

  read(): Literal | undefined {
    const value = this.value()
    this.space()
    return value !== undefined && this.i === this.text.length ? value : undefined
  }

  private space(): void {
    while (this.i < this.text.length && /\s/.test(this.text[this.i] as string)) this.i++
  }

  private value(): Literal | undefined {
    this.space()
    // A typed array as Node prints it: `Float64Array [1, 2]` or `Float64Array(2) [1, 2]`.
    const typed = TYPED.exec(this.text.slice(this.i))
    if (typed !== null) this.i += typed[0].length - 1
    const c = this.text[this.i]
    if (c === '[') return this.array()
    if (c === '{') return this.object()
    if (c === "'" || c === '"') {
      const s = this.string()
      return s === undefined ? undefined : { kind: 'atom', value: s }
    }
    return this.word()
  }

  private array(): Literal | undefined {
    this.i++
    const items: Literal[] = []
    for (;;) {
      this.space()
      if (this.text[this.i] === ']') {
        this.i++
        return { kind: 'array', items }
      }
      const item = this.value()
      if (item === undefined) return undefined
      items.push(item)
      this.space()
      if (this.text[this.i] === ',') this.i++
      else if (this.text[this.i] !== ']') return undefined
    }
  }

  private object(): Literal | undefined {
    this.i++
    const fields: Record<string, Literal> = {}
    for (;;) {
      this.space()
      if (this.text[this.i] === '}') {
        this.i++
        return { kind: 'object', fields }
      }
      const c = this.text[this.i]
      let key: string | undefined
      if (c === "'" || c === '"') key = this.string()
      else {
        const m = /^[A-Za-z_$][\w$]*|^\d+/.exec(this.text.slice(this.i))
        if (m !== null) {
          key = m[0]
          this.i += key.length
        }
      }
      if (key === undefined) return undefined
      this.space()
      if (this.text[this.i] !== ':') return undefined
      this.i++
      const value = this.value()
      if (value === undefined) return undefined
      fields[key] = value
      this.space()
      if (this.text[this.i] === ',') this.i++
      else if (this.text[this.i] !== '}') return undefined
    }
  }

  private string(): string | undefined {
    const quote = this.text[this.i]
    let out = ''
    for (this.i++; this.i < this.text.length; this.i++) {
      const c = this.text[this.i] as string
      if (c === quote) {
        this.i++
        try {
          return JSON.parse(`"${out}"`) as string
        } catch {
          return undefined
        }
      }
      if (c === '\\') {
        const next = this.text[++this.i] ?? ''
        out += next === "'" ? "'" : `\\${next}`
      } else out += c === '"' ? '\\"' : c
    }
    return undefined
  }

  private word(): Literal | undefined {
    const rest = this.text.slice(this.i)
    const num = /^[-+]?(?:Infinity|NaN|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)/.exec(rest)
    if (num !== null) {
      const text = num[0]
      if (/^[\w$]/.test(rest.slice(text.length))) return undefined
      this.i += text.length
      return statedNumber(text)
    }
    const atom = /^(true|false|null|undefined)(?![\w$])/.exec(rest)
    if (atom === null) return undefined
    this.i += atom[0].length
    const value = { true: true, false: false, null: null, undefined }[
      atom[0] as 'true' | 'false' | 'null' | 'undefined'
    ]
    return { kind: 'atom', value }
  }
}

/**
 * A number as written, with half a unit of its last digit: `1.25` states two decimals (`0.005`), `1.5e-3` states
 * four, and an integer such as `3` states none (it means exactly 3, up to rounding error).
 *
 * @param text - A numeric literal.
 * @returns The value and its stated half unit.
 */
export function statedNumber(text: string): StatedNumber {
  const value = Number(text)
  const m = /^[-+]?(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?$/.exec(text)
  if (m === null || !Number.isFinite(value)) return { kind: 'number', value, halfUnit: 0 }
  const decimals = (m[2] ?? '').length
  const exponent = Number(m[3] ?? '0')
  if (decimals === 0 && exponent >= 0) return { kind: 'number', value, halfUnit: 0 }
  return { kind: 'number', value, halfUnit: 0.5 * 10 ** (exponent - decimals) }
}

/**
 * Read an expected value written as a literal (numbers, strings, booleans, `null`, `undefined`, arrays, objects, and
 * typed arrays in Node's notation, `Float64Array [1, 2]`, which compare as arrays).
 *
 * @param text - The text after `// =>`.
 * @returns The literal tree, or `undefined` when the text is an expression rather than a literal.
 */
export function parseLiteral(text: string): Literal | undefined {
  return new LiteralReader(text.trim()).read()
}

/**
 * A value as plain data for comparison: tensors (objects with `kind: 'tensor'`, a `shape` and `data`) become nested
 * arrays, typed arrays become arrays; everything else is returned as is.
 *
 * @param value - Any value.
 * @returns The plain value.
 */
export function plain(value: unknown): unknown {
  if (ArrayBuffer.isView(value) && !(value instanceof DataView))
    return Array.from(value as unknown as ArrayLike<number>)
  if (typeof value !== 'object' || value === null) return value
  const t = value as { kind?: unknown; shape?: unknown; data?: unknown; dtype?: unknown }
  if (t.kind === 'tensor' && Array.isArray(t.shape) && ArrayBuffer.isView(t.data)) {
    const shape = t.shape as number[]
    const data = t.data as unknown as ArrayLike<number>
    const bool = t.dtype === 'bool'
    const build = (depth: number, offset: number): unknown => {
      if (depth === shape.length) return bool ? data[offset] !== 0 : data[offset]
      const inner = shape.slice(depth + 1).reduce((a, b) => a * b, 1)
      return Array.from({ length: shape[depth] ?? 0 }, (_, i) => build(depth + 1, offset + i * inner))
    }
    return build(0, 0)
  }
  return value
}

function closeEnough(actual: number, expected: number, halfUnit: number): boolean {
  if (Number.isNaN(expected)) return Number.isNaN(actual)
  if (!Number.isFinite(expected)) return actual === expected
  const slack = DEFAULT_RTOL * Math.max(1, Math.abs(expected))
  return Math.abs(actual - expected) <= halfUnit * (1 + 1e-9) + slack
}

function show(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'number' || typeof value === 'boolean' || value == null) return String(value)
  if (Array.isArray(value)) return `array(${value.length})`
  return Object.prototype.toString.call(value)
}

function fieldsOf(value: object): string[] {
  return Object.keys(value).filter((k) => (value as Record<string, unknown>)[k] !== undefined)
}

function compareTo(actual: unknown, expected: Literal | unknown, literal: boolean, path: string): string | null {
  const a = plain(actual)
  const where = path === '' ? '' : ` at ${path}`
  if (literal) {
    const e = expected as Literal
    switch (e.kind) {
      case 'number': {
        const n = typeof a === 'bigint' ? Number(a) : a
        if (typeof n !== 'number') return `expected ${e.value}${where}, got ${show(a)}`
        return closeEnough(n, e.value, e.halfUnit) ? null : `expected ${e.value}${where}, got ${n}`
      }
      case 'atom':
        return Object.is(a, e.value) ? null : `expected ${show(e.value)}${where}, got ${show(a)}`
      case 'array': {
        if (!Array.isArray(a)) return `expected an array${where}, got ${show(a)}`
        if (a.length !== e.items.length) return `expected ${e.items.length} items${where}, got ${a.length}`
        for (let i = 0; i < a.length; i++) {
          const r = compareTo(a[i], e.items[i], true, `${path}[${i}]`)
          if (r !== null) return r
        }
        return null
      }
      case 'object': {
        if (typeof a !== 'object' || a === null) return `expected an object${where}, got ${show(a)}`
        const want = Object.keys(e.fields).filter(
          (k) => !(e.fields[k]?.kind === 'atom' && e.fields[k]?.value === undefined),
        )
        const have = fieldsOf(a)
        const missing = want.filter((k) => !have.includes(k))
        const extra = have.filter((k) => !(k in e.fields))
        if (missing.length > 0) return `missing field ${missing[0]}${where}`
        if (extra.length > 0) return `unexpected field ${extra[0]}${where}`
        for (const k of want) {
          const r = compareTo((a as Record<string, unknown>)[k], e.fields[k], true, path === '' ? k : `${path}.${k}`)
          if (r !== null) return r
        }
        return null
      }
    }
  }
  const e = plain(expected)
  if (typeof e === 'number') {
    const n = typeof a === 'bigint' ? Number(a) : a
    if (typeof n !== 'number') return `expected ${e}${where}, got ${show(a)}`
    return closeEnough(n, e, 0) ? null : `expected ${e}${where}, got ${n}`
  }
  if (typeof e !== 'object' || e === null || typeof a !== 'object' || a === null)
    return Object.is(a, e) ? null : `expected ${show(e)}${where}, got ${show(a)}`
  if (Array.isArray(e)) {
    if (!Array.isArray(a)) return `expected an array${where}, got ${show(a)}`
    if (a.length !== e.length) return `expected ${e.length} items${where}, got ${a.length}`
    for (let i = 0; i < e.length; i++) {
      const r = compareTo(a[i], e[i], false, `${path}[${i}]`)
      if (r !== null) return r
    }
    return null
  }
  if (e instanceof Map || e instanceof Set) {
    if (!(a instanceof Map || a instanceof Set)) return `expected a ${e.constructor.name}${where}, got ${show(a)}`
    return compareTo([...a], [...e], false, path)
  }
  if (a === e) return null
  const want = fieldsOf(e)
  const have = fieldsOf(a)
  const missing = want.filter((k) => !have.includes(k))
  const extra = have.filter((k) => !want.includes(k))
  if (missing.length > 0) return `missing field ${missing[0]}${where}`
  if (extra.length > 0) return `unexpected field ${extra[0]}${where}`
  for (const k of want) {
    const r = compareTo(
      (a as Record<string, unknown>)[k],
      (e as Record<string, unknown>)[k],
      false,
      path === '' ? k : `${path}.${k}`,
    )
    if (r !== null) return r
  }
  return null
}

/**
 * Compare an example's value with its expectation. When the expected text is a literal, each number in it is matched
 * to the precision it is written with; otherwise the evaluated expected value is matched to {@link DEFAULT_RTOL}.
 * Objects must have the same defined fields; tensors and typed arrays compare as nested arrays.
 *
 * @param actual - What the expression gave.
 * @param expected - The evaluated expected value.
 * @param text - The expected value as written, read with {@link parseLiteral} when it is a literal.
 * @returns Whether they match, and where they first differ.
 */
export function compareValues(actual: unknown, expected: unknown, text?: string): Comparison {
  const literal = text === undefined ? undefined : parseLiteral(text)
  const message = literal === undefined ? compareTo(actual, expected, false, '') : compareTo(actual, literal, true, '')
  return { pass: message === null, message: message ?? '' }
}
