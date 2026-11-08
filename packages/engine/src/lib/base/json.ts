/**
 * Lossless JSON for anything that crosses a boundary (files, messages, traces, fixtures): the `Json` type and its
 * schema, number encoding that writes infinities and NaN as strings (the convention the Python oracle shares), and
 * `toJson` and `fromJson` for values that also hold -0, typed arrays, bigints and undefined fields.
 *
 * @index structure Lossless JSON for anything that crosses a boundary.
 */

import { z } from 'zod'
import { OmaiError } from './errors.ts'

/** A JSON value: what can cross a boundary (files, messages, traces) without loss. */
export type Json = null | boolean | number | string | readonly Json[] | { readonly [key: string]: Json }

/**
 * Schema for any JSON value, used inside larger boundary schemas.
 *
 * @returns Parses any JSON-compatible value.
 * @example
 * ```ts
 * import { JsonSchema } from 'omai-engine/base'
 * JsonSchema.parse({ a: [1, 'x', null] }) // => { a: [1, 'x', null] }
 * ```
 */
export const JsonSchema: z.ZodType<Json> = z.lazy(() =>
  z.union([z.null(), z.boolean(), z.number(), z.string(), z.array(JsonSchema), z.record(z.string(), JsonSchema)]),
)

/**
 * Encode a number for JSON, writing non-finite values as the strings `'inf'`, `'-inf'` and `'nan'` (the convention the
 * Python oracle uses too).
 *
 * @param x - A number.
 * @returns The number, or its string encoding when it is not finite.
 * @example
 * ```ts
 * import { encodeNumber } from 'omai-engine/base'
 * encodeNumber(-Infinity) // => '-inf'
 * encodeNumber(1.5) // => 1.5
 * ```
 */
export function encodeNumber(x: number): number | string {
  if (Number.isNaN(x)) return 'nan'
  if (x === Infinity) return 'inf'
  if (x === -Infinity) return '-inf'
  return x
}

/**
 * Decode a number written by {@link encodeNumber}.
 *
 * @param x - A number or one of `'inf'`, `'-inf'`, `'nan'`.
 * @returns The number.
 * @example
 * ```ts
 * import { decodeNumber } from 'omai-engine/base'
 * decodeNumber('inf') // => Infinity
 * ```
 */
export function decodeNumber(x: number | string): number {
  if (typeof x === 'number') return x
  if (x === 'inf') return Infinity
  if (x === '-inf') return -Infinity
  return Number.NaN
}

/** The typed-array classes {@link toJson} encodes. */
const TYPED_ARRAYS = {
  Float64Array,
  Float32Array,
  Int32Array,
  Int16Array,
  Int8Array,
  Uint32Array,
  Uint16Array,
  Uint8Array,
  Uint8ClampedArray,
} as const

type TypedArrayName = keyof typeof TYPED_ARRAYS

/** Values {@link toJson} accepts: JSON plus non-finite numbers, -0, typed arrays, bigints and undefined fields. */
export type Encodable =
  | null
  | undefined
  | boolean
  | number
  | bigint
  | string
  | ArrayLike<number>
  | readonly Encodable[]
  | { readonly [key: string]: Encodable }

function typedArrayName(x: object): TypedArrayName | undefined {
  if (!ArrayBuffer.isView(x)) return undefined
  for (const name of Object.keys(TYPED_ARRAYS) as TypedArrayName[]) if (x instanceof TYPED_ARRAYS[name]) return name
  return undefined
}

// Own-property writes, so a '__proto__' key is data rather than a prototype change (a plain assignment does for the rest).
function setOwn<T>(o: Record<string, T>, k: string, v: T): void {
  if (k === '__proto__') Object.defineProperty(o, k, { value: v, enumerable: true, writable: true, configurable: true })
  else o[k] = v
}

function encodeTagged(x: number): Json {
  if (Object.is(x, -0)) return { $number: '-0' }
  return Number.isFinite(x) ? x : { $number: encodeNumber(x) }
}

/**
 * Encode a value as JSON without loss. Plain JSON passes through; non-finite numbers and -0 become `{ $number: 'inf' }`
 * (also `'-inf'`, `'nan'`, `'-0'`), typed arrays `{ $typed: 'Float64Array', values: [...] }`, bigints
 * `{ $bigint: '123' }`, and objects with a key starting with `$` are wrapped as `{ $object: {...} }` so they cannot be
 * mistaken for tags. Undefined object fields are dropped, as `JSON.stringify` does. {@link fromJson} inverts it.
 *
 * @param x - The value.
 * @returns Its JSON encoding.
 * @throws OmaiError with code `value` for functions, symbols and class instances other than typed arrays.
 * @example
 * ```ts
 * import { toJson } from 'omai-engine/base'
 * toJson({ a: Infinity, b: new Float64Array([1, Number.NaN]) })
 * // => { a: { $number: 'inf' }, b: { $typed: 'Float64Array', values: [1, { $number: 'nan' }] } }
 * ```
 */
export function toJson(x: Encodable): Json {
  if (x === null || x === undefined) return null
  if (typeof x === 'boolean' || typeof x === 'string') return x
  if (typeof x === 'number') return encodeTagged(x)
  if (typeof x === 'bigint') return { $bigint: x.toString() }
  if (Array.isArray(x)) return (x as readonly Encodable[]).map(toJson)
  if (typeof x !== 'object') throw new OmaiError('value', `toJson: cannot encode a ${typeof x}`)
  const typed = typedArrayName(x)
  if (typed !== undefined) return { $typed: typed, values: Array.from(x as ArrayLike<number>, encodeTagged) }
  const proto = Object.getPrototypeOf(x) as unknown
  if (proto !== Object.prototype && proto !== null)
    throw new OmaiError('value', 'toJson: cannot encode a class instance')
  const out: Record<string, Json> = {}
  let tagLike = false
  for (const [k, v] of Object.entries(x as { readonly [key: string]: Encodable })) {
    if (v === undefined) continue
    if (k.startsWith('$')) tagLike = true
    setOwn(out, k, toJson(v))
  }
  return tagLike ? { $object: out } : out
}

/** Values {@link fromJson} returns. */
export type Decoded =
  | null
  | boolean
  | number
  | bigint
  | string
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Uint8ClampedArray
  | Decoded[]
  | { [key: string]: Decoded }

function decodeTagged(x: Json): number {
  if (typeof x === 'number') return x
  if (typeof x === 'object' && x !== null && !Array.isArray(x)) {
    const tag = (x as { readonly [key: string]: Json })['$number']
    if (tag === '-0') return -0
    if (tag === 'inf' || tag === '-inf' || tag === 'nan') return decodeNumber(tag)
  }
  throw new OmaiError('schema', `fromJson: expected a number, got ${JSON.stringify(x)}`)
}

/**
 * Decode JSON written by {@link toJson}.
 *
 * @param x - The JSON value.
 * @returns The value, with typed arrays, non-finite numbers, -0 and bigints restored.
 * @throws OmaiError with code `schema` for a malformed tag.
 * @example
 * ```ts
 * import { fromJson, toJson } from 'omai-engine/base'
 * fromJson(toJson([-Infinity, 'x', { $id: 1 }])) // => [-Infinity, 'x', { $id: 1 }]
 * fromJson(toJson(new Int32Array([4, 5]))) instanceof Int32Array // => true
 * ```
 */
export function fromJson(x: Json): Decoded {
  if (x === null || typeof x === 'boolean' || typeof x === 'number' || typeof x === 'string') return x
  if (Array.isArray(x)) return (x as readonly Json[]).map(fromJson)
  const o = x as { readonly [key: string]: Json }
  const keys = Object.keys(o)
  if (keys.length === 1 && keys[0] === '$number') return decodeTagged(o)
  if (keys.length === 1 && keys[0] === '$bigint') {
    const v = o['$bigint']
    if (typeof v !== 'string') throw new OmaiError('schema', 'fromJson: $bigint must be a string')
    return BigInt(v)
  }
  if (keys.length === 2 && '$typed' in o && 'values' in o) {
    const name = o['$typed']
    const values = o['values']
    if (typeof name !== 'string' || !(name in TYPED_ARRAYS) || !Array.isArray(values))
      throw new OmaiError('schema', `fromJson: bad typed array ${JSON.stringify(name)}`)
    return TYPED_ARRAYS[name as TypedArrayName].from((values as readonly Json[]).map(decodeTagged))
  }
  const body = keys.length === 1 && keys[0] === '$object' ? o['$object'] : o
  if (typeof body !== 'object' || body === null || Array.isArray(body))
    throw new OmaiError('schema', 'fromJson: $object must hold an object')
  const out: { [key: string]: Decoded } = {}
  for (const [k, v] of Object.entries(body as { readonly [key: string]: Json })) {
    if (body === o && k.startsWith('$')) throw new OmaiError('schema', `fromJson: unknown tag ${k}`)
    setOwn(out, k, fromJson(v))
  }
  return out
}
