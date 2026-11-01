/**
 * Generic state serialisation for checkpoints: `stateToJson` turns an algorithm state made of plain objects, arrays,
 * numbers, strings, keys, tensors and typed arrays into JSON, and `stateFromJson` turns it back, so an algorithm's
 * `serialise` and `deserialise` are one line each. Maps, sets, functions and class instances are refused.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { isTensor, type TensorJson, tensorFromJson, tensorToJson } from '../tensor/index.ts'

const TYPED = {
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

type TypedName = keyof typeof TYPED

function typedName(x: ArrayBufferView): TypedName | undefined {
  for (const name of Object.keys(TYPED) as TypedName[]) if (x instanceof TYPED[name]) return name
  return undefined
}

/**
 * An algorithm state as JSON: tensors become `{ "$tensor": ... }`, typed arrays `{ "$typed": name, data }`, and plain
 * objects and arrays recurse; `undefined` fields are dropped.
 *
 * @param x - The state.
 * @returns Its JSON.
 * @throws MdmlError with code `value` for a map, set, function or class instance.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { stateToJson } from 'mdml-engine/trace'
 * stateToJson({ n: 2, q: Float64Array.of(1, 2) }) // => { n: 2, q: { $typed: 'Float64Array', data: [1, 2] } }
 * ```
 */
export function stateToJson(x: unknown): Json {
  if (x === null || typeof x === 'number' || typeof x === 'string' || typeof x === 'boolean') return x
  if (x === undefined) return null
  if (isTensor(x)) return { $tensor: tensorToJson(x) as unknown as Json }
  if (ArrayBuffer.isView(x) && !(x instanceof DataView)) {
    const name = typedName(x)
    if (name === undefined) throw new MdmlError('value', 'stateToJson: unsupported typed array')
    return { $typed: name, data: Array.from(x as unknown as ArrayLike<number>) }
  }
  if (Array.isArray(x)) return x.map((v) => stateToJson(v))
  if (typeof x === 'object') {
    const proto = Object.getPrototypeOf(x) as object | null
    if (proto !== null && proto !== Object.prototype)
      throw new MdmlError('value', `stateToJson: cannot serialise a ${x.constructor?.name ?? 'class'} instance`)
    const out: Record<string, Json> = {}
    for (const [k, v] of Object.entries(x as Record<string, unknown>)) if (v !== undefined) out[k] = stateToJson(v)
    return out
  }
  throw new MdmlError('value', `stateToJson: cannot serialise a ${typeof x}`)
}

/**
 * The inverse of {@link stateToJson}.
 *
 * @param json - The JSON.
 * @returns The state.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { stateFromJson, stateToJson } from 'mdml-engine/trace'
 * const back = stateFromJson(stateToJson({ q: Float64Array.of(1, 2) })) as { q: Float64Array }
 * back.q instanceof Float64Array // => true
 * ```
 */
export function stateFromJson(json: Json): unknown {
  if (json === null || typeof json !== 'object') return json
  if (Array.isArray(json)) return json.map((v) => stateFromJson(v))
  const o = json as { readonly [key: string]: Json }
  if ('$tensor' in o && Object.keys(o).length === 1) return tensorFromJson(o['$tensor'] as unknown as TensorJson)
  if ('$typed' in o && 'data' in o && Object.keys(o).length === 2) {
    const name = o['$typed'] as TypedName
    const ctor = TYPED[name] as { from(data: readonly number[]): ArrayBufferView } | undefined
    if (ctor === undefined) throw new MdmlError('value', `stateFromJson: unknown typed array '${String(name)}'`)
    return ctor.from(o['data'] as readonly number[])
  }
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(o)) out[k] = stateFromJson(v)
  return out
}

/**
 * A number as JSON that survives a round trip through `JSON.stringify`: finite numbers stay numbers, `NaN` and the
 * infinities become the strings `'NaN'`, `'Infinity'` and `'-Infinity'` (plain `JSON.stringify` turns all three into
 * `null`). For the fields of an algorithm state that may be unset or unbounded.
 *
 * @param x - The number.
 * @returns A number or one of the three strings.
 * @example
 * ```ts
 * import { numberToJson } from 'mdml-engine/trace'
 * [numberToJson(1.5), numberToJson(Number.NaN), numberToJson(-Infinity)] // => [1.5, 'NaN', '-Infinity']
 * ```
 */
export function numberToJson(x: number): Json {
  return Number.isFinite(x) ? x : Number.isNaN(x) ? 'NaN' : x > 0 ? 'Infinity' : '-Infinity'
}

/**
 * The inverse of {@link numberToJson}; `null` (what a plain `JSON.stringify` makes of `NaN`) reads as `NaN`.
 *
 * @param j - A number, one of the three strings, or `null`.
 * @returns The number.
 * @example
 * ```ts
 * import { numberFromJson } from 'mdml-engine/trace'
 * [numberFromJson('Infinity'), numberFromJson(2), Number.isNaN(numberFromJson(null))] // => [Infinity, 2, true]
 * ```
 */
export function numberFromJson(j: Json | undefined): number {
  return typeof j === 'number' ? j : j === null || j === undefined ? Number.NaN : Number(j)
}

/**
 * A list of numbers as JSON with {@link numberToJson} applied to each.
 *
 * @param xs - The numbers.
 * @returns The JSON array.
 * @example
 * ```ts
 * import { numbersToJson } from 'mdml-engine/trace'
 * numbersToJson([1, Infinity]) // => [1, 'Infinity']
 * ```
 */
export function numbersToJson(xs: ArrayLike<number>): Json {
  return Array.from(xs, numberToJson)
}

/**
 * The inverse of {@link numbersToJson}.
 *
 * @param j - The JSON array.
 * @returns The numbers.
 * @example
 * ```ts
 * import { numbersFromJson } from 'mdml-engine/trace'
 * numbersFromJson([1, 'Infinity']) // => [1, Infinity]
 * ```
 */
export function numbersFromJson(j: Json | undefined): number[] {
  return Array.isArray(j) ? j.map((v) => numberFromJson(v)) : []
}
