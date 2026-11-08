/**
 * Generic state serialisation for checkpoints: `stateToJson` turns an algorithm state made of plain objects, arrays,
 * numbers, strings, keys, tensors and typed arrays into JSON, and `stateFromJson` turns it back, so an algorithm's
 * `serialise` and `deserialise` are one line each. Maps, sets, functions and class instances are refused. The codec is
 * the engine's one, `toJson` and `fromJson` of `base/json.ts` (non-finite numbers and -0 survive a trip through JSON
 * text, typed arrays are `{ $typed, values }`); the only thing added here is the tensor, which `base` does not know.
 */

import { decodeNumber, type Encodable, encodeNumber, fromJson, type Json, toJson } from '../base/index.ts'
import { isTensor, type TensorJson, tensorFromJson, tensorToJson } from '../tensor/index.ts'

function isPlain(x: unknown): x is Readonly<Record<string, unknown>> {
  if (typeof x !== 'object' || x === null) return false
  const proto = Object.getPrototypeOf(x) as object | null
  return proto === null || proto === Object.prototype
}

const isTag = (x: unknown): boolean => isPlain(x) && Object.keys(x).length === 1 && Object.hasOwn(x, '$tensor')

/** Whether a state holds a tensor (`found`) or a `$tensor` object (`isTag`) anywhere: a walk that allocates nothing. */
function holds(x: unknown, found: (x: unknown) => boolean): boolean {
  if (found(x)) return true
  if (Array.isArray(x)) return x.some((v) => holds(v, found))
  return isPlain(x) && Object.values(x).some((v) => holds(v, found))
}

/** Tensors as `{ $tensor: ... }` objects, anywhere in a state. */
function tensorsToTags(x: unknown): unknown {
  if (isTensor(x)) return { $tensor: tensorToJson(x) }
  if (Array.isArray(x)) return x.map(tensorsToTags)
  if (isPlain(x)) return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, tensorsToTags(v)]))
  return x
}

/** The inverse of {@link tensorsToTags}, over what `fromJson` returns. */
function tagsToTensors(x: unknown): unknown {
  if (isTag(x)) return tensorFromJson((x as { readonly $tensor: TensorJson }).$tensor)
  if (Array.isArray(x)) return x.map(tagsToTensors)
  if (isPlain(x)) return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, tagsToTensors(v)]))
  return x
}

/**
 * An algorithm state as JSON, in the encoding of `toJson`: non-finite numbers and -0 are tagged, typed arrays become
 * `{ $typed, values }`, plain objects and arrays recurse, `undefined` fields are dropped, and tensors become
 * `{ $tensor: ... }` objects (carried, like any object with a `$` key, under `$object`).
 *
 * @param x - The state.
 * @returns Its JSON.
 * @throws OmaiError with code `value` for a map, set, function or class instance.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { stateToJson } from 'omai-engine/trace'
 * stateToJson({ n: 2, q: Float64Array.of(1, 2) }) // => { n: 2, q: { $typed: 'Float64Array', values: [1, 2] } }
 * ```
 */
export function stateToJson(x: unknown): Json {
  return toJson((holds(x, isTensor) ? tensorsToTags(x) : x) as Encodable)
}

/**
 * The inverse of {@link stateToJson}.
 *
 * @param json - The JSON.
 * @returns The state.
 * @throws OmaiError with code `schema` for a malformed tag.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { stateFromJson, stateToJson } from 'omai-engine/trace'
 * const back = stateFromJson(stateToJson({ q: Float64Array.of(1, 2) })) as { q: Float64Array }
 * back.q instanceof Float64Array // => true
 * ```
 */
export function stateFromJson(json: Json): unknown {
  const state = fromJson(hasLegacy(json) ? fromLegacy(json) : json)
  return holds(state, isTag) ? tagsToTensors(state) : state
}

type JsonObject = { readonly [key: string]: Json }

/** Whether a JSON holds a state written before the codec was shared: a bare `{ $tensor }` or `{ $typed, data }`. */
function hasLegacy(j: Json): boolean {
  if (j === null || typeof j !== 'object') return false
  if (Array.isArray(j)) return j.some(hasLegacy)
  const o = j as JsonObject
  const keys = Object.keys(o)
  if (keys.length === 1 && keys[0] === '$object') return false
  if (keys.length === 1 && keys[0] === '$tensor') return true
  if (keys.length === 2 && '$typed' in o && 'data' in o) return true
  return Object.values(o).some(hasLegacy)
}

/** The current spelling of a state written before the codec was shared (tapes saved earlier still load). */
function fromLegacy(j: Json): Json {
  if (j === null || typeof j !== 'object') return j
  if (Array.isArray(j)) return j.map(fromLegacy)
  const o = j as JsonObject
  const keys = Object.keys(o)
  if (keys.length === 1 && keys[0] === '$object') return o
  if (keys.length === 1 && keys[0] === '$tensor') return { $object: o }
  if (keys.length === 2 && '$typed' in o && 'data' in o)
    return { $typed: o['$typed'] as Json, values: o['data'] as Json }
  return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, fromLegacy(v)]))
}

/**
 * A number as JSON that survives a round trip through `JSON.stringify`: finite numbers stay numbers, `NaN` and the
 * infinities become the strings of `encodeNumber` (`'nan'`, `'inf'`, `'-inf'`; plain `JSON.stringify` turns all three
 * into `null`). For the fields of an algorithm state that may be unset or unbounded.
 *
 * @param x - The number.
 * @returns A number or one of the three strings.
 * @example
 * ```ts
 * import { numberToJson } from 'omai-engine/trace'
 * [numberToJson(1.5), numberToJson(Number.NaN), numberToJson(-Infinity)] // => [1.5, 'nan', '-inf']
 * ```
 */
export function numberToJson(x: number): Json {
  return encodeNumber(x)
}

/**
 * The inverse of {@link numberToJson}; `null` (what a plain `JSON.stringify` makes of `NaN`) reads as `NaN`, and the
 * spellings `'Infinity'`, `'-Infinity'` and `'NaN'`, which some states still write themselves, read as what they say.
 *
 * @param j - A number, one of the strings, or `null`.
 * @returns The number.
 * @example
 * ```ts
 * import { numberFromJson } from 'omai-engine/trace'
 * [numberFromJson('inf'), numberFromJson(2), Number.isNaN(numberFromJson(null))] // => [Infinity, 2, true]
 * ```
 */
export function numberFromJson(j: Json | undefined): number {
  if (typeof j === 'number') return j
  if (j === null || j === undefined) return Number.NaN
  return j === 'inf' || j === '-inf' || j === 'nan' ? decodeNumber(j) : Number(j)
}

/**
 * A list of numbers as JSON with {@link numberToJson} applied to each.
 *
 * @param xs - The numbers.
 * @returns The JSON array.
 * @example
 * ```ts
 * import { numbersToJson } from 'omai-engine/trace'
 * numbersToJson([1, Infinity]) // => [1, 'inf']
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
 * import { numbersFromJson } from 'omai-engine/trace'
 * numbersFromJson([1, 'inf']) // => [1, Infinity]
 * ```
 */
export function numbersFromJson(j: Json | undefined): number[] {
  return Array.isArray(j) ? j.map((v) => numberFromJson(v)) : []
}
