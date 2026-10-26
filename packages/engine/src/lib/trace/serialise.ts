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
