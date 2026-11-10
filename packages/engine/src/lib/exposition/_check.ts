// Shape checks shared by the record constructors (private to the domain).
import { OmaiError } from '../base/index.ts'

/** Fail unless `n` equals `expected`, naming the record and the field. */
export function sameLength(kind: string, field: string, n: number, expected: number): void {
  if (n !== expected) throw new OmaiError('shape', `${kind}: ${field} has ${n} entries, expected ${expected}`)
}

/** Fail unless every row of a matrix has `width` entries. */
export function rowsOf(kind: string, field: string, rows: readonly (readonly number[])[], width: number): void {
  for (const [i, row] of rows.entries()) {
    if (row.length !== width)
      throw new OmaiError('shape', `${kind}: ${field}[${i}] has ${row.length} entries, expected ${width}`)
  }
}

/** How {@link seal} treats the numbers of a record. */
export interface SealOptions {
  /** Names of fields (the nearest property name on the path) that may hold NaN: a missing value shown as a gap. */
  readonly gaps?: readonly string[]
  /** Names of fields that may hold an infinity: an open bound, an unbounded limit. */
  readonly unbounded?: readonly string[]
  /** Names of fields that are copied by reference and not looked into (scenes, which the scene builders own). */
  readonly opaque?: readonly string[]
}

const isPlain = (x: object): boolean => {
  const proto = Object.getPrototypeOf(x)
  return proto === Object.prototype || proto === null
}

function check(kind: string, path: string, key: string, v: number, o: SealOptions): void {
  if (Number.isFinite(v)) return
  const allowed = Number.isNaN(v) ? o.gaps : o.unbounded
  if (allowed?.includes(key) === true) return
  throw new OmaiError('value', `${kind}: ${path} is ${v}, expected a finite number`)
}

function seal1(kind: string, path: string, key: string, value: unknown, o: SealOptions): unknown {
  if (typeof value === 'number') {
    check(kind, path, key, value, o)
    return value
  }
  if (Array.isArray(value)) {
    const out: unknown[] = new Array(value.length)
    for (let i = 0; i < value.length; i++) {
      const v: unknown = value[i]
      if (typeof v === 'number') {
        if (!Number.isFinite(v)) check(kind, `${path}[${i}]`, key, v, o)
        out[i] = v
      } else out[i] = seal1(kind, `${path}[${i}]`, key, v, o)
    }
    return out
  }
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const xs = value as unknown as ArrayLike<number | bigint> & { slice(): unknown }
    for (let i = 0; i < xs.length; i++) {
      const v = xs[i]
      if (typeof v === 'number' && !Number.isFinite(v)) check(kind, `${path}[${i}]`, key, v, o)
    }
    return xs.slice()
  }
  if (typeof value === 'object' && value !== null && isPlain(value)) {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value)) {
      out[k] = o.opaque?.includes(k) === true ? v : seal1(kind, path === '' ? k : `${path}.${k}`, k, v, o)
    }
    return out
  }
  return value
}

/**
 * The finished record of a constructor: a copy that shares no array with the caller's input, in which every number is
 * finite except the fields {@link SEAL_OPTIONS} lists for its kind as gaps (NaN) or unbounded (an infinity).
 *
 * @param record - The record with its `kind`.
 * @returns The copy.
 * @throws OmaiError `value` naming the kind and the path of the first non-finite number.
 */
export function seal<T extends { readonly kind: string }>(record: T): T {
  return seal1(record.kind, '', '', record, SEAL_OPTIONS[record.kind] ?? {}) as T
}

/**
 * The fields of each kind that may hold a non-finite number (a missing value drawn as a gap, an open bound); every
 * other number of every record must be finite. Keys are record kinds.
 */
export const SEAL_OPTIONS: Readonly<Record<string, SealOptions>> = {
  coefficientTable: { gaps: ['standardErrors', 'intervals', 'pValues'] },
  decisionSurface: { gaps: ['decision', 'values'], unbounded: ['decision'] },
  sceneSteps: { opaque: ['scene'] },
  structureSteps: { opaque: ['structure'] },
  diagram: { opaque: ['structure'] },
  modelRun: { opaque: ['structure'] },
  thresholdSweep: {
    gaps: ['tpr', 'fpr', 'precision', 'selectionRate', 'accuracy', 'rocAuc', 'averagePrecision', 'costs', 'cost'],
    unbounded: ['thresholds'],
  },
}
