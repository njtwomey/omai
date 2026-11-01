// Helpers for the states of the model-selection algorithms: scores may be NaN (a failed fold), which JSON cannot carry.
import type { Json } from '../../base/index.ts'

/** Scores as JSON: the non-finite ones (a failed fold's NaN) as strings. */
export function scoresToJson(x: ArrayLike<number>): Json {
  return Array.from(x, (v) => (Number.isFinite(v) ? v : v > 0 ? 'Infinity' : v < 0 ? '-Infinity' : 'NaN'))
}

/** The inverse of {@link scoresToJson}. */
export function scoresFromJson(j: Json): number[] {
  return (j as readonly (number | string)[]).map((v) => (typeof v === 'number' ? v : Number(v)))
}

/** A record of score lists as JSON. */
export function scoreRecordToJson(r: Readonly<Record<string, readonly number[]>>): Json {
  return Object.fromEntries(Object.entries(r).map(([k, v]) => [k, scoresToJson(v)]))
}

/** The inverse of {@link scoreRecordToJson}. */
export function scoreRecordFromJson(j: Json): Record<string, number[]> {
  return Object.fromEntries(Object.entries(j as Readonly<Record<string, Json>>).map(([k, v]) => [k, scoresFromJson(v)]))
}
