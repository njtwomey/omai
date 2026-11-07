/**
 * Records of example-based and contrastive explanations: a set of counterfactuals (or pertinent negatives and
 * positives) against the original row, with what changed, how far and whether each reaches the outcome; and a
 * dataset's prototypes and criticisms with the witness function that chose them.
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength, seal } from './_check.ts'

/** One counterfactual of a {@link CounterfactualSet}. */
export interface CounterfactualEntry {
  /** The point. */
  readonly x: readonly number[]
  /** The model's output there. */
  readonly prediction: number
  /** Its distance from the original (in the method's metric). */
  readonly distance: number
  /** Whether it reaches the desired outcome. */
  readonly valid: boolean
  /** The features it changes. */
  readonly changed: readonly number[]
  /** What it is (`counterfactual 1`, `pertinent negative`, `prototype-guided`). */
  readonly label?: string
  /** Its cost under the recourse costs, when given. */
  readonly cost?: number
  /** Actionability constraints it breaks. */
  readonly violations?: number
  /** The share of alternative models under which it still holds. */
  readonly robustness?: number
}

/**
 * Counterfactual (or contrastive) explanations of one row: the original and its prediction, the desired outcome,
 * and each counterfactual with the features it changes, its distance and validity; immutable features are marked.
 */
export interface CounterfactualSet {
  readonly kind: 'counterfactuals'
  /** The method (`growing spheres`, `DiCE`, `CEM`, ...). */
  readonly method: string
  readonly features: readonly string[]
  /** What the predictions are (`P(good credit)`). */
  readonly output: string
  readonly original: readonly number[]
  readonly originalPrediction: number
  /** The outcome sought (`P(good credit) > 0.5`). */
  readonly desired?: string
  readonly counterfactuals: readonly CounterfactualEntry[]
  /** Features that may not change. */
  readonly immutable?: readonly number[]
  /** Feature scales (MADs) the distances use, to colour changes comparably. */
  readonly scales?: readonly number[]
  /** Which row of the dataset was explained. */
  readonly index?: number
}

/**
 * A counterfactual-set record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the original, a counterfactual or the scales do not have one entry per feature;
 *   `value` when a changed or immutable feature is not named.
 * @example
 * ```ts
 * import { counterfactualSet } from 'mdml-engine/exposition'
 * const r = counterfactualSet({ method: 'growing spheres', features: ['income', 'debt'], output: 'P(approve)', original: [30, 10], originalPrediction: 0.3, counterfactuals: [{ x: [36, 10], prediction: 0.6, distance: 6, valid: true, changed: [0] }] })
 * r.counterfactuals[0].changed // => [0]
 * ```
 */
export function counterfactualSet(record: Omit<CounterfactualSet, 'kind'>): CounterfactualSet {
  const k = 'counterfactuals'
  const d = record.features.length
  sameLength(k, 'original', record.original.length, d)
  rowsOf(
    k,
    'counterfactuals.x',
    record.counterfactuals.map((c) => c.x),
    d,
  )
  if (record.scales !== undefined) sameLength(k, 'scales', record.scales.length, d)
  const named = (j: number): void => {
    if (!(Number.isInteger(j) && j >= 0 && j < d)) throw new MdmlError('value', `${k}: feature ${j} is not named`)
  }
  for (const c of record.counterfactuals) for (const j of c.changed) named(j)
  for (const j of record.immutable ?? []) named(j)
  return seal({ kind: k, ...record })
}

/**
 * The features two points differ in (beyond a relative tolerance).
 *
 * @param a - A point.
 * @param b - Another point.
 * @returns The indices where they differ.
 * @example
 * ```ts
 * import { changedFeatures } from 'mdml-engine/exposition'
 * changedFeatures([1, 2, 3], [1, 2.5, 3]) // => [1]
 * ```
 */
export function changedFeatures(a: readonly number[], b: readonly number[]): number[] {
  return a.flatMap((v, j) => (Math.abs((b[j] ?? 0) - v) > 1e-9 * (1 + Math.abs(v)) ? [j] : []))
}

/**
 * A dataset's prototypes and criticisms (MMD-critic): the rows that represent it and those it represents badly, the
 * witness function at every row, and the rows' coordinates for a scatter (two features or an embedding), or an image
 * shape when the rows are images.
 */
export interface PrototypeSet {
  readonly kind: 'prototypes'
  readonly method: string
  readonly features: readonly string[]
  /** Row indices of the prototypes and the criticisms. */
  readonly prototypes: readonly number[]
  readonly criticisms: readonly number[]
  /** The prototypes' and criticisms' feature values, in the same order. */
  readonly prototypeData: readonly (readonly number[])[]
  readonly criticismData: readonly (readonly number[])[]
  /** The witness function at every row (positive: under-represented by the prototypes). */
  readonly witness: readonly number[]
  /** Every row's 2-D coordinates for a scatter, and their axis names. */
  readonly points?: readonly (readonly [number, number])[]
  readonly axes?: readonly [string, string]
  /** A label per row (class names), to colour the scatter. */
  readonly labels?: readonly string[]
  /** Rows are images of this shape `[h, w]`. */
  readonly imageShape?: readonly [number, number]
}

/**
 * A prototype-set record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the data do not match the indices or the features, or the points, labels and
 *   witness differ in rows.
 * @example
 * ```ts
 * import { prototypeSet } from 'mdml-engine/exposition'
 * const r = prototypeSet({ method: 'MMD-critic', features: ['a'], prototypes: [0], criticisms: [2], prototypeData: [[0]], criticismData: [[9]], witness: [-0.1, -0.1, 0.3] })
 * r.criticisms // => [2]
 * ```
 */
export function prototypeSet(record: Omit<PrototypeSet, 'kind'>): PrototypeSet {
  const k = 'prototypes'
  const d = record.features.length
  sameLength(k, 'prototypeData', record.prototypeData.length, record.prototypes.length)
  sameLength(k, 'criticismData', record.criticismData.length, record.criticisms.length)
  rowsOf(k, 'prototypeData', record.prototypeData, d)
  rowsOf(k, 'criticismData', record.criticismData, d)
  const n = record.witness.length
  if (record.points !== undefined) sameLength(k, 'points', record.points.length, n)
  if (record.labels !== undefined) sameLength(k, 'labels', record.labels.length, n)
  if (record.imageShape !== undefined && record.imageShape[0] * record.imageShape[1] !== d)
    throw new MdmlError('shape', `${k}: images of ${record.imageShape.join('x')} for ${d} features`)
  return seal({ kind: k, ...record })
}
