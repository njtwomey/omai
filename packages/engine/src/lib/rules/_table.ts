// The training table the rule learners search: a row-major feature matrix with class indices, and the pool of
// candidate literals (every category of a nominal feature, thresholds between consecutive values of a numeric one)
// with their coverage masks, computed once. Rows in play are index arrays so covering can remove rows cheaply.

import type { LearnData } from '../learn/estimator.ts'
import { requireTargets } from '../learn/estimator.ts'
import { encodeLabels, matrixView } from '../learn/validation.ts'
import type { Literal } from './rule.ts'

/** Training data prepared for rule search. */
export type RuleData = {
  readonly n: number
  readonly d: number
  /** Row-major features. */
  readonly x: Float64Array
  /** Class index of each row. */
  readonly y: Int32Array
  readonly classes: number[]
  readonly k: number
  /** Features searched by equality (category codes) rather than by threshold. */
  readonly nominal: readonly boolean[]
}

/** Candidate literals and the rows each holds on. */
export type LiteralPool = { readonly literals: readonly Literal[]; readonly masks: readonly Uint8Array[] }

/**
 * Prepare learn data for rule search.
 *
 * @param data - Features and class labels.
 * @param nominal - Indices of features whose values are category codes (default none: all thresholded).
 * @param who - The caller, for errors.
 * @returns The prepared data.
 */
export function ruleData(data: LearnData, nominal: readonly number[], who: string): RuleData {
  const key = nominal.join(',')
  const cached = PREPARED.get(data)?.get(key)
  if (cached !== undefined) return cached
  const rd = prepare(data, nominal, who)
  const forData = PREPARED.get(data) ?? new Map<string, RuleData>()
  forData.set(key, rd)
  PREPARED.set(data, forData)
  return rd
}

const PREPARED = new WeakMap<LearnData, Map<string, RuleData>>()

function prepare(data: LearnData, nominal: readonly number[], who: string): RuleData {
  const m = matrixView(data.x, who, { allowNaN: true })
  const { classes, codes } = encodeLabels(requireTargets(data, who), who, m.n)
  return {
    n: m.n,
    d: m.d,
    x: m.data,
    y: codes,
    classes: [...classes],
    k: classes.length,
    nominal: Array.from({ length: m.d }, (_, j) => nominal.includes(j)),
  }
}

/**
 * The candidate literals of a table: `==` for every category of a nominal feature, and `<=` with `>` at up to
 * `maxThresholds` cut points (midpoints of consecutive distinct values, evenly spread over the sorted values) for a
 * numeric one.
 *
 * @param rd - The table.
 * @param maxThresholds - The most thresholds per numeric feature (default 24).
 * @returns The literals with their masks.
 */
export function literalPool(rd: RuleData, maxThresholds = 24): LiteralPool {
  const literals: Literal[] = []
  const masks: Uint8Array[] = []
  const add = (l: Literal): void => {
    const mask = new Uint8Array(rd.n)
    for (let i = 0; i < rd.n; i++) {
      const v = rd.x[i * rd.d + l.feature] as number
      mask[i] = !Number.isNaN(v) && (l.op === '==' ? v === l.value : l.op === '<=' ? v <= l.value : v > l.value) ? 1 : 0
    }
    literals.push(l)
    masks.push(mask)
  }
  for (let j = 0; j < rd.d; j++) {
    const values = [
      ...new Set(Array.from({ length: rd.n }, (_, i) => rd.x[i * rd.d + j] as number).filter((v) => !Number.isNaN(v))),
    ].sort((a, b) => a - b)
    if (rd.nominal[j] === true) {
      for (const v of values) add({ feature: j, op: '==', value: v })
      continue
    }
    const gaps = values.length - 1
    const picks = new Set<number>()
    if (gaps <= maxThresholds) for (let g = 0; g < gaps; g++) picks.add(g)
    else for (let t = 0; t < maxThresholds; t++) picks.add(Math.floor(((t + 0.5) * gaps) / maxThresholds))
    for (const g of [...picks].sort((a, b) => a - b)) {
      const cut = ((values[g] as number) + (values[g + 1] as number)) / 2
      add({ feature: j, op: '<=', value: cut })
      add({ feature: j, op: '>', value: cut })
    }
  }
  return { literals, masks }
}

/** Class counts among some rows. */
export function countClasses(rd: RuleData, rows: ArrayLike<number>): number[] {
  const out = new Array<number>(rd.k).fill(0)
  for (let i = 0; i < rows.length; i++)
    out[rd.y[rows[i] as number] as number] = (out[rd.y[rows[i] as number] as number] ?? 0) + 1
  return out
}

/** The rows (among `rows`) on which a literal holds. */
export function keepRows(pool: LiteralPool, rows: Int32Array, literal: number): Int32Array {
  const mask = pool.masks[literal] as Uint8Array
  const out = new Int32Array(rows.length)
  let m = 0
  for (let i = 0; i < rows.length; i++) if (mask[rows[i] as number] === 1) out[m++] = rows[i] as number
  return out.subarray(0, m)
}

/** The rows (among `rows`) a mask excludes. */
export function dropRows(rows: ArrayLike<number>, covered: Uint8Array): Int32Array {
  const out = new Int32Array(rows.length)
  let m = 0
  for (let i = 0; i < rows.length; i++) if (covered[rows[i] as number] !== 1) out[m++] = rows[i] as number
  return out.subarray(0, m)
}

/** The coverage of a conjunction of pool literals over all rows. */
export function conjunctionMask(rd: RuleData, pool: LiteralPool, literals: readonly number[]): Uint8Array {
  const out = new Uint8Array(rd.n).fill(1)
  for (const l of literals) {
    const mask = pool.masks[l] as Uint8Array
    for (let i = 0; i < rd.n; i++) if (mask[i] !== 1) out[i] = 0
  }
  return out
}

/** Whether adding a literal to a conjunction is pointless: the same literal, or an equality on a feature already tested by one. */
export function conflicts(pool: LiteralPool, literals: readonly number[], candidate: number): boolean {
  const c = pool.literals[candidate] as Literal
  for (const l of literals) {
    if (l === candidate) return true
    const o = pool.literals[l] as Literal
    if (c.op === '==' && o.feature === c.feature) return true
    if (o.op === '==' && o.feature === c.feature) return true
  }
  return false
}
