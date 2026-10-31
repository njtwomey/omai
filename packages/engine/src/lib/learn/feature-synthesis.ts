/**
 * Deep feature synthesis (Kanter and Veeramachaneni 2015), small: given a table of entities and tables of child rows that
 * point at them (orders of customers, sessions of users), generate candidate features by stacking primitives to a chosen
 * depth: aggregates of the child tables' columns per parent row (sum, mean, min, max, standard deviation, count) and
 * arithmetic on the columns of one table (add, subtract, multiply, divide). Every feature carries its name and
 * definition; selection by the score of a model keeps the ones that help. The point is the search space made explicit, so
 * it can be inspected, counted and pruned, not a feature store.
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { Estimator, LearnData } from './estimator.ts'
import { crossValScore } from './model-selection/cross-validation.ts'
import { fromView } from './validation.ts'

/** A table of rows with numeric columns. A child table names the parent table and the column holding the parent row's index. */
export interface EntityTable {
  readonly name: string
  /** Numeric columns, each with one value per row (the parent-key column is not used as a feature). */
  readonly columns: Readonly<Record<string, ArrayLike<number>>>
  /** For a child table: the parent table and the column whose value is the row index (0-based) of the parent row. */
  readonly parent?: { readonly table: string; readonly key: string }
}

/** Aggregations of child-row values per parent row. */
export type Aggregation = 'sum' | 'mean' | 'min' | 'max' | 'std' | 'count'

/** Arithmetic on two features of one table. */
export type Transform = 'add' | 'subtract' | 'multiply' | 'divide'

/** Options of {@link deepFeatureSynthesis}. */
export interface SynthesisOptions {
  /** The aggregations (default sum, mean, min, max, std; `count` of child rows is always added). */
  readonly aggregations?: readonly Aggregation[]
  /** The arithmetic transforms (default add, subtract, multiply, divide). */
  readonly transforms?: readonly Transform[]
  /** The deepest stack of primitives (default 1; 0 is the raw columns). */
  readonly maxDepth?: number
  /** Stop adding features past this many (default 5000). */
  readonly maxFeatures?: number
}

/** A synthesised feature of the target table. */
export interface SynthesisedFeature {
  /** Its name, such as `SUM(orders.amount)` or `age * income`. */
  readonly name: string
  readonly values: Float64Array
  /** The number of primitives stacked. */
  readonly depth: number
  /** The definition as a tree of primitive applications. */
  readonly definition: Json
}

const DEFAULT_AGGREGATIONS: readonly Aggregation[] = ['sum', 'mean', 'min', 'max', 'std']
const DEFAULT_TRANSFORMS: readonly Transform[] = ['add', 'subtract', 'multiply', 'divide']

function aggregate(kind: Aggregation, values: readonly number[]): number {
  if (kind === 'count') return values.length
  if (values.length === 0) return 0
  let sum = 0
  for (const v of values) sum += v
  if (kind === 'sum') return sum
  const mean = sum / values.length
  if (kind === 'mean') return mean
  if (kind === 'min') return Math.min(...values)
  if (kind === 'max') return Math.max(...values)
  let ss = 0
  for (const v of values) ss += (v - mean) ** 2
  return values.length > 1 ? Math.sqrt(ss / (values.length - 1)) : 0
}

function apply(kind: Transform, a: number, b: number): number {
  if (kind === 'add') return a + b
  if (kind === 'subtract') return a - b
  if (kind === 'multiply') return a * b
  return b === 0 ? 0 : a / b
}

const SYMBOL: Readonly<Record<Transform, string>> = { add: '+', subtract: '-', multiply: '*', divide: '/' }

/**
 * Generate candidate features for a target table by deep feature synthesis. Depth 0 is the table's own columns; each
 * level adds (a) per child table, every aggregation of each child feature of lower depth over the child rows belonging to
 * each parent row, and the child-row count, and (b) every arithmetic combination of two features of the table of lower depth
 * (symmetric transforms once per pair, the others in both orders). A feature's depth is one more than the deepest feature it
 * is built from. With the default primitives and one child table of c numeric columns the first level is about 5c + 1 aggregates
 * and the second multiplies the count roughly by its square, so `maxFeatures` and the choice of primitives are the pruning.
 *
 * @param tables - All tables, parents and children.
 * @param target - The name of the table to build features for.
 * @param options - Primitives, depth and the cap.
 * @returns The features of the target table, one value per row, by increasing depth.
 * @throws MdmlError with code `value` for an unknown table, a child pointing at a missing parent, or a parent index out of range.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { deepFeatureSynthesis } from 'mdml-engine/learn'
 * const customers = { name: 'customers', columns: { age: [30, 40] } }
 * const orders = { name: 'orders', columns: { customer: [0, 0, 1], amount: [5, 7, 2] }, parent: { table: 'customers', key: 'customer' } }
 * deepFeatureSynthesis([customers, orders], 'customers', { aggregations: ['sum'] }).map((f) => f.name) // => ['age', 'COUNT(orders)', 'SUM(orders.amount)']
 * ```
 */
export function deepFeatureSynthesis(
  tables: readonly EntityTable[],
  target: string,
  options: SynthesisOptions = {},
): SynthesisedFeature[] {
  const byName = new Map(tables.map((t) => [t.name, t]))
  if (!byName.has(target)) throw new MdmlError('value', `deepFeatureSynthesis: no table '${target}'`)
  const aggs = options.aggregations ?? DEFAULT_AGGREGATIONS
  const transforms = options.transforms ?? DEFAULT_TRANSFORMS
  const maxDepth = options.maxDepth ?? 1
  const cap = options.maxFeatures ?? 5000
  const rowsOf = (t: EntityTable): number => {
    const first = Object.values(t.columns)[0]
    return first?.length ?? 0
  }
  for (const t of tables) {
    if (t.parent === undefined) continue
    const parent = byName.get(t.parent.table)
    if (parent === undefined)
      throw new MdmlError('value', `deepFeatureSynthesis: '${t.name}' points at a missing table '${t.parent.table}'`)
    const key = t.columns[t.parent.key]
    if (key === undefined)
      throw new MdmlError('value', `deepFeatureSynthesis: '${t.name}' has no key column '${t.parent.key}'`)
    for (let i = 0; i < key.length; i++) {
      const k = key[i] ?? -1
      if (!(Number.isInteger(k) && k >= 0 && k < rowsOf(parent)))
        throw new MdmlError(
          'value',
          `deepFeatureSynthesis: '${t.name}.${t.parent.key}' has a row index outside '${parent.name}'`,
        )
    }
  }
  const cache = new Map<string, SynthesisedFeature[]>()

  /** The features of a table with depth at most `depth`, ordered by depth. */
  const featuresOf = (name: string, depth: number): SynthesisedFeature[] => {
    const id = `${name}@${depth}`
    const hit = cache.get(id)
    if (hit !== undefined) return hit
    const table = byName.get(name) as EntityTable
    const rows = rowsOf(table)
    const out: SynthesisedFeature[] = []
    const skip = table.parent?.key
    for (const [col, v] of Object.entries(table.columns)) {
      if (col === skip) continue
      out.push({ name: col, values: Float64Array.from(v), depth: 0, definition: col })
    }
    for (let level = 1; level <= depth && out.length < cap; level++) {
      const fresh: SynthesisedFeature[] = []
      // Aggregates of child features of depth <= level - 1.
      for (const child of tables) {
        if (child.parent?.table !== name) continue
        const key = child.columns[child.parent.key] as ArrayLike<number>
        const groups: number[][] = Array.from({ length: rows }, () => [])
        for (let i = 0; i < key.length; i++) (groups[key[i] ?? 0] as number[]).push(i)
        if (level === 1) {
          fresh.push({
            name: `COUNT(${child.name})`,
            values: Float64Array.from(groups, (g) => g.length),
            depth: 1,
            definition: { op: 'count', table: child.name },
          })
        }
        for (const f of featuresOf(child.name, level - 1)) {
          if (f.depth !== level - 1) continue
          for (const a of aggs) {
            const prefixed = f.name.startsWith(`${child.name}.`) ? f.name : `${child.name}.${f.name}`
            fresh.push({
              name: `${a.toUpperCase()}(${prefixed})`,
              values: Float64Array.from(groups, (g) =>
                aggregate(
                  a,
                  g.map((i) => f.values[i] ?? 0),
                ),
              ),
              depth: level,
              definition: { op: a, table: child.name, of: f.definition },
            })
          }
        }
      }
      // Transforms of two features of depth <= level - 1 (at least one of depth level - 1).
      const pool = out.filter((f) => f.depth <= level - 1 && f.depth >= 0)
      for (let i = 0; i < pool.length; i++)
        for (let j = 0; j < pool.length; j++) {
          const a = pool[i] as SynthesisedFeature
          const b = pool[j] as SynthesisedFeature
          if (i === j || Math.max(a.depth, b.depth) !== level - 1) continue
          for (const t of transforms) {
            const symmetric = t === 'add' || t === 'multiply'
            if (symmetric && j < i) continue
            if (t === 'subtract' && j < i) continue
            fresh.push({
              name: `${a.name} ${SYMBOL[t]} ${b.name}`,
              values: Float64Array.from(a.values, (v, r) => apply(t, v, b.values[r] ?? 0)),
              depth: level,
              definition: { op: t, left: a.definition, right: b.definition },
            })
            if (out.length + fresh.length >= cap) break
          }
        }
      out.push(...fresh.slice(0, Math.max(0, cap - out.length)))
    }
    cache.set(id, out)
    return out
  }
  return featuresOf(target, maxDepth)
}

/** Options of {@link selectSynthesised}. */
export interface SelectOptions {
  /** The estimator whose cross-validated score judges a feature set. */
  readonly estimator: Estimator
  /** The most features to keep (default 5). */
  readonly maxFeatures?: number
  /** Candidates considered for the greedy search, pre-ranked by absolute correlation with the target (default 30). */
  readonly candidates?: number
  /** Cross-validation folds (default 4). */
  readonly folds?: number
  /** Stop when the best addition improves the score by less than this (default 1e-4). */
  readonly minGain?: number
}

/** The outcome of {@link selectSynthesised}. */
export interface Selection {
  /** The chosen features, in the order added. */
  readonly features: readonly SynthesisedFeature[]
  /** The cross-validated score after each addition. */
  readonly scores: readonly number[]
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += (a[i] ?? 0) / n
    mb += (b[i] ?? 0) / n
  }
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    saa += ((a[i] ?? 0) - ma) ** 2
    sbb += ((b[i] ?? 0) - mb) ** 2
  }
  return saa === 0 || sbb === 0 ? 0 : sab / Math.sqrt(saa * sbb)
}

/**
 * Select synthesised features by the cross-validated score of a model: pre-rank the candidates by absolute correlation with
 * the target, then add greedily the feature that most improves the estimator's cross-validated score, stopping at
 * `maxFeatures` or when the gain falls below `minGain` (the first feature is always taken). Forward selection by the model's own score keeps redundant
 * aggregates out (the second feature must add to the first).
 *
 * @param features - The candidates (from {@link deepFeatureSynthesis}).
 * @param y - The target, one value per row.
 * @param options - The estimator and the search limits.
 * @returns The chosen features in order and the score after each.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { selectSynthesised } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { asTensor, full, item, mean } from 'mdml-engine/tensor'
 * const meanModel = defineEstimator({ name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y as never)) }), methods: (p) => ({ predict: (x) => full([asTensor(x).shape[0] as number], p.m) }) }, {})
 * const f = { name: 'a', values: Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]), depth: 0, definition: 'a' }
 * selectSynthesised([f], [1, 2, 3, 4, 5, 6, 7, 8], { estimator: meanModel, maxFeatures: 1, folds: 2 }).features.length // => 1
 * ```
 */
export function selectSynthesised(
  features: readonly SynthesisedFeature[],
  y: ArrayLike<number>,
  options: SelectOptions,
): Selection {
  const maxFeatures = options.maxFeatures ?? 5
  const folds = options.folds ?? 4
  const minGain = options.minGain ?? 1e-4
  const n = y.length
  const ranked = [...features]
    .map((f) => ({ f, c: Math.abs(correlation(f.values, y)) }))
    .sort((a, b) => b.c - a.c)
    .slice(0, options.candidates ?? 30)
    .map((r) => r.f)
  const chosen: SynthesisedFeature[] = []
  const scores: number[] = []
  let best = Number.NEGATIVE_INFINITY
  const dataFor = (cols: readonly SynthesisedFeature[]): LearnData => ({
    x: fromView({
      n,
      d: cols.length,
      data: Float64Array.from(
        { length: n * cols.length },
        (_, k) => cols[k % cols.length]?.values[Math.floor(k / cols.length)] ?? 0,
      ),
    }),
    y: Float64Array.from(y),
  })
  while (chosen.length < maxFeatures) {
    let pick: SynthesisedFeature | undefined
    let pickScore = Number.NEGATIVE_INFINITY
    for (const f of ranked) {
      if (chosen.includes(f)) continue
      const folded = crossValScore(options.estimator, dataFor([...chosen, f]), { cv: folds })
      const score = folded.reduce((a, b) => a + b, 0) / folded.length
      if (score > pickScore) {
        pickScore = score
        pick = f
      }
    }
    if (pick === undefined || (chosen.length > 0 && pickScore - best < minGain)) break
    chosen.push(pick)
    scores.push(pickScore)
    best = pickScore
  }
  return { features: chosen, scores }
}
