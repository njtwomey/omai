/**
 * Uplift trees (Rzepakowski and Jaroszewicz): decision trees whose splits maximise the divergence between the treated
 * and control outcome distributions in the children rather than the purity of the outcome, so each leaf holds a group
 * whose response to treatment differs from its neighbours'. The criteria are the Kullback-Leibler, squared Euclidean and
 * chi-squared divergences of the two response rates (binary outcomes) and the squared difference of the children's mean
 * effects (any outcome). A leaf predicts its treated minus control mean. Trees are grown greedily on all thresholds, with
 * a minimum of units of each arm in a leaf; a bagged average of them (`upliftForest`) steadies the estimate.
 */

import { OmaiError } from '../base/index.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import type { CausalData } from './data.ts'

/** How a split is scored. */
export type UpliftCriterion = 'KL' | 'ED' | 'Chi' | 'tau'

/** Options of {@link upliftTree}. */
export interface UpliftTreeOptions {
  /** `'KL'`, `'ED'` or `'Chi'` for 0/1 outcomes; `'tau'` (default `'ED'` for 0/1 outcomes, else `'tau'`). */
  readonly criterion?: UpliftCriterion
  /** The deepest level (default 4). */
  readonly maxDepth?: number
  /** The fewest units of each arm in a leaf (default 20). */
  readonly minLeaf?: number
  /** The smallest gain worth a split (default 0). */
  readonly minGain?: number
}

/** A node of an uplift tree. */
export type UpliftNode = {
  /** The feature split on (-1 for a leaf). */
  readonly feature: number
  readonly threshold: number
  readonly left: number
  readonly right: number
  readonly treated: number
  readonly control: number
  /** The mean outcome of the treated and control units here. */
  readonly treatedMean: number
  readonly controlMean: number
  /** The estimated uplift: treated mean minus control mean. */
  readonly uplift: number
  /** The gain of the split (0 for a leaf). */
  readonly gain: number
  readonly depth: number
}

/** A fitted uplift tree. */
export interface UpliftTree {
  readonly nodes: readonly UpliftNode[]
  readonly criterion: UpliftCriterion
  /** The estimated uplift at each row of `x`. */
  readonly predict: (x: TensorLike) => Float64Array
  /** The leaf each row reaches. */
  readonly apply: (x: TensorLike) => Int32Array
  /** The total gain of the splits on each feature. */
  readonly importances: readonly number[]
}

const divergence = (criterion: UpliftCriterion, pt: number, pc: number): number => {
  const a = Math.min(1 - 1e-9, Math.max(1e-9, pt))
  const b = Math.min(1 - 1e-9, Math.max(1e-9, pc))
  if (criterion === 'KL') return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b))
  if (criterion === 'Chi') return (a - b) ** 2 / b + (a - b) ** 2 / (1 - b)
  return 2 * (a - b) ** 2
}

interface Stats {
  nt: number
  nc: number
  st: number
  sc: number
}

const rate = (s: number, n: number): number => (s + 0.5) / (n + 1)

function nodeScore(criterion: UpliftCriterion, a: Stats): number {
  return criterion === 'tau' ? a.st / a.nt - a.sc / a.nc : divergence(criterion, rate(a.st, a.nt), rate(a.sc, a.nc))
}

function grow(
  x: Float64Array,
  d: number,
  t: Float64Array,
  y: Float64Array,
  rows: readonly number[],
  depth: number,
  options: Required<UpliftTreeOptions>,
  nodes: UpliftNode[],
  gains: Float64Array,
): number {
  const total: Stats = { nt: 0, nc: 0, st: 0, sc: 0 }
  for (const r of rows) {
    if (t[r] === 1) {
      total.nt++
      total.st += y[r] ?? 0
    } else {
      total.nc++
      total.sc += y[r] ?? 0
    }
  }
  const id = nodes.length
  const leafOf = (gain: number): UpliftNode => ({
    feature: -1,
    threshold: 0,
    left: -1,
    right: -1,
    treated: total.nt,
    control: total.nc,
    treatedMean: total.nt === 0 ? Number.NaN : total.st / total.nt,
    controlMean: total.nc === 0 ? Number.NaN : total.sc / total.nc,
    uplift: total.nt === 0 || total.nc === 0 ? 0 : total.st / total.nt - total.sc / total.nc,
    gain,
    depth,
  })
  nodes.push(leafOf(0))
  if (depth >= options.maxDepth || total.nt < 2 * options.minLeaf || total.nc < 2 * options.minLeaf) return id
  const before = options.criterion === 'tau' ? 0 : nodeScore(options.criterion, total)
  let best = { gain: options.minGain, feature: -1, threshold: 0 }
  for (let f = 0; f < d; f++) {
    const order = [...rows].sort((a, b) => (x[a * d + f] ?? 0) - (x[b * d + f] ?? 0))
    const left: Stats = { nt: 0, nc: 0, st: 0, sc: 0 }
    for (let k = 0; k < order.length - 1; k++) {
      const r = order[k] ?? 0
      if (t[r] === 1) {
        left.nt++
        left.st += y[r] ?? 0
      } else {
        left.nc++
        left.sc += y[r] ?? 0
      }
      const here = x[r * d + f] ?? 0
      const next = x[(order[k + 1] ?? 0) * d + f] ?? 0
      if (next === here) continue
      const right: Stats = {
        nt: total.nt - left.nt,
        nc: total.nc - left.nc,
        st: total.st - left.st,
        sc: total.sc - left.sc,
      }
      if (
        left.nt < options.minLeaf ||
        left.nc < options.minLeaf ||
        right.nt < options.minLeaf ||
        right.nc < options.minLeaf
      )
        continue
      const nl = left.nt + left.nc
      const nr = right.nt + right.nc
      const n = nl + nr
      const gain =
        options.criterion === 'tau'
          ? (nl * nr * (nodeScore('tau', left) - nodeScore('tau', right)) ** 2) / (n * n)
          : (nl * nodeScore(options.criterion, left) + nr * nodeScore(options.criterion, right)) / n - before
      if (gain > best.gain) best = { gain, feature: f, threshold: (here + next) / 2 }
    }
  }
  if (best.feature < 0) return id
  const leftRows = rows.filter((r) => (x[r * d + best.feature] ?? 0) <= best.threshold)
  const rightRows = rows.filter((r) => (x[r * d + best.feature] ?? 0) > best.threshold)
  gains[best.feature] = (gains[best.feature] ?? 0) + best.gain * rows.length
  const l = grow(x, d, t, y, leftRows, depth + 1, options, nodes, gains)
  const r = grow(x, d, t, y, rightRows, depth + 1, options, nodes, gains)
  nodes[id] = { ...leafOf(best.gain), feature: best.feature, threshold: best.threshold, left: l, right: r }
  return id
}

function applyNodes(nodes: readonly UpliftNode[], x: Float64Array, d: number, n: number): Int32Array {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    let node = 0
    while ((nodes[node]?.feature ?? -1) >= 0) {
      const nd = nodes[node]
      if (nd === undefined) break
      node = (x[i * d + nd.feature] ?? 0) <= nd.threshold ? nd.left : nd.right
    }
    out[i] = node
  }
  return out
}

/**
 * Grow an uplift tree. Each split maximises the weighted divergence between the treated and control response rates of
 * the two children over the divergence in the parent (or, for `'tau'`, the squared difference of the children's effects).
 *
 * @param data - Covariates, treatment and outcome (0/1 for the divergence criteria).
 * @param options - Criterion, depth, the least units of each arm per leaf and the least gain.
 * @returns The tree.
 * @example
 * ```ts
 * import { upliftTree } from 'omai-engine/causal'
 * const x = Array.from({ length: 80 }, (_, i) => [i < 40 ? 0 : 1])
 * const treatment = x.map((_, i) => i % 2)
 * const outcome = x.map((r, i) => ((r[0] ?? 0) === 1 && i % 2 === 1 ? 1 : 0))
 * upliftTree({ x, treatment, outcome }, { minLeaf: 5, maxDepth: 1 }).predict([[0], [1]]) // => [0, 1]
 * ```
 */
export function upliftTree(data: CausalData, options: UpliftTreeOptions = {}): UpliftTree {
  const m = matrixView(data.x, 'upliftTree')
  const t = targetVector(data.treatment, 'upliftTree', m.n)
  const y = targetVector(data.outcome, 'upliftTree', m.n)
  const binary = y.every((v) => v === 0 || v === 1)
  const criterion = options.criterion ?? (binary ? 'ED' : 'tau')
  if (criterion !== 'tau' && !binary)
    throw new OmaiError('value', `upliftTree: the ${criterion} criterion needs a 0/1 outcome`)
  const full: Required<UpliftTreeOptions> = {
    criterion,
    maxDepth: options.maxDepth ?? 4,
    minLeaf: options.minLeaf ?? 20,
    minGain: options.minGain ?? 0,
  }
  const nodes: UpliftNode[] = []
  const gains = new Float64Array(m.d)
  grow(
    m.data,
    m.d,
    t,
    y,
    Array.from({ length: m.n }, (_, i) => i),
    0,
    full,
    nodes,
    gains,
  )
  const sum = gains.reduce((a, b) => a + b, 0) || 1
  const reach = (q: TensorLike): Int32Array => {
    const v = matrixView(q, 'upliftTree')
    if (v.d !== m.d) throw new OmaiError('shape', `upliftTree: expected ${m.d} covariates, got ${v.d}`)
    return applyNodes(nodes, v.data, m.d, v.n)
  }
  return {
    nodes,
    criterion,
    predict: (q) => Float64Array.from(reach(q), (node) => nodes[node]?.uplift ?? 0),
    apply: reach,
    importances: Array.from(gains, (g) => g / sum),
  }
}

/** A bagged average of uplift trees. */
export interface UpliftForest {
  readonly trees: readonly UpliftTree[]
  /** The mean estimated uplift over the trees. */
  readonly predict: (x: TensorLike) => Float64Array
}

/**
 * A bagged forest of uplift trees: each tree is grown on a bootstrap resample of the units, and the estimated uplift is
 * their average, which removes much of a single tree's instability.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Tree options and `trees` (default 50), `key`.
 * @returns The forest.
 * @example
 * ```ts
 * import { upliftForest } from 'omai-engine/causal'
 * const x = Array.from({ length: 80 }, (_, i) => [i < 40 ? 0 : 1])
 * const treatment = x.map((_, i) => i % 2)
 * const outcome = x.map((r, i) => ((r[0] ?? 0) === 1 && i % 2 === 1 ? 1 : 0))
 * upliftForest({ x, treatment, outcome }, { trees: 5, minLeaf: 5, maxDepth: 1 }).trees.length // => 5
 * ```
 */
export function upliftForest(
  data: CausalData,
  options: UpliftTreeOptions & { readonly trees?: number; readonly key?: Key } = {},
): UpliftForest {
  const m = matrixView(data.x, 'upliftForest')
  const t = targetVector(data.treatment, 'upliftForest', m.n)
  const y = targetVector(data.outcome, 'upliftForest', m.n)
  const key = options.key ?? makeKey(0)
  const trees = Array.from({ length: options.trees ?? 50 }, (_, b) => {
    const rows = toFlat(bootstrapIndices(foldIn(key, b), m.n))
    const x = Array.from(rows, (r) => Array.from(m.data.subarray(r * m.d, (r + 1) * m.d)))
    return upliftTree(
      { x, treatment: Array.from(rows, (r) => t[r] ?? 0), outcome: Array.from(rows, (r) => y[r] ?? 0) },
      options,
    )
  })
  return {
    trees,
    predict: (q) => {
      const all = trees.map((tr) => tr.predict(q))
      return Float64Array.from(all[0] ?? [], (_, i) => all.reduce((s, a) => s + (a[i] ?? 0), 0) / all.length)
    },
  }
}
