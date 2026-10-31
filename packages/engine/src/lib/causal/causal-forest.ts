/**
 * Causal forests (Wager and Athey; Athey, Tibshirani and Wager's generalised random forests) for the conditional average
 * treatment effect. Outcome and treatment are first centred by cross-fitted nuisance models; each tree is grown on a
 * subsample, its splits chosen by a regression tree (`trees`) on the gradient pseudo-outcomes of the residual-on-residual
 * moment, and made honest by re-estimating the leaf effects on units held out of the splitting. The forest's estimate at a
 * point is the residual-on-residual regression weighted by how often each training unit shares a leaf with the point, so
 * the trees are an adaptive neighbourhood. The forest grows one tree per step as an algorithm, and its out-of-bag
 * predictions give a doubly robust average effect.
 */

import { MdmlError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { foldIn, key as makeKey, permutation } from '../random/index.ts'
import { fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { decisionTreeRegressor } from '../trees/decision-tree.ts'
import { applyTree, type Tree } from '../trees/tree.ts'
import { mean, type Prepared, pickRows, prepare } from './_dense.ts'
import { type CausalData, type EffectEstimate, effectFromInfluence } from './data.ts'
import { nuisancesFor } from './estimation.ts'
import type { NuisanceOptions } from './nuisance.ts'

/** Options of {@link causalForestAlgorithm}. */
export interface CausalForestOptions extends NuisanceOptions {
  /** The number of trees (default 200). */
  readonly trees?: number
  /** The share of units each tree sees (default 0.5). */
  readonly subsample?: number
  /** Use half of each subsample to choose splits and the other half to estimate the leaf effects (default true). */
  readonly honest?: boolean
  /** The fewest units of the splitting sample in a leaf (default 5). */
  readonly minLeaf?: number
  /** The deepest level of a tree (default 12). */
  readonly maxDepth?: number
  /** Features tried per split: a count, or `'sqrt'`/`'log2'`/`'all'` (default `'all'`). */
  readonly maxFeatures?: number | 'sqrt' | 'log2' | 'all'
  /** Cross-fitting folds of the centring models (default 0: fit and predict on all units). */
  readonly nuisanceFolds?: number
}

/** One tree of the forest: the tree, the leaf moments from its estimation sample, and the units it did not see. */
export type CausalTreeRecord = {
  readonly tree: Tree
  /** Per leaf (by node id): the sum of `T~ Y~`, the sum of `T~^2` and the number of units. */
  readonly leaves: Readonly<Record<string, readonly [number, number, number]>>
  /** The rows outside the tree's subsample. */
  readonly outOfBag: number[]
}

/** The state of the forest algorithm: the centred data and the trees grown so far. */
export type CausalForestState = {
  readonly grown: number
  readonly trees: CausalTreeRecord[]
  /** Outcome residual `Y - m(X)` per unit. */
  readonly outcomeResidual: number[]
  /** Treatment residual `T - e(X)` per unit. */
  readonly treatmentResidual: number[]
  /** The propensity per unit. */
  readonly propensity: number[]
}

/** A fitted causal forest. */
export interface CausalForest {
  readonly trees: number
  /** The estimated effect at each row of `x`. */
  readonly predict: (x: TensorLike) => Float64Array
  /** The out-of-bag estimate at each training unit (trees that did not see the unit). */
  readonly predictOutOfBag: () => Float64Array
  /** The doubly robust average treatment effect from the out-of-bag effects. */
  readonly averageEffect: () => EffectEstimate
  /** The share of the forest's splits made on each covariate. */
  readonly importances: readonly number[]
}

function leafMoments(
  tree: Tree,
  x: Float64Array,
  d: number,
  rows: readonly number[],
  ty: ArrayLike<number>,
  tt: ArrayLike<number>,
): Record<string, [number, number, number]> {
  const leaves = applyTree(tree, fromStorage(pickRows(x, d, rows), [rows.length, d], 'float64'))
  const out: Record<string, [number, number, number]> = {}
  rows.forEach((r, i) => {
    const id = String(leaves[i] ?? 0)
    const cell = (out[id] ??= [0, 0, 0])
    cell[0] += (ty[r] ?? 0) * (tt[r] ?? 0)
    cell[1] += (tt[r] ?? 0) ** 2
    cell[2] += 1
  })
  return out
}

/** The effect at the rows of `x` from the listed trees: ratio of the summed leaf moments per leaf size. */
function forestEffect(trees: readonly CausalTreeRecord[], x: TensorLike, d: number, fallback: number): Float64Array {
  const m = matrixView(x, 'causalForest')
  if (m.d !== d) throw new MdmlError('shape', `causalForest: expected ${d} covariates, got ${m.d}`)
  const num = new Float64Array(m.n)
  const den = new Float64Array(m.n)
  for (const t of trees) {
    const leaf = applyTree(t.tree, x)
    for (let i = 0; i < m.n; i++) {
      const cell = t.leaves[String(leaf[i] ?? 0)]
      if (cell === undefined || cell[2] === 0) continue
      num[i] = (num[i] ?? 0) + cell[0] / cell[2]
      den[i] = (den[i] ?? 0) + cell[1] / cell[2]
    }
  }
  return Float64Array.from(num, (v, i) => ((den[i] ?? 0) > 1e-12 ? v / (den[i] ?? 1) : fallback))
}

/**
 * The causal forest as an algorithm that grows one tree per step: the centred outcome and treatment are computed in
 * `init` (cross-fitted when `nuisanceFolds` is at least two), and each step draws a subsample, splits it into a
 * splitting half and an estimation half, grows a regression tree on the gradient pseudo-outcomes of the splitting half
 * and records the residual moments of the estimation half in its leaves. The state serialises.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Trees, subsample, honesty, tree size, centring learners and key.
 * @returns The algorithm; it observes the number of trees `grown`.
 * @example
 * ```ts
 * import { causalForestAlgorithm } from 'mdml-engine/causal'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 60 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const treatment = x.map((_, i) => (Math.sin(5 * i) > 0 ? 1 : 0))
 * const outcome = x.map((r, i) => (r[0] ?? 0) + (treatment[i] ?? 0) * (1 + (r[1] ?? 0)))
 * runAlgorithm(causalForestAlgorithm({ x, treatment, outcome }, { trees: 4 }), { maxSteps: 10 }).state.grown // => 4
 * ```
 */
export function causalForestAlgorithm(
  data: CausalData,
  options: CausalForestOptions = {},
): Algorithm<CausalForestState> {
  const p = prepare(data.x, data.treatment, data.outcome, 'causalForest')
  const total = options.trees ?? 200
  const share = options.subsample ?? 0.5
  const honest = options.honest ?? true
  const minLeaf = options.minLeaf ?? 5
  const maxDepth = options.maxDepth ?? 12
  const maxFeatures = options.maxFeatures ?? 'all'
  const centre = (): CausalForestState => {
    const nu = nuisancesFor(p, { ...options, folds: options.nuisanceFolds ?? 0, pooled: true })
    const m = nu.m ?? new Float64Array(p.n)
    return {
      grown: 0,
      trees: [],
      outcomeResidual: Array.from(p.y, (y, i) => y - (m[i] ?? 0)),
      treatmentResidual: Array.from(p.t, (t, i) => t - (nu.e[i] ?? 0)),
      propensity: Array.from(nu.e),
    }
  }
  return {
    name: 'causal-forest',
    config: { trees: total, subsample: share, honest, minLeaf, maxDepth, n: p.n },
    schema: channels({ metrics: { grown: { description: 'trees grown so far' } } }),
    init: () => centre(),
    step: (s, ctx) => {
      if (s.grown >= total) return s
      const k = foldIn(ctx.key, s.grown)
      const order = Array.from(toFlat(permutation(k, p.n)))
      const size = Math.max(2 * minLeaf + 2, Math.min(p.n, Math.round(share * p.n)))
      const sample = order.slice(0, size)
      const splitRows = honest ? sample.slice(0, Math.floor(size / 2)) : sample
      const estimateRows = honest ? sample.slice(Math.floor(size / 2)) : sample
      // Gradient pseudo-outcomes at the root of the splitting sample.
      let a = 0
      let b = 0
      for (const r of splitRows) {
        a += (s.treatmentResidual[r] ?? 0) * (s.outcomeResidual[r] ?? 0)
        b += (s.treatmentResidual[r] ?? 0) ** 2
      }
      const tau = b > 1e-12 ? a / b : 0
      const variationT = b / splitRows.length || 1
      const rho = splitRows.map(
        (r) =>
          ((s.treatmentResidual[r] ?? 0) * ((s.outcomeResidual[r] ?? 0) - tau * (s.treatmentResidual[r] ?? 0))) /
          variationT,
      )
      const fitted = decisionTreeRegressor({ minSamplesLeaf: minLeaf, maxDepth, maxFeatures }).fit(
        {
          x: fromStorage(pickRows(p.x, p.d, splitRows), [splitRows.length, p.d], 'float64'),
          y: Float64Array.from(rho),
        },
        { key: foldIn(k, 1), explain: false },
      )
      const tree = fitted.params
      const inSample = new Set(sample)
      const record: CausalTreeRecord = {
        tree,
        leaves: leafMoments(tree, p.x, p.d, estimateRows, s.outcomeResidual, s.treatmentResidual),
        outOfBag: Array.from({ length: p.n }, (_, i) => i).filter((i) => !inSample.has(i)),
      }
      return { ...s, grown: s.grown + 1, trees: [...s.trees, record] }
    },
    done: (s) => (s.grown >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { grown: s.grown } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CausalForestState,
  }
}

/**
 * The prediction functions of a grown forest state.
 *
 * @param state - The state of {@link causalForestAlgorithm}, after any number of steps.
 * @param data - The training data the forest was grown on.
 * @returns The forest.
 */
export function causalForestOf(state: CausalForestState, data: CausalData): CausalForest {
  const p: Prepared = prepare(data.x, data.treatment, data.outcome, 'causalForest')
  let a = 0
  let b = 0
  for (let i = 0; i < p.n; i++) {
    a += (state.treatmentResidual[i] ?? 0) * (state.outcomeResidual[i] ?? 0)
    b += (state.treatmentResidual[i] ?? 0) ** 2
  }
  const fallback = b > 0 ? a / b : 0
  const importances = new Array<number>(p.d).fill(0)
  for (const t of state.trees)
    t.tree.feature.forEach((f, node) => {
      if (f >= 0 && (t.tree.left[node] ?? -1) >= 0) importances[f] = (importances[f] ?? 0) + 1
    })
  const total = importances.reduce((s, v) => s + v, 0) || 1
  const oob = (): Float64Array => {
    // Each unit is predicted by the trees that did not see it: invert the out-of-bag lists.
    const byUnit: CausalTreeRecord[][] = Array.from({ length: p.n }, () => [])
    for (const t of state.trees) for (const i of t.outOfBag) byUnit[i]?.push(t)
    const out = new Float64Array(p.n)
    for (let i = 0; i < p.n; i++) {
      const row = fromStorage(pickRows(p.x, p.d, [i]), [1, p.d], 'float64')
      out[i] = forestEffect(byUnit[i] ?? [], row, p.d, fallback)[0] ?? fallback
    }
    return out
  }
  return {
    trees: state.trees.length,
    predict: (x) => forestEffect(state.trees, x, p.d, fallback),
    predictOutOfBag: oob,
    averageEffect: () => {
      const tau = oob()
      const score = Float64Array.from(tau, (t, i) => {
        const e = state.propensity[i] ?? 0.5
        const tt = state.treatmentResidual[i] ?? 0
        return t + (tt / (e * (1 - e))) * ((state.outcomeResidual[i] ?? 0) - t * tt)
      })
      const value = mean(score)
      return effectFromInfluence(
        'ate',
        value,
        score.map((v) => v - value),
        { method: 'causal forest (out-of-bag, doubly robust)' },
      )
    },
    importances: importances.map((v) => v / total),
  }
}

/**
 * A causal forest, grown to the end: honest trees on subsamples with the residual-on-residual moment, centred by
 * cross-fitted nuisance models.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Trees, subsample, honesty, tree size, centring learners and key.
 * @returns The forest.
 * @example
 * ```ts
 * import { causalForest } from 'mdml-engine/causal'
 * const x = Array.from({ length: 80 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const treatment = x.map((_, i) => (Math.sin(5 * i) > 0 ? 1 : 0))
 * const outcome = x.map((r, i) => (r[0] ?? 0) + (treatment[i] ?? 0) * (1 + (r[1] ?? 0)))
 * causalForest({ x, treatment, outcome }, { trees: 10 }).trees // => 10
 * ```
 */
export function causalForest(data: CausalData, options: CausalForestOptions = {}): CausalForest {
  const alg = causalForestAlgorithm(data, options)
  const run = runAlgorithm(alg, { key: options.key ?? makeKey(0), maxSteps: options.trees ?? 200 })
  return causalForestOf(run.state, data)
}
