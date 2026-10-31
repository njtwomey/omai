/**
 * Cyclic gradient boosting, the booster behind explainable boosting machines (Lou, Caruana and Gehrke 2012 and 2013;
 * InterpretML's EBM): a generalised additive model with a few pairwise interactions, f(x) = b + sum_j f_j(x_j) +
 * sum_(j,k) f_jk(x_j, x_k), learnt by visiting every feature in turn and moving its shape function a small step towards
 * the mean residual in each of its quantile bins. The booster is a stepped algorithm, one cycle over the features per
 * step, then (for interactions) one cycle over the pairs; `explainableBoosting` runs it. The model is a table per term,
 * so each prediction is a sum of readable contributions. Moved here from `explain`, which re-exports it.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import type { FittedFacts, LearnData, Verbs } from '../learn/estimator.ts'
import { withDefaultScore } from '../learn/estimator.ts'
import { columnOf, encodeLabels, matrixView, targetVector } from '../learn/validation.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  Log,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** A term of a glass-box model: a shape function of one feature, or a surface over a pair, piecewise constant. */
export type GlassBoxTerm =
  | {
      readonly features: readonly [number]
      /** Bin edges (cut points between bins), ascending. */
      readonly edges: readonly number[]
      /** The term's value in each bin (one more than the edges). */
      readonly values: readonly number[]
    }
  | {
      readonly features: readonly [number, number]
      readonly edges: readonly [readonly number[], readonly number[]]
      /** values[i][j] for bin i of the first feature and bin j of the second. */
      readonly values: readonly (readonly number[])[]
    }

/** An explainable boosting machine: an intercept plus its terms (on the link scale for classification). */
export type GlassBoxModel = {
  readonly task: 'regression' | 'classification'
  readonly intercept: number
  readonly terms: readonly GlassBoxTerm[]
}

/** Options of {@link explainableBoosting}. */
export interface ExplainableBoostingOptions {
  /** `regression` (squared error) or `classification` (binary log loss; labels 0/1). Default regression. */
  readonly task?: 'regression' | 'classification'
  /** Quantile bins per feature (default 32) and per feature of a pair (default 8). */
  readonly bins?: number
  readonly pairBins?: number
  /** Boosting rounds over all features (default 200) and the learning rate (default 0.05). */
  readonly rounds?: number
  readonly learningRate?: number
  /** Pairwise terms added after the main effects, the strongest by residual gain (default 0). */
  readonly interactions?: number
  /** Rounds for the pairwise terms (default 100). */
  readonly pairRounds?: number
}

/**
 * Quantile cut points of a column: between consecutive values when there are few, else at evenly spaced quantiles.
 *
 * @param col - The column.
 * @param bins - The most bins.
 * @returns The ascending cut points (one fewer than the bins).
 * @example
 * ```ts
 * import { quantileEdges } from 'mdml-engine/ensembles'
 * quantileEdges([1, 2, 3, 4], 2) // => [2.5]
 * quantileEdges([1, 2, 3], 8) // => [1.5, 2.5]
 * ```
 */
export function quantileEdges(col: ArrayLike<number>, bins: number): number[] {
  const sorted = Float64Array.from(col).sort()
  const n = sorted.length
  const distinct = Array.from(new Set(sorted))
  // Few values: cut between each pair of consecutive values, so no bin is empty.
  if (distinct.length <= bins) return distinct.slice(1).map((v, i) => ((distinct[i] ?? 0) + v) / 2)
  const cuts: number[] = []
  for (let b = 1; b < bins; b++) {
    const h = (b / bins) * (n - 1)
    const lo = Math.floor(h)
    const v = (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, n - 1)] ?? 0) - (sorted[lo] ?? 0))
    if (v > (sorted[0] ?? 0) && v <= (sorted[n - 1] ?? 0) && (cuts.length === 0 || v > (cuts[cuts.length - 1] ?? 0)))
      cuts.push(v)
  }
  return cuts
}

/**
 * The bin of a value among ascending cut points: the number of cut points at or below it.
 *
 * @param edges - The cut points.
 * @param v - The value.
 * @returns The bin index.
 * @example
 * ```ts
 * import { binIndex } from 'mdml-engine/ensembles'
 * binIndex([1, 2], 1.5) // => 1
 * binIndex([1, 2], 5) // => 2
 * ```
 */
export function binIndex(edges: readonly number[], v: number): number {
  let lo = 0
  let hi = edges.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (v < (edges[mid] ?? 0)) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * The paths of the cyclic booster: the cycle count and training loss it logs, and the derived `model/shapeFunctions`.
 *
 * @example
 * ```ts
 * import type { CyclicPath } from 'mdml-engine/ensembles'
 * const path: CyclicPath = 'boost/round'
 * path // => 'boost/round'
 * ```
 */
export type CyclicPath = 'boost/round' | 'boost/trainLoss' | 'model/shapeFunctions'

/** The state of the cyclic booster. */
export interface CyclicState {
  /** The current scores on the training rows (the link scale for classification). */
  readonly f: Float64Array
  /** The shape of each feature (a value per bin), before centring. */
  readonly shapes: readonly (readonly number[])[]
  /** The pairs chosen for interactions (null until the pair phase starts) and their tables. */
  readonly pairs: readonly { readonly j: number; readonly k: number }[] | null
  readonly tables: readonly (readonly (readonly number[])[])[]
  /** Cycles done: over the features first, then over the pairs. */
  readonly round: number
}

/** A cyclic booster: the algorithm, and the model any of its states describes. */
export interface CyclicBooster {
  readonly algorithm: Algorithm<CyclicState, CyclicPath>
  /** The glass-box model of a state: terms centred over the data, the intercept carrying the rest. */
  model(state: CyclicState): GlassBoxModel
}

/**
 * The cyclic booster as an algorithm: each step is one cycle over the features (each moves its shape function by a small
 * step towards the mean residual per bin; for classification the step is divided by the summed hessian p (1 - p)), and
 * after `rounds` of those, `pairRounds` cycles over the strongest pairs (ranked by how much a coarse 2-D table of bin
 * means reduces the residual sum of squares). Logs `boost/round` and `boost/trainLoss`; the probe `model/shapeFunctions`
 * (see {@link explainableBoostingRegressor}) is the model of a state.
 *
 * @param data - Training rows and targets (0/1 for classification).
 * @param options - task, bins, pairBins, rounds, learningRate, interactions and pairRounds.
 * @returns The algorithm and the model-of-a-state function.
 * @concept boosting
 * @example
 * ```ts
 * import { cyclicBoosting } from 'mdml-engine/ensembles'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 20 }, (_, i) => [i % 4])
 * const booster = cyclicBoosting({ x, y: x.map(([a = 0]) => 2 * a) }, { bins: 4, rounds: 200, learningRate: 0.3 })
 * const state = runAlgorithm(booster.algorithm, { maxSteps: 200, seed: 0 }).state
 * Math.round(booster.model(state).terms[0].values[3] - booster.model(state).terms[0].values[0]) // => 6
 * ```
 */
export function cyclicBoosting(data: LearnData, options: ExplainableBoostingOptions = {}): CyclicBooster {
  const m = matrixView(data.x, 'explainableBoosting')
  if (data.y === undefined) throw new MdmlError('value', 'explainableBoosting: needs targets')
  const y = targetVector(data.y, 'explainableBoosting', m.n)
  const task = options.task ?? 'regression'
  const classification = task === 'classification'
  const lr = options.learningRate ?? 0.05
  const rounds = options.rounds ?? 200
  const nPairs = options.interactions ?? 0
  const pairRounds = nPairs > 0 && m.d >= 2 ? (options.pairRounds ?? 100) : 0
  const cols = Array.from({ length: m.d }, (_, j) => columnOf(m, j))
  const edges = cols.map((c) => quantileEdges(c, options.bins ?? 32))
  const bins = cols.map((c, j) => Array.from(c, (v) => binIndex(edges[j] ?? [], v)))
  const mean = y.reduce((a, b) => a + b, 0) / m.n
  const intercept0 = classification ? Math.log(Math.max(1e-6, mean) / Math.max(1e-6, 1 - mean)) : mean
  const pb = options.pairBins ?? 8
  const pEdges = cols.map((c) => quantileEdges(c, pb))
  const pBins = cols.map((c, j) => Array.from(c, (v) => binIndex(pEdges[j] ?? [], v)))
  const residual = (f: Float64Array): Float64Array =>
    Float64Array.from(f, (v, i) => (y[i] ?? 0) - (classification ? 1 / (1 + Math.exp(-v)) : v))
  const lossOf = (f: Float64Array): number => {
    let s = 0
    for (let i = 0; i < m.n; i++) {
      const z = f[i] ?? 0
      s += classification
        ? Math.log1p(Math.exp(-Math.abs(z))) + Math.max(z, 0) - (y[i] ?? 0) * z
        : ((y[i] ?? 0) - z) ** 2
    }
    return s / m.n
  }
  /** FAST: rank pairs by the residual sum of squares a 2-D table of bin means removes. */
  const choosePairs = (f: Float64Array): { j: number; k: number }[] => {
    const r0 = residual(f)
    const gains: { j: number; k: number; gain: number }[] = []
    for (let j = 0; j < m.d; j++)
      for (let k = j + 1; k < m.d; k++) {
        const nj = (pEdges[j]?.length ?? 0) + 1
        const nk = (pEdges[k]?.length ?? 0) + 1
        const s = new Float64Array(nj * nk)
        const c = new Float64Array(nj * nk)
        for (let i = 0; i < m.n; i++) {
          const cell = (pBins[j]?.[i] ?? 0) * nk + (pBins[k]?.[i] ?? 0)
          s[cell] = (s[cell] ?? 0) + (r0[i] ?? 0)
          c[cell] = (c[cell] ?? 0) + 1
        }
        let gain = 0
        for (let t = 0; t < s.length; t++) if ((c[t] ?? 0) > 0) gain += (s[t] ?? 0) ** 2 / (c[t] ?? 1)
        gains.push({ j, k, gain })
      }
    gains.sort((a, b) => b.gain - a.gain || a.j - b.j || a.k - b.k)
    return gains.slice(0, nPairs).map(({ j, k }) => ({ j, k }))
  }
  const algorithm: Algorithm<CyclicState, CyclicPath> = {
    name: 'cyclicBoosting',
    config: { task, rounds, pairRounds, learningRate: lr },
    init: () => ({
      f: new Float64Array(m.n).fill(intercept0),
      shapes: edges.map((e) => new Array<number>(e.length + 1).fill(0)),
      pairs: null,
      tables: [],
      round: 0,
    }),
    step: (s, ctx) => {
      const f = Float64Array.from(s.f)
      if (s.round < rounds) {
        const shapes = s.shapes.map((sh) => [...sh])
        for (let j = 0; j < m.d; j++) {
          const r = residual(f)
          const nb = (edges[j]?.length ?? 0) + 1
          const sum = new Float64Array(nb)
          const count = new Float64Array(nb)
          const hess = new Float64Array(nb)
          for (let i = 0; i < m.n; i++) {
            const b = bins[j]?.[i] ?? 0
            sum[b] = (sum[b] ?? 0) + (r[i] ?? 0)
            count[b] = (count[b] ?? 0) + 1
            if (classification) {
              const p = 1 / (1 + Math.exp(-(f[i] ?? 0)))
              hess[b] = (hess[b] ?? 0) + p * (1 - p)
            }
          }
          const step = Array.from(sum, (v, b) => {
            const den = classification ? (hess[b] ?? 0) + 1e-9 : (count[b] ?? 0)
            return den > 0 ? (lr * v) / den : 0
          })
          const shape = shapes[j] as number[]
          for (let b = 0; b < nb; b++) shape[b] = (shape[b] ?? 0) + (step[b] ?? 0)
          for (let i = 0; i < m.n; i++) f[i] = (f[i] ?? 0) + (step[bins[j]?.[i] ?? 0] ?? 0)
        }
        if (ctx.wants('boost/round')) ctx.log('boost/round', s.round + 1)
        if (ctx.wants('boost/trainLoss')) ctx.log('boost/trainLoss', lossOf(f))
        return { ...s, f, shapes, round: s.round + 1 }
      }
      const chosen = s.pairs ?? choosePairs(f)
      const tables = (
        s.pairs === null
          ? chosen.map(({ j, k }) =>
              Array.from({ length: (pEdges[j]?.length ?? 0) + 1 }, () =>
                new Array<number>((pEdges[k]?.length ?? 0) + 1).fill(0),
              ),
            )
          : s.tables.map((t) => t.map((row) => [...row]))
      ) as number[][][]
      chosen.forEach(({ j, k }, t) => {
        const r = residual(f)
        const table = tables[t] as number[][]
        const nk = table[0]?.length ?? 1
        const sums = new Float64Array(table.length * nk)
        const c = new Float64Array(table.length * nk)
        for (let i = 0; i < m.n; i++) {
          const cell = (pBins[j]?.[i] ?? 0) * nk + (pBins[k]?.[i] ?? 0)
          sums[cell] = (sums[cell] ?? 0) + (r[i] ?? 0)
          if (classification) {
            const p = 1 / (1 + Math.exp(-(f[i] ?? 0)))
            c[cell] = (c[cell] ?? 0) + p * (1 - p)
          } else c[cell] = (c[cell] ?? 0) + 1
        }
        for (let a = 0; a < table.length; a++)
          for (let b = 0; b < nk; b++) {
            const cell = a * nk + b
            if ((c[cell] ?? 0) > 0)
              (table[a] as number[])[b] = (table[a]?.[b] ?? 0) + (lr * (sums[cell] ?? 0)) / ((c[cell] ?? 0) + 1e-9)
          }
        for (let i = 0; i < m.n; i++) {
          const cell = (pBins[j]?.[i] ?? 0) * nk + (pBins[k]?.[i] ?? 0)
          if ((c[cell] ?? 0) > 0) f[i] = (f[i] ?? 0) + (lr * (sums[cell] ?? 0)) / ((c[cell] ?? 0) + 1e-9)
        }
      })
      if (ctx.wants('boost/round')) ctx.log('boost/round', s.round + 1)
      if (ctx.wants('boost/trainLoss')) ctx.log('boost/trainLoss', lossOf(f))
      return { ...s, f, pairs: chosen, tables, round: s.round + 1 }
    },
    done: (s) => (s.round >= rounds + pairRounds ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CyclicState,
  }
  const model = (s: CyclicState): GlassBoxModel => {
    const terms: GlassBoxTerm[] = []
    let intercept = intercept0
    for (let j = 0; j < m.d; j++) {
      const shape = s.shapes[j] ?? []
      let c = 0
      for (let i = 0; i < m.n; i++) c += (shape[bins[j]?.[i] ?? 0] ?? 0) / m.n
      intercept += c
      terms.push({ features: [j] as const, edges: edges[j] ?? [], values: shape.map((v) => v - c) })
    }
    s.pairs?.forEach(({ j, k }, t) => {
      const table = s.tables[t] as readonly (readonly number[])[]
      let c = 0
      for (let i = 0; i < m.n; i++) c += (table[pBins[j]?.[i] ?? 0]?.[pBins[k]?.[i] ?? 0] ?? 0) / m.n
      intercept += c
      terms.push({
        features: [j, k] as const,
        edges: [pEdges[j] ?? [], pEdges[k] ?? []] as const,
        values: table.map((r) => r.map((v) => v - c)),
      })
    })
    return { task, intercept, terms }
  }
  return { algorithm, model }
}

/**
 * Explainable boosting (Lou, Caruana and Gehrke 2012 and 2013; InterpretML's ExplainableBoostingMachine): a
 * generalised additive model with a few pairwise interactions, f(x) = b + sum_j f_j(x_j) + sum_(j,k) f_jk(x_j, x_k),
 * fitted by cyclic gradient boosting: each round visits every feature in turn and moves its shape function by a small
 * step towards the mean residual (the negative gradient of the loss) in each of its quantile bins. Pairs are then
 * ranked by how much a coarse 2-D bin table reduces the residuals (FAST) and boosted the same way. Each term is
 * centred to mean zero over the data, the intercept carrying the rest, so a term reads as the feature's effect. This
 * runs {@link cyclicBoosting} to the end.
 *
 * @param data - Training rows and targets (0/1 for classification).
 * @param options - task, bins, pairBins, rounds, learningRate, interactions and pairRounds.
 * @returns The model.
 * @concept boosting
 * @example
 * ```ts
 * import { explainableBoosting } from 'mdml-engine/ensembles'
 * const x = Array.from({ length: 40 }, (_, i) => [i % 4, Math.floor(i / 4) % 2])
 * const y = x.map(([a, b]) => 2 * a + (b === 1 ? 5 : 0))
 * const m = explainableBoosting({ x, y }, { bins: 4, rounds: 300, learningRate: 0.2 })
 * // The shape of the second feature: a step of 5 between its two values.
 * Math.round(m.terms[1].values[1] - m.terms[1].values[0]) // => 5
 * ```
 */
export function explainableBoosting(data: LearnData, options: ExplainableBoostingOptions = {}): GlassBoxModel {
  const booster = cyclicBoosting(data, options)
  const steps =
    (options.rounds ?? 200) +
    (options.interactions !== undefined && options.interactions > 0 ? (options.pairRounds ?? 100) : 0)
  return booster.model(runAlgorithm(booster.algorithm, { maxSteps: steps, seed: 0 }).state)
}

/**
 * Each row's contribution from each term of a glass-box model: the term's value at the row (its local explanation,
 * exact by construction), so the intercept plus a row's contributions is its score.
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns contributions[row][term].
 * @example
 * ```ts
 * import { glassBoxContributions } from 'mdml-engine/ensembles'
 * const model = { task: 'regression', intercept: 1, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * glassBoxContributions(model, [[0], [1]]) // => [[-1], [1]]
 * ```
 */
export function glassBoxContributions(model: GlassBoxModel, x: TensorLike): number[][] {
  const m = matrixView(x, 'glassBoxContributions')
  return Array.from({ length: m.n }, (_, i) =>
    model.terms.map((t): number => {
      const v = (j: number): number => m.data[i * m.d + j] ?? 0
      if (t.features.length === 1)
        return (t.values as readonly number[])[binIndex(t.edges as readonly number[], v(t.features[0]))] ?? 0
      const [ex, ey] = t.edges as readonly [readonly number[], readonly number[]]
      const [a, b] = t.features as readonly [number, number]
      return (t.values as readonly (readonly number[])[])[binIndex(ex, v(a))]?.[binIndex(ey, v(b))] ?? 0
    }),
  )
}

/**
 * A glass-box model's predictions: the intercept plus the terms (a probability through the logistic function for
 * classification).
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns One prediction per row.
 * @example
 * ```ts
 * import { glassBoxPredict } from 'mdml-engine/ensembles'
 * const model = { task: 'regression', intercept: 1, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * glassBoxPredict(model, [[0], [1]]) // => [0, 2]
 * ```
 */
export function glassBoxPredict(model: GlassBoxModel, x: TensorLike): number[] {
  return glassBoxContributions(model, x).map((c) => {
    const s = c.reduce((a, b) => a + b, model.intercept)
    return model.task === 'classification' ? 1 / (1 + Math.exp(-s)) : s
  })
}

/**
 * The importance of each term (InterpretML's term importances): the mean absolute contribution over the rows.
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns One importance per term.
 * @example
 * ```ts
 * import { termImportances } from 'mdml-engine/ensembles'
 * const model = { task: 'regression', intercept: 0, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * termImportances(model, [[0], [1], [1]]) // => [1]
 * ```
 */
export function termImportances(model: GlassBoxModel, x: TensorLike): number[] {
  const c = glassBoxContributions(model, x)
  return model.terms.map((_, t) => c.reduce((s, row) => s + Math.abs(row[t] ?? 0), 0) / Math.max(1, c.length))
}

/** Parameters of a fitted explainable boosting machine: the model and the class labels it maps to 0 and 1. */
export type EbmParams = { readonly model: GlassBoxModel; readonly classes: number[]; readonly nFeatures: number }

/** Verbs and facts of a fitted explainable boosting machine. */
export interface EbmVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Each row's contribution from each term, `[n][terms]`. */
  readonly contributions: (x: TensorLike) => number[][]
  readonly nFeaturesIn: number
}

/** An explainable boosting machine estimator. */
export type EbmEstimator = IterativeEstimator<ExplainableBoostingOptions, CyclicState, EbmParams, EbmVerbs, CyclicPath>

function ebmEstimator(name: string, classifier: boolean, options: ExplainableBoostingOptions): EbmEstimator {
  const task = classifier ? 'classification' : 'regression'
  const dataOf = (data: LearnData): LearnData =>
    classifier ? { ...data, y: Array.from(encodeLabels(data.y ?? [], name).codes) } : data
  return iterative<ExplainableBoostingOptions, CyclicState, EbmParams, EbmVerbs, CyclicPath>({
    name,
    kind: classifier ? 'classifier' : 'regressor',
    defaults: options,
    hyperSpace: (h) =>
      dict({
        rounds: int(1, Math.max(500, h.rounds ?? 200), { default: h.rounds ?? 200, label: 'rounds' }),
        learningRate: real(0.005, 1, { log: true, default: h.learningRate ?? 0.05, label: 'learning rate' }),
        bins: int(2, Math.max(64, h.bins ?? 32), { default: h.bins ?? 32, label: 'bins per feature' }),
      }),
    algorithm: (data, h) => cyclicBoosting(dataOf(data), { ...h, task }).algorithm,
    params: (state, data, h) => ({
      model: cyclicBoosting(dataOf(data), { ...h, task }).model(state),
      classes: classifier ? [...encodeLabels(data.y ?? [], name).classes] : [],
      nFeatures: matrixView(data.x, name).d,
    }),
    verbs: (p) => {
      const proba = (x: TensorLike): Tensor => {
        const q = glassBoxPredict(p.model, x)
        return fromStorage(Float64Array.from(q.flatMap((v) => [1 - v, v])), [q.length, 2], 'float64')
      }
      if (!classifier)
        return withDefaultScore('regressor', {
          predict: (x: TensorLike) =>
            fromStorage(Float64Array.from(glassBoxPredict(p.model, x)), [matrixView(x, name).n], 'float64'),
          contributions: (x: TensorLike) => glassBoxContributions(p.model, x),
          nFeaturesIn: p.nFeatures,
        })
      return withDefaultScore('classifier', {
        predict: (x: TensorLike) => {
          const q = glassBoxPredict(p.model, x)
          return fromStorage(
            Float64Array.from(q, (v) => p.classes[v > 0.5 ? 1 : 0] ?? 0),
            [q.length],
            'float64',
          )
        },
        predictProba: proba,
        contributions: (x: TensorLike) => glassBoxContributions(p.model, x),
        classes: p.classes,
        nFeaturesIn: p.nFeatures,
      })
    },
    maxSteps: (h) =>
      (h.rounds ?? 200) + (h.interactions !== undefined && h.interactions > 0 ? (h.pairRounds ?? 100) : 0),
    contract: (data, h): Contract<CyclicState, CyclicPath> =>
      defineContract({
        'boost/round': scalar({ description: 'cycles over the features (then the pairs) so far' }),
        'boost/trainLoss': scalar({ description: 'the training loss (squared error or log loss)', direction: 'lower' }),
        'model/shapeFunctions': derived<CyclicState>({
          description: 'the terms of the model at this cycle: a shape function per feature, tables for the pairs',
          compute: (state) => cyclicBoosting(dataOf(data), { ...h, task }).model(state) as never,
        }),
      }),
    logs: { 'boost/round': Log(), 'boost/trainLoss': Log() },
  })
}

/**
 * An explainable boosting machine for regression as an estimator on the iterative base: a session runs one cycle over
 * the features per step, logging `boost/round` and `boost/trainLoss`, and the probe `model/shapeFunctions` is the
 * model (a shape function per feature and tables for the interaction pairs) at any cycle: the shape functions as they
 * were learnt, round by round.
 *
 * @param options - bins, rounds, learning rate, interactions and pair rounds.
 * @returns The estimator.
 * @concept boosting
 * @example
 * ```ts
 * import { explainableBoostingRegressor } from 'mdml-engine/ensembles'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i % 4, Math.floor(i / 4) % 2])
 * const y = x.map(([a = 0, b = 0]) => 2 * a + 5 * b)
 * toFlat(explainableBoostingRegressor({ bins: 4, rounds: 200, learningRate: 0.3 }).fit({ x, y }).predict([[3, 1]])).map(Math.round) // => [11]
 * ```
 */
export function explainableBoostingRegressor(options: Omit<ExplainableBoostingOptions, 'task'> = {}): EbmEstimator {
  return ebmEstimator('explainable-boosting-regressor', false, options)
}

/**
 * An explainable boosting machine for binary classification (labels any two values, mapped to 0 and 1): the log-odds
 * is a sum of shape functions. Logging and probes as {@link explainableBoostingRegressor}.
 *
 * @param options - bins, rounds, learning rate, interactions and pair rounds.
 * @returns The estimator.
 * @concept boosting
 * @example
 * ```ts
 * import { explainableBoostingClassifier } from 'mdml-engine/ensembles'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * toFlat(explainableBoostingClassifier({ bins: 8, rounds: 100, learningRate: 0.3 }).fit({ x, y }).predict([[0.1], [0.9]])) // => [0, 1]
 * ```
 */
export function explainableBoostingClassifier(options: Omit<ExplainableBoostingOptions, 'task'> = {}): EbmEstimator {
  return ebmEstimator('explainable-boosting-classifier', true, options)
}
