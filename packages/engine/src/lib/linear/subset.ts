/**
 * Variable selection for linear regression by an information criterion: best-subset selection (every subset of each
 * size, exactly), forward selection, backward elimination, and stepwise selection in both directions, scored by AIC,
 * BIC, the small-sample AICc, Mallows' Cp or adjusted R-squared. The search is an `Algorithm` with one move (or one
 * subset size) per step, so the path of criteria can be watched; the subsets are scored from the Gram matrix of the
 * centred data, so a candidate costs one small Cholesky solve.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type IterativeEstimator, iterative, type LearnData, modelSelection } from '../learn/index.ts'
import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { choice, dict } from '../space/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { centre, designOf } from './design.ts'

/** How the subsets are searched. */
export type SubsetMethod = 'best' | 'forward' | 'backward' | 'stepwise'

/** The criterion that picks among subsets. */
export type SubsetCriterion = 'aic' | 'bic' | 'aicc' | 'cp' | 'adjusted-r2'

/** The hyperparameters of {@link subsetRegression}. */
export interface SubsetHyper {
  readonly method: SubsetMethod
  readonly criterion: SubsetCriterion
  /** The largest subset to consider (default all the features; best-subset needs at most 24 features). */
  readonly maxSize: number | null
  readonly fitIntercept: boolean
}

/** One subset on the path: its members and its criterion. */
export type SubsetStep = {
  readonly subset: number[]
  /** The criterion in its natural scale (AIC, BIC, AICc, Cp, or adjusted R-squared). */
  readonly criterion: number
  readonly rss: number
}

/** The state of {@link subsetAlgorithm}. */
export type SubsetState = {
  /** The current subset (sorted feature indices). */
  readonly subset: number[]
  /** The criterion of the current subset (to be minimised, adjusted R-squared negated). */
  readonly score: number
  /** The best subset seen (the answer for `best`, equal to `subset` for the greedy methods). */
  readonly best: number[]
  readonly bestScore: number
  /** Every subset visited as the current one, in order. */
  readonly path: SubsetStep[]
  readonly iteration: number
  readonly finished: boolean
}

/** The sufficient statistics of the centred data: enough to fit any subset. */
interface Gram {
  readonly n: number
  readonly d: number
  readonly g: Float64Array
  readonly c: Float64Array
  readonly yy: number
  /** Parameters beyond the subset's slopes in the information criteria: one (statsmodels: `df_model + 1`). */
  readonly extra: number
  readonly sigma2Full: number
}

function gramOf(
  data: LearnData,
  fitIntercept: boolean,
): Gram & { readonly xMean: Float64Array; readonly yMean: number } {
  const design = designOf(data, 'subsetRegression')
  const { n, d } = design.x
  const cen = centre(design, fitIntercept)
  const g = new Float64Array(d * d)
  const c = new Float64Array(d)
  let yy = 0
  for (let i = 0; i < n; i++) {
    const yi = cen.y[i] ?? 0
    yy += yi * yi
    for (let j = 0; j < d; j++) {
      const xj = cen.x[i * d + j] ?? 0
      c[j] = (c[j] ?? 0) + xj * yi
      for (let k = 0; k <= j; k++) g[j * d + k] = (g[j * d + k] ?? 0) + xj * (cen.x[i * d + k] ?? 0)
    }
  }
  for (let j = 0; j < d; j++) for (let k = j + 1; k < d; k++) g[j * d + k] = g[k * d + j] ?? 0
  const extra = 1
  const base = { n, d, g, c, yy, extra, sigma2Full: Number.NaN }
  const full = fitSubset(
    base,
    Array.from({ length: d }, (_, j) => j),
  )
  const dfFull = n - d - (fitIntercept ? 1 : 0)
  return { ...base, sigma2Full: dfFull > 0 ? full.rss / dfFull : Number.NaN, xMean: cen.xMean, yMean: cen.yMean }
}

/** The least-squares fit of a subset from the Gram matrix: coefficients and residual sum of squares. */
function fitSubset(
  gram: Pick<Gram, 'd' | 'g' | 'c' | 'yy'>,
  subset: readonly number[],
): { coef: Float64Array; rss: number } {
  const k = subset.length
  if (k === 0) return { coef: new Float64Array(0), rss: gram.yy }
  const a = new Float64Array(k * k)
  const b = new Float64Array(k)
  subset.forEach((j, r) => {
    b[r] = gram.c[j] ?? 0
    subset.forEach((l, s) => {
      a[r * k + s] = gram.g[j * gram.d + l] ?? 0
    })
  })
  const chol = dpotrf(a, k)
  if (chol.info !== 0) return { coef: new Float64Array(k), rss: Number.POSITIVE_INFINITY }
  const coef = dpotrs(chol.factor, b, k)
  let fit = 0
  for (let r = 0; r < k; r++) fit += (coef[r] ?? 0) * (b[r] ?? 0)
  return { coef, rss: Math.max(gram.yy - fit, 0) }
}

/** The criterion of a subset in its natural scale, and the value to minimise. */
function criterionOf(
  gram: Gram,
  rss: number,
  size: number,
  criterion: SubsetCriterion,
  fitIntercept: boolean,
): { natural: number; score: number } {
  const { n } = gram
  if (!Number.isFinite(rss)) return { natural: Number.NaN, score: Number.POSITIVE_INFINITY }
  const k = size + gram.extra
  const llf = modelSelection.gaussianLogLikelihood(Math.max(rss, 1e-300), n)
  let natural: number
  switch (criterion) {
    case 'aic':
      natural = modelSelection.aic(llf, k)
      break
    case 'bic':
      natural = modelSelection.bic(llf, k, n)
      break
    case 'aicc':
      natural = n > k + 1 ? modelSelection.aicc(llf, k, n) : Number.POSITIVE_INFINITY
      break
    case 'cp':
      natural = modelSelection.mallowsCp(rss, gram.sigma2Full, n, size + (fitIntercept ? 1 : 0))
      break
    case 'adjusted-r2': {
      // R-squared about the mean (or about zero without an intercept): 1 - (rss / df) / (tss / (n - 1)).
      const dfResid = n - size - (fitIntercept ? 1 : 0)
      natural = dfResid > 0 ? 1 - rss / dfResid / (gram.yy / (n - (fitIntercept ? 1 : 0))) : Number.NEGATIVE_INFINITY
      break
    }
  }
  return { natural, score: criterion === 'adjusted-r2' ? -natural : natural }
}

/** Every subset of `0..d-1` with exactly `size` members, in lexicographic order. */
function* combinations(d: number, size: number): Generator<number[]> {
  const idx = Array.from({ length: size }, (_, i) => i)
  if (size > d) return
  while (true) {
    yield [...idx]
    let i = size - 1
    while (i >= 0 && idx[i] === d - size + i) i--
    if (i < 0) return
    idx[i] = (idx[i] ?? 0) + 1
    for (let j = i + 1; j < size; j++) idx[j] = (idx[j - 1] ?? 0) + 1
  }
}

/** Options of {@link subsetAlgorithm}. */
export interface SubsetOptions {
  readonly method?: SubsetMethod
  readonly criterion?: SubsetCriterion
  readonly maxSize?: number | null
  readonly fitIntercept?: boolean
}

/**
 * Subset selection as a traced algorithm. For `best` each step finds the best subset of the next size (so step `k` is
 * the best `k`-subset) and the answer is the best over the sizes; for `forward` each step adds the feature that most
 * improves the criterion and stops when none does; `backward` starts from every feature and removes one at a time;
 * `stepwise` tries the best addition and the best removal each step and takes the better, if it improves.
 *
 * @param data - Features and targets.
 * @param options - The method, criterion, size limit and intercept.
 * @returns The algorithm.
 * @throws OmaiError `value` for best-subset search over more than 24 features or Cp without residual degrees of freedom.
 * @example
 * ```ts
 * import { subsetAlgorithm } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(3 * i), (i % 5) / 5])
 * const y = x.map((r, i) => 2 * (r[0] ?? 0) + 0.05 * Math.sin(7 * i))
 * runAlgorithm(subsetAlgorithm({ x, y }, { method: 'forward', criterion: 'bic' }), { maxSteps: 10 }).state.best // => [0]
 * ```
 */
export function subsetAlgorithm(data: LearnData, options: SubsetOptions = {}): Algorithm<SubsetState> {
  const method = options.method ?? 'forward'
  const criterion = options.criterion ?? 'bic'
  const fitIntercept = options.fitIntercept ?? true
  const gram = gramOf(data, fitIntercept)
  const { d } = gram
  const maxSize = Math.min(options.maxSize ?? d, d)
  if (method === 'best' && d > 24)
    throw new OmaiError('value', `subsetRegression: best-subset search over ${d} features is too large (at most 24)`)
  if (criterion === 'cp' && !Number.isFinite(gram.sigma2Full))
    throw new OmaiError('value', 'subsetRegression: Cp needs more rows than features for the full-model variance')
  const evaluate = (subset: readonly number[]): { natural: number; score: number; rss: number } => {
    const { rss } = fitSubset(gram, subset)
    return { ...criterionOf(gram, rss, subset.length, criterion, fitIntercept), rss }
  }
  const stepOf = (subset: number[]): SubsetStep => {
    const e = evaluate(subset)
    return { subset, criterion: e.natural, rss: e.rss }
  }
  const startSubset = method === 'backward' ? Array.from({ length: Math.min(d, maxSize) }, (_, j) => j) : []
  return {
    name: `subset-${method}`,
    config: { method, criterion, maxSize, fitIntercept } as Json,
    schema: channels({
      metrics: {
        criterion: { description: `the ${criterion} of the current subset` },
        size: { description: 'the number of features in the current subset' },
      },
      tensors: { support: { role: 'state', description: 'whether each feature is in the current subset', shape: [d] } },
    }),
    init: () => {
      const e = evaluate(startSubset)
      return {
        subset: startSubset,
        score: e.score,
        best: startSubset,
        bestScore: e.score,
        path: [{ subset: startSubset, criterion: e.natural, rss: e.rss }],
        iteration: 0,
        finished: false,
      }
    },
    step: (s) => {
      if (method === 'best') {
        const size = s.iteration + 1
        if (size > maxSize) return { ...s, finished: true }
        let winner: number[] = []
        let winnerScore = Number.POSITIVE_INFINITY
        for (const subset of combinations(d, size)) {
          const e = evaluate(subset)
          if (e.score < winnerScore) {
            winner = subset
            winnerScore = e.score
          }
        }
        const better = winnerScore < s.bestScore
        return {
          subset: winner,
          score: winnerScore,
          best: better ? winner : s.best,
          bestScore: better ? winnerScore : s.bestScore,
          path: [...s.path, stepOf(winner)],
          iteration: s.iteration + 1,
          finished: size >= maxSize,
        }
      }
      type Move = { subset: number[]; score: number }
      let move: Move | undefined
      const consider = (subset: number[]): void => {
        const e = evaluate(subset)
        if (e.score < (move?.score ?? Number.POSITIVE_INFINITY)) move = { subset, score: e.score }
      }
      const addAllowed = method !== 'backward' && s.subset.length < maxSize
      const dropAllowed = method !== 'forward' && s.subset.length > 0
      if (addAllowed)
        for (let j = 0; j < d; j++) if (!s.subset.includes(j)) consider([...s.subset, j].sort((a, b) => a - b))
      if (dropAllowed) for (const j of s.subset) consider(s.subset.filter((v) => v !== j))
      const chosen = move as Move | undefined
      // Stop at a local optimum of the criterion; stepwise must also not undo its last move forever (strict gain).
      if (chosen === undefined || !(chosen.score < s.score - 1e-12)) return { ...s, finished: true }
      return {
        subset: chosen.subset,
        score: chosen.score,
        best: chosen.subset,
        bestScore: chosen.score,
        path: [...s.path, stepOf(chosen.subset)],
        iteration: s.iteration + 1,
        finished: false,
      }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { criterion: s.path[s.path.length - 1]?.criterion ?? Number.NaN, size: s.subset.length },
      tensors: {
        support: {
          role: 'state',
          value: Array.from({ length: d }, (_, j) => (s.subset.includes(j) ? 1 : 0)),
          shape: [d],
        },
      },
    }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as SubsetState,
  }
}

/** The fitted parameters of {@link subsetRegression}. */
export type SubsetParams = {
  readonly method: SubsetMethod
  readonly criterion: SubsetCriterion
  /** The selected feature indices, sorted. */
  readonly selected: number[]
  readonly support: boolean[]
  /** The coefficients over all features, zero for those left out. */
  readonly coef: number[]
  readonly intercept: number
  /** The criterion of the selected subset, in its natural scale. */
  readonly criterionValue: number
  /** The subsets visited, with their criteria. */
  readonly path: SubsetStep[]
}

/**
 * Linear regression with the features chosen by an information criterion: `best` examines every subset (exact),
 * `forward`, `backward` and `stepwise` move greedily; the criterion is AIC, BIC, AICc, Mallows' Cp or adjusted
 * R-squared, with the Gaussian likelihood and parameter counts of statsmodels' OLS (`df_model + 1`). The model is then ordinary least squares on the selected features.
 *
 * @param hyper - `method`, `criterion`, `maxSize`, `fitIntercept`.
 * @returns The estimator.
 * @reference A. Miller, Subset Selection in Regression (2002); G. Furnival and R. Wilson, Regressions by leaps and
 * bounds, Technometrics 16 (1974).
 * @example
 * ```ts
 * import { subsetRegression } from 'omai-engine/linear'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(2.3 * i), Math.sin(5.1 * i)])
 * const y = x.map((r, i) => 3 * (r[0] ?? 0) - 2 * (r[2] ?? 0) + 0.05 * Math.cos(9 * i))
 * subsetRegression({ method: 'best', criterion: 'bic' }).fit({ x, y }).params.selected // => [0, 2]
 * ```
 */
export function subsetRegression(
  hyper: Partial<SubsetHyper> = {},
): IterativeEstimator<SubsetHyper, SubsetState, SubsetParams, AffineVerbs> {
  const defaults: SubsetHyper = { method: 'forward', criterion: 'bic', maxSize: null, fitIntercept: true }
  return iterative<SubsetHyper, SubsetState, SubsetParams, AffineVerbs>(
    {
      name: 'subset-regression',
      kind: 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          method: choice(['best', 'forward', 'backward', 'stepwise'], { default: h.method }),
          criterion: choice(['aic', 'bic', 'aicc', 'cp', 'adjusted-r2'], { default: h.criterion }),
        }),
      algorithm: (data, h) => subsetAlgorithm(data, h),
      maxSteps: (h, data) => (h.maxSize ?? Math.max(designOf(data, 'subsetRegression').x.d, 1)) * 2 + 2,
      params: (s, data, h) => {
        const gram = gramOf(data, h.fitIntercept)
        const { coef } = fitSubset(gram, s.best)
        const full = new Array<number>(gram.d).fill(0)
        s.best.forEach((j, r) => {
          full[j] = coef[r] ?? 0
        })
        const e = criterionOf(gram, fitSubset(gram, s.best).rss, s.best.length, h.criterion, h.fitIntercept)
        return {
          method: h.method,
          criterion: h.criterion,
          selected: [...s.best],
          support: Array.from({ length: gram.d }, (_, j) => s.best.includes(j)),
          coef: full,
          intercept: h.fitIntercept ? gram.yMean - full.reduce((a, b, j) => a + b * (gram.xMean[j] ?? 0), 0) : 0,
          criterionValue: e.natural,
          path: s.path,
        }
      },
      verbs: (par) => affineVerbs(par.coef, par.intercept, 'subsetRegression'),
      logs: { 'solver/criterion': Log(), 'solver/size': Log(), 'solver/support': Log({ detail: 'full' }) },
    },
    { ...defaults, ...hyper },
  )
}
