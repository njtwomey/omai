/**
 * Choosing the graphical-lasso penalty: `selectGraphicalLassoPenalty` runs the penalty path of the graphical lasso
 * (each penalty warm-started from the previous one, on the library's `graphicalLassoAlgorithm`) and picks a penalty by
 * one of three rules, the `method` slot: `'cv'` (scikit-learn's `GraphicalLassoCV`: the held-out Gaussian
 * log-likelihood over k folds, on a grid that is refined around the best penalty), `'ebic'` (Foygel and Drton's extended
 * BIC, which charges each edge `log n + 4 gamma log p`) and `'stars'` (Liu, Roeder and Wasserman's stability approach:
 * the sparsest graph whose edges are stable under subsampling). The selection is one traced `Algorithm` with a step per
 * penalty and a final refit, and its exposition is the criterion along the penalties beside the path of the partial
 * correlations.
 */

import { OmaiError } from '../base/index.ts'
import {
  type CrossValidationCurve,
  crossValidationCurve,
  type RegularisationPath,
  regularisationPath,
} from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  featureNamesOf,
  type IterativeEstimator,
  type IterativeFitted,
  iterative,
  matrixView,
} from '../learn/index.ts'
import { kFold, type Splitter, splitIndices } from '../learn/model-selection/index.ts'
import { linalg } from '../numerics/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { bool, choice, dict, int } from '../space/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { asTensor, item, type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { precisionGraph } from './evaluation.ts'
import { graphicalLasso } from './graphical-lasso.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs } from './shrinkage.ts'

/** The rule that picks the penalty: held-out likelihood, extended BIC or stability under subsampling. */
export type PenaltySelector = 'cv' | 'ebic' | 'stars'

/** Options of {@link graphicalLassoSelectionAlgorithm} and {@link selectGraphicalLassoPenalty}. */
export interface PenaltySelectionOptions {
  /** The rule (default `'cv'`). */
  readonly method?: PenaltySelector
  /**
   * The penalties, or how many to put on the grid: log-spaced from the smallest penalty with an empty graph (the largest
   * off-diagonal covariance; for `'stars'` the largest over the subsamples too) down to `minRatio` times it (default 4
   * for `'cv'`, as scikit-learn, whose grid is then refined; 15 for `'ebic'`; 20 for `'stars'`). An explicit list is
   * walked from the largest down and is not refined.
   */
  readonly alphas?: readonly number[] | number
  /** The smallest penalty of a grid given by count, as a fraction of the largest (default 0.01; 0.1 for `'stars'`, as `huge`). */
  readonly minRatio?: number
  /** The grid refinements of `'cv'` for a grid given by count (default 4, as scikit-learn). */
  readonly nRefinements?: number
  /** The cross-validation splits of `'cv'` (default `kFold(5)`: contiguous folds, as scikit-learn). */
  readonly cv?: Splitter
  /** The weight of the edge-count term of `'ebic'` (default 0.5; 0 is the BIC). */
  readonly gamma?: number
  /** The number of subsamples of `'stars'` (default 20). */
  readonly nSubsamples?: number
  /** The subsample size of `'stars'` (default `floor(10 sqrt(n))` above 144 rows, else `floor(0.8 n)`; below `n`). */
  readonly subsampleSize?: number
  /** The instability the chosen graph may have, for `'stars'` (default 0.05). */
  readonly beta?: number
  /** The graphical lasso's duality-gap tolerance (default 1e-4). */
  readonly tol?: number
  /** The inner lasso tolerance (default 1e-4). */
  readonly enetTol?: number
  /** The sweep cap of the final fit (default 100); the cross-validation paths use a tenth of it, as scikit-learn. */
  readonly maxIter?: number
}

/** One penalty of the path: its criterion and the full-data fit there. */
export interface PenaltyEntry {
  readonly alpha: number
  /** `'cv'`: the held-out log-likelihood on each fold; empty for the other rules. */
  readonly scores: number[]
  /** `'cv'`: the mean held-out log-likelihood (higher is better); `'ebic'`: the EBIC (lower); `'stars'`: the instability `D` (lower). */
  readonly criterion: number
  /** The number of edges of the full-data fit. */
  readonly edges: number
  /** The duality gap of the full-data fit. */
  readonly gap: number
  /** The precision matrix of the full-data fit. */
  readonly precision: number[][]
}

/** The state of {@link graphicalLassoSelectionAlgorithm}. */
export interface PenaltySelectionState {
  /** `'path'` while penalties remain, `'final'` for the refit step, `'done'` after it. */
  readonly stage: 'path' | 'final' | 'done'
  /** The grid refinement (0 is the first grid). */
  readonly refinement: number
  /** The penalties of the grid being walked, descending, and the position in it. */
  readonly grid: number[]
  readonly cursor: number
  /** The penalties evaluated so far, in the order of evaluation. */
  readonly entries: PenaltyEntry[]
  /** The covariance each fit unit ended its last penalty with (the warm start of the next); unit 0 is the full data. */
  readonly warm: (number[][] | null)[]
  /** The training and test rows of each fold (`'cv'`). */
  readonly folds: { readonly train: number[]; readonly test: number[] }[]
  /** The rows of each subsample (`'stars'`). */
  readonly subsamples: number[][]
  /** The chosen penalty (NaN before the final step). */
  readonly chosen: number
  /** The refit at the chosen penalty. */
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly gap: number
  readonly objective: number
  readonly nIter: number
  /** `'cv'`: the held-out log-likelihood of the unpenalised (empirical covariance) fit on each fold. */
  readonly unpenalised: number[]
}

/**
 * The mean Gaussian log-likelihood per observation of a precision matrix under a covariance (scikit-learn's
 * `log_likelihood`): `(-tr(S Theta) + log det Theta - p log 2 pi) / 2`.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param precision - The precision matrix `[p][p]`.
 * @returns The mean log-likelihood (`-Infinity` when the precision is not positive definite).
 * @example
 * ```ts
 * import { gaussianLogLikelihood } from 'omai-engine/covariance'
 * gaussianLogLikelihood([[1]], [[1]]) // => -1.4189385332046727
 * ```
 */
export function gaussianLogLikelihood(
  s: readonly (readonly number[])[],
  precision: readonly (readonly number[])[],
): number {
  const p = s.length
  let trace = 0
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) trace += (s[a]?.[b] ?? 0) * (precision[b]?.[a] ?? 0)
  const { sign, logabsdet } = linalg.slogdet(precision as number[][])
  if (item(sign) <= 0) return Number.NEGATIVE_INFINITY
  return (-trace + item(logabsdet) - p * Math.log(2 * Math.PI)) / 2
}

/**
 * Foygel and Drton's extended BIC of a Gaussian graphical model: `-2 l + |E| (log n + 4 gamma log p)`, with `l` the
 * log-likelihood of the sample at the estimate and `|E|` the number of edges. With `gamma = 0` it is the BIC (counting
 * the edges alone, the diagonal's `p` parameters being the same everywhere); a positive `gamma` raises the
 * penalty on dense graphs, which keeps false edges out when `p` is large.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param precision - The estimated precision matrix `[p][p]`.
 * @param n - The number of observations behind `s`.
 * @param gamma - The weight of the edge-count term in `[0, 1]` (default 0.5).
 * @returns The criterion (lower is better).
 * @example
 * ```ts
 * import { graphicalLassoEbic } from 'omai-engine/covariance'
 * graphicalLassoEbic([[1, 0], [0, 1]], [[1, 0], [0, 1]], 100, 0) // => 567.575413281869
 * ```
 */
export function graphicalLassoEbic(
  s: readonly (readonly number[])[],
  precision: readonly (readonly number[])[],
  n: number,
  gamma = 0.5,
): number {
  const p = s.length
  let edges = 0
  for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if (Math.abs(precision[a]?.[b] ?? 0) > 1e-10) edges++
  return -2 * n * gaussianLogLikelihood(s, precision) + edges * (Math.log(n) + 4 * gamma * Math.log(p))
}

function covarianceOfRows(rows: readonly (readonly number[])[], assumeCentered = false): number[][] {
  return toArray(empiricalCovariance(asTensor(rows as number[][]), { assumeCentered })) as number[][]
}

/** scikit-learn's `alpha_max`: the largest off-diagonal magnitude, the smallest penalty with an empty graph. */
function alphaMax(s: readonly (readonly number[])[]): number {
  let m = 0
  for (let a = 0; a < s.length; a++)
    for (let b = 0; b < s.length; b++) if (a !== b) m = Math.max(m, Math.abs(s[a]?.[b] ?? 0))
  return m
}

/** `count` penalties log-spaced from `high` down to `low` (descending). */
function logGrid(high: number, low: number, count: number): number[] {
  if (count === 1) return [high]
  const a = Math.log10(high)
  const b = Math.log10(low)
  return Array.from({ length: count }, (_, i) => 10 ** (a + ((b - a) * i) / (count - 1)))
}

/** The entries sorted by descending penalty. */
function sortedEntries(entries: readonly PenaltyEntry[]): PenaltyEntry[] {
  return [...entries].sort((a, b) => b.alpha - a.alpha)
}

/** The monotone instability of StARS: the running maximum from the largest penalty down. */
function monotone(values: readonly number[]): number[] {
  let top = Number.NEGATIVE_INFINITY
  return values.map((v) => {
    top = Math.max(top, v)
    return top
  })
}

/** The criterion of each penalty (penalties descending), as the rule compares them, and the index it picks. */
function pick(
  method: PenaltySelector,
  path: readonly PenaltyEntry[],
  beta: number,
): { readonly criterion: number[]; readonly best: number } {
  if (method === 'stars') {
    const criterion = monotone(path.map((e) => e.criterion))
    let best = 0
    for (let i = 0; i < criterion.length; i++) if ((criterion[i] ?? Number.POSITIVE_INFINITY) <= beta) best = i
    return { criterion, best }
  }
  const criterion = path.map((e) => e.criterion)
  const sign = method === 'cv' ? 1 : -1
  let bestScore = Number.NEGATIVE_INFINITY
  let best = 0
  // As scikit-learn: ties go to the smaller penalty; an unusable (non-finite or absurdly large) score never wins.
  for (const [i, c] of criterion.entries()) {
    const score = method === 'cv' && c >= 0.1 / Number.EPSILON ? Number.NaN : sign * c
    if (score >= bestScore) {
      bestScore = score
      best = i
    }
  }
  return { criterion, best }
}

/** The next grid of a refinement of `'cv'`, by scikit-learn's rules around the best penalty. */
function refinedGrid(path: readonly PenaltyEntry[], best: number, count: number): number[] {
  const alpha = (i: number): number => path[i]?.alpha ?? 0
  let finite = 0
  for (const [i, e] of path.entries()) {
    const c = e.criterion
    if (Number.isFinite(c) && c < 0.1 / Number.EPSILON) finite = i
  }
  let high: number
  let low: number
  if (best === 0) {
    high = alpha(0)
    low = alpha(1)
  } else if (best === finite && best !== path.length - 1) {
    high = alpha(best)
    low = alpha(best + 1)
  } else if (best === path.length - 1) {
    high = alpha(best)
    low = 0.01 * alpha(best)
  } else {
    high = alpha(best - 1)
    low = alpha(best + 1)
  }
  return logGrid(high, low, count + 2).slice(1, -1)
}

/**
 * The graphical-lasso penalty selection as a traced {@link Algorithm}: each step evaluates one penalty on the grid
 * (the full-data fit, warm-started from the previous penalty, and the rule's own fits: the held-out likelihood on each
 * fold, the EBIC of the full fit, the edge instability over subsamples), a `'cv'` grid is refined around its best penalty
 * as scikit-learn does, and the last step refits at the chosen penalty. It records the penalty, the criterion, the
 * number of edges and the duality gap of each step.
 *
 * @param x - Observations `[n, p]`.
 * @param options - The rule, the grid and the rule's settings.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for fewer than 4 rows, or data with no correlation to select a penalty for.
 * @example
 * ```ts
 * import { graphicalLassoSelectionAlgorithm } from 'omai-engine/covariance'
 * graphicalLassoSelectionAlgorithm([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], { method: 'ebic' }).name // => 'graphical-lasso-selection'
 * ```
 */
export function graphicalLassoSelectionAlgorithm(
  x: TensorLike,
  options: PenaltySelectionOptions = {},
): Algorithm<PenaltySelectionState> {
  const m = matrixView(x, 'selectGraphicalLassoPenalty', { minSamples: 4 })
  const { n, d: p } = m
  if (p < 2) throw new OmaiError('shape', 'selectGraphicalLassoPenalty: need at least two variables')
  const rows: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: p }, (__, j) => m.data[i * p + j] ?? 0),
  )
  const method = options.method ?? 'cv'
  const maxIter = options.maxIter ?? 100
  const pathIter = Math.max(1, Math.floor(0.1 * maxIter))
  const explicit = Array.isArray(options.alphas)
  const count = typeof options.alphas === 'number' ? options.alphas : method === 'cv' ? 4 : method === 'ebic' ? 15 : 20
  const minRatio = options.minRatio ?? (method === 'stars' ? 0.1 : 0.01)
  const nRefinements = explicit || method !== 'cv' ? 1 : (options.nRefinements ?? 4)
  const gamma = options.gamma ?? 0.5
  const beta = options.beta ?? 0.05
  const nSubsamples = options.nSubsamples ?? 20
  if (!(count >= 1) || !Number.isInteger(count))
    throw new OmaiError('value', 'selectGraphicalLassoPenalty: alphas must be a list or a positive count')
  if (explicit)
    for (const a of options.alphas as readonly number[])
      if (!(a >= 0)) throw new OmaiError('value', 'selectGraphicalLassoPenalty: penalties must be non-negative')
  const full = covarianceOfRows(rows)
  const top = alphaMax(full)
  if (!explicit && !(top > 0))
    throw new OmaiError('value', 'selectGraphicalLassoPenalty: the variables are uncorrelated')
  const tol = options.tol ?? 1e-4
  const enetTol = options.enetTol ?? 1e-4
  const fit = (s: number[][], alpha: number, warm: number[][] | null, sweeps: number) => {
    try {
      const r = graphicalLasso(s, alpha, {
        tol,
        enetTol,
        maxIter: sweeps,
        ...(warm === null ? {} : { start: { covariance: warm } }),
      })
      return r.precision.every((row) => row.every(Number.isFinite)) ? r : null
    } catch {
      return null
    }
  }
  const edgeCount = (prec: number[][]): number => {
    let e = 0
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if (Math.abs(prec[a]?.[b] ?? 0) > 1e-10) e++
    return e
  }
  const subsampleSize = Math.min(
    n - 1,
    Math.max(2, options.subsampleSize ?? (n > 144 ? Math.floor(10 * Math.sqrt(n)) : Math.floor(0.8 * n))),
  )
  return {
    name: 'graphical-lasso-selection',
    config: { method, gamma, beta, nSubsamples, count, nRefinements, explicit },
    schema: channels({
      metrics: {
        alpha: { description: 'the penalty being evaluated' },
        criterion: {
          description:
            "the rule's criterion at the penalty: held-out log-likelihood (cv), EBIC or instability (lower is better)",
        },
        edges: { description: 'the number of edges of the full-data fit at the penalty' },
        gap: { description: 'the duality gap of the full-data fit', direction: 'lower' },
      },
      tensors: {
        precision: { role: 'param', description: 'the full-data precision at the penalty', shape: ['p', 'p'] },
      },
    }),
    init: (key: Key) => {
      let folds: PenaltySelectionState['folds'] = []
      let subsamples: number[][] = []
      let units = 1
      if (method === 'cv') {
        folds = splitIndices(options.cv ?? kFold(5), { n }, foldIn(key, 0)).map((f) => ({
          train: Array.from(f.train),
          test: Array.from(f.test),
        }))
        units += folds.length
      } else if (method === 'stars') {
        subsamples = Array.from({ length: nSubsamples }, (_, b) =>
          Array.from(permutation(foldIn(foldIn(key, 1), b), n).data.subarray(0, subsampleSize)).sort((a, c) => a - c),
        )
        units += subsamples.length
      }
      // Under StARS the first penalty should give an empty graph on every subsample too.
      const high = subsamples.reduce(
        (m, idx) => Math.max(m, alphaMax(covarianceOfRows(idx.map((i) => rows[i] as number[])))),
        top,
      )
      const grid = explicit
        ? [...(options.alphas as readonly number[])].sort((a, b) => b - a)
        : logGrid(high, minRatio * high, count)
      return {
        stage: 'path',
        refinement: 0,
        grid,
        cursor: 0,
        entries: [],
        warm: new Array<number[][] | null>(units).fill(null),
        folds,
        subsamples,
        chosen: Number.NaN,
        covariance: [],
        precision: [],
        gap: Number.NaN,
        objective: Number.NaN,
        nIter: 0,
        unpenalised: [],
      } satisfies PenaltySelectionState
    },
    step: (st) => {
      if (st.stage === 'final') {
        const path = sortedEntries(st.entries)
        const { best } = pick(method, path, beta)
        const chosen = path[best]?.alpha ?? 0
        const out = graphicalLasso(full, chosen, { tol, enetTol, maxIter })
        const unpenalised =
          method === 'cv'
            ? st.folds.map((f) => {
                const train = f.train.map((i) => rows[i] as number[])
                const mu = Array.from(
                  { length: p },
                  (_, j) => train.reduce((s, r) => s + (r[j] ?? 0), 0) / train.length,
                )
                const test = covarianceOfRows(
                  f.test.map((i) => (rows[i] as number[]).map((v, j) => v - (mu[j] ?? 0))),
                  true,
                )
                const precision = toArray(linalg.pinv(covarianceOfRows(train))) as number[][]
                return gaussianLogLikelihood(test, precision)
              })
            : []
        return {
          ...st,
          stage: 'done',
          chosen,
          covariance: out.covariance,
          precision: out.precision,
          gap: out.gap,
          objective: out.objective,
          nIter: out.nIter,
          unpenalised,
        }
      }
      const alpha = st.grid[st.cursor] ?? 0
      const warm = [...st.warm]
      const base = fit(full, alpha, warm[0] ?? null, maxIter)
      if (base !== null) warm[0] = base.covariance
      const precision = base?.precision ?? full.map((r, a) => r.map((_, b) => (a === b ? 1 : 0)))
      let scores: number[] = []
      let criterion: number
      if (method === 'cv') {
        scores = st.folds.map((f, k) => {
          const train = covarianceOfRows(f.train.map((i) => rows[i] as number[]))
          const test = covarianceOfRows(f.test.map((i) => rows[i] as number[]))
          const r = fit(train, alpha, warm[k + 1] ?? null, pathIter)
          if (r === null) return Number.NEGATIVE_INFINITY
          warm[k + 1] = r.covariance
          const score = gaussianLogLikelihood(test, r.precision)
          return Number.isFinite(score) ? score : Number.NEGATIVE_INFINITY
        })
        criterion = scores.reduce((a, v) => a + v, 0) / scores.length
      } else if (method === 'ebic') {
        criterion = base === null ? Number.POSITIVE_INFINITY : graphicalLassoEbic(full, precision, n, gamma)
      } else {
        const frequency = Array.from({ length: p }, () => new Array<number>(p).fill(0))
        for (const [b, idx] of st.subsamples.entries()) {
          const r = fit(covarianceOfRows(idx.map((i) => rows[i] as number[])), alpha, warm[b + 1] ?? null, maxIter)
          if (r === null) continue
          warm[b + 1] = r.covariance
          for (let a = 0; a < p; a++)
            for (let c = a + 1; c < p; c++)
              if (Math.abs(r.precision[a]?.[c] ?? 0) > 1e-10)
                (frequency[a] as number[])[c] = (frequency[a]?.[c] ?? 0) + 1 / nSubsamples
        }
        let total = 0
        for (let a = 0; a < p; a++)
          for (let c = a + 1; c < p; c++) total += 2 * (frequency[a]?.[c] ?? 0) * (1 - (frequency[a]?.[c] ?? 0))
        criterion = total / ((p * (p - 1)) / 2)
      }
      const entry: PenaltyEntry = {
        alpha,
        scores,
        criterion,
        edges: edgeCount(precision),
        gap: base?.gap ?? Number.NaN,
        precision,
      }
      const entries = [...st.entries, entry]
      const cursor = st.cursor + 1
      if (cursor < st.grid.length) return { ...st, entries, warm, cursor, precision }
      const reset = new Array<number[][] | null>(warm.length).fill(null)
      if (method === 'cv' && st.refinement + 1 < nRefinements) {
        const path = sortedEntries(entries)
        const { best } = pick(method, path, beta)
        return {
          ...st,
          entries,
          warm: reset,
          grid: refinedGrid(path, best, count),
          cursor: 0,
          refinement: st.refinement + 1,
          precision,
        }
      }
      return { ...st, entries, warm: reset, cursor, stage: 'final', precision }
    },
    done: (st) => (st.stage === 'done' ? 'converged' : undefined),
    observe: (st) => {
      const last = st.entries[st.entries.length - 1]
      return {
        metrics:
          last === undefined
            ? {}
            : {
                alpha: last.alpha,
                criterion: last.criterion,
                edges: last.edges,
                ...(Number.isFinite(last.gap) ? { gap: last.gap } : {}),
              },
        tensors:
          st.precision.length === 0 ? {} : { precision: { role: 'param', value: st.precision.flat(), shape: [p, p] } },
      }
    },
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as PenaltySelectionState,
  }
}

/** The outcome of a penalty selection. */
export interface PenaltySelection {
  readonly method: PenaltySelector
  /** The chosen penalty. */
  readonly alpha: number
  /** The penalties evaluated, descending (a refined `'cv'` grid is the union of its grids). */
  readonly alphas: number[]
  /** The rule's criterion at each penalty: the mean held-out log-likelihood, the EBIC, or the monotone StARS instability. */
  readonly criterion: number[]
  /** Whether a higher (`'max'`, held-out likelihood) or lower (`'min'`) criterion is better; StARS takes the sparsest graph under `beta`. */
  readonly direction: 'max' | 'min'
  /** The index of `alpha` in `alphas`. */
  readonly bestIndex: number
  /** `'cv'`: the held-out log-likelihood of each penalty on each fold `[alphas][folds]` (empty for the other rules). */
  readonly scores: number[][]
  /** `'cv'`: the same for the unpenalised (empirical covariance) fit, scikit-learn's `alpha = 0` row `[folds]`. */
  readonly unpenalisedScores: number[]
  /** `'stars'`: the raw (not monotonised) instability at each penalty. */
  readonly instability: number[]
  /** The number of edges of the full-data fit at each penalty. */
  readonly edges: number[]
  /** The full-data precision matrix at each penalty (a path of warm starts, so loosely converged under `'cv'`). */
  readonly precisions: number[][][]
  /** The refit at the chosen penalty. */
  readonly covariance: number[][]
  readonly precision: number[][]
  /** The conditional-independence graph of the refit `[p][p]`. */
  readonly graph: boolean[][]
  readonly gap: number
  readonly objective: number
  /** The sweeps of the final fit. */
  readonly nIter: number
}

/** The selection a finished state holds. */
export function penaltySelectionOf(
  state: PenaltySelectionState,
  options: PenaltySelectionOptions = {},
): PenaltySelection {
  const method = options.method ?? 'cv'
  const path = sortedEntries(state.entries)
  const { criterion, best } = pick(method, path, options.beta ?? 0.05)
  return {
    method,
    alpha: state.chosen,
    alphas: path.map((e) => e.alpha),
    criterion,
    direction: method === 'cv' ? 'max' : 'min',
    bestIndex: best,
    scores: method === 'cv' ? path.map((e) => e.scores) : [],
    unpenalisedScores: state.unpenalised,
    instability: method === 'stars' ? path.map((e) => e.criterion) : [],
    edges: path.map((e) => e.edges),
    precisions: path.map((e) => e.precision),
    covariance: state.covariance,
    precision: state.precision,
    graph: precisionGraph(state.precision),
    gap: state.gap,
    objective: state.objective,
    nIter: state.nIter,
  }
}

/**
 * The exposition of a selection: the criterion along the penalties as a `crossValidationCurve` (the penalty chosen
 * marked; for `'cv'` with its fold scores and their spread) and the path of the full-data partial correlations as a
 * `regularisationPath` with the chosen penalty marked.
 *
 * @param selection - The selection.
 * @param names - The variable names (default `x0`, `x1`, ...).
 * @returns The two records.
 * @example
 * ```ts
 * import { selectGraphicalLassoPenalty, selectionRecords } from 'omai-engine/covariance'
 * const x = [[1, 0.9, 0], [-1, -1.1, 0.3], [0.5, 0.4, -1], [-0.5, -0.2, 1], [0.2, 0.3, 0.1], [-0.2, -0.1, -0.5]]
 * selectionRecords(selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: 4 }))[0]?.kind // => 'crossValidationCurve'
 * ```
 */
export function selectionRecords(
  selection: PenaltySelection,
  names?: readonly string[],
): readonly [CrossValidationCurve, RegularisationPath] {
  const p = selection.precision.length
  const labels = names ?? Array.from({ length: p }, (_, j) => `x${j}`)
  const pairs: [number, number][] = []
  for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) pairs.push([a, b])
  const folds = selection.scores
  const deviation = (v: readonly number[]): number => {
    const finite = v.filter(Number.isFinite)
    if (finite.length < 2) return 0
    const mean = finite.reduce((a, c) => a + c, 0) / finite.length
    return Math.sqrt(finite.reduce((a, c) => a + (c - mean) ** 2, 0) / finite.length / finite.length)
  }
  const coef = selection.precisions.map((prec) =>
    pairs.map(([a, b]) => -(prec[a]?.[b] ?? 0) / Math.sqrt((prec[a]?.[a] ?? 1) * (prec[b]?.[b] ?? 1))),
  )
  return [
    crossValidationCurve({
      parameter: 'alpha',
      values: selection.alphas,
      meanScores: selection.criterion,
      standardErrors: selection.method === 'cv' ? folds.map(deviation) : selection.alphas.map(() => 0),
      ...(selection.method === 'cv' ? { foldScores: folds } : {}),
      higherIsBetter: selection.direction === 'max',
      best: selection.alpha,
      scoreName:
        selection.method === 'cv'
          ? 'held-out log likelihood'
          : selection.method === 'ebic'
            ? 'extended BIC'
            : 'monotone edge instability (StARS)',
      penalty: true,
    }),
    regularisationPath({
      penalties: selection.alphas,
      features: pairs.map(([a, b]) => `${labels[a]}~${labels[b]}`),
      coefficients: coef,
      active: coef.map((r) => r.flatMap((v, j) => (Math.abs(v) > 1e-10 ? [j] : []))),
      chosen: selection.alpha,
    }),
  ]
}

/**
 * Choose the graphical-lasso penalty, by held-out likelihood (`'cv'`, scikit-learn's `GraphicalLassoCV`), extended BIC
 * (`'ebic'`) or stability under subsampling (`'stars'`), and return the chosen penalty, the criterion along the whole
 * penalty path and the estimated precision matrix and graph. The grid is traced step by step by
 * {@link graphicalLassoSelectionAlgorithm}; the records of {@link selectionRecords} plot the curve and the path.
 *
 * @param x - Observations `[n, p]`.
 * @param options - The rule, the grid and the rule's settings.
 * @param key - The key of the random folds (`'cv'` with a shuffling splitter) and subsamples (`'stars'`; default key 0).
 * @returns The selection, with its exposition records.
 * @index function Choose the graphical-lasso penalty by CV, EBIC or StARS; returns the path and the graph.
 * @example
 * ```ts
 * import { selectGraphicalLassoPenalty } from 'omai-engine/covariance'
 * const x = [[1, 0.9, 0], [-1, -1.1, 0.3], [0.5, 0.4, -1], [-0.5, -0.2, 1], [0.2, 0.3, 0.1], [-0.2, -0.1, -0.5]]
 * const out = selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: [0.5, 0.1, 0.01] })
 * out.alphas // => [0.5, 0.1, 0.01]
 * ```
 */
export function selectGraphicalLassoPenalty(
  x: TensorLike,
  options: PenaltySelectionOptions = {},
  key?: Key,
): PenaltySelection & { readonly records: readonly [CrossValidationCurve, RegularisationPath] } {
  const algorithm = graphicalLassoSelectionAlgorithm(x, options)
  const out = runAlgorithm(algorithm, { maxSteps: 10_000, ...(key === undefined ? {} : { key }) })
  const selection = penaltySelectionOf(out.state, options)
  return { ...selection, records: selectionRecords(selection) }
}

/** Hyperparameters of the graphical-lasso penalty selection estimator. */
export interface GraphicalLassoSelectionHyper {
  /** The rule (default `'cv'`). */
  readonly method?: PenaltySelector
  /** The penalties, or the number of them on the grid (see {@link PenaltySelectionOptions.alphas}). */
  readonly alphas?: readonly number[] | number
  /** The grid refinements of `'cv'` (default 4). */
  readonly nRefinements?: number
  /** The number of contiguous (or shuffled) cross-validation folds (default 5). */
  readonly folds?: number
  /** Shuffle the rows into the folds, with the run's key (default false, as scikit-learn). */
  readonly shuffle?: boolean
  /** The EBIC weight (default 0.5). */
  readonly gamma?: number
  /** The StARS subsamples (default 20) and instability bound (default 0.05). */
  readonly nSubsamples?: number
  readonly beta?: number
  /** The duality-gap tolerance (default 1e-4), inner lasso tolerance (1e-4) and sweep cap (100). */
  readonly tol?: number
  readonly enetTol?: number
  readonly maxIter?: number
}

/** The fitted parameters of the selection estimator: the covariance estimate and the penalty path. */
export type GraphicalLassoSelectionParams = CovarianceParams & {
  readonly method: PenaltySelector
  readonly alpha: number
  readonly alphas: number[]
  readonly criterion: number[]
  readonly edges: number[]
  readonly gap: number
  readonly objective: number
}

/** A fitted graphical-lasso penalty selection. */
export type FittedGraphicalLassoSelection = IterativeFitted<GraphicalLassoSelectionParams, CovarianceVerbs>

function optionsOf(h: GraphicalLassoSelectionHyper): PenaltySelectionOptions {
  return {
    method: h.method ?? 'cv',
    ...(h.alphas === undefined ? {} : { alphas: h.alphas }),
    ...(h.nRefinements === undefined ? {} : { nRefinements: h.nRefinements }),
    cv: kFold(h.folds ?? 5, { shuffle: h.shuffle ?? false }),
    ...(h.gamma === undefined ? {} : { gamma: h.gamma }),
    ...(h.nSubsamples === undefined ? {} : { nSubsamples: h.nSubsamples }),
    ...(h.beta === undefined ? {} : { beta: h.beta }),
    ...(h.tol === undefined ? {} : { tol: h.tol }),
    ...(h.enetTol === undefined ? {} : { enetTol: h.enetTol }),
    ...(h.maxIter === undefined ? {} : { maxIter: h.maxIter }),
  }
}

/**
 * The graphical lasso with its penalty chosen by cross-validation, EBIC or StARS, on the iterative base: the selection
 * of {@link graphicalLassoSelectionAlgorithm} traced penalty by penalty. Its exposition is the criterion curve with the
 * chosen penalty marked and the path of the partial correlations.
 *
 * @param hyper - The rule and its settings.
 * @returns The iterative estimator.
 * @index estimator Graphical lasso with the penalty chosen by cross-validation, EBIC or StARS.
 * @example
 * ```ts
 * import { graphicalLassoSelectionEstimator } from 'omai-engine/covariance'
 * const x = [[1, 0.9, 0], [-1, -1.1, 0.3], [0.5, 0.4, -1], [-0.5, -0.2, 1], [0.2, 0.3, 0.1], [-0.2, -0.1, -0.5]]
 * graphicalLassoSelectionEstimator({ method: 'ebic', alphas: [0.5, 0.1] }).fit({ x }).params.alphas // => [0.5, 0.1]
 * ```
 */
export function graphicalLassoSelectionEstimator(
  hyper: GraphicalLassoSelectionHyper = {},
): IterativeEstimator<
  GraphicalLassoSelectionHyper,
  PenaltySelectionState,
  GraphicalLassoSelectionParams,
  CovarianceVerbs
> {
  return iterative<GraphicalLassoSelectionHyper, PenaltySelectionState, GraphicalLassoSelectionParams, CovarianceVerbs>(
    {
      name: 'graphical-lasso-selection',
      kind: 'density',
      defaults: {
        ...hyper,
        method: hyper.method ?? 'cv',
        folds: hyper.folds ?? 5,
        shuffle: hyper.shuffle ?? false,
        maxIter: hyper.maxIter ?? 100,
      },
      hyperSpace: (h) =>
        dict({
          method: choice(['cv', 'ebic', 'stars'], { default: h.method ?? 'cv', label: 'rule' }),
          folds: int(2, 20, { default: h.folds ?? 5, label: 'folds' }),
          shuffle: bool({ default: h.shuffle ?? false, label: 'shuffle the folds' }),
        }),
      algorithm: (data, h) => graphicalLassoSelectionAlgorithm(data.x, optionsOf(h)),
      maxSteps: () => 10_000,
      params: (st, data, h) => {
        const m = matrixView(data.x, 'graphicalLassoSelection', { minSamples: 4 })
        const location = Array.from({ length: m.d }, (_, j) => {
          let t = 0
          for (let i = 0; i < m.n; i++) t += m.data[i * m.d + j] ?? 0
          return t / m.n
        })
        const sel = penaltySelectionOf(st, optionsOf(h))
        return {
          location,
          covariance: st.covariance,
          precision: st.precision,
          shrinkage: 0,
          method: sel.method,
          alpha: sel.alpha,
          alphas: sel.alphas,
          criterion: sel.criterion,
          edges: sel.edges,
          gap: st.gap,
          objective: st.objective,
        }
      },
      verbs: covarianceVerbs,
      records: (st, data, h) => {
        const names = featureNamesOf(data, st.precision.length)
        return selectionRecords(penaltySelectionOf(st, optionsOf(h)), names)
      },
      logs: { 'solver/alpha': Log(), 'solver/criterion': Log(), 'solver/edges': Log(), 'solver/gap': Log() },
    },
  )
}

/**
 * Cross-validated graphical lasso as a scikit-learn-style class wrapper (`GraphicalLassoCV`): the held-out likelihood
 * over contiguous folds on a grid of penalties refined around the best, then a refit; `method` switches the rule to
 * `'ebic'` or `'stars'`.
 *
 * @example
 * ```ts
 * import { GraphicalLassoCV } from 'omai-engine/covariance'
 * const x = [[1, 0.9, 0], [-1, -1.1, 0.3], [0.5, 0.4, -1], [-0.5, -0.2, 1], [0.2, 0.3, 0.1], [-0.2, -0.1, -0.5]]
 * new GraphicalLassoCV({ folds: 3, alphas: [0.3, 0.1] }).fit(x).model.params.alphas // => [0.3, 0.1]
 * ```
 */
export class GraphicalLassoCV extends ExplainedEstimatorClass<
  GraphicalLassoSelectionHyper,
  Fitted<GraphicalLassoSelectionParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link graphicalLassoSelectionEstimator}.
   */
  constructor(hyper: GraphicalLassoSelectionHyper = {}) {
    super(graphicalLassoSelectionEstimator(hyper))
  }
}
