/**
 * Vector error-correction models, statsmodels' `VECM`: dy_t = alpha beta' y_(t-1) + Gamma_1 dy_(t-1) + ... + deterministic
 * terms + u_t, with the cointegration rank r given. The likelihood concentrated on beta is a smooth objective
 * (log|beta'(S11 - S10 S00^-1 S01)beta| - log|beta'S11 beta|, beta normalised to [I; B]), so the fit takes its solver
 * from the slot: Johansen's reduced-rank eigen-solve is the model's own mode, and every generic smooth solver minimises
 * the same objective to the same beta. Forecasts iterate the VAR form of the model, with intervals from its
 * moving-average representation.
 */

import { MdmlError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import {
  analyticObjective,
  familySolvers,
  type SlotState,
  type SmoothProblem,
  slotAlgorithm,
  type VectorParams,
} from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { dict, type SpaceValue } from '../space/index.ts'
import { item } from '../tensor/index.ts'
import { type Algorithm, type ModeSpec, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  add,
  columnsOf,
  fromColumns,
  identity,
  invert,
  logDeterminant,
  type Matrix,
  multiply,
  residualsOn,
  transpose,
} from './_matrix.ts'
import { leastSquares } from './_ml.ts'
import { reducedRankEigen } from './cointegration.ts'

/** The deterministic terms of a VECM: none, or a constant / linear trend outside or inside the cointegration relation. */
export type VecmDeterministic = 'n' | 'co' | 'ci' | 'lo' | 'li' | 'cili' | 'colo' | 'cilo' | 'coli'

/** Options of {@link vecmProblem} and {@link vecmFit}. */
export interface VecmOptions {
  /** The lagged differences (default 1; the VAR in levels has this plus one lags). */
  readonly kArDiff?: number
  /** The cointegration rank (default 1). */
  readonly rank?: number
  /** The deterministic terms (default `'n'`). */
  readonly deterministic?: VecmDeterministic
}

/** The data of a VECM in the form its estimation needs (Luetkepohl's matrices, one array per row). */
export interface VecmProblem {
  readonly nVars: number
  readonly rank: number
  readonly kArDiff: number
  readonly deterministic: VecmDeterministic
  /** Observations after the presample. */
  readonly nobs: number
  /** The differences, `nVars` rows of `nobs`. */
  readonly dy: readonly Float64Array[]
  /** The lagged levels with the deterministic terms inside the relation, `nVars + inside` rows. */
  readonly yLag: readonly Float64Array[]
  /** The lagged differences and the deterministic terms outside the relation. */
  readonly dx: readonly Float64Array[]
  readonly s00: Matrix
  readonly s01: Matrix
  readonly s11: Matrix
  /** The last `kArDiff + 1` levels, to start forecasts from. */
  readonly last: Matrix
}

function terms(det: VecmDeterministic): { co: boolean; ci: boolean; lo: boolean; li: boolean } {
  const has = (s: string): boolean => det.includes(s)
  if (has('co') && has('ci')) throw new MdmlError('value', "vecm: both 'co' and 'ci' given; choose one")
  return { co: has('co'), ci: has('ci'), lo: has('lo'), li: has('li') }
}

/**
 * The matrices of Johansen's estimation for a series (statsmodels' `_endog_matrices` and `_sij`): differences, lagged
 * levels with the trend terms inside the relation, lagged differences with the terms outside, and the product moments
 * of their residuals.
 *
 * @param y - The series as rows `[time][series]`.
 * @param options - Lags, rank and deterministic terms.
 * @returns The problem data.
 * @example
 * ```ts
 * import { vecmProblem } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 30 }, (_, i) => [Math.sin(i) + i * 0.1, Math.cos(i * 1.3) + i * 0.1])
 * vecmProblem(y, { kArDiff: 1, rank: 1 }).nobs // => 28
 * ```
 */
export function vecmProblem(y: ArrayLike<ArrayLike<number>>, options: VecmOptions = {}): VecmProblem {
  const kd = options.kArDiff ?? 1
  const rank = options.rank ?? 1
  const deterministic = options.deterministic ?? 'n'
  const t = terms(deterministic)
  const rows = Array.from(y, (r) => Array.from(r))
  const cols = columnsOf(rows)
  const K = cols.length
  const p = kd + 1
  const n = rows.length
  const T = n - p
  if (T < K * (kd + 2) + 2) throw new MdmlError('value', 'vecm: too few observations')
  if (!(Number.isInteger(rank) && rank >= 1 && rank <= K)) throw new MdmlError('value', 'vecm: rank must be 1 to nVars')
  const dyAll = cols.map((c) => Float64Array.from({ length: n - 1 }, (_, i) => (c[i + 1] ?? 0) - (c[i] ?? 0)))
  const dy = dyAll.map((c) => c.slice(p - 1))
  const yLag: Float64Array[] = cols.map((c) => c.slice(p - 1, n - 1))
  if (t.ci) yLag.push(new Float64Array(T).fill(1))
  if (t.li) yLag.push(Float64Array.from({ length: T }, (_, i) => i + p))
  const dx: Float64Array[] = []
  // delta_x column j holds dy_(j+p-2), ..., dy_(j) (most recent lag first), all series for each lag.
  for (let lag = 1; lag <= kd; lag++)
    for (const c of dyAll) dx.push(Float64Array.from({ length: T }, (_, j) => c[j + p - 1 - lag] ?? 0))
  if (t.co) dx.push(new Float64Array(T).fill(1))
  if (t.lo) dx.push(Float64Array.from({ length: T }, (_, i) => i + p + 1))
  const r0 = residualsOn(dy, dx)
  const r1 = residualsOn(yLag, dx)
  const mom = (a: readonly Float64Array[], b: readonly Float64Array[]): Matrix =>
    a.map((u) =>
      b.map((v) => {
        let s = 0
        for (let i = 0; i < T; i++) s += (u[i] ?? 0) * (v[i] ?? 0)
        return s / T
      }),
    )
  return {
    nVars: K,
    rank,
    kArDiff: kd,
    deterministic,
    nobs: T,
    dy,
    yLag,
    dx,
    s00: mom(r0, r0),
    s01: mom(r0, r1),
    s11: mom(r1, r1),
    last: rows.slice(n - p),
  }
}

/** beta from its free block B: the rows of the identity on top, B below (`m x r`). */
function betaOf(p: VecmProblem, x: ArrayLike<number>): Matrix {
  const m = p.yLag.length
  const r = p.rank
  const out: Matrix = []
  for (let i = 0; i < r; i++) out.push(Array.from({ length: r }, (_, j) => (i === j ? 1 : 0)))
  for (let i = r; i < m; i++) out.push(Array.from({ length: r }, (_, j) => x[(i - r) * r + j] ?? 0))
  return out
}

/** The concentrated objective log|beta'A beta| - log|beta'S11 beta| and its gradient in B. */
function concentrated(p: VecmProblem, x: ArrayLike<number>): { value: number; grad: Float64Array } {
  const A = add(p.s11, multiply(multiply(transpose(p.s01), invert(p.s00)), p.s01), -1)
  const beta = betaOf(p, x)
  const bt = transpose(beta)
  const bAb = multiply(multiply(bt, A), beta)
  const bSb = multiply(multiply(bt, p.s11), beta)
  const value = logDeterminant(bAb) - logDeterminant(bSb)
  const gA = multiply(multiply(A, beta), invert(bAb))
  const gS = multiply(multiply(p.s11, beta), invert(bSb))
  const m = p.yLag.length
  const r = p.rank
  const grad = new Float64Array((m - r) * r)
  for (let i = r; i < m; i++)
    for (let j = 0; j < r; j++) grad[(i - r) * r + j] = 2 * ((gA[i]?.[j] ?? 0) - (gS[i]?.[j] ?? 0))
  return { value, grad }
}

/** The start: beta from the regressions of the first `rank` lagged levels on the others (the Engle-Granger step). */
function engleGrangerStart(p: VecmProblem): number[] {
  const m = p.yLag.length
  const r = p.rank
  const levels = residualsFor(p, 'yLag')
  const rest = levels.slice(r)
  const x = new Array<number>((m - r) * r).fill(0)
  for (let j = 0; j < r; j++) {
    const coef = leastSquares(rest, levels[j] ?? new Float64Array())
    for (let i = 0; i < m - r; i++) x[i * r + j] = -(coef[i] ?? 0)
  }
  return x
}

/**
 * The VECM likelihood concentrated on beta as a smooth problem for the solver slot: the free block B of
 * beta = [I; B] and the objective log|beta'(S11 - S10 S00^-1 S01)beta| - log|beta'S11 beta| with its analytic gradient.
 *
 * @param problem - The data from {@link vecmProblem}.
 * @returns The objective and the start (the regression of the first levels on the rest).
 * @example
 * ```ts
 * import { vecmProblem, vecmSmoothProblem } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 30 }, (_, i) => [Math.sin(i) + i * 0.1, Math.cos(i * 1.3) + i * 0.1])
 * vecmSmoothProblem(vecmProblem(y)).x0.length // => 1
 * ```
 */
export function vecmSmoothProblem(problem: VecmProblem): SmoothProblem {
  return {
    objective: analyticObjective({
      value: (x) => concentrated(problem, x).value,
      gradient: (x) => concentrated(problem, x).grad,
    }),
    x0: engleGrangerStart(problem),
  }
}

/** The state of Johansen's closed-form mode: the solution found at once. */
interface ClosedState {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
}

/** Johansen's reduced-rank eigen-solve: beta from the leading eigenvectors of S11^-1/2 S10 S00^-1 S01 S11^-1/2. */
function johansenBeta(p: VecmProblem): Float64Array {
  const r = p.rank
  const m = p.yLag.length
  // Eigenproblem S10 S00^-1 S01 v = lambda S11 v, by the Cholesky route of the cointegration tests.
  const { vectors } = reducedRankEigen(p.dy.length === 0 ? [] : residualsFor(p, 'dy'), residualsFor(p, 'yLag'))
  const lead: Matrix = vectors.map((row) => row.slice(0, r))
  const top: Matrix = lead.slice(0, r)
  const beta = multiply(lead, invert(top))
  const x = new Float64Array((m - r) * r)
  for (let i = r; i < m; i++) for (let j = 0; j < r; j++) x[(i - r) * r + j] = beta[i]?.[j] ?? 0
  return x
}

const residualCache = new WeakMap<VecmProblem, { dy: Float64Array[]; yLag: Float64Array[] }>()
function residualsFor(p: VecmProblem, which: 'dy' | 'yLag'): Float64Array[] {
  let c = residualCache.get(p)
  if (c === undefined) {
    c = { dy: residualsOn(p.dy, p.dx), yLag: residualsOn(p.yLag, p.dx) }
    residualCache.set(p, c)
  }
  return c[which]
}

const johansenMode: ModeSpec<VecmProblem, ClosedState, VectorParams> = {
  description:
    "Johansen's reduced-rank regression: the eigenvectors of the squared canonical correlations, in one step",
  space: dict({}),
  make: (p): Algorithm<ClosedState> => {
    const point = (): ClosedState => {
      const x = johansenBeta(p)
      const { value, grad } = concentrated(p, x)
      return { x, value, grad, step: 0 }
    }
    return {
      name: 'vecm-johansen',
      config: { rank: p.rank } as never,
      init: () => point(),
      step: (s) => s,
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as ClosedState,
    }
  },
  params: (s) => ({ x: Array.from(s.x) }),
}

/**
 * The ways to fit a VECM: `johansen` (the closed-form eigen-solve of the reduced-rank regression), then `newton`,
 * `bfgs`, `lbfgs`, `conjugateGradient`, `gradientDescent`, `coordinateDescent` and the stochastic `gradient` mode, each
 * minimising the concentrated likelihood to the same beta.
 */
export const vecmSolvers: ReturnType<typeof familySolvers<VecmProblem>> = familySolvers(
  { johansen: johansenMode },
  vecmSmoothProblem,
  { label: 'solver', description: 'how the concentrated VECM likelihood is minimised', default: 'johansen' },
)

/** A fitted VECM: plain data. */
export interface VecmFit {
  readonly nVars: number
  readonly rank: number
  readonly kArDiff: number
  readonly deterministic: VecmDeterministic
  /** The adjustment coefficients, `nVars x rank`. */
  readonly alpha: Matrix
  /** The cointegrating vectors, `nVars x rank`, normalised so that the first `rank` rows are the identity. */
  readonly beta: Matrix
  /** The short-run coefficients `Gamma_1 ... Gamma_kArDiff`, `nVars x (nVars kArDiff)`. */
  readonly gamma: Matrix
  /** The coefficients of the deterministic terms outside the relation (constant, then trend), `nVars x terms`. */
  readonly detCoef: Matrix
  /** The coefficients of the deterministic terms inside the relation (constant, then trend), `terms x rank`. */
  readonly detCoefCoint: Matrix
  /** The maximum-likelihood innovation covariance. */
  readonly sigma: Matrix
  readonly logLikelihood: number
  readonly nobs: number
  /** The last `kArDiff + 1` observations. */
  readonly last: Matrix
  /** The solver that fitted it. */
  readonly solver: string
}

/** The model's quantities from beta. */
function assemble(p: VecmProblem, x: ArrayLike<number>, solver: string): VecmFit {
  const K = p.nVars
  const r = p.rank
  const T = p.nobs
  const beta = betaOf(p, x)
  const dyM = fromColumns(p.dy) // T x K
  const yLagM = fromColumns(p.yLag) // T x m
  const dxM = fromColumns(p.dx) // T x q
  const bt = transpose(beta)
  const alpha = multiply(multiply(p.s01, beta), invert(multiply(multiply(bt, p.s11), beta)))
  // dy - alpha beta' yLag, per time row: K x T as rows of the transpose.
  const eq = multiply(multiply(yLagM, beta), transpose(alpha)) // T x K
  const resid1 = add(dyM, eq, -1) // T x K
  const q = p.dx.length
  let gammaAll: Matrix = []
  let resid: Matrix = resid1
  if (q > 0) {
    const dxt = transpose(dxM)
    const coef = multiply(multiply(transpose(resid1), dxM), invert(multiply(dxt, dxM))) // K x q
    gammaAll = coef
    resid = add(resid1, multiply(dxM, transpose(coef)), -1)
  }
  const sigma = multiply(transpose(resid), resid).map((row) => row.map((v) => v / T))
  const nGamma = K * p.kArDiff
  const gamma = gammaAll.map((row) => row.slice(0, nGamma))
  const detCoef = gammaAll.map((row) => row.slice(nGamma))
  const insideRows = beta.slice(K)
  const llf =
    -(K * T * Math.log(2 * Math.PI)) / 2 - (T * (logDeterminant(p.s00) + concentrated(p, x).value)) / 2 - (K * T) / 2
  return {
    nVars: K,
    rank: r,
    kArDiff: p.kArDiff,
    deterministic: p.deterministic,
    alpha,
    beta: beta.slice(0, K),
    gamma,
    detCoef,
    detCoefCoint: insideRows,
    sigma,
    logLikelihood: llf,
    nobs: T,
    last: p.last,
    solver,
  }
}

/** The state of {@link vecmAlgorithm}. */
export type VecmState = SlotState & {
  /** The free block of beta, row-major. */
  readonly u: number[]
  readonly objective: number
  readonly gradient: number
}

/** Options of {@link vecmAlgorithm} and {@link vecmFit}. */
export interface VecmSolveOptions extends VecmOptions {
  /** The solver, a value of `vecmSolvers.space` (default `johansen`). */
  readonly solver?: SpaceValue
  /** Stop when the largest gradient entry falls below this (default 1e-8). */
  readonly tol?: number
}

/**
 * The VECM fit as a traced algorithm over the solver slot. With the `johansen` mode it is one step; with a generic
 * solver each step is one of its iterations on the concentrated likelihood. It logs `objective`, `gradient`, `gradNorm`
 * and `step`, and the tensor `beta`.
 *
 * @param problem - The data from {@link vecmProblem}.
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { vecmAlgorithm, vecmProblem } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 30 }, (_, i) => [Math.sin(i) + i * 0.1, Math.cos(i * 1.3) + i * 0.1])
 * vecmAlgorithm(vecmProblem(y)).name // => 'vecm'
 * ```
 */
export function vecmAlgorithm(problem: VecmProblem, options: VecmSolveOptions = {}): Algorithm<VecmState> {
  const solver = options.solver ?? { kind: 'johansen' }
  const tol = options.tol ?? 1e-8
  const m = problem.yLag.length - problem.rank
  return slotAlgorithm<VecmState, VecmProblem>({
    name: 'vecm',
    config: { rank: problem.rank, kArDiff: problem.kArDiff, deterministic: problem.deterministic, solver } as never,
    schema: {
      metrics: {
        objective: { description: 'log|beta A beta| - log|beta S11 beta|', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: {
        beta: { role: 'param', description: 'the free block of the cointegrating vectors', shape: [m * problem.rank] },
      },
    },
    problem,
    table: vecmSolvers,
    solver,
    tol,
    view: (r) => ({
      u: Array.from(r.x),
      objective: r.value,
      gradient: r.grad.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { beta: { role: 'param', value: s.u, shape: [m * problem.rank] } },
    }),
  })
}

/**
 * Fit a vector error-correction model by maximum likelihood (statsmodels `VECM(...).fit()`), by Johansen's
 * eigen-solve or any solver of the slot.
 *
 * @param y - The series as rows `[time][series]`.
 * @param options - Lags, rank, deterministic terms, solver.
 * @returns The fitted model.
 * @example
 * ```ts
 * import { vecmFit } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 40 }, (_, i) => [Math.sin(i) + i * 0.1, Math.cos(i * 1.3) + i * 0.1])
 * vecmFit(y, { rank: 1 }).beta.length // => 2
 * ```
 */
export function vecmFit(y: ArrayLike<ArrayLike<number>>, options: VecmSolveOptions = {}): VecmFit {
  const problem = vecmProblem(y, options)
  const run = runAlgorithm(vecmAlgorithm(problem, options), { maxSteps: 500, key: makeKey(0) })
  return assemble(problem, run.state.u, vecmSolvers.modeOf(options.solver ?? { kind: 'johansen' }).name)
}

/** Forecasts of a VECM with normal intervals. */
export interface VecmForecast {
  readonly mean: Matrix
  readonly lower: Matrix
  readonly upper: Matrix
}

/** The VAR(p) matrices of the model (statsmodels' `var_rep`): A_1 ... A_p with p = kArDiff + 1. */
export function vecmVarRepresentation(fit: Pick<VecmFit, 'nVars' | 'kArDiff' | 'alpha' | 'beta' | 'gamma'>): Matrix[] {
  const K = fit.nVars
  const p = fit.kArDiff + 1
  const pi = multiply(fit.alpha, transpose(fit.beta))
  const A: Matrix[] = Array.from({ length: p }, () => Array.from({ length: K }, () => new Array<number>(K).fill(0)))
  A[0] = add(pi, identity(K))
  if (fit.kArDiff > 0) {
    const g = (i: number): Matrix => fit.gamma.map((row) => row.slice(K * i, K * (i + 1)))
    A[0] = add(A[0] as Matrix, g(0))
    A[p - 1] = g(p - 2).map((row) => row.map((v) => -v))
    for (let i = 1; i < p - 1; i++) A[i] = add(g(i), g(i - 1), -1)
  }
  return A
}

/**
 * Forecast `steps` ahead (statsmodels `VECMResults.predict`): iterate the VAR form of the model from the last
 * observations with the deterministic terms continued, intervals from the moving-average representation and the
 * innovation covariance.
 *
 * @param fit - The fitted model.
 * @param steps - How many steps ahead.
 * @param alpha - The significance level of the intervals (default 0.05).
 * @returns Means and bounds, `[step][series]`.
 * @example
 * ```ts
 * import { vecmFit, vecmForecast } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 40 }, (_, i) => [Math.sin(i) + i * 0.1, Math.cos(i * 1.3) + i * 0.1])
 * vecmForecast(vecmFit(y), 3).mean.length // => 3
 * ```
 */
export function vecmForecast(fit: VecmFit, steps: number, alpha = 0.05): VecmForecast {
  const K = fit.nVars
  const p = fit.kArDiff + 1
  const t = terms(fit.deterministic)
  const A = vecmVarRepresentation(fit)
  // The coefficient columns: outside terms (constant, trend) and the inside ones through alpha.
  let col = 0
  const constOut = t.co ? fit.detCoef.map((r) => r[col] ?? 0) : null
  if (t.co) col++
  const trendOut = t.lo ? fit.detCoef.map((r) => r[col] ?? 0) : null
  let row = 0
  const constIn = t.ci ? fit.detCoefCoint[row++] : undefined
  const trendIn = t.li ? fit.detCoefCoint[row] : undefined
  const alphaTimes = (v: readonly number[] | undefined): number[] =>
    fit.alpha.map((a) => a.reduce((s, x, j) => s + x * (v?.[j] ?? 0), 0))
  const cIn = constIn === undefined ? null : alphaTimes(constIn)
  const tIn = trendIn === undefined ? null : alphaTimes(trendIn)
  const history: Matrix = fit.last.map((r) => r.slice())
  const mean: Matrix = []
  for (let h = 0; h < steps; h++) {
    const next = new Array<number>(K).fill(0)
    for (let i = 1; i <= p; i++) {
      const prev = h - i < 0 ? (history[history.length + h - i] ?? []) : (mean[h - i] ?? [])
      const a = A[i - 1] as Matrix
      for (let e = 0; e < K; e++)
        for (let j = 0; j < K; j++) next[e] = (next[e] ?? 0) + (a[e]?.[j] ?? 0) * (prev[j] ?? 0)
    }
    for (let e = 0; e < K; e++) {
      next[e] = (next[e] ?? 0) + (constOut?.[e] ?? 0) + (cIn?.[e] ?? 0)
      next[e] = (next[e] ?? 0) + (trendOut?.[e] ?? 0) * (fit.nobs + p + 1 + h) + (tIn?.[e] ?? 0) * (fit.nobs + p + h)
    }
    mean.push(next)
  }
  const phis: Matrix[] = [identity(K)]
  for (let i = 1; i < steps; i++) {
    let acc: Matrix = Array.from({ length: K }, () => new Array<number>(K).fill(0))
    for (let j = 1; j <= Math.min(i, p); j++) acc = add(acc, multiply(A[j - 1] as Matrix, phis[i - j] as Matrix))
    phis.push(acc)
  }
  const q = item(ndtri(1 - alpha / 2))
  let cov: Matrix = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const lower: Matrix = []
  const upper: Matrix = []
  for (let h = 0; h < steps; h++) {
    const phi = phis[h] as Matrix
    cov = add(cov, multiply(multiply(phi, fit.sigma), transpose(phi)))
    lower.push((mean[h] as number[]).map((v, e) => v - q * Math.sqrt(cov[e]?.[e] ?? 0)))
    upper.push((mean[h] as number[]).map((v, e) => v + q * Math.sqrt(cov[e]?.[e] ?? 0)))
  }
  return { mean, lower, upper }
}
