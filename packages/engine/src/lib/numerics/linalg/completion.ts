/**
 * Matrix completion from observed entries as traced algorithms with serialisable state: biased factorisation by
 * stochastic gradient descent (one epoch per step), alternating least squares (one half-sweep of ridge solves per
 * step), implicit-feedback ALS with confidence weights (Hu, Koren and Volinsky), and soft-impute (iterated
 * soft-thresholded SVD, Mazumder, Hastie and Tibshirani). Each reports its objective and the RMSE on the observed
 * entries; predictions and the completed matrix come from the state.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { type Key, normal, permutation } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../../trace/index.ts'
import { solveKernel, svdKernel } from './_kernels.ts'

/** Observed entries of an `m x n` matrix in coordinate form: entry `k` is `values[k]` at `(rows[k], cols[k])`. */
export interface ObservedEntries {
  readonly shape: readonly [number, number]
  readonly rows: ArrayLike<number>
  readonly cols: ArrayLike<number>
  readonly values: ArrayLike<number>
}

/** The factors of a completion: `u` is `m x rank`, `v` is `n x rank`, both row-major. */
export interface FactorState {
  readonly u: readonly number[]
  readonly v: readonly number[]
  /** Row and column biases and the global mean (the biased SGD model; zero for the others). */
  readonly rowBias: readonly number[]
  readonly colBias: readonly number[]
  readonly mean: number
  /** The half-sweep or epoch just made (0 at the start). */
  readonly iteration: number
  /** Which factor the next ALS half-sweep updates. */
  readonly next: 'rows' | 'cols'
  /** The regularised objective being minimised. */
  readonly objective: number
  /** Root mean squared error on the observed entries. */
  readonly rmse: number
}

/** Options shared by the factorisation algorithms. */
export interface FactorOptions {
  /** The number of latent factors. */
  readonly rank: number
  /** The ridge penalty on the factors (default 0.1). */
  readonly lambda?: number
  /** Starting factors `{ u: [m, rank], v: [n, rank] }` (default: Gaussian scaled by `initScale` from the run's key). */
  readonly init?: { readonly u: TensorLike; readonly v: TensorLike }
  /** Standard deviation of the Gaussian start (default 0.1). */
  readonly initScale?: number
}

function validate(obs: ObservedEntries): { m: number; n: number; count: number } {
  const [m, n] = obs.shape
  const count = obs.values.length
  if (obs.rows.length !== count || obs.cols.length !== count)
    throw new OmaiError('shape', 'ObservedEntries: rows, cols and values need the same length')
  for (let k = 0; k < count; k++) {
    const i = obs.rows[k] ?? -1
    const j = obs.cols[k] ?? -1
    if (!Number.isInteger(i) || !Number.isInteger(j) || i < 0 || i >= m || j < 0 || j >= n)
      throw new OmaiError('value', `ObservedEntries: entry ${k} lies outside the ${m} x ${n} matrix`)
  }
  return { m, n, count }
}

function startFactors(options: FactorOptions, m: number, n: number, key: Key): { u: number[]; v: number[] } {
  const r = options.rank
  if (options.init !== undefined) {
    const u = concreteOrThrow(promoteFloat(options.init.u)[0] as Tensor, 'completion')
    const v = concreteOrThrow(promoteFloat(options.init.v)[0] as Tensor, 'completion')
    if (u.shape[0] !== m || u.shape[1] !== r || v.shape[0] !== n || v.shape[1] !== r)
      throw new OmaiError('shape', `completion: init factors must be [${m}, ${r}] and [${n}, ${r}]`)
    return { u: Array.from(u.data), v: Array.from(v.data) }
  }
  const scale = options.initScale ?? 0.1
  const draw = normal(key, [(m + n) * r]).data
  return {
    u: Array.from(draw.subarray(0, m * r), (x) => x * scale),
    v: Array.from(draw.subarray(m * r), (x) => x * scale),
  }
}

function checkRank(rank: number): void {
  if (!Number.isInteger(rank) || rank < 1)
    throw new OmaiError('value', 'completion needs an integer rank of at least 1')
}

function byRow(obs: ObservedEntries, m: number): number[][] {
  const groups: number[][] = Array.from({ length: m }, () => [])
  for (let k = 0; k < obs.values.length; k++) groups[obs.rows[k] ?? 0]?.push(k)
  return groups
}

function byCol(obs: ObservedEntries, n: number): number[][] {
  const groups: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < obs.values.length; k++) groups[obs.cols[k] ?? 0]?.push(k)
  return groups
}

/** The residual statistics on the observed entries under the model `mean + rowBias + colBias + u . v`. */
function fit(
  obs: ObservedEntries,
  s: Pick<FactorState, 'u' | 'v' | 'rowBias' | 'colBias' | 'mean'>,
  r: number,
): { sse: number; rmse: number } {
  let sse = 0
  const count = obs.values.length
  for (let k = 0; k < count; k++) {
    const i = obs.rows[k] ?? 0
    const j = obs.cols[k] ?? 0
    let p = s.mean + (s.rowBias[i] ?? 0) + (s.colBias[j] ?? 0)
    for (let c = 0; c < r; c++) p += (s.u[i * r + c] ?? 0) * (s.v[j * r + c] ?? 0)
    sse += ((obs.values[k] ?? 0) - p) ** 2
  }
  return { sse, rmse: count === 0 ? 0 : Math.sqrt(sse / count) }
}

const squares = (x: readonly number[]): number => x.reduce((t, v) => t + v * v, 0)

const SCHEMA = channels({
  metrics: {
    objective: { description: 'the regularised objective', direction: 'lower' },
    rmse: { description: 'root mean squared error on the observed entries', direction: 'lower' },
  },
  tensors: {
    userFactors: { role: 'state', description: 'the row factors, m by rank' },
    itemFactors: { role: 'state', description: 'the column factors, n by rank' },
  },
})

function observeFactors(rows: number, cols: number, r: number) {
  return (s: FactorState) => ({
    metrics: { objective: s.objective, rmse: s.rmse },
    tensors: {
      userFactors: { role: 'state' as const, value: s.u, shape: [rows, r] },
      itemFactors: { role: 'state' as const, value: s.v, shape: [cols, r] },
    },
  })
}

const serialiseFactors = (s: FactorState): Json =>
  ({
    ...s,
    u: [...s.u],
    v: [...s.v],
    rowBias: [...s.rowBias],
    colBias: [...s.colBias],
  }) as Json
const deserialiseFactors = (j: Json): FactorState => j as unknown as FactorState

/**
 * Alternating least squares for the factorisation `X ~ U V^T` on the observed entries, minimising
 * `sum (x_ij - u_i . v_j)^2 + lambda (|U|^2 + |V|^2)`. One step is one half-sweep: all rows of `U` (each a ridge
 * solve over the entries observed in that row, with `V` fixed), then all rows of `V`, alternately, starting with `U`.
 * The objective never increases. A row with no observed entries gets a zero factor.
 *
 * @param obs - The observed entries.
 * @param options - Rank, ridge penalty and start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { alsCompletionAlgorithm } from 'omai-engine/numerics/linalg'
 * alsCompletionAlgorithm({ shape: [2, 2], rows: [0, 1], cols: [0, 1], values: [1, 2] }, { rank: 1 }).name // => 'als-completion'
 * ```
 */
export function alsCompletionAlgorithm(obs: ObservedEntries, options: FactorOptions): Algorithm<FactorState> {
  checkRank(options.rank)
  const { m, n } = validate(obs)
  const r = options.rank
  const lambda = options.lambda ?? 0.1
  const rowsOf = byRow(obs, m)
  const colsOf = byCol(obs, n)
  const zeros = (len: number): number[] => new Array<number>(len).fill(0)
  const stateOf = (u: number[], v: number[], iteration: number, next: 'rows' | 'cols'): FactorState => {
    const base = { u, v, rowBias: zeros(m), colBias: zeros(n), mean: 0 }
    const { sse, rmse } = fit(obs, base, r)
    return { ...base, iteration, next, objective: sse + lambda * (squares(u) + squares(v)), rmse }
  }
  /** Solve the ridge problem of every row of `target` given the fixed `other` factors. */
  const solveSide = (
    groups: number[][],
    other: readonly number[],
    indexOf: ArrayLike<number>,
    count: number,
  ): number[] => {
    const out = zeros(count * r)
    groups.forEach((entries, row) => {
      if (entries.length === 0) return
      const gram = new Float64Array(r * r)
      const rhs = new Float64Array(r)
      for (const k of entries) {
        const o = indexOf[k] ?? 0
        for (let a = 0; a < r; a++) {
          rhs[a] = (rhs[a] ?? 0) + (obs.values[k] ?? 0) * (other[o * r + a] ?? 0)
          for (let b = 0; b < r; b++)
            gram[a * r + b] = (gram[a * r + b] ?? 0) + (other[o * r + a] ?? 0) * (other[o * r + b] ?? 0)
        }
      }
      for (let a = 0; a < r; a++) gram[a * r + a] = (gram[a * r + a] ?? 0) + lambda
      const x = solveKernel(gram, rhs, r, 1)
      for (let a = 0; a < r; a++) out[row * r + a] = x[a] ?? 0
    })
    return out
  }
  return {
    name: 'als-completion',
    config: { rank: r, lambda, shape: [m, n], observed: obs.values.length } as Json,
    schema: SCHEMA,
    init: (key) => {
      const { u, v } = startFactors(options, m, n, key)
      return stateOf(u, v, 0, 'rows')
    },
    step: (s) => {
      if (s.next === 'rows') return stateOf(solveSide(rowsOf, s.v, obs.cols, m), [...s.v], s.iteration + 1, 'cols')
      return stateOf([...s.u], solveSide(colsOf, s.u, obs.rows, n), s.iteration + 1, 'rows')
    },
    observe: observeFactors(m, n, r),
    serialise: serialiseFactors,
    deserialise: deserialiseFactors,
  }
}

/** Options of the biased SGD factorisation. */
export interface SgdCompletionOptions extends FactorOptions {
  /** The step size (default 0.02). */
  readonly learningRate?: number
  /** The step size is multiplied by this after every epoch (default 1). */
  readonly decay?: number
  /** Visit the entries in a fresh random order each epoch (`'shuffled'`, default) or in the given order. */
  readonly order?: 'shuffled' | 'sequential'
  /** Fit row and column biases and the global mean (default true). */
  readonly biases?: boolean
}

/**
 * Biased matrix factorisation by stochastic gradient descent (Funk's SVD, as in the Netflix prize): the model
 * `mu + b_i + c_j + u_i . v_j` is fitted entry by entry, `e = x - prediction`, with updates `b_i += lr (e - reg b_i)`
 * and `u_i += lr (e v_j - reg u_i)` (the old `u_i` used for `v_j`). One step is one epoch over the observed entries,
 * in a random order drawn from the step's key. The objective is the squared error plus `lambda` times the squared
 * biases and factors.
 *
 * @param obs - The observed entries.
 * @param options - Rank, penalty, step size and decay, order and whether to fit biases.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { sgdCompletionAlgorithm } from 'omai-engine/numerics/linalg'
 * sgdCompletionAlgorithm({ shape: [2, 2], rows: [0, 1], cols: [0, 1], values: [1, 2] }, { rank: 1 }).name // => 'sgd-completion'
 * ```
 */
export function sgdCompletionAlgorithm(obs: ObservedEntries, options: SgdCompletionOptions): Algorithm<FactorState> {
  checkRank(options.rank)
  const { m, n, count } = validate(obs)
  const r = options.rank
  const lambda = options.lambda ?? 0.1
  const lr0 = options.learningRate ?? 0.02
  const decay = options.decay ?? 1
  const order = options.order ?? 'shuffled'
  const useBiases = options.biases ?? true
  let mean = 0
  for (let k = 0; k < count; k++) mean += obs.values[k] ?? 0
  mean = useBiases && count > 0 ? mean / count : 0
  const objectiveOf = (s: Omit<FactorState, 'objective' | 'rmse'>): { objective: number; rmse: number } => {
    const { sse, rmse } = fit(obs, s, r)
    return { objective: sse + lambda * (squares(s.u) + squares(s.v) + squares(s.rowBias) + squares(s.colBias)), rmse }
  }
  return {
    name: 'sgd-completion',
    config: {
      rank: r,
      lambda,
      learningRate: lr0,
      decay,
      order,
      biases: useBiases,
      shape: [m, n],
      observed: count,
    } as Json,
    schema: SCHEMA,
    init: (key) => {
      const { u, v } = startFactors(options, m, n, key)
      const base = {
        u,
        v,
        rowBias: new Array<number>(m).fill(0),
        colBias: new Array<number>(n).fill(0),
        mean,
        iteration: 0,
        next: 'rows' as const,
      }
      return { ...base, ...objectiveOf(base) }
    },
    step: (s, ctx) => {
      const u = Array.from(s.u)
      const v = Array.from(s.v)
      const bi = Array.from(s.rowBias)
      const bj = Array.from(s.colBias)
      const lr = lr0 * decay ** s.iteration
      const visit =
        order === 'shuffled' ? Array.from(permutation(ctx.key, count).data) : Array.from({ length: count }, (_, k) => k)
      for (const k of visit) {
        const i = obs.rows[k] ?? 0
        const j = obs.cols[k] ?? 0
        let p = mean + (bi[i] ?? 0) + (bj[j] ?? 0)
        for (let c = 0; c < r; c++) p += (u[i * r + c] ?? 0) * (v[j * r + c] ?? 0)
        const e = (obs.values[k] ?? 0) - p
        if (useBiases) {
          bi[i] = (bi[i] ?? 0) + lr * (e - lambda * (bi[i] ?? 0))
          bj[j] = (bj[j] ?? 0) + lr * (e - lambda * (bj[j] ?? 0))
        }
        for (let c = 0; c < r; c++) {
          const uic = u[i * r + c] ?? 0
          const vjc = v[j * r + c] ?? 0
          u[i * r + c] = uic + lr * (e * vjc - lambda * uic)
          v[j * r + c] = vjc + lr * (e * uic - lambda * vjc)
        }
      }
      const base = { u, v, rowBias: bi, colBias: bj, mean, iteration: s.iteration + 1, next: 'rows' as const }
      return { ...base, ...objectiveOf(base) }
    },
    observe: observeFactors(m, n, r),
    serialise: serialiseFactors,
    deserialise: deserialiseFactors,
  }
}

/** Options of implicit-feedback ALS. */
export interface ImplicitAlsOptions extends FactorOptions {
  /** The confidence scale: an observed entry with count `r` has confidence `1 + alpha r` (default 40). */
  readonly alpha?: number
}

/**
 * Implicit-feedback ALS (Hu, Koren and Volinsky): every cell has a preference `p_ij` (1 where an entry is observed, 0
 * elsewhere) and a confidence `c_ij = 1 + alpha x_ij` (1 where unobserved); the factors minimise
 * `sum_{all i,j} c_ij (p_ij - u_i . v_j)^2 + lambda (|U|^2 + |V|^2)`. A half-sweep solves each row exactly using the
 * identity `V^T C_i V = V^T V + V^T (C_i - I) V`, which touches only the observed entries. Observed values are
 * the interaction counts (positive).
 *
 * @param obs - The observed counts (positive values).
 * @param options - Rank, penalty, confidence scale and start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { implicitAlsAlgorithm } from 'omai-engine/numerics/linalg'
 * implicitAlsAlgorithm({ shape: [2, 2], rows: [0, 1], cols: [0, 1], values: [3, 1] }, { rank: 1 }).name // => 'implicit-als'
 * ```
 */
export function implicitAlsAlgorithm(obs: ObservedEntries, options: ImplicitAlsOptions): Algorithm<FactorState> {
  checkRank(options.rank)
  const { m, n, count } = validate(obs)
  for (let k = 0; k < count; k++)
    if (!((obs.values[k] ?? 0) > 0)) throw new OmaiError('value', 'implicitAls needs positive counts')
  const r = options.rank
  const lambda = options.lambda ?? 0.1
  const alpha = options.alpha ?? 40
  const rowsOf = byRow(obs, m)
  const colsOf = byCol(obs, n)
  const zeros = (len: number): number[] => new Array<number>(len).fill(0)
  /** The objective over every cell: unobserved cells have preference 0 and confidence 1. */
  const objectiveOf = (u: readonly number[], v: readonly number[]): { objective: number; rmse: number } => {
    const observed = new Map<number, number>()
    for (let k = 0; k < count; k++)
      observed.set((obs.rows[k] ?? 0) * n + (obs.cols[k] ?? 0), 1 + alpha * (obs.values[k] ?? 0))
    let total = 0
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let p = 0
        for (let c = 0; c < r; c++) p += (u[i * r + c] ?? 0) * (v[j * r + c] ?? 0)
        const conf = observed.get(i * n + j)
        total += conf === undefined ? p * p : conf * (1 - p) ** 2
      }
    const { rmse } = fit(
      { ...obs, values: Array.from({ length: count }, () => 1) },
      { u, v, rowBias: zeros(m), colBias: zeros(n), mean: 0 },
      r,
    )
    return { objective: total + lambda * (squares(u) + squares(v)), rmse }
  }
  const stateOf = (u: number[], v: number[], iteration: number, next: 'rows' | 'cols'): FactorState => ({
    u,
    v,
    rowBias: zeros(m),
    colBias: zeros(n),
    mean: 0,
    iteration,
    next,
    ...objectiveOf(u, v),
  })
  const solveSide = (
    groups: number[][],
    other: readonly number[],
    indexOf: ArrayLike<number>,
    total: number,
    otherCount: number,
  ): number[] => {
    const gramAll = new Float64Array(r * r)
    for (let o = 0; o < otherCount; o++)
      for (let a = 0; a < r; a++)
        for (let b = 0; b < r; b++)
          gramAll[a * r + b] = (gramAll[a * r + b] ?? 0) + (other[o * r + a] ?? 0) * (other[o * r + b] ?? 0)
    const out = zeros(total * r)
    groups.forEach((entries, row) => {
      const gram = Float64Array.from(gramAll)
      const rhs = new Float64Array(r)
      for (const k of entries) {
        const o = indexOf[k] ?? 0
        const extra = alpha * (obs.values[k] ?? 0)
        for (let a = 0; a < r; a++) {
          rhs[a] = (rhs[a] ?? 0) + (1 + extra) * (other[o * r + a] ?? 0)
          for (let b = 0; b < r; b++)
            gram[a * r + b] = (gram[a * r + b] ?? 0) + extra * (other[o * r + a] ?? 0) * (other[o * r + b] ?? 0)
        }
      }
      for (let a = 0; a < r; a++) gram[a * r + a] = (gram[a * r + a] ?? 0) + lambda
      const x = solveKernel(gram, rhs, r, 1)
      for (let a = 0; a < r; a++) out[row * r + a] = x[a] ?? 0
    })
    return out
  }
  return {
    name: 'implicit-als',
    config: { rank: r, lambda, alpha, shape: [m, n], observed: count } as Json,
    schema: SCHEMA,
    init: (key) => {
      const { u, v } = startFactors(options, m, n, key)
      return stateOf(u, v, 0, 'rows')
    },
    step: (s) => {
      if (s.next === 'rows') return stateOf(solveSide(rowsOf, s.v, obs.cols, m, n), [...s.v], s.iteration + 1, 'cols')
      return stateOf([...s.u], solveSide(colsOf, s.u, obs.rows, n, m), s.iteration + 1, 'rows')
    },
    observe: observeFactors(m, n, r),
    serialise: serialiseFactors,
    deserialise: deserialiseFactors,
  }
}

/**
 * The prediction `mu + b_i + c_j + u_i . v_j` of a factorisation state.
 *
 * @param state - A state of {@link alsCompletionAlgorithm}, {@link sgdCompletionAlgorithm} or {@link implicitAlsAlgorithm}.
 * @param rank - The number of factors.
 * @param i - Row.
 * @param j - Column.
 * @returns The predicted entry.
 * @example
 * ```ts
 * import { predictEntry } from 'omai-engine/numerics/linalg'
 * const state = { u: [1, 2], v: [3, 4], rowBias: [0], colBias: [0], mean: 1, iteration: 0, next: 'rows', objective: 0, rmse: 0 } as const
 * predictEntry(state, 2, 0, 0) // => 12
 * ```
 */
export function predictEntry(state: FactorState, rank: number, i: number, j: number): number {
  let p = state.mean + (state.rowBias[i] ?? 0) + (state.colBias[j] ?? 0)
  for (let c = 0; c < rank; c++) p += (state.u[i * rank + c] ?? 0) * (state.v[j * rank + c] ?? 0)
  return p
}

/** The state of soft-impute: the completed matrix and the singular values that built it. */
export interface SoftImputeState {
  /** The current completed matrix `Z`, row-major `m x n`. */
  readonly z: readonly number[]
  /** The soft-thresholded singular values of `Z` (the nonzero ones, descending). */
  readonly singularValues: readonly number[]
  readonly iteration: number
  /** The objective `1/2 |P(X - Z)|^2 + lambda |Z|_*` (nuclear norm). */
  readonly objective: number
  /** `|Z_new - Z|^2 / |Z|^2` of the last step. */
  readonly relativeChange: number
  /** RMSE on the observed entries. */
  readonly rmse: number
}

/** Options of soft-impute. */
export interface SoftImputeOptions {
  /** The nuclear-norm penalty (singular values are shrunk by this much); default a tenth of {@link softImputeLambdaMax}. */
  readonly lambda?: number
  /** Stop when the relative change of `Z` is below this (default 1e-9). */
  readonly tol?: number
}

function denseSvd(a: Float64Array, m: number, n: number): { u: Float64Array; s: Float64Array; v: Float64Array } {
  return svdKernel(a, m, n, 'golub-kahan')
}

/**
 * The smallest penalty at which soft-impute returns the zero matrix: the largest singular value of the zero-filled
 * observed matrix. Penalties below it give solutions of increasing rank.
 *
 * @param obs - The observed entries.
 * @returns The largest singular value.
 * @example
 * ```ts
 * import { softImputeLambdaMax } from 'omai-engine/numerics/linalg'
 * softImputeLambdaMax({ shape: [2, 2], rows: [0, 1], cols: [0, 1], values: [3, 4] }) // => 4
 * ```
 */
export function softImputeLambdaMax(obs: ObservedEntries): number {
  const { m, n } = validate(obs)
  const x = new Float64Array(m * n)
  for (let k = 0; k < obs.values.length; k++) {
    const at = (obs.rows[k] ?? 0) * n + (obs.cols[k] ?? 0)
    x[at] = obs.values[k] ?? 0
  }
  return denseSvd(x, m, n).s[0] ?? 0
}

/**
 * Soft-impute: starting from the zero matrix, repeatedly fill the unobserved cells with the current estimate, take the
 * SVD, and shrink the singular values by `lambda`. This is a proximal gradient method for the nuclear-norm
 * regularised completion `1/2 |P(X - Z)|^2 + lambda |Z|_*`, whose objective falls at every step. One step is one SVD.
 *
 * @param obs - The observed entries (where an entry is listed twice, the last value counts).
 * @param options - The penalty and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { softImputeAlgorithm } from 'omai-engine/numerics/linalg'
 * softImputeAlgorithm({ shape: [2, 2], rows: [0, 1], cols: [0, 1], values: [3, 4] }).name // => 'soft-impute'
 * ```
 */
export function softImputeAlgorithm(obs: ObservedEntries, options: SoftImputeOptions = {}): Algorithm<SoftImputeState> {
  const { m, n, count } = validate(obs)
  const lambda = options.lambda ?? 0.1 * softImputeLambdaMax(obs)
  const tol = options.tol ?? 1e-9
  const measure = (z: ArrayLike<number>, singularValues: readonly number[]): { objective: number; rmse: number } => {
    let sse = 0
    for (let k = 0; k < count; k++)
      sse += ((obs.values[k] ?? 0) - (z[(obs.rows[k] ?? 0) * n + (obs.cols[k] ?? 0)] ?? 0)) ** 2
    return {
      objective: 0.5 * sse + lambda * singularValues.reduce((t, x) => t + x, 0),
      rmse: count === 0 ? 0 : Math.sqrt(sse / count),
    }
  }
  return {
    name: 'soft-impute',
    config: { lambda, tol, shape: [m, n], observed: count } as Json,
    schema: channels({
      metrics: {
        objective: {
          description: 'half the squared error on the observed entries plus lambda times the nuclear norm',
          direction: 'lower',
        },
        rmse: { description: 'root mean squared error on the observed entries', direction: 'lower' },
        rank: { description: 'the number of singular values above the threshold' },
        relativeChange: { description: 'relative change of the completed matrix in the last step', direction: 'lower' },
      },
      tensors: { completed: { role: 'state', description: 'the completed matrix' } },
    }),
    init: () => {
      const z = new Array<number>(m * n).fill(0)
      return { z, singularValues: [], iteration: 0, relativeChange: Number.POSITIVE_INFINITY, ...measure(z, []) }
    },
    step: (s) => {
      const filled = Float64Array.from(s.z)
      // Observed cells take the data (the last entry wins on duplicates).
      for (let k = 0; k < count; k++) filled[(obs.rows[k] ?? 0) * n + (obs.cols[k] ?? 0)] = obs.values[k] ?? 0
      const { u, s: sv, v } = denseSvd(filled, m, n)
      const k = sv.length
      const shrunk = Array.from(sv, (x) => Math.max(x - lambda, 0))
      const z = new Array<number>(m * n).fill(0)
      for (let c = 0; c < k; c++) {
        const w = shrunk[c] ?? 0
        if (w === 0) continue
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++) z[i * n + j] = (z[i * n + j] ?? 0) + w * (u[i * k + c] ?? 0) * (v[j * k + c] ?? 0)
      }
      let change = 0
      let norm = 0
      for (let i = 0; i < m * n; i++) {
        change += ((z[i] ?? 0) - (s.z[i] ?? 0)) ** 2
        norm += (z[i] ?? 0) ** 2
      }
      const kept = shrunk.filter((x) => x > 0)
      return {
        z,
        singularValues: kept,
        iteration: s.iteration + 1,
        relativeChange: norm === 0 ? change : change / norm,
        ...measure(z, kept),
      }
    },
    done: (s) => (s.relativeChange <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        rmse: s.rmse,
        rank: s.singularValues.length,
        relativeChange: Number.isFinite(s.relativeChange) ? s.relativeChange : 1,
      },
      tensors: { completed: { role: 'state' as const, value: s.z, shape: [m, n] } },
    }),
    serialise: (s) =>
      ({
        ...s,
        z: [...s.z],
        singularValues: [...s.singularValues],
        relativeChange: Number.isFinite(s.relativeChange) ? s.relativeChange : 'inf',
      }) as Json,
    deserialise: (j) => {
      const o = j as unknown as Omit<SoftImputeState, 'relativeChange'> & { relativeChange: number | string }
      return {
        ...o,
        relativeChange: o.relativeChange === 'inf' ? Number.POSITIVE_INFINITY : (o.relativeChange as number),
      }
    },
  }
}

/** The result of a completion run. */
export interface CompletionResult {
  /** The completed matrix `m x n` (the model's prediction in every cell). */
  readonly completed: Tensor
  readonly iterations: number
  readonly converged: boolean
  /** How the run ended (`converged`, `completed` at the iteration cap, ...). */
  readonly status: RunStatus
  /** The objective after each step (the first is the start's). */
  readonly history: readonly number[]
}

/**
 * Complete a matrix with soft-impute until the completed matrix stops changing.
 *
 * @param obs - The observed entries.
 * @param options - Penalty, tolerance, iteration cap (default 500) and trace options.
 * @returns The completed matrix and the objective history.
 * @example
 * ```ts
 * import { softImpute } from 'omai-engine/numerics/linalg'
 * softImpute({ shape: [2, 2], rows: [0, 0, 1], cols: [0, 1, 0], values: [1, 2, 2] }, { lambda: 0.01 }).completed.shape // => [2, 2]
 * ```
 */
export function softImpute(
  obs: ObservedEntries,
  options: SoftImputeOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): CompletionResult {
  const { lambda, tol, maxIterations, ...run } = options
  const algorithm = softImputeAlgorithm(obs, {
    ...(lambda === undefined ? {} : { lambda }),
    ...(tol === undefined ? {} : { tol }),
  })
  const history: number[] = []
  const watched: Algorithm<SoftImputeState> = {
    ...algorithm,
    init: (key) => {
      const s = algorithm.init(key)
      history.push(s.objective)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.objective)
      return next
    },
  }
  const result = runAlgorithm(watched, { ...run, maxSteps: maxIterations ?? 500 })
  const [m, n] = obs.shape
  return {
    completed: fromStorage(Float64Array.from(result.state.z), [m, n], 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
    status: result.status,
    history,
  }
}

/**
 * The matrix a factorisation state predicts in every cell.
 *
 * @param state - A factorisation state.
 * @param rank - The number of factors.
 * @returns The `m x n` matrix.
 * @example
 * ```ts
 * import { factorsToMatrix } from 'omai-engine/numerics/linalg'
 * const state = { u: [1, 2], v: [3, 4], rowBias: [0, 0], colBias: [0, 0], mean: 0, iteration: 0, next: 'rows', objective: 0, rmse: 0 } as const
 * factorsToMatrix(state, 1).shape // => [2, 2]
 * ```
 */
export function factorsToMatrix(state: FactorState, rank: number): Tensor {
  const m = state.u.length / rank
  const n = state.v.length / rank
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = predictEntry(state, rank, i, j)
  return fromStorage(out, [m, n], 'float64')
}
