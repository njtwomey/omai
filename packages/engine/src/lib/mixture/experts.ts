/**
 * The mixture of linear experts (Jacobs, Jordan, Nowlan and Hinton) fitted by expectation-maximisation: a gating
 * network, a softmax of linear functions of the features with a bias, shares each row among linear-Gaussian experts,
 * each with its own coefficients and noise variance. Each step computes every expert's posterior responsibility for
 * every row (E), then refits each expert by least squares weighted by its responsibilities and moves the gate by a few
 * damped Newton steps on the weighted multinomial logistic objective (M), so the log likelihood never decreases. The
 * state keeps no past (the tape records it).
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { type MatrixView, matrixView, targetVector } from '../learn/index.ts'
import { leastSquaresSolve, matrixTensor, vectorTensor } from '../linear/index.ts'
import { cholesky, choSolve, ddot } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { normaliseLogDensities } from './gaussian.ts'

/**
 * The parameters of a mixture of linear experts. Every coefficient vector runs over `[1, x_1, ..., x_d]`, intercept
 * first. The gate is `softmax(gate x)` over the experts; the last expert's row is held at zero (the softmax does not
 * change when every row moves by the same amount).
 */
export interface MoeParameters {
  /** gate[k]: expert k's gate weights, `[k][d + 1]`. */
  readonly gate: number[][]
  /** experts[k]: expert k's regression coefficients, `[k][d + 1]`. */
  readonly experts: number[][]
  /** variances[k]: expert k's noise variance. */
  readonly variances: number[]
}

/**
 * The state of EM for a mixture of experts: the parameters, the mean log likelihood of the step before them and how
 * far it moved, and the responsibilities that gave them. Before the first step the likelihood is `-Number.MAX_VALUE`
 * and the change `Number.MAX_VALUE` (finite, so the state is plain JSON).
 */
export interface MoeState extends MoeParameters {
  readonly logLikelihood: number
  readonly change: number
  /** resp[i][k]: the posterior probability that expert k produced row i. */
  readonly resp: number[][]
}

/** Options of {@link mixtureOfExpertsAlgorithm}. */
export interface MoeOptions {
  /** The number of experts (default 2). */
  readonly nExperts?: number
  /** The change in mean log likelihood that stops EM (default 1e-6). */
  readonly tol?: number
  /** Newton steps on the gate per EM step (default 3). */
  readonly gateSteps?: number
  /**
   * The ridge penalty on the gate's feature weights (not its intercepts), default 0.01. Without it a gate that
   * separates the experts perfectly sharpens for ever; with it the gate settles. EM raises the penalised likelihood,
   * and `0` is plain maximum likelihood.
   */
  readonly gateRidge?: number
  /** The floor of every expert's noise variance, as a share of the variance of the targets (default 1e-4). */
  readonly minVariance?: number
  /** Start from these parameters instead of random chords through the data. */
  readonly start?: MoeParameters
}

/** The row `[1, x_i...]` of the design with an intercept column, row-major `[n, d + 1]`. */
function augmentedOf(m: MatrixView): Float64Array {
  const q = m.d + 1
  const a = new Float64Array(m.n * q)
  for (let i = 0; i < m.n; i++) {
    a[i * q] = 1
    for (let j = 0; j < m.d; j++) a[i * q + j + 1] = m.data[i * m.d + j] ?? 0
  }
  return a
}

/** Log-softmax gate probabilities `[n][k]` of an augmented design under the gate's weights. */
function gateLogProbs(a: Float64Array, n: number, gate: readonly (readonly number[])[]): number[][] {
  const q = gate[0]?.length ?? 1
  return Array.from({ length: n }, (_, i) => {
    const row = a.subarray(i * q, (i + 1) * q)
    const eta = gate.map((w) => ddot(w, row))
    const top = maxOf(eta)
    const logZ = top + Math.log(eta.reduce((s, v) => s + Math.exp(v - top), 0))
    return eta.map((v) => v - logZ)
  })
}

/** Each expert's mean `[n][k]` of an augmented design. */
function expertMeansOf(a: Float64Array, n: number, experts: readonly (readonly number[])[]): number[][] {
  const q = experts[0]?.length ?? 1
  return Array.from({ length: n }, (_, i) => {
    const row = a.subarray(i * q, (i + 1) * q)
    return experts.map((w) => ddot(w, row))
  })
}

function checkParameters(p: MoeParameters, d: number, who: string): void {
  const k = p.experts.length
  if (k < 1 || p.gate.length !== k || p.variances.length !== k)
    throw new MdmlError('shape', `${who}: gate, experts and variances must have one entry per expert`)
  for (const row of [...p.gate, ...p.experts])
    if (row.length !== d + 1)
      throw new MdmlError('shape', `${who}: expected ${d + 1} coefficients per expert (intercept and ${d} features)`)
}

/**
 * The gate: the probability each expert is the one to ask at each row, a softmax of linear functions of the features.
 *
 * @param x - Points `[m, d]`.
 * @param gate - The gate weights `[k][d + 1]` (intercept first).
 * @returns Probabilities `[m][k]`, each row summing to one.
 * @throws MdmlError with code `shape` when the weights do not match the features.
 * @example
 * ```ts
 * import { moeGate } from 'mdml-engine/mixture'
 * moeGate([[0], [10]], [[1, 0], [0, 0]])[0] // => [0.7310585786300049, 0.2689414213699951]
 * ```
 */
export function moeGate(x: TensorLike, gate: readonly (readonly number[])[]): number[][] {
  const m = matrixView(x, 'moeGate')
  if (gate.length < 1 || gate.some((w) => w.length !== m.d + 1))
    throw new MdmlError('shape', `moeGate: expected gate weights of length ${m.d + 1} (intercept and ${m.d} features)`)
  return gateLogProbs(augmentedOf(m), m.n, gate).map((row) => row.map(Math.exp))
}

/**
 * Each expert's own prediction at each row: its line, whatever the gate thinks of it.
 *
 * @param x - Points `[m, d]`.
 * @param experts - The coefficients `[k][d + 1]` (intercept first).
 * @returns Predictions `[m][k]`.
 * @throws MdmlError with code `shape` when the coefficients do not match the features.
 * @example
 * ```ts
 * import { moeExpertMeans } from 'mdml-engine/mixture'
 * moeExpertMeans([[2]], [[1, 1], [0, -1]]) // => [[3, -2]]
 * ```
 */
export function moeExpertMeans(x: TensorLike, experts: readonly (readonly number[])[]): number[][] {
  const m = matrixView(x, 'moeExpertMeans')
  if (experts.length < 1 || experts.some((w) => w.length !== m.d + 1))
    throw new MdmlError('shape', `moeExpertMeans: expected ${m.d + 1} coefficients per expert`)
  return expertMeansOf(augmentedOf(m), m.n, experts)
}

/**
 * The mixture's mean at each row: the experts' lines averaged by the gate.
 *
 * @param x - Points `[m, d]`.
 * @param params - The gate, the experts and their variances.
 * @returns The predictions `[m]`.
 * @example
 * ```ts
 * import { moePredict } from 'mdml-engine/mixture'
 * const p = { gate: [[0, 0], [0, 0]], experts: [[0, 2], [4, 0]], variances: [1, 1] }
 * moePredict([[1]], p) // => [3]
 * ```
 */
export function moePredict(x: TensorLike, params: MoeParameters): number[] {
  const m = matrixView(x, 'moePredict')
  checkParameters(params, m.d, 'moePredict')
  const a = augmentedOf(m)
  const g = gateLogProbs(a, m.n, params.gate)
  const mu = expertMeansOf(a, m.n, params.experts)
  return g.map((row, i) => row.reduce((s, lp, k) => s + Math.exp(lp) * (mu[i]?.[k] ?? 0), 0))
}

/** The log joint of every row and expert, `log g_k(x_i) + log N(y_i | mu_ik, sigma_k^2)`, `[n][k]`. */
function logJoint(a: Float64Array, y: ArrayLike<number>, n: number, p: MoeParameters): number[][] {
  const g = gateLogProbs(a, n, p.gate)
  const mu = expertMeansOf(a, n, p.experts)
  return g.map((row, i) =>
    row.map((lg, k) => {
      const v = p.variances[k] ?? 1
      const r = (y[i] ?? 0) - (mu[i]?.[k] ?? 0)
      return lg - 0.5 * Math.log(2 * Math.PI * v) - (r * r) / (2 * v)
    }),
  )
}

/**
 * The E-step: the posterior responsibility of each expert for each row, and each row's log density under the mixture.
 *
 * @param x - Points `[n, d]`.
 * @param y - Targets `[n]`.
 * @param params - The gate, the experts and their variances.
 * @returns Responsibilities `[n][k]` and the log density of each row.
 * @example
 * ```ts
 * import { moePosterior } from 'mdml-engine/mixture'
 * const p = { gate: [[0, 0], [0, 0]], experts: [[0, 1], [0, -1]], variances: [0.1, 0.1] }
 * moePosterior([[2]], [2], p).resp[0][0].toFixed(3) // => '1.000'
 * ```
 */
export function moePosterior(
  x: TensorLike,
  y: TensorLike,
  params: MoeParameters,
): { readonly resp: number[][]; readonly logNorm: number[] } {
  const m = matrixView(x, 'moePosterior')
  checkParameters(params, m.d, 'moePosterior')
  return normaliseLogDensities(logJoint(augmentedOf(m), targetVector(y, 'moePosterior', m.n), m.n, params))
}

/** The most responsible expert of each row. */
function mostResponsible(resp: readonly (readonly number[])[]): number[] {
  return resp.map((r) => r.indexOf(maxOf(r)))
}

/**
 * The most responsible expert of each row: the hard assignment behind the soft responsibilities.
 *
 * @param x - Points `[n, d]`.
 * @param y - Targets `[n]`.
 * @param params - The gate, the experts and their variances.
 * @returns The expert index per row.
 * @example
 * ```ts
 * import { moeAssignments } from 'mdml-engine/mixture'
 * const p = { gate: [[0, 0], [0, 0]], experts: [[0, 1], [0, -1]], variances: [0.1, 0.1] }
 * moeAssignments([[2], [2]], [2, -2], p) // => [0, 1]
 * ```
 */
export function moeAssignments(x: TensorLike, y: TensorLike, params: MoeParameters): number[] {
  return mostResponsible(moePosterior(x, y, params).resp)
}

/** One Newton step for the gate: ridge-penalised multinomial logistic regression on soft labels, from `w` (`[k - 1][q]`). */
function gateNewton(
  a: Float64Array,
  n: number,
  q: number,
  resp: readonly (readonly number[])[],
  w: number[][],
  ridge: number,
): number[][] {
  const k = w.length + 1
  const free = k - 1
  const size = free * q
  const withLast = (rows: readonly (readonly number[])[]): number[][] => [
    ...rows.map((r) => [...r]),
    new Array<number>(q).fill(0),
  ]
  const objectiveOf = (rows: readonly (readonly number[])[]): number => {
    const lp = gateLogProbs(a, n, withLast(rows))
    let s = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) s += (resp[i]?.[j] ?? 0) * (lp[i]?.[j] ?? 0)
    for (const row of rows) for (let u = 1; u < q; u++) s -= (ridge / 2) * (row[u] ?? 0) ** 2
    return s
  }
  const probs = gateLogProbs(a, n, withLast(w)).map((row) => row.map(Math.exp))
  const grad = new Float64Array(size)
  const hess = new Float64Array(size * size)
  for (let i = 0; i < n; i++) {
    const row = a.subarray(i * q, (i + 1) * q)
    const p = probs[i] ?? []
    for (let j = 0; j < free; j++) {
      const rj = (resp[i]?.[j] ?? 0) - (p[j] ?? 0)
      for (let u = 0; u < q; u++) grad[j * q + u] = (grad[j * q + u] ?? 0) + rj * (row[u] ?? 0)
      for (let l = 0; l <= j; l++) {
        const c = (p[j] ?? 0) * ((j === l ? 1 : 0) - (p[l] ?? 0))
        for (let u = 0; u < q; u++)
          for (let v = 0; v < q; v++) {
            const at = (j * q + u) * size + l * q + v
            hess[at] = (hess[at] ?? 0) + c * (row[u] ?? 0) * (row[v] ?? 0)
          }
      }
    }
  }
  for (let j = 0; j < free; j++)
    for (let u = 1; u < q; u++) {
      grad[j * q + u] = (grad[j * q + u] ?? 0) - ridge * (w[j]?.[u] ?? 0)
      hess[(j * q + u) * size + j * q + u] = (hess[(j * q + u) * size + j * q + u] ?? 0) + ridge
    }
  // The Hessian is symmetric: fill the upper triangle from the lower.
  for (let r = 0; r < size; r++) for (let c = r + 1; c < size; c++) hess[r * size + c] = hess[c * size + r] ?? 0
  let scale = 0
  for (let r = 0; r < size; r++) scale = Math.max(scale, hess[r * size + r] ?? 0)
  // Levenberg damping on the Hessian only (the objective is the plain likelihood), raised until the matrix factors.
  let direction: Float64Array | undefined
  for (let damp = 1e-8 * (scale + 1); direction === undefined && damp < 1e8; damp *= 100) {
    const damped = Float64Array.from(hess)
    for (let r = 0; r < size; r++) damped[r * size + r] = (damped[r * size + r] ?? 0) + damp
    try {
      const sol = Float64Array.from(
        toFlat(primalOf(choSolve(cholesky(matrixTensor(damped, size, size)), vectorTensor(grad)))),
      )
      if (sol.every(Number.isFinite)) direction = sol
    } catch {
      // Not positive definite at this damping: raise it.
    }
  }
  if (direction === undefined) return w
  const before = objectiveOf(w)
  for (let t = 1; t > 1e-4; t /= 2) {
    const next = w.map((row, j) => row.map((v, u) => v + t * (direction[j * q + u] ?? 0)))
    if (objectiveOf(next) >= before) return next
  }
  return w
}

/**
 * The gate's M-step: a few damped Newton steps on the weighted multinomial logistic objective
 * `sum_i sum_k r_ik log g_k(x_i)` less the ridge penalty, each accepted only if it does not lower the objective. The
 * last expert's row is held at zero.
 */
function fitGate(
  a: Float64Array,
  n: number,
  q: number,
  resp: readonly (readonly number[])[],
  gate: readonly (readonly number[])[],
  steps: number,
  ridge: number,
): number[][] {
  const k = gate.length
  if (k === 1) return [new Array<number>(q).fill(0)]
  let w = gate.slice(0, k - 1).map((row) => [...row])
  for (let s = 0; s < steps; s++) w = gateNewton(a, n, q, resp, w, ridge)
  return [...w, new Array<number>(q).fill(0)]
}

/** The M-step: each expert by least squares weighted by its responsibilities, then its variance, then the gate. */
function mStep(
  m: MatrixView,
  a: Float64Array,
  y: Float64Array,
  resp: readonly (readonly number[])[],
  gate: readonly (readonly number[])[],
  varianceFloor: number,
  gateSteps: number,
  gateRidge: number,
): MoeParameters {
  const k = gate.length
  const experts: number[][] = []
  const variances: number[] = []
  for (let j = 0; j < k; j++) {
    const w = Float64Array.from(resp, (r) => r[j] ?? 0)
    const solved = leastSquaresSolve({ x: m, y, w }, 0, true, 'cholesky')
    const beta = [solved.intercept, ...solved.coef]
    let rss = 0
    let total = 0
    for (let i = 0; i < m.n; i++) {
      const r = (y[i] ?? 0) - ddot(beta, a.subarray(i * (m.d + 1), (i + 1) * (m.d + 1)))
      rss += (w[i] ?? 0) * r * r
      total += w[i] ?? 0
    }
    experts.push(beta)
    variances.push(Math.max(varianceFloor, total > 0 ? rss / total : varianceFloor))
  }
  return { gate: fitGate(a, m.n, m.d + 1, resp, gate, gateSteps, gateRidge), experts, variances }
}

/** `nExperts` random chords: each expert interpolates `d + 1` random rows; the variances start at the targets'. */
function randomStart(m: MatrixView, y: Float64Array, nExperts: number, varianceFloor: number, key: Key): MoeParameters {
  const q = m.d + 1
  const mean = y.reduce((s, v) => s + v, 0) / m.n
  const spread = Math.max(varianceFloor, y.reduce((s, v) => s + (v - mean) ** 2, 0) / m.n)
  const experts = Array.from({ length: nExperts }, (_, j) => {
    const rows = Array.from(toFlat(permutation(foldIn(key, j), m.n)).slice(0, Math.min(m.n, q)))
    const data = new Float64Array(rows.length * m.d)
    for (const [i, r] of rows.entries()) data.set(m.data.subarray(r * m.d, (r + 1) * m.d), i * m.d)
    const design = {
      x: { n: rows.length, d: m.d, data },
      y: Float64Array.from(rows, (r) => y[r] ?? 0),
      w: new Float64Array(rows.length).fill(1),
    }
    // A tiny ridge keeps the chord defined when two of the rows share a feature value.
    const solved = leastSquaresSolve(design, 1e-8, true, 'cholesky')
    return [solved.intercept, ...solved.coef]
  })
  return {
    gate: Array.from({ length: nExperts }, () => new Array<number>(q).fill(0)),
    experts,
    variances: new Array<number>(nExperts).fill(spread),
  }
}

/** The same mixture with the last expert's gate row subtracted from every row, so that it is zero. */
function normalisedGate(p: MoeParameters): MoeParameters {
  const last = p.gate[p.gate.length - 1] ?? []
  return { ...p, gate: p.gate.map((row) => row.map((v, u) => v - (last[u] ?? 0))) }
}

const emSchema = channels({
  metrics: {
    logLikelihood: {
      description: 'mean log likelihood per row under the parameters before this step',
      direction: 'higher',
    },
    change: { description: 'how much the mean log likelihood moved this step' },
  },
  tensors: {
    responsibilities: { role: 'state', description: 'the posterior of each expert for each row', shape: ['n', 'k'] },
    gate: {
      role: 'param',
      description: 'the gate weights, intercept first (the last expert is held at zero)',
      shape: ['k', 'd+1'],
    },
    experts: { role: 'param', description: 'the experts’ coefficients, intercept first', shape: ['k', 'd+1'] },
    variances: { role: 'param', description: 'the experts’ noise variances', shape: ['k'] },
  },
})

/**
 * EM for a mixture of linear experts as a traced algorithm: each step computes the experts' responsibilities and the
 * mean log likelihood under the current parameters (E), then refits every expert by weighted least squares, its
 * noise variance, and the gate by damped Newton steps (M). No step lowers the likelihood. Converged when the mean log
 * likelihood changes by less than tol. The state keeps no past: the tape records the log likelihood, its change, the
 * responsibilities, the gate, the experts and their variances.
 *
 * @param x - Points `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - The number of experts, tolerance, Newton steps on the gate, variance floor and an optional start.
 * @param key - Randomness for the start (random chords through the data).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when there are more experts than rows.
 * @example
 * ```ts
 * import { mixtureOfExpertsAlgorithm, moePredict } from 'mdml-engine/mixture'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // One expert is ordinary least squares; the gate has nothing to decide.
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const y = [0, 1, 2, 3, 2, 1]
 * const alg = mixtureOfExpertsAlgorithm(x, y, { nExperts: 1 }, key(0))
 * moePredict([[2]], runAlgorithm(alg, { maxSteps: 20 }).state).map((v) => Math.round(v * 100) / 100) // => [1.37]
 * ```
 */
export function mixtureOfExpertsAlgorithm(
  x: TensorLike,
  y: TensorLike,
  options: MoeOptions,
  key: Key,
): Algorithm<MoeState> {
  const m = matrixView(x, 'mixtureOfExperts')
  const targets = targetVector(y, 'mixtureOfExperts', m.n)
  const k = options.nExperts ?? options.start?.experts.length ?? 2
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `mixtureOfExperts: nExperts must be in 1..${m.n}`)
  const tol = options.tol ?? 1e-6
  const gateSteps = options.gateSteps ?? 3
  const gateRidge = options.gateRidge ?? 0.01
  const mean = targets.reduce((s, v) => s + v, 0) / m.n
  const floor = Math.max(
    1e-12,
    (options.minVariance ?? 1e-4) * (targets.reduce((s, v) => s + (v - mean) ** 2, 0) / m.n),
  )
  const a = augmentedOf(m)
  if (options.start !== undefined) checkParameters(options.start, m.d, 'mixtureOfExperts')
  return {
    name: 'mixture-of-experts-em',
    schema: emSchema,
    config: { nExperts: k, tol, gateSteps, gateRidge, minVariance: options.minVariance ?? 1e-4 },
    init: () => {
      const start = options.start === undefined ? randomStart(m, targets, k, floor, key) : normalisedGate(options.start)
      return {
        ...start,
        logLikelihood: -Number.MAX_VALUE,
        change: Number.MAX_VALUE,
        resp: normaliseLogDensities(logJoint(a, targets, m.n, start)).resp,
      }
    },
    step: (s) => {
      const { resp, logNorm } = normaliseLogDensities(logJoint(a, targets, m.n, s))
      const ll = logNorm.reduce((acc, v) => acc + v, 0) / m.n
      const next = mStep(m, a, targets, resp, s.gate, floor, gateSteps, gateRidge)
      return {
        ...next,
        logLikelihood: ll,
        change: s.logLikelihood === -Number.MAX_VALUE ? Number.MAX_VALUE : Math.abs(ll - s.logLikelihood),
        resp,
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics:
        s.logLikelihood === -Number.MAX_VALUE
          ? {}
          : { logLikelihood: s.logLikelihood, ...(s.change === Number.MAX_VALUE ? {} : { change: s.change }) },
      tensors: {
        responsibilities: { role: 'state', value: s.resp.flat(), shape: [m.n, k] },
        gate: { role: 'param', value: s.gate.flat(), shape: [k, m.d + 1] },
        experts: { role: 'param', value: s.experts.flat(), shape: [k, m.d + 1] },
        variances: { role: 'param', value: s.variances, shape: [k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MoeState,
  }
}
