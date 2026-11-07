/**
 * Generalised linear models as a problem and a slot. `GlmProblem` is the penalised GLM objective, made from a design,
 * a response family with its link, prior weights, an offset and a smooth penalty (none, ridge, or a general
 * quadratic penalty matrix as an additive model needs); `glmSolvers` is the table of ways to minimise it: iteratively
 * reweighted least squares (Fisher scoring) with step halving, glmnet-style coordinate descent, and every generic
 * smooth solver of `smoothSolvers` (Newton with the observed information, BFGS, L-BFGS, conjugate gradient, gradient
 * descent and the stochastic `gradient` mode with its optimisers, schedules and batches) applied to the same objective.
 * A model that is a GLM states its objective once and takes the solver as a hyperparameter; the fitted coefficients
 * are the same whichever mode runs, so the modes are compared on one problem and the tuners see all their knobs (at
 * the scale the generic modes assume: their gradient tolerance is absolute, IRLS stops on the relative change of the
 * coefficients, so a design in very small units ends differently under the two).
 */

import type { Json } from '../base/index.ts'
import { MdmlError } from '../base/index.ts'
import type { DispersionFamily, Link } from '../dists/families/index.ts'
import { ddot, dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import { dict, int, real, type SpaceValue } from '../space/index.ts'
import {
  add,
  fromStorage,
  item,
  matmul,
  mul,
  primalOf,
  square,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type ModeSpec,
  type ModeTable,
  modes,
  type RunOptions,
  type RunStatus,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { matrixOf, vectorOf } from './_vector.ts'
import { type SmoothProblem, smoothSolvers, type VectorParams } from './solvers.ts'
import type { BatchObjective } from './stochastic.ts'

/**
 * The penalty of a GLM objective. The intercept (when the problem fits one) and the `unpenalised` coordinates carry no
 * `l2` weight; `matrix` is over every coordinate, so a design with its own intercept column puts zeros there.
 */
export interface GlmPenalty {
  /** The ridge weight: `(l2 / 2) sum_j beta_j^2` (default 0). */
  readonly l2?: number
  /** A quadratic penalty `beta' S beta / 2`, `S` a `[p, p]` matrix (a smoothing penalty of an additive model). */
  readonly matrix?: TensorLike
  /** Coordinates the `l2` weight skips. */
  readonly unpenalised?: readonly number[]
}

/**
 * A penalised GLM problem: minimise `D(beta) / (2 s) + l2 |beta|^2 / 2 + beta' S beta / 2` where `D` is
 * the deviance `sum_i w_i d(y_i, mu_i)` of the family at `mu = g^-1(x beta + b + offset)`, `b` the intercept (the last
 * coordinate, when fitted) and `s` the sum of the weights (scikit-learn's scaling of its GLMs and elastic net) or 1.
 * The parameter vector is the slopes then the intercept. The data are tensor-likes (migration, D-009: `x` was row-major
 * `ArrayLike<number>` with `n` and `d` beside it, which are now the shape of `x`; `y`, `weights`, `offset` and `start`
 * were `ArrayLike<number>`).
 */
export interface GlmProblem {
  /** The design, an `[n, d]` matrix. */
  readonly x: TensorLike
  /** The responses, a vector of `n`. */
  readonly y: TensorLike
  /** Prior weights, a vector of `n` (default 1). */
  readonly weights?: TensorLike
  /** A known offset added to the linear predictor, a vector of `n` (default none). */
  readonly offset?: TensorLike
  readonly family: DispersionFamily
  readonly link: Link
  /** Append an unpenalised intercept as the last coordinate (default `false`). */
  readonly fitIntercept?: boolean
  readonly penalty?: GlmPenalty
  /** Divide the deviance by the sum of the weights, as scikit-learn does (default `true`); `false` divides by one. */
  readonly normalise?: boolean
  /** Where to start (default zeros, with the intercept at the link of the family's starting mean). */
  readonly start?: TensorLike
}

/** The state of the IRLS and coordinate-descent modes: the iterate and what the logging convention reads. */
export interface GlmState {
  /** The coefficients: slopes then intercept. */
  readonly x: Float64Array
  /** The penalised objective at `x` (infinite before the first step of a cold start). */
  readonly value: number
  /** The gradient of the objective. */
  readonly grad: Float64Array
  /** The length of the last step. */
  readonly step: number
  /** The largest coefficient change relative to the coefficients' size. */
  readonly change: number
  /** The step halvings the last step needed. */
  readonly halvings: number
  /** Steps taken. */
  readonly iteration: number
}

/** What {@link glmProblemOf} prepares once: the augmented design and the penalty by coordinate. */
interface Prepared {
  readonly a: Float64Array
  readonly n: number
  readonly d: number
  readonly p: number
  readonly y: Float64Array
  readonly w: Float64Array
  readonly offset: Float64Array
  readonly scale: number
  readonly l2: Float64Array
  readonly matrix: Float64Array | undefined
  readonly start: Float64Array
  readonly warm: boolean
  readonly family: DispersionFamily
  readonly link: Link
}

const vec = (a: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(a), [a.length], 'float64')
const flat = (v: Parameters<typeof primalOf>[0]): Float64Array => Float64Array.from(toFlat(primalOf(v)))

const cache = new WeakMap<GlmProblem, Prepared>()

function prepared(problem: GlmProblem): Prepared {
  const hit = cache.get(problem)
  if (hit !== undefined) return hit
  const { family, link } = problem
  const { data: x, rows: n, cols: d } = matrixOf(problem.x, 'glm: x')
  const intercept = problem.fitIntercept === true
  const p = d + (intercept ? 1 : 0)
  const yIn = toVector(problem.y, 'glm: y')
  if (yIn.length !== n) throw new MdmlError('shape', `glm: y has ${yIn.length} entries for ${n} rows`)
  const a = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) a[i * p + j] = x[i * d + j] ?? 0
    if (intercept) a[i * p + d] = 1
  }
  const w =
    problem.weights === undefined
      ? new Float64Array(n).fill(1)
      : Float64Array.from(toVector(problem.weights, 'glm: weights'))
  if (w.length !== n) throw new MdmlError('shape', `glm: ${w.length} weights for ${n} rows`)
  const offset =
    problem.offset === undefined ? new Float64Array(n) : Float64Array.from(toVector(problem.offset, 'glm: offset'))
  if (offset.length !== n) throw new MdmlError('shape', `glm: ${offset.length} offsets for ${n} rows`)
  const penalty = problem.penalty ?? {}
  const l2 = penalty.l2 ?? 0
  if (!(l2 >= 0)) throw new MdmlError('value', 'glm: the penalty weight must be non-negative')
  const free = new Set(penalty.unpenalised ?? [])
  const l2v = Float64Array.from({ length: p }, (_, j) => (j < d && !free.has(j) ? l2 : 0))
  const matrix = penalty.matrix === undefined ? undefined : matrixOf(penalty.matrix, 'glm: penalty').data
  if (matrix !== undefined && matrix.length !== p * p)
    throw new MdmlError('shape', `glm: the penalty matrix has ${matrix.length} entries for [${p}, ${p}]`)
  let sw = 0
  for (const v of w) sw += v
  const scale = problem.normalise === false ? 1 : sw
  const y = Float64Array.from(yIn)
  let start: Float64Array
  if (problem.start !== undefined) {
    start = Float64Array.from(toVector(problem.start, 'glm: start'))
    if (start.length !== p) throw new MdmlError('shape', `glm: the start has ${start.length} entries for ${p}`)
  } else {
    start = new Float64Array(p)
    if (intercept && sw > 0) {
      const eta = flat(link.link(family.startingMu(vec(y))))
      let s = 0
      for (let i = 0; i < n; i++) s += (w[i] ?? 0) * ((eta[i] ?? 0) - (offset[i] ?? 0))
      start[d] = Number.isFinite(s) ? s / sw : 0
    }
  }
  const out: Prepared = {
    a,
    n,
    d,
    p,
    y,
    w,
    offset,
    scale,
    l2: l2v,
    matrix,
    start,
    warm: problem.start !== undefined,
    family,
    link,
  }
  cache.set(problem, out)
  return out
}

/** `A beta + offset`. */
function predictor(P: Prepared, beta: ArrayLike<number>): Float64Array {
  const eta = new Float64Array(P.n)
  for (let i = 0; i < P.n; i++) eta[i] = (P.offset[i] ?? 0) + ddot(P.a.subarray(i * P.p, (i + 1) * P.p), beta)
  return eta
}

/** The penalty at `beta`. */
function penaltyValue(P: Prepared, beta: ArrayLike<number>): number {
  let v = 0
  for (let j = 0; j < P.p; j++) {
    const b = beta[j] ?? 0
    v += ((P.l2[j] ?? 0) * b * b) / 2
  }
  if (P.matrix !== undefined) {
    for (let j = 0; j < P.p; j++) {
      let row = 0
      for (let k = 0; k < P.p; k++) row += (P.matrix[j * P.p + k] ?? 0) * (beta[k] ?? 0)
      v += ((beta[j] ?? 0) * row) / 2
    }
  }
  return v
}

function valueAt(P: Prepared, beta: ArrayLike<number>): number {
  const mu = P.link.inverse(vec(predictor(P, beta)))
  const deviance = item(P.family.deviance(vec(P.y), mu, vec(P.w)))
  return deviance / (2 * P.scale) + penaltyValue(P, beta)
}

/** The gradient of the objective. */
function gradientAt(P: Prepared, beta: ArrayLike<number>): Float64Array {
  const mu = P.link.inverse(vec(predictor(P, beta)))
  const m = flat(mu)
  const variance = flat(P.family.variance(mu))
  const slope = flat(P.link.derivative(mu))
  const u = new Float64Array(P.n)
  for (let i = 0; i < P.n; i++) {
    const v = ((P.w[i] ?? 0) * ((m[i] ?? 0) - (P.y[i] ?? 0))) / ((variance[i] ?? 1) * (slope[i] ?? 1))
    u[i] = Number.isFinite(v) ? v : 0
  }
  const g = new Float64Array(P.p)
  for (let i = 0; i < P.n; i++) {
    const ui = u[i] ?? 0
    if (ui === 0) continue
    for (let j = 0; j < P.p; j++) g[j] = (g[j] ?? 0) + ui * (P.a[i * P.p + j] ?? 0)
  }
  for (let j = 0; j < P.p; j++) {
    let t = (g[j] ?? 0) / P.scale + (P.l2[j] ?? 0) * (beta[j] ?? 0)
    if (P.matrix !== undefined) for (let k = 0; k < P.p; k++) t += (P.matrix[j * P.p + k] ?? 0) * (beta[k] ?? 0)
    g[j] = t
  }
  return g
}

/**
 * The penalised objective of a GLM problem at a coefficient vector. Takes the coefficients as a tensor-like
 * (migration, D-009: was `ArrayLike<number>`).
 *
 * @param problem - The problem.
 * @param beta - The coefficients, a vector: slopes then the intercept.
 * @returns The objective.
 * @throws MdmlError with code `shape` unless `beta` has rank 1, and `not-differentiable` for a traced one.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { glmValue } from 'mdml-engine/optim'
 * const problem = { x: [[0], [1]], y: [0, 1], family: families.binomial(), link: families.logit() }
 * glmValue(problem, [0]).toFixed(4) // => '0.6931'
 * ```
 */
export function glmValue(problem: GlmProblem, beta: TensorLike): number {
  return valueAt(prepared(problem), toVector(beta, 'glmValue'))
}

/**
 * The gradient of a GLM problem's objective at a coefficient vector. Takes the coefficients as a tensor-like and returns
 * a tensor (migration, D-009: was `ArrayLike<number>` in, `Float64Array` out).
 *
 * @param problem - The problem.
 * @param beta - The coefficients, a vector: slopes then the intercept.
 * @returns The gradient, a `[p]` tensor.
 * @throws MdmlError with code `shape` unless `beta` has rank 1, and `not-differentiable` for a traced one.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { glmGradient } from 'mdml-engine/optim'
 * import { toFlat } from 'mdml-engine/tensor'
 * const problem = { x: [[0], [1]], y: [0, 1], family: families.binomial(), link: families.logit() }
 * toFlat(glmGradient(problem, [0])).map((v) => v.toFixed(2)) // => ['-0.25']
 * ```
 */
export function glmGradient(problem: GlmProblem, beta: TensorLike): Tensor<'float64', [number]> {
  return vectorOf(gradientAt(prepared(problem), toVector(beta, 'glmGradient')))
}

/**
 * The problem as a smooth problem for the generic solvers: the objective as a differentiable function of the
 * coefficient vector (gradient and the observed-information Hessian by automatic differentiation), the start, and
 * the number of rows so that the stochastic mode can draw batches (a batch's deviance is scaled to estimate the whole
 * set's, the penalty is kept whole).
 *
 * @param problem - The problem.
 * @returns The smooth problem.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { glmSmoothProblem } from 'mdml-engine/optim'
 * const problem = { x: [[0], [1]], y: [0, 1], family: families.binomial(), link: families.logit() }
 * glmSmoothProblem(problem).batches?.n // => 2
 * ```
 */
export function glmSmoothProblem(problem: GlmProblem): SmoothProblem {
  const P = prepared(problem)
  const { n, p } = P
  const design = fromStorage(P.a, [n, p], 'float64')
  const y = vec(P.y)
  const w = vec(P.w)
  const offset = P.offset.some((v) => v !== 0) ? vec(P.offset) : undefined
  const l2 = P.l2.some((v) => v > 0) ? vec(P.l2) : undefined
  const S = P.matrix === undefined ? undefined : fromStorage(P.matrix, [p, p], 'float64')
  const objective: BatchObjective = (params, batch) => {
    const beta = params as Value
    let a = design
    let yb = y
    let wb = w
    let ob = offset
    let scale = 1 / (2 * P.scale)
    if (batch !== undefined) {
      const k = batch.length
      const ab = new Float64Array(k * p)
      const yv = new Float64Array(k)
      const wv = new Float64Array(k)
      const ov = new Float64Array(k)
      batch.forEach((i, r) => {
        ab.set(P.a.subarray(i * p, (i + 1) * p), r * p)
        yv[r] = P.y[i] ?? 0
        wv[r] = P.w[i] ?? 0
        ov[r] = P.offset[i] ?? 0
      })
      a = fromStorage(ab, [k, p], 'float64')
      yb = vec(yv)
      wb = vec(wv)
      ob = offset === undefined ? undefined : vec(ov)
      scale *= n / k
    }
    const lin = matmul(a, beta)
    const mu = P.link.inverse(ob === undefined ? lin : add(lin, ob))
    let f: Value = mul(P.family.deviance(yb, mu, wb), scale)
    if (l2 !== undefined) f = add(f, mul(sum(mul(l2, square(beta))), 0.5))
    if (S !== undefined) f = add(f, mul(sum(mul(beta, matmul(S, beta))), 0.5))
    return f
  }
  return { objective, x0: P.start, batches: { n } }
}

// ------------------------------------------------------------------------------------------ the weighted solves

/** The ridge-type normal equations of one IRLS step, `(A' W A + s (diag l2 + S)) beta = A' W z`, by Cholesky. */
function solveNormal(P: Prepared, ww: Float64Array, z: Float64Array): Float64Array {
  const { n, p } = P
  const g = new Float64Array(p * p)
  const rhs = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    const wi = ww[i] ?? 0
    if (wi === 0) continue
    const row = P.a.subarray(i * p, (i + 1) * p)
    const zi = z[i] ?? 0
    for (let j = 0; j < p; j++) {
      const rj = wi * (row[j] ?? 0)
      rhs[j] = (rhs[j] ?? 0) + rj * zi
      for (let k = 0; k <= j; k++) g[j * p + k] = (g[j * p + k] ?? 0) + rj * (row[k] ?? 0)
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) g[k * p + j] = g[j * p + k] ?? 0
    g[j * p + j] = (g[j * p + j] ?? 0) + P.scale * (P.l2[j] ?? 0)
  }
  if (P.matrix !== undefined) for (let k = 0; k < p * p; k++) g[k] = (g[k] ?? 0) + P.scale * (P.matrix[k] ?? 0)
  // A singular system (collinear columns, no penalty) is solved with the smallest Levenberg shift that makes it definite.
  let diag = 0
  for (let j = 0; j < p; j++) diag = Math.max(diag, g[j * p + j] ?? 0)
  let shift = 0
  for (let attempt = 0; attempt < 30; attempt++) {
    const a = Float64Array.from(g)
    for (let j = 0; j < p; j++) a[j * p + j] = (a[j * p + j] ?? 0) + shift
    const { factor, info } = dpotrf(a, p)
    if (info === 0) {
      const beta = dpotrs(factor, rhs, p)
      if (beta.every((v) => Number.isFinite(v))) return beta
    }
    shift = shift === 0 ? 1e-12 * Math.max(diag, 1) : shift * 10
  }
  throw new MdmlError('singular', 'glm: the weighted normal equations are singular')
}

// ------------------------------------------------------------------------------------------ the IRLS family

interface ReweightingKnobs {
  readonly tol: number
  readonly maxHalvings: number
}

function reweighting(problem: GlmProblem, name: string, knobs: ReweightingKnobs): Algorithm<GlmState> {
  const P = prepared(problem)
  const { n, p } = P
  const yT = vec(P.y)
  const wT = vec(P.w)
  const relative = (a: Float64Array, b: Float64Array): number => {
    let delta = 0
    let size = 0
    for (let j = 0; j < p; j++) {
      delta = Math.max(delta, Math.abs((a[j] ?? 0) - (b[j] ?? 0)))
      size = Math.max(size, Math.abs(a[j] ?? 0))
    }
    return delta / (1 + size)
  }
  const norm = (v: ArrayLike<number>): number => Math.sqrt(ddot(v, v))
  return {
    name,
    config: {
      family: P.family.name,
      link: P.link.name,
      tol: knobs.tol,
      maxHalvings: knobs.maxHalvings,
    } as Json,
    init: () =>
      P.warm
        ? {
            x: Float64Array.from(P.start),
            value: valueAt(P, P.start),
            grad: gradientAt(P, P.start),
            step: 0,
            change: Number.POSITIVE_INFINITY,
            halvings: 0,
            iteration: 0,
          }
        : {
            x: Float64Array.from(P.start),
            value: Number.POSITIVE_INFINITY,
            grad: new Float64Array(p),
            step: 0,
            change: Number.POSITIVE_INFINITY,
            halvings: 0,
            iteration: 0,
          },
    step: (s) => {
      const first = !Number.isFinite(s.value)
      const eta = first ? flat(P.link.link(P.family.startingMu(yT))) : predictor(P, s.x)
      const mu = P.link.inverse(vec(eta))
      const muFlat = flat(mu)
      const ww = flat(P.family.irlsWeights(mu, P.link, wT))
      for (let i = 0; i < n; i++) if (!Number.isFinite(ww[i])) ww[i] = 0
      const slope = flat(P.link.derivative(mu))
      const z = new Float64Array(n)
      for (let i = 0; i < n; i++)
        z[i] = (eta[i] ?? 0) - (P.offset[i] ?? 0) + ((P.y[i] ?? 0) - (muFlat[i] ?? 0)) * (slope[i] ?? 0)
      let x = solveNormal(P, ww, z)
      let value = valueAt(P, x)
      let halvings = 0
      while (!first && !(value <= s.value + 1e-12 * Math.abs(s.value)) && halvings < knobs.maxHalvings) {
        halvings++
        x = Float64Array.from(x, (v, j) => 0.5 * (v + (s.x[j] ?? 0)))
        value = valueAt(P, x)
      }
      const moved = Float64Array.from(x, (v, j) => v - (s.x[j] ?? 0))
      return {
        x,
        value,
        grad: gradientAt(P, x),
        step: norm(moved),
        change: first ? Number.POSITIVE_INFINITY : relative(x, s.x),
        halvings,
        iteration: s.iteration + 1,
      }
    },
    done: (s) => (s.change <= knobs.tol ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GlmState,
  }
}

const vector = (s: GlmState): VectorParams => ({ x: Array.from(s.x) })

const tol = real(1e-14, 1e-3, { log: true, default: 1e-10, label: 'relative coefficient change' })
const maxHalvings = int(0, 60, { default: 30, label: 'step halvings allowed' })

const irls: ModeSpec<GlmProblem, GlmState, VectorParams> = {
  description:
    'iteratively reweighted least squares (Fisher scoring): each step solves the penalised weighted least-squares problem at the working weights and response exactly by Cholesky, halving the step while the objective rises',
  space: dict({ tol, maxHalvings }),
  make: (p, k) =>
    reweighting(p, 'glm-irls', {
      tol: (k['tol'] as number | undefined) ?? 1e-10,
      maxHalvings: (k['maxHalvings'] as number | undefined) ?? 30,
    }),
  params: vector,
}

/** A generic smooth solver of `smoothSolvers`, named, applied to the GLM objective. */
function throughSmooth(name: string): ModeSpec<GlmProblem, unknown, VectorParams> {
  const base = smoothSolvers.modeOf({ kind: name })
  return {
    description: `${base.description ?? name}, on the GLM objective`,
    space: base.space,
    make: (p, k, key) => {
      return base.make(glmSmoothProblem(p), k, key)
    },
    params: base.params,
  }
}

/**
 * The solvers of a smooth penalised GLM as one slot (no penalty, ridge, or a quadratic penalty matrix; the non-smooth
 * penalties, lasso and elastic net, are particular algorithms of `linear`). `irls` (Fisher scoring with step halving),
 * `newton` (the observed information), `bfgs`, `lbfgs`, `conjugateGradient`, `gradientDescent` and the stochastic `gradient` mode (an optimiser, a schedule and a batch size) all minimise the
 * same objective of a {@link GlmProblem} and yield the same fitted vector.
 */
export const glmSolvers: ModeTable<GlmProblem, VectorParams> = modes(
  {
    irls,
    newton: throughSmooth('newton'),
    bfgs: throughSmooth('bfgs'),
    lbfgs: throughSmooth('lbfgs'),
    conjugateGradient: throughSmooth('conjugateGradient'),
    gradientDescent: throughSmooth('gradientDescent'),
    gradient: throughSmooth('gradient'),
  },
  { label: 'solver', description: 'how the penalised GLM objective is minimised', default: 'irls' },
)

/** A fitted GLM problem. */
export interface GlmFit {
  /** The coefficient vector: slopes then the intercept. */
  readonly x: Float64Array
  /** The slopes. */
  readonly coef: Float64Array
  /** The intercept (zero when the problem fits none). */
  readonly intercept: number
  /** The penalised objective at the solution. */
  readonly objective: number
  /** The objective after each step. */
  readonly history: readonly number[]
  /** The iterate after each step. */
  readonly path: readonly Float64Array[]
  readonly steps: number
  readonly converged: boolean
  /** How the run ended in the one vocabulary: `converged`, or `completed` when the iteration cap was reached. */
  readonly status: RunStatus
  /** The mode that ran. */
  readonly solver: string
}

/**
 * Minimise a GLM problem with the configured solver and report the fit.
 *
 * @param problem - The problem.
 * @param config - A value of `glmSolvers.space` (default IRLS).
 * @param options - Steps (default 100 for `irls` and `newton`, else 1000), the key and run options.
 * @returns The fit.
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { fitGlm } from 'mdml-engine/optim'
 * const problem = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 1, 0, 1, 1], family: families.binomial(), link: families.logit(), fitIntercept: true }
 * const irls = fitGlm(problem, { kind: 'irls' })
 * const lbfgs = fitGlm(problem, { kind: 'lbfgs', gradTol: 1e-10 })
 * Math.abs(irls.coef[0] - lbfgs.coef[0]) < 1e-4 // => true
 * ```
 */
export function fitGlm(
  problem: GlmProblem,
  config: SpaceValue = { kind: 'irls' },
  options: Partial<Omit<RunOptions, 'start'>> & { readonly key?: Key } = {},
): GlmFit {
  const P = prepared(problem)
  const kind = glmSolvers.modeOf(config).name
  const history: number[] = []
  const path: Float64Array[] = []
  const key = options.key ?? makeKey(options.seed ?? 0)
  const inner = glmSolvers.make(problem, config, key)
  const tracked: Algorithm<unknown> = {
    ...inner,
    step: (state, ctx) => {
      const next = inner.step(state, ctx)
      const r = next as { readonly x: Float64Array; readonly value: number }
      history.push(r.value)
      path.push(Float64Array.from(r.x))
      return next
    },
  }
  const result = runAlgorithm(tracked, {
    ...options,
    key,
    maxSteps: options.maxSteps ?? (['irls', 'newton'].includes(kind) ? 100 : 1000),
  })
  const x = Float64Array.from((result.state as { readonly x: Float64Array }).x)
  return {
    x,
    coef: x.slice(0, P.d),
    intercept: P.p > P.d ? (x[P.d] ?? 0) : 0,
    objective: valueAt(P, x),
    history,
    path,
    steps: result.steps,
    converged: result.status === 'converged',
    status: result.status,
    solver: kind,
  }
}
