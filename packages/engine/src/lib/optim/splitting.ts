/**
 * Operator splitting methods for problems with a smooth part and a nonsmooth part known through its proximal operator:
 * proximal gradient descent with backtracking (ISTA), its accelerated form FISTA with optional restarts, and the
 * Chambolle-Pock primal-dual hybrid gradient method for min F(K x) + G(x), all as traced algorithms.
 */

import { WhiteboxError } from '../base/index.ts'
import { daxpy, ddot, dnrm2 } from '../numerics/linalg/index.ts'
import { asOperator, type CsrMatrix, type LinearOperator } from '../numerics/sparse/index.ts'
import { key, normal } from '../random/index.ts'
import { asTensor, fromStorage, item, primalOf, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './minimize/_shared.ts'
import {
  flatObjective,
  infNorm,
  type MinimizeResult,
  type Objective,
  type VectorMethodOptions,
} from './minimize/objective.ts'
import type { Prox } from './proximal.ts'

/** Options of {@link proximalGradientAlgorithm}, {@link ista} and {@link fista}. */
export interface ProximalGradientOptions extends VectorMethodOptions {
  /** A fixed step 1 / L; without it the step is found by backtracking from `initialStep`. */
  readonly stepSize?: number
  /** The first step tried by backtracking (default 1). */
  readonly initialStep?: number
  /** Backtracking factor (default 0.5). */
  readonly shrink?: number
  /** The nonsmooth term's value, so the traced objective is f + g (otherwise f alone is recorded). */
  readonly g?: (x: Value) => Value
  /** Stop once the relative change |x+ - x| / max(1, |x|) falls to this (default 1e-12). */
  readonly tol?: number
  /** Nesterov acceleration (FISTA). */
  readonly accelerate?: boolean
  /** Adaptive restart of the momentum (O'Donoghue and Candes 2015): on objective increase, or on gradient test. */
  readonly restart?: 'none' | 'function' | 'gradient'
}

/** State of the proximal gradient methods. */
export interface ProximalGradientState extends VectorState {
  /** The extrapolated point, its smooth value and gradient (equal to x without acceleration). */
  readonly y: Float64Array
  readonly fy: number
  readonly gy: Float64Array
  /** The smooth part at x. */
  readonly fx: number
  /** The momentum parameter t_k. */
  readonly tk: number
  readonly restarts: number
  readonly trials: number
}

/**
 * Proximal gradient descent for f(x) + g(x), f smooth (differentiated by autodiff) and g given by its proximal
 * operator: x+ = prox_{t g}(y - t grad f(y)), with backtracking on the quadratic upper bound (Beck and Teboulle 2009)
 * and optional Nesterov acceleration (FISTA) with adaptive restart. The recorded gradient is the gradient mapping
 * (y - x+) / t, whose norm measures stationarity.
 *
 * @param f - The smooth part.
 * @param prox - The proximal operator of the nonsmooth part.
 * @param x0 - The starting point (a tensor of any shape; the proximal operator sees that shape).
 * @param options - Step, backtracking, acceleration, restart, the value of g and tolerances.
 * @returns The algorithm.
 * @concept proximal-gradient
 * @example
 * ```ts
 * import { proximalGradientAlgorithm, proxL1 } from 'whitebox-engine/optim'
 * import { square, sub, sum, type Value } from 'whitebox-engine/tensor'
 * proximalGradientAlgorithm((x) => sum(square(sub(x as Value, 2))), proxL1, [0]).name // => 'ista'
 * ```
 */
export function proximalGradientAlgorithm(
  f: Objective,
  prox: Prox,
  x0: TensorLike,
  options: ProximalGradientOptions = {},
): Algorithm<ProximalGradientState> {
  const obj = flatObjective(f, x0)
  const accelerate = options.accelerate ?? false
  const restart = options.restart ?? 'none'
  const backtracking = options.stepSize === undefined
  const shrink = options.shrink ?? 0.5
  const gradTol = options.gradTol ?? 0
  const tol = options.tol ?? 1e-12
  const proxFlat = (v: Float64Array, t: number): Float64Array =>
    Float64Array.from(toFlat(primalOf(prox(obj.unravel(v) as Value, t))))
  const gOf = (x: Float64Array): number =>
    options.g === undefined ? 0 : item(primalOf(options.g(obj.unravel(x) as Value)))
  return {
    name: accelerate ? 'fista' : 'ista',
    config: configOf({ ...options, accelerate, restart, g: undefined }),
    schema: vectorSchema({
      metrics: {
        smooth: { description: 'smooth part f(x)' },
        momentum: { description: 'momentum parameter t_k' },
        trials: { description: 'backtracking trials in the step' },
      },
    }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      const { value, grad } = obj.valueAndGrad(x)
      return {
        x,
        value: value + gOf(x),
        grad: new Float64Array(x.length).fill(Infinity),
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: options.stepSize ?? options.initialStep ?? 1,
        y: x,
        fy: value,
        gy: grad,
        fx: value,
        tk: 1,
        restarts: 0,
        trials: 0,
      }
    },
    step: (s) => {
      let t = s.step
      let nfev = s.nfev
      let trials = 0
      let xp: Float64Array
      let fxp: number | undefined
      for (;;) {
        trials += 1
        xp = proxFlat(daxpy(-t, s.gy, s.y), t)
        if (!backtracking) break
        const d = daxpy(-1, s.y, xp)
        fxp = obj.value(xp)
        nfev += 1
        if (fxp <= s.fy + ddot(s.gy, d) + ddot(d, d) / (2 * t) + 1e-12 * Math.abs(s.fy) || t < 1e-20) break
        t *= shrink
      }
      const mapping = Float64Array.from(s.y, (v, i) => (v - (xp[i] ?? 0)) / t)
      let ngev = s.ngev
      let tk = 1
      let y = xp
      let fy: number
      let gy: Float64Array
      let fx: number
      let restarts = s.restarts
      if (accelerate) {
        if (fxp === undefined) {
          fxp = obj.value(xp)
          nfev += 1
        }
        fx = fxp
        const value = fx + gOf(xp)
        const dx = daxpy(-1, s.x, xp)
        const reset =
          (restart === 'function' && value > s.value) || (restart === 'gradient' && ddot(daxpy(-1, xp, s.y), dx) > 0)
        if (reset) restarts += 1
        tk = reset ? 1 : (1 + Math.sqrt(1 + 4 * s.tk * s.tk)) / 2
        y = reset ? xp : daxpy((s.tk - 1) / tk, dx, xp)
        const r = obj.valueAndGrad(y)
        fy = r.value
        gy = r.grad
      } else {
        const r = obj.valueAndGrad(xp)
        fy = r.value
        gy = r.grad
        fx = r.value
      }
      nfev += 1
      ngev += 1
      const change = dnrm2(daxpy(-1, s.x, xp)) / Math.max(1, dnrm2(s.x))
      const value = fx + gOf(xp)
      const gm = infNorm(mapping)
      const stop =
        gm <= gradTol
          ? ({ status: 'converged', message: 'gradient mapping below tolerance' } as const)
          : change <= tol
            ? ({ status: 'converged', message: 'relative change below tolerance' } as const)
            : undefined
      return {
        x: xp,
        value,
        grad: mapping,
        nfev,
        ngev,
        nhev: 0,
        step: t,
        y,
        fy,
        gy,
        fx,
        tk,
        restarts,
        trials,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { smooth: s.fx, momentum: s.tk, trials: s.trials } }),
  }
}

/**
 * ISTA, the proximal gradient method (iterative shrinkage-thresholding when g is the L1 norm), with backtracking.
 *
 * @param f - The smooth part.
 * @param prox - The proximal operator of the nonsmooth part.
 * @param x0 - The starting point.
 * @param options - Step, backtracking, the value of g, tolerances, iteration cap (default 10000) and run options.
 * @returns The result; `grad` is the gradient mapping at the solution.
 * @concept proximal-gradient
 * @example
 * ```ts
 * import { ista, proxL1 } from 'whitebox-engine/optim'
 * import { square, sub, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * // argmin (x - 2)^2 + |x|: the soft threshold of 2 at 1/2.
 * toArray(ista((x) => sum(square(sub(x as Value, 2))), proxL1, [0], { stepSize: 0.5 }).x as Value) // => [1.5]
 * ```
 */
export function ista(f: Objective, prox: Prox, x0: TensorLike, options: ProximalGradientOptions = {}): MinimizeResult {
  return runVector(
    proximalGradientAlgorithm(f, prox, x0, { ...options, accelerate: false }),
    flatObjective(f, x0),
    options,
    10000,
  )
}

/**
 * FISTA (Beck and Teboulle 2009): proximal gradient with Nesterov momentum, O(1/k^2), with backtracking and optional
 * adaptive restart (`restart: 'function'` or `'gradient'`).
 *
 * @param f - The smooth part.
 * @param prox - The proximal operator of the nonsmooth part.
 * @param x0 - The starting point.
 * @param options - Step, backtracking, restart, the value of g, tolerances, iteration cap (default 10000).
 * @returns The result.
 * @concept proximal-gradient
 * @example
 * ```ts
 * import { fista, proxL1 } from 'whitebox-engine/optim'
 * import { square, sub, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * const r = fista((x) => sum(square(sub(x as Value, 2))), proxL1, [0], { restart: 'function' })
 * toArray(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [1.5]
 * ```
 */
export function fista(f: Objective, prox: Prox, x0: TensorLike, options: ProximalGradientOptions = {}): MinimizeResult {
  return runVector(
    proximalGradientAlgorithm(f, prox, x0, { ...options, accelerate: true }),
    flatObjective(f, x0),
    options,
    10000,
  )
}

/** Options of the primal-dual method {@link chambollePockAlgorithm}. */
export interface ChambollePockOptions extends VectorMethodOptions {
  /** Primal and dual steps with tau sigma |K|^2 < 1 (default both 0.95 / |K|, |K| by power iteration). */
  readonly tau?: number
  readonly sigma?: number
  /** Extrapolation (default 1). */
  readonly theta?: number
  /** The starting dual point (default 0). */
  readonly y0?: ArrayLike<number>
  /** Stop once the primal and dual residuals fall to this (default 1e-10). */
  readonly tol?: number
  /** The objective F(K x) + G(x), for the trace. */
  readonly objective?: (x: Value) => Value
}

/** The problem min_x F(K x) + G(x) for {@link chambollePockAlgorithm}. */
export interface PrimalDualProblem {
  /** The linear operator K (dense matrix, CSR or operator with `rmatvec`). */
  readonly K: CsrMatrix | TensorLike | LinearOperator
  /** The proximal operator of G. */
  readonly proxG: Prox
  /** The proximal operator of F (its conjugate's is derived by the Moreau decomposition), or that of F* directly. */
  readonly proxF?: Prox
  readonly proxFConjugate?: Prox
}

/** State of {@link chambollePockAlgorithm}. */
export interface ChambollePockState extends VectorState {
  readonly y: Float64Array
  readonly xBar: Float64Array
  readonly primalResidual: number
  readonly dualResidual: number
}

const asVec = (x: ArrayLike<number>): Value => fromStorage(Float64Array.from(x), [x.length], 'float64')
const flat = (v: Value): Float64Array => Float64Array.from(toFlat(primalOf(v)))

/** |K| by power iteration on K^T K. */
function operatorNorm(op: LinearOperator, iterations = 50): number {
  const n = op.shape[1]
  // A fixed random start: a constant vector can lie in the null space (difference operators).
  const start = Float64Array.from(normal(key(0), [n]).data)
  const s0 = dnrm2(start)
  let v: Float64Array = Float64Array.from(start, (e) => e / s0)
  let lambda = 0
  for (let i = 0; i < iterations; i++) {
    const w = flat((op.rmatvec as (x: Value) => Value)(op.matvec(asVec(v))))
    lambda = dnrm2(w)
    if (lambda === 0) return 0
    v = Float64Array.from(w, (e) => e / lambda)
  }
  return Math.sqrt(lambda)
}

/**
 * The primal-dual hybrid gradient method of Chambolle and Pock (2011) for min_x F(K x) + G(x):
 * y+ = prox_{sigma F*}(y + sigma K xBar), x+ = prox_{tau G}(x - tau K^T y+), xBar = x+ + theta (x+ - x).
 * Records the primal and dual residuals (Goldstein et al.) and, when given, the objective.
 *
 * @param problem - K and the proximal operators of G and F (or F*).
 * @param x0 - The starting primal point, a vector.
 * @param options - Steps, extrapolation, starting dual point, tolerance and the objective for the trace.
 * @returns The algorithm.
 * @concept primal-dual
 * @example
 * ```ts
 * import { chambollePockAlgorithm, proxL1, proxSquaredL2 } from 'whitebox-engine/optim'
 * chambollePockAlgorithm({ K: [[1, 0], [0, 1]], proxG: proxSquaredL2, proxF: proxL1 }, [1, 1]).name // => 'chambollePock'
 * ```
 */
export function chambollePockAlgorithm(
  problem: PrimalDualProblem,
  x0: TensorLike,
  options: ChambollePockOptions = {},
): Algorithm<ChambollePockState> {
  const op = asOperator(problem.K)
  const [m, n] = op.shape
  const rmatvec = op.rmatvec
  if (rmatvec === undefined) throw new WhiteboxError('value', 'chambollePock: the operator needs rmatvec (K^T)')
  const K = (x: Float64Array): Float64Array => flat(op.matvec(asVec(x)))
  const Kt = (y: Float64Array): Float64Array => flat(rmatvec(asVec(y)))
  const proxF = problem.proxF
  const proxFConj: Prox =
    problem.proxFConjugate ??
    ((v, s) => {
      if (proxF === undefined) throw new WhiteboxError('value', 'chambollePock: give proxF or proxFConjugate')
      const vv = flat(v)
      const inner = flat(proxF(asVec(vv.map((e) => e / s)), 1 / s))
      return asVec(vv.map((e, i) => e - s * (inner[i] ?? 0)))
    })
  const normK = options.tau === undefined || options.sigma === undefined ? operatorNorm(op) : 0
  const tau = options.tau ?? 0.95 / Math.max(normK, 1e-300)
  const sigma = options.sigma ?? 0.95 / Math.max(normK, 1e-300)
  const theta = options.theta ?? 1
  const tol = options.tol ?? 1e-10
  const objective = (x: Float64Array): number =>
    options.objective === undefined ? Number.NaN : item(primalOf(options.objective(asVec(x))))
  return {
    name: 'chambollePock',
    config: configOf({ tau, sigma, theta, tol, m, n }),
    schema: vectorSchema({
      metrics: {
        primalResidual: { description: 'primal residual |(x - x+) / tau - K^T (y - y+)|' },
        dualResidual: { description: 'dual residual |(y - y+) / sigma - K (x - x+)|' },
      },
      tensors: { y: { role: 'state', description: 'dual variable' } },
    }),
    init: () => {
      const x = flat(asTensor(x0))
      const y = options.y0 === undefined ? new Float64Array(m) : Float64Array.from(options.y0)
      return {
        x,
        value: objective(x),
        grad: new Float64Array(0),
        nfev: 0,
        ngev: 0,
        nhev: 0,
        step: tau,
        y,
        xBar: x,
        primalResidual: Infinity,
        dualResidual: Infinity,
      }
    },
    step: (s) => {
      const y = flat(proxFConj(asVec(daxpy(sigma, K(s.xBar), s.y)), sigma))
      const x = flat(problem.proxG(asVec(daxpy(-tau, Kt(y), s.x)), tau))
      const dx = daxpy(-1, x, s.x)
      const dy = daxpy(-1, y, s.y)
      const primalResidual = dnrm2(
        daxpy(
          -1,
          Kt(dy),
          Float64Array.from(dx, (e) => e / tau),
        ),
      )
      const dualResidual = dnrm2(
        daxpy(
          -1,
          K(dx),
          Float64Array.from(dy, (e) => e / sigma),
        ),
      )
      const xBar = Float64Array.from(x, (e, i) => e + theta * (e - (s.x[i] ?? 0)))
      const stop =
        Math.max(primalResidual, dualResidual) <= tol
          ? ({ status: 'converged', message: 'primal and dual residuals below tolerance' } as const)
          : undefined
      return {
        x,
        value: objective(x),
        grad: new Float64Array(0),
        nfev: s.nfev,
        ngev: 0,
        nhev: 0,
        step: tau,
        y,
        xBar,
        primalResidual,
        dualResidual,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: (s) => (s.stop !== undefined ? s.stop.status : undefined),
    observe: (s) =>
      observeVector(s, {
        metrics: { primalResidual: s.primalResidual, dualResidual: s.dualResidual },
        tensors: { y: { role: 'state', value: s.y } },
      }),
  }
}

/** Result of {@link chambollePock}. */
export interface PrimalDualResult extends MinimizeResult {
  /** The dual solution. */
  readonly y: Value
  readonly primalResidual: number
  readonly dualResidual: number
}

/**
 * Solve min_x F(K x) + G(x) by the Chambolle-Pock primal-dual method.
 *
 * @param problem - K and the proximal operators.
 * @param x0 - The starting point, a vector.
 * @param options - Steps, extrapolation, tolerance, objective, iteration cap (default 10000) and run options.
 * @returns The primal and dual solutions with residuals; `fun` is the objective when given.
 * @concept primal-dual
 * @example
 * ```ts
 * import { chambollePock, proxL1, proxSquaredL2 } from 'whitebox-engine/optim'
 * import { add, sub, toArray, type Value } from 'whitebox-engine/tensor'
 * // min |x|_2^2 / 2 + |x - 3|_1 written as F(Kx) with K = I and F = |. - 3|_1.
 * const r = chambollePock(
 *   { K: [[1, 0], [0, 1]], proxG: proxSquaredL2, proxF: (v, t) => add(proxL1(sub(v, 3), t), 3) },
 *   [0, 0],
 * )
 * toArray(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [1, 1]
 * r.converged // => true
 * ```
 */
export function chambollePock(
  problem: PrimalDualProblem,
  x0: TensorLike,
  options: ChambollePockOptions = {},
): PrimalDualResult {
  const { maxIterations, gradTol: _g, tau, sigma, theta, y0, tol, objective, ...run } = options
  const alg = chambollePockAlgorithm(problem, x0, options)
  const result = runAlgorithm(alg, { ...run, maxSteps: maxIterations ?? 10000 })
  const s = result.state
  const shape = asTensor(x0).shape
  return {
    x: fromStorage(s.x, shape, 'float64'),
    fun: s.value,
    gradNorm: Number.NaN,
    steps: result.steps,
    nfev: s.nfev,
    ngev: 0,
    nhev: 0,
    status: result.status,
    converged: result.status === 'converged',
    message: s.stop?.message ?? 'maximum number of iterations reached',
    y: asVec(s.y),
    primalResidual: s.primalResidual,
    dualResidual: s.dualResidual,
  }
}
