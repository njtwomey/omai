import { WhiteboxError } from '../../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../../numerics/differentiate/index.ts'
import type { TensorLike, Value } from '../../tensor/index.ts'
import { floatState, numbersOf, rms } from './_vec.ts'
import type { Stepper, VectorField } from './problem.ts'
import { type OdeMethodName, solveIvp } from './solve-ivp.ts'

/** The coefficients a, b, c of any Runge-Kutta method (explicit, diagonally implicit or fully implicit). */
export interface RungeKuttaCoefficients {
  readonly a: readonly (readonly number[])[]
  readonly b: readonly number[]
  readonly c: readonly number[]
}

/** A rooted tree as the list of its subtrees (the empty list is the single node). */
export type RootedTree = readonly RootedTree[]

function treeOrder(t: RootedTree): number {
  return 1 + t.reduce((s, u) => s + treeOrder(u), 0)
}

function treeKey(t: RootedTree): string {
  return `[${t.map(treeKey).sort().join(',')}]`
}

/**
 * All rooted trees with at most `maxOrder` nodes (1, 1, 2, 4, 9, 20, 48, 115 of orders 1 to 8): the index set of the
 * Runge-Kutta order conditions (Butcher).
 *
 * @param maxOrder - The largest order (1 to 10).
 * @returns The trees, grouped by order.
 * @throws WhiteboxError with code `value` outside 1..10.
 * @concept butcher-trees
 * @example
 * ```ts
 * import { rootedTrees } from 'whitebox-engine/dynamics/ode'
 * rootedTrees(5).map((ts) => ts.length) // => [1, 1, 2, 4, 9]
 * ```
 */
export function rootedTrees(maxOrder: number): RootedTree[][] {
  if (!Number.isInteger(maxOrder) || maxOrder < 1 || maxOrder > 10) {
    throw new WhiteboxError('value', 'rootedTrees: the order is 1 to 10')
  }
  const byOrder: RootedTree[][] = [[], [[]]]
  for (let n = 2; n <= maxOrder; n++) {
    // A tree of order n is a root with a multiset of subtrees whose orders sum to n - 1.
    const seen = new Set<string>()
    const out: RootedTree[] = []
    const build = (remaining: number, maxChild: number, children: RootedTree[]): void => {
      if (remaining === 0) {
        const key = treeKey(children)
        if (!seen.has(key)) {
          seen.add(key)
          out.push(children)
        }
        return
      }
      for (let k = Math.min(remaining, maxChild); k >= 1; k--) {
        for (const sub of byOrder[k] ?? []) build(remaining - k, k, [...children, sub])
      }
    }
    build(n - 1, n - 1, [])
    byOrder.push(out)
  }
  return byOrder.slice(1)
}

function stageWeights(m: RungeKuttaCoefficients, t: RootedTree): number[] {
  // Phi_i(t) = prod over subtrees u of sum_j a_ij Phi_j(u).
  const s = m.b.length
  const out = new Array<number>(s).fill(1)
  for (const u of t) {
    const inner = stageWeights(m, u)
    for (let i = 0; i < s; i++) {
      let acc = 0
      for (let j = 0; j < s; j++) acc += (m.a[i]?.[j] ?? 0) * (inner[j] ?? 0)
      out[i] = (out[i] ?? 0) * acc
    }
  }
  return out
}

function density(t: RootedTree): number {
  return treeOrder(t) * t.reduce((p, u) => p * density(u), 1)
}

/** The outcome of {@link orderConditions}. */
export interface OrderReport {
  /** The highest order p with every condition of order <= p satisfied. */
  readonly order: number
  /** The largest residual |b^T Phi(t) - 1/gamma(t)| at each order. */
  readonly residuals: readonly number[]
}

/**
 * Check the Runge-Kutta order conditions b^T Phi(t) = 1/gamma(t) over every rooted tree up to `maxOrder`.
 *
 * @param method - The coefficients.
 * @param maxOrder - The highest order checked (default 8).
 * @param tol - The tolerance on each residual (default 1e-10).
 * @returns The order reached and the largest residual per order.
 * @concept butcher-trees
 * @example
 * ```ts
 * import { orderConditions, RK4 } from 'whitebox-engine/dynamics/ode'
 * orderConditions(RK4, 6).order // => 4
 * ```
 */
export function orderConditions(method: RungeKuttaCoefficients, maxOrder = 8, tol = 1e-10): OrderReport {
  const trees = rootedTrees(maxOrder)
  const residuals = trees.map((ts) =>
    ts.reduce((worst, t) => {
      const phi = stageWeights(method, t)
      let v = 0
      for (let i = 0; i < method.b.length; i++) v += (method.b[i] ?? 0) * (phi[i] ?? 0)
      return Math.max(worst, Math.abs(v - 1 / density(t)))
    }, 0),
  )
  let order = 0
  while (order < residuals.length && (residuals[order] ?? 1) <= tol) order++
  return { order, residuals }
}

/** The result of a convergence study. */
export interface ConvergenceStudy {
  readonly steps: readonly number[]
  readonly errors: readonly number[]
  /** The least-squares slope of log error against log step. */
  readonly order: number
  /** The order between consecutive steps. */
  readonly localOrders: readonly number[]
}

/**
 * The observed order of a discretisation: errors at several step sizes, fitted as error ~ C h^p.
 *
 * @param error - The error for a step h.
 * @param steps - The step sizes (at least two, positive).
 * @returns The errors, fitted order and consecutive local orders.
 * @throws WhiteboxError with code `value` with fewer than two steps.
 * @concept order-of-convergence
 * @example
 * ```ts
 * import { convergenceOrder } from 'whitebox-engine/dynamics/ode'
 * Math.round(convergenceOrder((h) => 3 * h ** 2, [0.1, 0.05, 0.025]).order) // => 2
 * ```
 */
export function convergenceOrder(error: (h: number) => number, steps: readonly number[]): ConvergenceStudy {
  if (steps.length < 2 || steps.some((h) => !(h > 0))) {
    throw new WhiteboxError('value', 'convergenceOrder: give at least two positive steps')
  }
  const errors = steps.map((h) => error(h))
  const xs = steps.map(Math.log)
  const ys = errors.map((e) => Math.log(e))
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length
  const my = ys.reduce((a, b) => a + b, 0) / ys.length
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < xs.length; i++) {
    sxy += ((xs[i] ?? 0) - mx) * ((ys[i] ?? 0) - my)
    sxx += ((xs[i] ?? 0) - mx) ** 2
  }
  const localOrders = steps
    .slice(1)
    .map((h, i) => Math.log((errors[i] ?? 1) / (errors[i + 1] ?? 1)) / Math.log((steps[i] ?? 1) / h))
  return { steps: [...steps], errors, order: sxy / sxx, localOrders }
}

/**
 * The observed order of a fixed-step stepper on a problem with a known solution: the RMS error at the end of
 * [t0, t1] for each step size.
 *
 * @param stepper - Builds the stepper for a step size.
 * @param f - The vector field.
 * @param tspan - The interval.
 * @param y0 - The initial state.
 * @param exact - The exact final state.
 * @param steps - The step sizes.
 * @returns The convergence study.
 * @concept order-of-convergence
 * @example
 * ```ts
 * import { rk4, stepperOrder } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * Math.round(stepperOrder((dt) => rk4({ dt }), (_t, y) => neg(y), [0, 1], [1], [Math.exp(-1)], [0.2, 0.1, 0.05]).order) // => 4
 * ```
 */
export function stepperOrder(
  stepper: (dt: number) => Stepper,
  f: VectorField,
  tspan: readonly [number, number],
  y0: TensorLike,
  exact: TensorLike,
  steps: readonly number[],
): ConvergenceStudy {
  const want = numbersOf(floatState(exact))
  return convergenceOrder((dt) => {
    const sol = solveIvp(f, tspan, y0, { method: stepper(dt), maxSteps: 10_000_000 })
    const got = numbersOf(sol.ys[sol.ys.length - 1] as Value)
    return rms(Float64Array.from(got, (v, i) => v - (want[i] ?? 0)))
  }, steps)
}

function complexSolve(m: ComplexNumber[][], rhs: ComplexNumber[]): ComplexNumber[] {
  const n = rhs.length
  const a = m.map((row) => [...row])
  const b = [...rhs]
  for (let k = 0; k < n; k++) {
    let p = k
    for (let i = k + 1; i < n; i++) {
      const ai = a[i]?.[k] ?? C.of(0)
      const ap = a[p]?.[k] ?? C.of(0)
      if (Math.hypot(ai.re, ai.im) > Math.hypot(ap.re, ap.im)) p = i
    }
    ;[a[k], a[p]] = [a[p] as ComplexNumber[], a[k] as ComplexNumber[]]
    ;[b[k], b[p]] = [b[p] as ComplexNumber, b[k] as ComplexNumber]
    const pivot = a[k]?.[k] ?? C.of(0)
    for (let i = k + 1; i < n; i++) {
      const factor = C.div(a[i]?.[k] ?? C.of(0), pivot)
      for (let j = k; j < n; j++)
        (a[i] as ComplexNumber[])[j] = C.sub(a[i]?.[j] ?? C.of(0), C.mul(factor, a[k]?.[j] ?? C.of(0)))
      b[i] = C.sub(b[i] ?? C.of(0), C.mul(factor, b[k] ?? C.of(0)))
    }
  }
  const x = new Array<ComplexNumber>(n).fill(C.of(0))
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i] ?? C.of(0)
    for (let j = i + 1; j < n; j++) s = C.sub(s, C.mul(a[i]?.[j] ?? C.of(0), x[j] ?? C.of(0)))
    x[i] = C.div(s, a[i]?.[i] ?? C.of(1))
  }
  return x
}

/**
 * The stability function of a Runge-Kutta method, R(z) = 1 + z b^T (I - z A)^-1 1: one step of y' = lambda y
 * multiplies y by R(h lambda).
 *
 * @param method - The coefficients.
 * @param z - h lambda.
 * @returns R(z).
 * @concept a-stability
 * @example
 * ```ts
 * import { stabilityFunction, EULER } from 'whitebox-engine/dynamics/ode'
 * stabilityFunction(EULER, { re: -1, im: 0 }) // => { re: 0, im: 0 }
 * ```
 */
export function stabilityFunction(method: RungeKuttaCoefficients, z: ComplexNumber): ComplexNumber {
  const s = method.b.length
  const m = Array.from({ length: s }, (_, i) =>
    Array.from({ length: s }, (_, j) => C.sub(C.of(i === j ? 1 : 0), C.mul(z, method.a[i]?.[j] ?? 0))),
  )
  const x = complexSolve(
    m,
    Array.from({ length: s }, () => C.of(1)),
  )
  let acc = C.of(0)
  for (let i = 0; i < s; i++) acc = C.add(acc, C.mul(x[i] ?? C.of(0), method.b[i] ?? 0))
  return C.add(C.of(1), C.mul(z, acc))
}

/** A stability region sampled on a grid of h lambda. */
export interface StabilityRegion {
  readonly re: readonly number[]
  readonly im: readonly number[]
  /** |R(z)| at [i][j] = (im[i], re[j]). */
  readonly modulus: readonly (readonly number[])[]
  /** Whether |R(z)| <= 1. */
  readonly stable: readonly (readonly boolean[])[]
}

/**
 * The region of absolute stability {z : |R(z)| <= 1} of a Runge-Kutta method on a grid (for figures).
 *
 * @param method - The coefficients.
 * @param options - Real and imaginary ranges and the grid size (default [-5, 2] x [-4, 4], 81 x 81).
 * @returns The grid, |R| and the stable mask.
 * @concept a-stability
 * @example
 * ```ts
 * import { stabilityRegion, RK4 } from 'whitebox-engine/dynamics/ode'
 * stabilityRegion(RK4, { n: 5 }).stable.length // => 5
 * ```
 */
export function stabilityRegion(
  method: RungeKuttaCoefficients,
  options: {
    readonly re?: readonly [number, number]
    readonly im?: readonly [number, number]
    readonly n?: number
  } = {},
): StabilityRegion {
  const n = options.n ?? 81
  const [r0, r1] = options.re ?? [-5, 2]
  const [i0, i1] = options.im ?? [-4, 4]
  const re = Array.from({ length: n }, (_, k) => r0 + ((r1 - r0) * k) / Math.max(1, n - 1))
  const im = Array.from({ length: n }, (_, k) => i0 + ((i1 - i0) * k) / Math.max(1, n - 1))
  const modulus = im.map((y) =>
    re.map((x) => {
      const r = stabilityFunction(method, C.of(x, y))
      return Math.hypot(r.re, r.im)
    }),
  )
  return { re, im, modulus, stable: modulus.map((row) => row.map((v) => v <= 1 + 1e-12)) }
}

/** The outcome of {@link aStability}. */
export interface AStabilityReport {
  /** |R| <= 1 on the whole left half-plane (checked on the imaginary axis, with no poles on the left). */
  readonly aStable: boolean
  /** A-stable with R(z) -> 0 as z -> -infinity. */
  readonly lStable: boolean
  /** The largest |R(iy)| found on the imaginary axis. */
  readonly maxOnAxis: number
  /** |R(-10^8)|, the damping of very stiff components. */
  readonly atInfinity: number
}

/**
 * Check A-stability and L-stability of a Runge-Kutta method numerically: |R(iy)| <= 1 on a log-spaced sample of the
 * imaginary axis, R analytic in the left half-plane (I - zA singular only for Re z > 0, that is the eigenvalues of A
 * in the right half-plane, checked by sampling the left half-plane), and |R(-infinity)| for L-stability.
 *
 * @param method - The coefficients.
 * @returns The report.
 * @concept a-stability
 * @example
 * ```ts
 * import { aStability, BACKWARD_EULER, RK4 } from 'whitebox-engine/dynamics/ode'
 * [aStability(BACKWARD_EULER).lStable, aStability(RK4).aStable] // => [true, false]
 * ```
 */
export function aStability(method: RungeKuttaCoefficients): AStabilityReport {
  let maxOnAxis = 0
  for (let k = -40; k <= 40; k++) {
    const y = 10 ** (k / 10)
    for (const sgn of [1, -1]) {
      const r = stabilityFunction(method, C.of(0, sgn * y))
      maxOnAxis = Math.max(maxOnAxis, Math.hypot(r.re, r.im))
    }
  }
  let leftOk = true
  for (let k = 0; k <= 16 && leftOk; k++) {
    const radius = 10 ** (k / 4 - 1)
    for (let j = 1; j < 12; j++) {
      const theta = Math.PI / 2 + (Math.PI * j) / 12
      const r = stabilityFunction(method, C.of(radius * Math.cos(theta), radius * Math.sin(theta)))
      if (!(Math.hypot(r.re, r.im) <= 1 + 1e-9)) {
        leftOk = false
        break
      }
    }
  }
  const inf = stabilityFunction(method, C.of(-1e8))
  const atInfinity = Math.hypot(inf.re, inf.im)
  const aStable = maxOnAxis <= 1 + 1e-9 && leftOk
  return { aStable, lStable: aStable && atInfinity < 1e-6, maxOnAxis, atInfinity }
}

/**
 * The boundary locus of a linear multistep method sum alpha_j y_{n+j} = h sum beta_j f_{n+j}: the curve
 * z(theta) = rho(e^{i theta}) / sigma(e^{i theta}) that bounds its stability region.
 *
 * @param alpha - Coefficients of rho, lowest power first.
 * @param beta - Coefficients of sigma, lowest power first.
 * @param n - Points on the curve (default 200).
 * @returns The curve as real and imaginary parts.
 * @concept linear-multistep
 * @example
 * ```ts
 * import { boundaryLocus } from 'whitebox-engine/dynamics/ode'
 * // Forward Euler: y_{n+1} - y_n = h f_n traces the circle |z + 1| = 1.
 * Math.round(boundaryLocus([-1, 1], [1, 0], 4).re[2] ?? 0) // => -2
 * ```
 */
export function boundaryLocus(
  alpha: readonly number[],
  beta: readonly number[],
  n = 200,
): { readonly re: number[]; readonly im: number[] } {
  const re: number[] = []
  const im: number[] = []
  const poly = (c: readonly number[], z: ComplexNumber): ComplexNumber => {
    let acc = C.of(0)
    for (let j = c.length - 1; j >= 0; j--) acc = C.add(C.mul(acc, z), c[j] ?? 0)
    return acc
  }
  for (let k = 0; k < n; k++) {
    const theta = (2 * Math.PI * k) / n
    const zeta = C.of(Math.cos(theta), Math.sin(theta))
    const z = C.div(poly(alpha, zeta), poly(beta, zeta))
    re.push(z.re)
    im.push(z.im)
  }
  return { re, im }
}

/** One point of a work-precision diagram. */
export interface WorkPrecisionPoint {
  readonly method: string
  readonly tol: number
  /** RMS error of the final state against the reference. */
  readonly error: number
  readonly nfev: number
  readonly steps: number
}

/**
 * A work-precision diagram: for each method and tolerance, the error of the final state against a reference and the
 * work (vector field evaluations and steps).
 *
 * @param f - The vector field.
 * @param tspan - The interval.
 * @param y0 - The initial state.
 * @param reference - The (accurate) final state.
 * @param methods - Methods by name or stepper factories of a tolerance.
 * @param tolerances - Relative tolerances (atol is rtol / 100).
 * @returns One point per method and tolerance.
 * @concept work-precision
 * @example
 * ```ts
 * import { workPrecision } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * workPrecision((_t, y) => neg(y), [0, 1], [1], [Math.exp(-1)], ['RK45', 'DOP853'], [1e-4, 1e-8]).length // => 4
 * ```
 */
export function workPrecision(
  f: VectorField,
  tspan: readonly [number, number],
  y0: TensorLike,
  reference: TensorLike,
  methods: readonly (OdeMethodName | ((tol: number) => Stepper))[],
  tolerances: readonly number[],
): WorkPrecisionPoint[] {
  const want = numbersOf(floatState(reference))
  const out: WorkPrecisionPoint[] = []
  for (const m of methods) {
    for (const tol of tolerances) {
      const sol =
        typeof m === 'string'
          ? solveIvp(f, tspan, y0, { method: m, rtol: tol, atol: tol / 100 })
          : solveIvp(f, tspan, y0, { method: m(tol) })
      const got = numbersOf(sol.ys[sol.ys.length - 1] as Value)
      out.push({
        method: typeof m === 'string' ? m : 'custom',
        tol,
        error: rms(Float64Array.from(got, (v, i) => v - (want[i] ?? 0))),
        nfev: sol.nfev,
        steps: sol.nSteps,
      })
    }
  }
  return out
}
