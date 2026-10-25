/**
 * Numerical solution of stochastic differential equations dY = f dt + g dW, like sdeint and torchsde: Euler-Maruyama,
 * Milstein, stochastic Runge-Kutta and Heun schemes for several noise types, `sdeint` as a traced path simulation,
 * strong and weak convergence studies, conversion between Ito and Stratonovich forms, and the reverse-time SDE and
 * probability-flow ODE that score-based diffusion models are built on.
 */

import { jvp } from '../autodiff/index.ts'
import { type Estimate, estimate, MdmlError, maxOf } from '../base/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import {
  abs,
  add,
  asFloat,
  asTensor,
  mean,
  mul,
  reshape,
  sliceAxis,
  square,
  squeeze,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
  zerosLike,
} from '../tensor/index.ts'
import { asTimes, positiveInteger, sampleMean, type TimesLike, traceOf } from './_shared.ts'
import { brownianIncrements } from './brownian.ts'
import { coarsenIncrements, type Paths, simulatePaths, type TraceOptions, timeGrid, valuesAt } from './paths.ts'

/** A drift or diffusion coefficient: a function of the time and the state `[paths, ...state]`. */
export type SdeFunction = (t: number, y: Value) => Value

/**
 * How the Brownian motion enters: `diagonal` (default; g(t, y) has y's shape and multiplies an independent Brownian
 * motion per coordinate), `scalar` (one Brownian motion per path; g has y's shape), `general` (g(t, y) is
 * `[...y.shape, m]`, applied to an m-dimensional Brownian motion).
 */
export type NoiseType = 'diagonal' | 'scalar' | 'general'

/** A stochastic differential equation dY = f(t, Y) dt + g(t, Y) dW, read in the Ito or Stratonovich sense. */
export interface Sde {
  readonly drift: SdeFunction
  readonly diffusion: SdeFunction
  readonly noise?: NoiseType
  /** The Brownian dimension m of `general` noise. */
  readonly noiseDim?: number
  /** The calculus the equation is written in (default `ito`). */
  readonly calculus?: 'ito' | 'stratonovich'
}

/**
 * A scheme: `euler` (Euler-Maruyama, strong order 1/2, weak order 1), `milstein` (strong order 1 for diagonal or
 * scalar noise, derivative by forward-mode autodiff), `srk` (Platen's derivative-free stochastic Runge-Kutta, strong
 * order 1, Ito), `heun` (the stochastic Heun scheme for Stratonovich equations, strong order 1 for commutative noise).
 */
export type SdeMethod = 'euler' | 'milstein' | 'srk' | 'heun'

/** The shape of one step's Brownian increment for a state shape. */
function noiseShape(sde: Sde, y: readonly number[]): number[] {
  const kind = sde.noise ?? 'diagonal'
  if (kind === 'diagonal') return [...y]
  if (kind === 'scalar') return [y[0] ?? 1]
  return [y[0] ?? 1, positiveInteger(sde.noiseDim ?? 0, 'noiseDim', 'sde (general noise)')]
}

/** g dW for the noise type. */
function applyNoise(sde: Sde, g: Value, dw: Value, rank: number): Value {
  const kind = sde.noise ?? 'diagonal'
  if (kind === 'diagonal') return mul(g, dw)
  if (kind === 'scalar') return mul(g, reshape(dw, [dw.shape[0] ?? 1, ...new Array<number>(rank - 1).fill(1)]))
  const m = dw.shape[1] ?? 1
  return sum(mul(g, reshape(dw, [dw.shape[0] ?? 1, ...new Array<number>(rank - 1).fill(1), m])), -1)
}

/** The scalar-or-diagonal increment broadcast against the state (for the Milstein correction). */
function alongState(sde: Sde, dw: Value, rank: number): Value {
  return (sde.noise ?? 'diagonal') === 'scalar'
    ? reshape(dw, [dw.shape[0] ?? 1, ...new Array<number>(rank - 1).fill(1)])
    : dw
}

function checkScheme(sde: Sde, method: SdeMethod): void {
  const calculus = sde.calculus ?? 'ito'
  const noise = sde.noise ?? 'diagonal'
  if (method === 'heun' && calculus !== 'stratonovich') {
    throw new MdmlError('value', 'sde: the stochastic Heun scheme solves Stratonovich equations')
  }
  if ((method === 'euler' || method === 'srk') && calculus !== 'ito') {
    throw new MdmlError('value', `sde: ${method} solves Ito equations (use heun or milstein for Stratonovich)`)
  }
  if ((method === 'milstein' || method === 'srk') && noise === 'general') {
    throw new MdmlError('not-implemented', `sde: ${method} needs diagonal or scalar noise (Levy areas are not drawn)`)
  }
}

/**
 * One step of a scheme from (t, y) over dt with the Brownian increment dw: the building block of {@link sdeint}.
 *
 * The Milstein correction is 1/2 (g' g)(dW^2 - dt) (Ito) or 1/2 (g' g) dW^2 (Stratonovich), with g' g the directional
 * derivative of g along g, exact for diagonal noise whose i-th coefficient depends on the i-th coordinate and for
 * scalar noise. Platen's scheme replaces the derivative by (g(y + f dt + g sqrt dt) - g(y)) / sqrt dt.
 *
 * @param sde - The equation.
 * @param method - The scheme.
 * @param t - The time.
 * @param y - The state `[paths, ...state]`.
 * @param dt - The step.
 * @param dw - The Brownian increment (`y`'s shape for diagonal noise, `[paths]` scalar, `[paths, m]` general).
 * @returns The state at t + dt.
 * @throws MdmlError when the scheme does not apply to the equation's calculus or noise.
 * @concept stochastic-differential-equations
 * @example
 * ```ts
 * import { sdeStep } from 'mdml-engine/stochastic'
 * import { tensor, toArray, mul } from 'mdml-engine/tensor'
 * const sde = { drift: (_t, y) => mul(y, 0.5), diffusion: (_t, y) => mul(y, 0.2) }
 * toArray(sdeStep(sde, 'euler', 0, tensor([1]), 0.1, tensor([0.5]))).map((v) => Math.round(v * 1e12) / 1e12) // => [1.15]
 * ```
 */
export function sdeStep(sde: Sde, method: SdeMethod, t: number, y: Value, dt: number, dw: Value): Value {
  checkScheme(sde, method)
  const rank = y.shape.length
  const f = sde.drift(t, y)
  const g = sde.diffusion(t, y)
  const ito = (sde.calculus ?? 'ito') === 'ito'
  if (method === 'euler') return add(add(y, mul(f, dt)), applyNoise(sde, g, dw, rank))
  if (method === 'milstein') {
    const ggPrime = jvp((x: Value) => sde.diffusion(t, x), y, g).tangent
    const w = alongState(sde, dw, rank)
    const correction = mul(mul(ggPrime, 0.5), ito ? sub(square(w), dt) : square(w))
    return add(add(add(y, mul(f, dt)), mul(g, w)), correction)
  }
  if (method === 'srk') {
    const h = Math.sqrt(dt)
    const support = add(add(y, mul(f, dt)), mul(g, h))
    const w = alongState(sde, dw, rank)
    const correction = mul(sub(sde.diffusion(t, support), g), div2(sub(square(w), dt), 2 * h))
    return add(add(add(y, mul(f, dt)), mul(g, w)), correction)
  }
  // Stochastic Heun: predictor by Euler, corrector by the trapezoid in drift and diffusion.
  const predictor = add(add(y, mul(f, dt)), applyNoise(sde, g, dw, rank))
  const f2 = sde.drift(t + dt, predictor)
  const g2 = sde.diffusion(t + dt, predictor)
  return add(add(y, mul(add(f, f2), dt / 2)), applyNoise(sde, mul(add(g, g2), 0.5), dw, rank))
}

function div2(x: Value, d: number): Value {
  return mul(x, 1 / d)
}

/** Options of {@link sdeint}. */
export interface SdeintOptions extends TraceOptions {
  /** The scheme (default `euler` for Ito equations, `heun` for Stratonovich ones). */
  readonly method?: SdeMethod
  /** Scheme steps per grid interval (default 1); the paths are reported on the grid. */
  readonly substeps?: number
  /** Brownian increments `[paths, steps * substeps, ...noise]` to drive the scheme instead of drawing them. */
  readonly increments?: TensorLike
}

function defaultMethod(sde: Sde): SdeMethod {
  return (sde.calculus ?? 'ito') === 'ito' ? 'euler' : 'heun'
}

/**
 * Integrate an SDE on a grid (like sdeint and torchsde.sdeint), as a traced path simulation: each grid interval is one
 * algorithm step of `substeps` scheme steps whose Brownian increments come from the step key (or from `increments`, so
 * several schemes can be driven by the same Brownian path). Built from tensor operations, so the paths differentiate in
 * the initial state and in any parameters the coefficients close over (pathwise derivatives).
 *
 * @param key - The key.
 * @param sde - The equation.
 * @param y0 - The initial state `[paths, ...state]`.
 * @param times - The grid.
 * @param options - Scheme, substeps, increments and tracing.
 * @returns The paths `[paths, times.length, ...state]`.
 * @throws MdmlError with code `shape` when the increments do not match.
 * @concept stochastic-differential-equations
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { sdeint, timeGrid } from 'mdml-engine/stochastic'
 * import { mul, zeros } from 'mdml-engine/tensor'
 * const ou = { drift: (_t, y) => mul(y, -1), diffusion: (_t, y) => mul(zeros(y.shape), 0) }
 * const p = sdeint(key(0), ou, [1, 2], timeGrid(1, 1000), { method: 'milstein' })
 * p.values.shape // => [2, 1001]
 * Math.abs(p.values.data[1000] - Math.exp(-1)) < 1e-3 // => true
 * ```
 */
export function sdeint(key: Key, sde: Sde, y0: TensorLike, times: TimesLike, options: SdeintOptions = {}): Paths {
  const t = asTimes(times, 'sdeint')
  const method = options.method ?? defaultMethod(sde)
  checkScheme(sde, method)
  const substeps = positiveInteger(options.substeps ?? 1, 'substeps', 'sdeint')
  const x0 = asFloat(asTensor(y0))
  if (x0.shape.length === 0) throw new MdmlError('shape', 'sdeint: y0 must be [paths, ...state]')
  const shape = noiseShape(sde, x0.shape)
  const given = options.increments === undefined ? undefined : asTensor(options.increments)
  const total = (t.length - 1) * substeps
  if (given !== undefined && (given.shape[0] !== shape[0] || given.shape[1] !== total)) {
    throw new MdmlError('shape', `sdeint: increments must be [${shape[0]}, ${total}, ...noise]`)
  }
  return simulatePaths(
    key,
    {
      name: `sde-${method}`,
      config: { method, substeps, steps: t.length - 1, noise: sde.noise ?? 'diagonal' },
      times: t,
      x0,
      step: (y, c) => {
        const h = c.dt / substeps
        let state = y
        for (let s = 0; s < substeps; s++) {
          const j = c.i * substeps + s
          const dw =
            given === undefined
              ? mul(normal(foldIn(c.key, s), shape), Math.sqrt(h))
              : squeeze(sliceAxis(given, 1, j, j + 1), 1)
          state = sdeStep(sde, method, c.t + s * h, state, h, dw)
        }
        return state
      },
    },
    traceOf(options),
  )
}

/**
 * An SDE solver in the shape `nn.blocks.SDEBlock` expects (diagonal noise): `(f, g, y0, times, key) => states`, one
 * state per time, integrated by {@link sdeint} with the chosen scheme and substeps.
 *
 * @param method - The scheme (default `euler`).
 * @param options - Substeps per interval (default 1).
 * @returns The solver.
 * @concept stochastic-differential-equations
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { sdeSolver } from 'mdml-engine/stochastic'
 * import { mul, tensor } from 'mdml-engine/tensor'
 * const solve = sdeSolver('milstein', { substeps: 4 })
 * solve((_t, y) => mul(y, -1), (_t, y) => mul(y, 0.1), tensor([[1, 2]]), [0, 0.5, 1], key(0)).length // => 3
 * ```
 */
export function sdeSolver(
  method: SdeMethod = 'euler',
  options: { readonly substeps?: number } = {},
): (f: SdeFunction, g: SdeFunction, y0: Value, times: readonly number[], key: Key) => Value[] {
  return (f, g, y0, times, key) => {
    const scalar = y0.shape.length === 0
    const start = scalar ? reshape(y0, [1]) : y0
    const sde: Sde = scalar
      ? { drift: (t, y) => reshape(f(t, reshape(y, [])), [1]), diffusion: (t, y) => reshape(g(t, reshape(y, [])), [1]) }
      : { drift: f, diffusion: g }
    const p = sdeint(key, sde, start, times, {
      method,
      ...(options.substeps === undefined ? {} : { substeps: options.substeps }),
    })
    return Array.from({ length: times.length }, (_, i) => {
      const v = valuesAt(p, i)
      return scalar ? reshape(v, []) : v
    })
  }
}

/** One row of a convergence study. */
export interface ConvergenceStudy {
  /** The numbers of steps tried. */
  readonly steps: readonly number[]
  /** The step sizes. */
  readonly dt: readonly number[]
  /** The error estimates at each step size. */
  readonly errors: readonly Estimate[]
  /** The least-squares slope of log error against log dt: the empirical order. */
  readonly order: number
}

/** Options of the convergence studies. */
export interface ConvergenceOptions {
  /** The scheme. */
  readonly method?: SdeMethod
  /** The numbers of steps (default [8, 16, 32, 64]). */
  readonly steps?: readonly number[]
  /** The number of paths. */
  readonly paths?: number
}

function slope(x: readonly number[], y: readonly number[]): number {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my)
    sxx += ((x[i] ?? 0) - mx) ** 2
  }
  return sxx === 0 ? Number.NaN : sxy / sxx
}

function checkSteps(steps: readonly number[], where: string): void {
  if (steps.length < 2) throw new MdmlError('value', `${where}: needs at least two step counts`)
  for (const s of steps) positiveInteger(s, 'steps', where)
}

/**
 * A strong convergence study: E|Y_N(T) - X(T)| for several step counts, every scheme driven by the same Brownian path
 * (increments on the finest grid, summed onto the coarser ones), against the exact terminal value given that path.
 * Euler-Maruyama shows order 1/2, Milstein and SRK order 1 on equations with multiplicative noise.
 *
 * @param key - The key.
 * @param sde - The equation.
 * @param y0 - The initial state of one path (the same for all), shape `[...state]`.
 * @param t1 - The horizon (from 0).
 * @param exact - The exact X(T) given the finest increments `[paths, n, ...noise]` and the finest grid.
 * @param options - Scheme, step counts (each dividing the finest) and number of paths (default 200).
 * @returns The errors with standard errors and the fitted order.
 * @concept strong-convergence
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { strongConvergence } from 'mdml-engine/stochastic'
 * import { exp, mul, sum } from 'mdml-engine/tensor'
 * const gbm = { drift: (_t, y) => mul(y, 0.5), diffusion: (_t, y) => mul(y, 0.8) }
 * // X_T = exp((mu - sigma^2 / 2) T + sigma W_T), and W_T is the sum of the increments
 * const exact = (dw) => mul(exp(mul(sum(dw, 1), 0.8)), Math.exp(0.5 - 0.32))
 * const s = strongConvergence(key(0), gbm, [1], 1, exact, { method: 'milstein' })
 * s.order > 0.8 // => true
 * ```
 */
export function strongConvergence(
  key: Key,
  sde: Sde,
  y0: TensorLike,
  t1: number,
  exact: (increments: Value, times: Float64Array) => Value,
  options: ConvergenceOptions = {},
): ConvergenceStudy {
  const steps = options.steps ?? [8, 16, 32, 64]
  checkSteps(steps, 'strongConvergence')
  const paths = positiveInteger(options.paths ?? 200, 'paths', 'strongConvergence')
  const finest = maxOf(steps)
  const start = broadcastPaths(y0, paths)
  const fine = timeGrid(t1, finest)
  const dw = brownianIncrements(key, fine, { paths, shape: noiseShape(sde, start.shape).slice(1) })
  const target = exact(dw, fine)
  const errors: Estimate[] = []
  for (const n of steps) {
    if (finest % n !== 0) throw new MdmlError('value', `strongConvergence: ${n} does not divide ${finest}`)
    const coarse = coarsenIncrements(dw, finest / n)
    const p = sdeint(key, sde, start, timeGrid(t1, n), {
      ...(options.method === undefined ? {} : { method: options.method }),
      increments: coarse,
    })
    const err = abs(sub(valuesAt(p, -1), target))
    const perPath = toFlat(reshape(mean(reshape(err, [paths, -1]), 1), [paths]))
    errors.push(sampleMean(perPath))
  }
  const dts = steps.map((n) => t1 / n)
  return { steps, dt: dts, errors, order: fitOrder(dts, errors) }
}

function broadcastPaths(y0: TensorLike, paths: number): Value {
  const one = asFloat(asTensor(y0))
  return add(zeros([paths, ...one.shape]), one)
}

function fitOrder(dts: readonly number[], errors: readonly Estimate[]): number {
  return slope(
    dts.map((d) => Math.log(d)),
    errors.map((e) => Math.log(Math.max(e.value, Number.MIN_VALUE))),
  )
}

/**
 * A weak convergence study: |E phi(Y_N(T)) - E phi(X(T))| for several step counts, with independent paths per step
 * count and the exact expectation given. Euler-Maruyama shows weak order 1.
 *
 * @param key - The key.
 * @param sde - The equation.
 * @param y0 - The initial state of one path, shape `[...state]`.
 * @param t1 - The horizon.
 * @param test - The test function phi, mapping the terminal states `[paths, ...state]` to `[paths]`.
 * @param expected - The exact E phi(X(T)).
 * @param options - Scheme, step counts and number of paths (default 20000).
 * @returns The errors (with the Monte Carlo standard error of each) and the fitted order.
 * @concept weak-convergence
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { weakConvergence } from 'mdml-engine/stochastic'
 * import { mul } from 'mdml-engine/tensor'
 * const gbm = { drift: (_t, y) => mul(y, 1.5), diffusion: (_t, y) => mul(y, 0.1) }
 * const w = weakConvergence(key(0), gbm, [1], 1, (y) => y, Math.exp(1.5), { steps: [2, 4, 8], paths: 2000 })
 * w.order > 0.7 // => true
 * ```
 */
export function weakConvergence(
  key: Key,
  sde: Sde,
  y0: TensorLike,
  t1: number,
  test: (y: Value) => Value,
  expected: number,
  options: ConvergenceOptions = {},
): ConvergenceStudy {
  const steps = options.steps ?? [8, 16, 32, 64]
  checkSteps(steps, 'weakConvergence')
  const paths = positiveInteger(options.paths ?? 20000, 'paths', 'weakConvergence')
  const start = broadcastPaths(y0, paths)
  const errors: Estimate[] = []
  for (const [i, n] of steps.entries()) {
    const p = sdeint(foldIn(key, i), sde, start, timeGrid(t1, n), {
      ...(options.method === undefined ? {} : { method: options.method }),
    })
    const phi = toFlat(test(valuesAt(p, -1)))
    const m = sampleMean(phi)
    errors.push(estimate(Math.abs(m.value - expected), m.standardError))
  }
  const dts = steps.map((n) => t1 / n)
  return { steps, dt: dts, errors, order: fitOrder(dts, errors) }
}

/** A score: the gradient of the log density of the time-t marginal, s(t, y) = grad log p_t(y). */
export type ScoreFunction = (t: number, y: Value) => Value

/** The divergence term d_i = d(g_i^2)/dy_i of a diagonal diffusion (zero when g does not depend on y). */
function diffusionDivergence(sde: Sde, t: number, y: Value): Value {
  const g2 = (x: Value): Value => square(sde.diffusion(t, x))
  return jvp(g2, y, add(zerosLike(y), 1)).tangent
}

function checkReversible(sde: Sde, where: string): void {
  if ((sde.calculus ?? 'ito') !== 'ito' || (sde.noise ?? 'diagonal') !== 'diagonal') {
    throw new MdmlError('value', `${where}: needs an Ito equation with diagonal noise`)
  }
}

/**
 * The reverse-time SDE of Anderson (1982) given the score of the forward marginals: with s = T - t running forwards,
 * dY = [-f(T - s, Y) + g^2 grad log p_{T-s}(Y) + d(g^2)/dy] ds + g dB, whose marginals are those of the forward process
 * reversed (the sampler of score-based diffusion models). Integrate it with {@link sdeint} from samples of p_T.
 *
 * @param sde - The forward equation (Ito, diagonal noise whose i-th coefficient depends on the i-th coordinate).
 * @param score - The score of the forward marginals.
 * @param t1 - The horizon T of the forward process.
 * @returns The reverse-time equation in the time s = T - t.
 * @throws MdmlError with code `value` for Stratonovich equations or non-diagonal noise.
 * @concept score-based-diffusion
 * @reference B. D. O. Anderson, Reverse-time diffusion equation models, Stochastic Processes and their Applications 12
 *   (1982); Y. Song et al., Score-based generative modeling through stochastic differential equations (ICLR 2021).
 * @example
 * ```ts
 * import { reverseSde } from 'mdml-engine/stochastic'
 * import { tensor, toArray, mul, neg, add, zeros } from 'mdml-engine/tensor'
 * const ou = { drift: (_t, y) => neg(y), diffusion: (_t, y) => add(zeros(y.shape), Math.SQRT2) }
 * const back = reverseSde(ou, (_t, y) => neg(y), 1)
 * toArray(back.drift(0, tensor([2]))).map((v) => Math.round(v * 1e12) / 1e12) // => [-2]
 * ```
 */
export function reverseSde(sde: Sde, score: ScoreFunction, t1: number): Sde {
  checkReversible(sde, 'reverseSde')
  return {
    drift: (s, y) => {
      const t = t1 - s
      const g = sde.diffusion(t, y)
      return add(sub(mul(square(g), score(t, y)), sde.drift(t, y)), diffusionDivergence(sde, t, y))
    },
    diffusion: (s, y) => sde.diffusion(t1 - s, y),
    noise: 'diagonal',
    calculus: 'ito',
  }
}

/**
 * The probability-flow ODE drift f - 1/2 d(g^2)/dy - 1/2 g^2 grad log p_t of an SDE: the deterministic flow with the
 * same marginals (Song et al. 2021). Integrate it with any ODE solver, or with {@link sdeint} and zero diffusion.
 *
 * @param sde - The equation (Ito, diagonal noise).
 * @param score - The score of its marginals.
 * @returns The ODE vector field (t, y) => dy/dt.
 * @throws MdmlError with code `value` for Stratonovich equations or non-diagonal noise.
 * @concept score-based-diffusion
 * @example
 * ```ts
 * import { probabilityFlow } from 'mdml-engine/stochastic'
 * import { tensor, toArray, neg, add, zeros } from 'mdml-engine/tensor'
 * const ou = { drift: (_t, y) => neg(y), diffusion: (_t, y) => add(zeros(y.shape), Math.SQRT2) }
 * Math.abs(toArray(probabilityFlow(ou, (_t, y) => neg(y))(0, tensor([2])))[0]) < 1e-12 // => true
 * ```
 */
export function probabilityFlow(sde: Sde, score: ScoreFunction): SdeFunction {
  checkReversible(sde, 'probabilityFlow')
  return (t, y) => {
    const g = sde.diffusion(t, y)
    return sub(sde.drift(t, y), mul(add(mul(square(g), score(t, y)), diffusionDivergence(sde, t, y)), 0.5))
  }
}

/**
 * The drift of the Stratonovich equation equivalent to an Ito one (or back): f_S = f_I - 1/2 g' g for diagonal or
 * scalar noise, so the same process can be solved by Heun.
 *
 * @param sde - The equation.
 * @returns The equation in the other calculus, with the same paths.
 * @throws MdmlError with code `not-implemented` for general noise.
 * @concept stratonovich-integral
 * @example
 * ```ts
 * import { convertCalculus } from 'mdml-engine/stochastic'
 * import { tensor, toArray, mul } from 'mdml-engine/tensor'
 * const gbm = { drift: (_t, y) => mul(y, 0.5), diffusion: (_t, y) => mul(y, 1) }
 * toArray(convertCalculus(gbm).drift(0, tensor([2]))) // => [0]
 * ```
 */
export function convertCalculus(sde: Sde): Sde {
  if ((sde.noise ?? 'diagonal') === 'general') {
    throw new MdmlError('not-implemented', 'convertCalculus: needs diagonal or scalar noise')
  }
  const toStratonovich = (sde.calculus ?? 'ito') === 'ito'
  return {
    ...sde,
    drift: (t, y) => {
      const ggPrime = jvp((x: Value) => sde.diffusion(t, x), y, sde.diffusion(t, y)).tangent
      return toStratonovich ? sub(sde.drift(t, y), mul(ggPrime, 0.5)) : add(sde.drift(t, y), mul(ggPrime, 0.5))
    },
    calculus: toStratonovich ? 'stratonovich' : 'ito',
  }
}
