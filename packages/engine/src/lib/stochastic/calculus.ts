import { grad } from '../autodiff/index.ts'
import { type Estimate, estimate, WhiteboxError } from '../base/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import type { Key } from '../random/index.ts'
import {
  add,
  asTensor,
  div,
  exp,
  fromStorage,
  log,
  mean,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { positiveInteger, primalNumbers, sampleMean, stepsOf, valuesOf } from './_shared.ts'
import { type Paths, pathIncrements, timeGrid } from './paths.ts'
import { type Sde, type SdeMethod, sdeint } from './sde.ts'

/** The integrand at the left end of each interval: `[paths, n - 1, ...]` from n grid values (or given as such). */
function leftPoints(h: Value, n: number, where: string): Value {
  const m = h.shape[1] ?? 0
  if (m === n - 1) return h
  if (m === n) return sliceAxis(h, 1, 0, n - 1)
  throw new WhiteboxError('shape', `${where}: the integrand needs ${n} or ${n - 1} time points, got ${m}`)
}

function cumulative(terms: Value): Value {
  const out: Value[] = [zeros([terms.shape[0] ?? 1, ...terms.shape.slice(2)])]
  for (let i = 0; i < (terms.shape[1] ?? 0); i++) {
    out.push(add(out[i] as Value, squeeze(sliceAxis(terms, 1, i, i + 1), 1)))
  }
  return stack(out, 1)
}

/** Options of the stochastic integrals. */
export interface IntegralOptions {
  /** Return the running integral at every grid time (starting at 0) instead of the total. */
  readonly cumulative?: boolean
}

/**
 * The Ito integral int H dX by left-point Riemann sums, sum_i H_{t_i} (X_{t_{i+1}} - X_{t_i}): the limit that makes the
 * integral a martingale. For H = W and X = W the sum is exactly (W_T^2 - [W]_T) / 2. Differentiable in both.
 *
 * @param h - The integrand on the grid `[paths, n, ...]` (or at the n - 1 left points).
 * @param x - The integrator: paths or values `[paths, n, ...]`.
 * @param options - `cumulative` for the running integral.
 * @returns `[paths, ...]`, or `[paths, n, ...]` when cumulative.
 * @concept ito-integral
 * @example
 * ```ts
 * import { itoIntegral } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(itoIntegral([[0, 1, 3]], [[0, 1, 3]])) // => [2]
 * ```
 */
export function itoIntegral(h: Paths | TensorLike, x: Paths | TensorLike, options: IntegralOptions = {}): Value {
  const xv = valuesOf(x)
  const n = xv.shape[1] ?? 0
  const terms = mul(leftPoints(valuesOf(h), n, 'itoIntegral'), pathIncrements(xv))
  return options.cumulative === true ? cumulative(terms) : sum(terms, 1)
}

/**
 * The Stratonovich integral int H o dX by the trapezoid (midpoint average) rule, sum_i (H_{t_i} + H_{t_{i+1}}) / 2 dX_i,
 * which obeys the ordinary chain rule: int W o dW = W_T^2 / 2 exactly. Differentiable in both.
 *
 * @param h - The integrand on the grid `[paths, n, ...]`.
 * @param x - The integrator: paths or values `[paths, n, ...]`.
 * @param options - `cumulative` for the running integral.
 * @returns `[paths, ...]`, or `[paths, n, ...]` when cumulative.
 * @concept stratonovich-integral
 * @example
 * ```ts
 * import { stratonovichIntegral } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(stratonovichIntegral([[0, 1, 3]], [[0, 1, 3]])) // => [4.5]
 * ```
 */
export function stratonovichIntegral(
  h: Paths | TensorLike,
  x: Paths | TensorLike,
  options: IntegralOptions = {},
): Value {
  const xv = valuesOf(x)
  const hv = valuesOf(h)
  const n = xv.shape[1] ?? 0
  if ((hv.shape[1] ?? 0) !== n) throw new WhiteboxError('shape', 'stratonovichIntegral: H must be on the same grid')
  const mid = mul(add(sliceAxis(hv, 1, 0, n - 1), sliceAxis(hv, 1, 1, n)), 0.5)
  const terms = mul(mid, pathIncrements(xv))
  return options.cumulative === true ? cumulative(terms) : sum(terms, 1)
}

/**
 * The time integral int H dt by left-point sums on a grid.
 *
 * @param h - Values `[paths, n, ...]` on the grid.
 * @param times - The grid.
 * @returns `[paths, ...]`.
 * @concept ito-integral
 * @example
 * ```ts
 * import { timeIntegral } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(timeIntegral([[1, 2, 3]], [0, 0.5, 1])) // => [1.5]
 * ```
 */
export function timeIntegral(h: Paths | TensorLike, times: ArrayLike<number>): Value {
  const dts = stepsOf(Float64Array.from(times))
  const hv = leftPoints(valuesOf(h), dts.length + 1, 'timeIntegral')
  const shape = [1, dts.length, ...new Array<number>(Math.max(0, hv.shape.length - 2)).fill(1)]
  return sum(mul(hv, fromStorage(dts, shape, 'float64')), 1)
}

/** The two sides of Ito's formula on a path. */
export interface ItoLemmaCheck {
  /** f(T, X_T) - f(t_0, X_{t_0}) per path. */
  readonly lhs: Value
  /** int f_t dt + int f_x dX + 1/2 int f_xx d[X] by left-point sums per path. */
  readonly rhs: Value
  /** The largest |lhs - rhs| over paths: it vanishes as the grid is refined. */
  readonly residual: number
}

/**
 * Ito's lemma checked numerically on simulated paths: for a smooth f(t, x) applied elementwise and a scalar Ito process
 * X, f(T, X_T) - f(0, X_0) = int f_t dt + int f_x dX + 1/2 int f_xx d[X]. The derivatives come from autodiff and the
 * integrals from left-point sums with the realised quadratic variation, so the residual is the discretisation error.
 *
 * @param f - A function of the time and the state, applied elementwise.
 * @param x - Scalar paths `[paths, n]` of an Ito process.
 * @returns Both sides per path and the largest residual.
 * @concept ito-lemma
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { brownianMotion, itoLemma, timeGrid } from 'whitebox-engine/stochastic'
 * import { square } from 'whitebox-engine/tensor'
 * itoLemma((_t, x) => square(x), brownianMotion(key(0), timeGrid(1, 100), { paths: 3 })).residual < 1e-12 // => true
 * ```
 */
export function itoLemma(f: (t: Value, x: Value) => Value, x: Paths): ItoLemmaCheck {
  const v = valuesOf(x)
  if (v.shape.length !== 2) throw new WhiteboxError('shape', 'itoLemma: needs scalar paths [paths, n]')
  const n = v.shape[1] ?? 0
  const p = v.shape[0] ?? 1
  const times = x.times
  const tCol = (i: number): Value => add(zeros([p]), times[i] ?? 0)
  const at = (i: number): Value => squeeze(sliceAxis(v, 1, i, i + 1), 1)
  const fx = (t: Value, y: Value): Value => grad((z: Value) => sum(f(t, z)))(y)
  const fxx = (t: Value, y: Value): Value => grad((z: Value) => sum(fx(t, z)))(y)
  const ft = (t: Value, y: Value): Value => grad((s: Value) => sum(f(s, y)))(t)
  let rhs: Value = zeros([p])
  for (let i = 0; i + 1 < n; i++) {
    const t = tCol(i)
    const y = at(i)
    const dx = sub(at(i + 1), y)
    const dt = (times[i + 1] ?? 0) - (times[i] ?? 0)
    rhs = add(rhs, add(add(mul(ft(t, y), dt), mul(fx(t, y), dx)), mul(fxx(t, y), mul(square(dx), 0.5))))
  }
  const lhs = sub(f(tCol(n - 1), at(n - 1)), f(tCol(0), at(0)))
  const diff = primalNumbers(sub(lhs, rhs)).map(Math.abs)
  let residual = 0
  for (const d of diff) residual = Math.max(residual, d)
  return { lhs, rhs, residual }
}

/** Both sides of the Ito isometry, as Monte Carlo estimates. */
export interface IsometryCheck {
  /** E[(int H dW)^2]. */
  readonly lhs: Estimate
  /** E[int H^2 dt]. */
  readonly rhs: Estimate
}

/**
 * The Ito isometry E[(int H dW)^2] = E[int H^2 dt] for an adapted integrand, both sides estimated from the same paths.
 *
 * @param h - The integrand on the grid `[paths, n]` (adapted: H_{t_i} may depend on W up to t_i).
 * @param w - Brownian paths `[paths, n]`.
 * @returns Both sides with their standard errors.
 * @concept ito-isometry
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { brownianMotion, itoIsometry, timeGrid } from 'whitebox-engine/stochastic'
 * const w = brownianMotion(key(0), timeGrid(1, 50), { paths: 4000 })
 * const c = itoIsometry(w.values, w)
 * Math.abs(c.lhs.value - c.rhs.value) < 4 * c.lhs.standardError // => true
 * ```
 */
export function itoIsometry(h: Paths | TensorLike, w: Paths): IsometryCheck {
  const integral = itoIntegral(h, w)
  const energy = timeIntegral(square(valuesOf(h)), w.times)
  return { lhs: sampleMean(primalNumbers(square(integral))), rhs: sampleMean(primalNumbers(energy)) }
}

/**
 * Girsanov likelihood ratios dQ/dP = exp(int theta dW - 1/2 int theta^2 dt) on paths of a P-Brownian motion: under Q the
 * process W_t - int theta ds is Brownian, so Q-expectations of functionals of a drifted Brownian motion are P-weighted
 * averages. Differentiable in theta.
 *
 * @param theta - The drift: a scalar, or values on the grid `[paths, n]` (adapted).
 * @param w - Brownian paths `[paths, n]` under P.
 * @returns The weights `[paths]` (mean one under P).
 * @concept girsanov-theorem
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { brownianMotion, girsanovWeights, timeGrid } from 'whitebox-engine/stochastic'
 * import { mean, item } from 'whitebox-engine/tensor'
 * const w = brownianMotion(key(0), timeGrid(1, 20), { paths: 5000 })
 * Math.abs(item(mean(girsanovWeights(0.5, w))) - 1) < 0.05 // => true
 * ```
 */
export function girsanovWeights(theta: TensorLike, w: Paths): Value {
  const wv = valuesOf(w)
  const n = wv.shape[1] ?? 0
  const th = asTensor(theta)
  const grid = th.shape.length === 0 ? mul(add(zeros(wv.shape), 1), th) : valuesOf(th)
  const logZ = sub(itoIntegral(grid, wv), mul(timeIntegral(square(leftPoints(grid, n, 'girsanov')), w.times), 0.5))
  return exp(logZ)
}

/**
 * A weighted Monte Carlo expectation sum w_i f_i / n (or self-normalised, sum w_i f_i / sum w_i) with its standard error,
 * for importance sampling by Girsanov weights.
 *
 * @param values - The functional per path `[paths]`.
 * @param weights - The likelihood ratios `[paths]`.
 * @param options - `selfNormalised` (default false).
 * @returns The estimate.
 * @concept girsanov-theorem
 * @example
 * ```ts
 * import { weightedExpectation } from 'whitebox-engine/stochastic'
 * weightedExpectation([1, 3], [1, 1]).value // => 2
 * weightedExpectation([1, 3], [3, 1], { selfNormalised: true }).value // => 1.5
 * ```
 */
export function weightedExpectation(
  values: TensorLike,
  weights: TensorLike,
  options: { readonly selfNormalised?: boolean } = {},
): Estimate {
  const f = primalNumbers(values)
  const w = primalNumbers(weights)
  if (f.length !== w.length || f.length === 0) {
    throw new WhiteboxError('shape', 'weightedExpectation: values and weights must have the same non-zero length')
  }
  if (options.selfNormalised !== true) return sampleMean(f.map((x, i) => x * (w[i] ?? 0)))
  const sw = w.reduce((a, b) => a + b, 0)
  const m = f.reduce((a, x, i) => a + x * (w[i] ?? 0), 0) / sw
  // Delta-method standard error of the ratio estimator.
  const v = f.reduce((a, x, i) => a + ((w[i] ?? 0) * (x - m)) ** 2, 0) / (sw * sw)
  return estimate(m, Math.sqrt(v))
}

/** The problem of {@link feynmanKac}: u(t0, x) = E[exp(-int V dt) g(X_T) + int exp(-int V) k dt | X_t0 = x]. */
export interface FeynmanKacProblem {
  /** The diffusion X. */
  readonly sde: Sde
  /** The terminal condition g(X_T), mapping states `[paths, ...state]` to `[paths]`. */
  readonly payoff: (y: Value) => Value
  /** The potential (killing or discount rate) V(t, X_t) `[paths]` (default 0). */
  readonly potential?: (t: number, y: Value) => Value
  /** The source term k(t, X_t) `[paths]` (default 0). */
  readonly source?: (t: number, y: Value) => Value
}

/** The result of {@link feynmanKac}. */
export interface FeynmanKacResult {
  /** The Monte Carlo estimate with its standard error. */
  readonly estimate: Estimate
  /** The mean as a value, differentiable in parameters of the problem (pathwise derivatives). */
  readonly value: Value
  /** The discounted functional per path `[paths]`. */
  readonly samples: Value
}

/**
 * The Feynman-Kac formula by Monte Carlo: the solution of the backward PDE u_t + L u - V u + k = 0, u(T, x) = g(x), with
 * L the generator of the diffusion, is u(t0, x) = E[exp(-int V) g(X_T) + int exp(-int V) k ds]. Paths come from
 * {@link sdeint}, the integrals from left-point sums. With V = r and geometric Brownian motion this prices options
 * (compare {@link blackScholes}).
 *
 * @param key - The key.
 * @param problem - The diffusion, terminal condition, potential and source.
 * @param x0 - The starting state `[...state]` (the same for every path).
 * @param horizon - The time span T - t0 (time runs from 0).
 * @param options - Steps (default 100), paths (default 10000) and scheme.
 * @returns The estimate, the differentiable mean and the per-path samples.
 * @concept feynman-kac
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { feynmanKac } from 'whitebox-engine/stochastic'
 * import { add, square, sum, zeros } from 'whitebox-engine/tensor'
 * // the heat equation u_t + u_xx / 2 = 0, u(T, x) = x^2 has u(0, x) = x^2 + T
 * const heat = { drift: (_t, y) => zeros(y.shape), diffusion: (_t, y) => add(zeros(y.shape), 1) }
 * const r = feynmanKac(key(0), { sde: heat, payoff: (y) => sum(square(y), 1) }, [1], 2, { steps: 4, paths: 20000 })
 * Math.abs(r.estimate.value - 3) < 4 * r.estimate.standardError // => true
 * ```
 */
export function feynmanKac(
  key: Key,
  problem: FeynmanKacProblem,
  x0: TensorLike,
  horizon: number,
  options: { readonly steps?: number; readonly paths?: number; readonly method?: SdeMethod } = {},
): FeynmanKacResult {
  const steps = positiveInteger(options.steps ?? 100, 'steps', 'feynmanKac')
  const paths = positiveInteger(options.paths ?? 10000, 'paths', 'feynmanKac')
  const one = promoteFloat(x0)[0] as Value
  const start = add(zeros([paths, ...one.shape]), one)
  const times = timeGrid(horizon, steps)
  const p = sdeint(key, problem.sde, start, times, options.method === undefined ? {} : { method: options.method })
  const v = asTensor(p.values)
  const at = (i: number): Value => squeeze(sliceAxis(v, 1, i, i + 1), 1)
  let killed: Value = zeros([paths])
  let running: Value = zeros([paths])
  for (let i = 0; i < steps; i++) {
    const t = times[i] ?? 0
    const dt = (times[i + 1] ?? 0) - t
    const y = at(i)
    if (problem.source !== undefined) running = add(running, mul(mul(exp(neg(killed)), problem.source(t, y)), dt))
    if (problem.potential !== undefined) killed = add(killed, mul(problem.potential(t, y), dt))
  }
  const samples = add(mul(exp(neg(killed)), problem.payoff(at(steps))), running)
  return { estimate: sampleMean(primalNumbers(samples)), value: mean(samples), samples }
}

/** Options of {@link blackScholes}. */
export interface BlackScholesOptions {
  /** `call` (default) or `put`. */
  readonly kind?: 'call' | 'put'
  /** The continuous dividend yield (default 0). */
  readonly dividend?: TensorLike
}

/**
 * The Black-Scholes price of a European option: the Feynman-Kac solution of the Black-Scholes PDE,
 * C = S e^{-q T} N(d1) - K e^{-r T} N(d2), d1 = (log(S / K) + (r - q + sigma^2 / 2) T) / (sigma sqrt T), d2 = d1 -
 * sigma sqrt T (puts by parity). Differentiable in every argument, so the Greeks come from autodiff.
 *
 * @param spot - The spot price S.
 * @param strike - The strike K.
 * @param maturity - The time to maturity T.
 * @param rate - The risk-free rate r.
 * @param sigma - The volatility.
 * @param options - Call or put, dividend yield.
 * @returns The price.
 * @concept black-scholes
 * @example
 * ```ts
 * import { blackScholes } from 'whitebox-engine/stochastic'
 * import { item } from 'whitebox-engine/tensor'
 * Math.round(item(blackScholes(100, 100, 1, 0.05, 0.2)) * 1e4) / 1e4 // => 10.4506
 * ```
 */
export function blackScholes(
  spot: TensorLike,
  strike: TensorLike,
  maturity: TensorLike,
  rate: TensorLike,
  sigma: TensorLike,
  options: BlackScholesOptions = {},
): Value {
  const [s, k, t, r, vol, q] = promoteFloat(spot, strike, maturity, rate, sigma, options.dividend ?? 0) as [
    Value,
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  const sd = mul(vol, sqrt(t))
  const d1 = div(add(log(div(s, k)), mul(add(sub(r, q), mul(square(vol), 0.5)), t)), sd)
  const d2 = sub(d1, sd)
  const fs = mul(s, exp(neg(mul(q, t))))
  const fk = mul(k, exp(neg(mul(r, t))))
  if ((options.kind ?? 'call') === 'call') return sub(mul(fs, ndtr(d1)), mul(fk, ndtr(d2)))
  return sub(mul(fk, ndtr(neg(d2))), mul(fs, ndtr(neg(d1))))
}
