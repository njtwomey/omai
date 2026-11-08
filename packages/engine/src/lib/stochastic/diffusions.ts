/**
 * Named diffusion processes with their closed forms where they exist (exact transition laws, conditional moments,
 * stationary laws), simulated exactly or by an SDE scheme: Ornstein-Uhlenbeck and Vasicek, geometric Brownian motion,
 * Cox-Ingersoll-Ross with their bond prices, Bessel processes, overdamped and underdamped Langevin dynamics, and jump
 * diffusions including Merton's model with its return density and option price.
 */

import { grad } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { bijectors, compose, type Distribution, gamma, logNormal, noncentralChi2, normal } from '../dists/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { foldIn, type Key, normal as normalDraws, poisson } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  concat,
  div,
  exp,
  expm1,
  fromStorage,
  log,
  logsumexp,
  maximum,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type TensorLike,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { asTimes, positiveInteger, primalNumbers, type TimesLike, traceOf } from './_shared.ts'
import { blackScholes } from './calculus.ts'
import { type Paths, simulatePaths, type TraceOptions } from './paths.ts'
import { type Sde, type SdeMethod, sdeStep } from './sde.ts'

/**
 * A named diffusion: an Ito SDE (diagonal noise) with, where they exist in closed form, its exact transition law, the
 * moments of X_t given X_0 and its stationary law. Parameters are values, so every closed form differentiates in them.
 */
export interface Diffusion extends Sde {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  /** The law of X_{t + dt} given X_t = x (batch shape of x). */
  readonly transition?: (x: Value, dt: number) => Distribution
  /** E[X_t | X_0 = x0]. */
  readonly mean?: (x0: TensorLike, t: TensorLike) => Value
  /** Var[X_t | X_0 = x0]. */
  readonly variance?: (x0: TensorLike, t: TensorLike) => Value
  /** The stationary law, when the process is ergodic. */
  readonly stationary?: () => Distribution
}

/** A diffusion with an exact transition law and closed-form conditional moments. */
export type ExactDiffusion = Diffusion & {
  readonly transition: (x: Value, dt: number) => Distribution
  readonly mean: (x0: TensorLike, t: TensorLike) => Value
  readonly variance: (x0: TensorLike, t: TensorLike) => Value
}

/** An exact diffusion with a stationary law. */
export type ErgodicDiffusion = ExactDiffusion & { readonly stationary: () => Distribution }

/** Options of {@link simulateDiffusion}. */
export interface DiffusionSimulationOptions extends TraceOptions {
  /** The number of paths (default 1). */
  readonly paths?: number
  /** Sample the exact transition (default, when the diffusion has one) or integrate the SDE. */
  readonly exact?: boolean
  /** The scheme when integrating (default `milstein`). */
  readonly method?: SdeMethod
  /** Scheme steps per grid interval when integrating (default 1). */
  readonly substeps?: number
}

/**
 * Simulate a diffusion on a grid as a traced path simulation: by its exact transition (any grid, no discretisation
 * error; reparameterised where the law allows, so paths differentiate in the parameters) or, without one, by an SDE
 * scheme.
 *
 * @param key - The key.
 * @param diffusion - The process.
 * @param x0 - The start, broadcast to `[paths, ...state]`.
 * @param times - The grid.
 * @param options - Paths, exact or scheme, substeps and tracing.
 * @returns The paths.
 * @concept diffusion-processes
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { ornsteinUhlenbeck, simulateDiffusion, timeGrid } from 'omai-engine/stochastic'
 * simulateDiffusion(key(0), ornsteinUhlenbeck({ theta: 2, mu: 1, sigma: 0.5 }), 0, timeGrid(5, 10), { paths: 3 }).values.shape // => [3, 11]
 * ```
 */
export function simulateDiffusion(
  key: Key,
  diffusion: Diffusion,
  x0: TensorLike,
  times: TimesLike,
  options: DiffusionSimulationOptions = {},
): Paths {
  const t = asTimes(times, 'simulateDiffusion')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'simulateDiffusion')
  const one = asFloat(asTensor(x0))
  const start = one.shape[0] === paths && one.shape.length >= 1 ? one : add(zeros([paths, ...one.shape]), one)
  const exact = (options.exact ?? true) && diffusion.transition !== undefined
  const method = options.method ?? 'milstein'
  const substeps = positiveInteger(options.substeps ?? 1, 'substeps', 'simulateDiffusion')
  const transition = diffusion.transition
  return simulatePaths(
    key,
    {
      name: `${diffusion.name}${exact ? '-exact' : `-${method}`}`,
      config: { paths, steps: t.length - 1, exact },
      times: t,
      x0: start,
      step: (x, c) => {
        if (exact && transition !== undefined) {
          const law = transition(x, c.dt)
          return law.hasRsample ? law.rsample(c.key) : law.sample(c.key)
        }
        const h = c.dt / substeps
        let y = x
        for (let s = 0; s < substeps; s++) {
          const dw = mul(normalDraws(foldIn(c.key, s), y.shape), Math.sqrt(h))
          y = sdeStep(diffusion, method, c.t + s * h, y, h, dw)
        }
        return y
      },
    },
    traceOf(options),
  )
}

/** Parameters of {@link ornsteinUhlenbeck}. */
export interface OuParams {
  /** The mean-reversion rate (positive). */
  readonly theta: TensorLike
  /** The long-run mean (default 0). */
  readonly mu?: TensorLike
  /** The volatility (positive). */
  readonly sigma: TensorLike
}

/**
 * The Ornstein-Uhlenbeck process dX = theta (mu - X) dt + sigma dW: Gaussian transitions
 * N(mu + (x - mu) e^{-theta dt}, sigma^2 (1 - e^{-2 theta dt}) / (2 theta)) and stationary law N(mu, sigma^2 / (2 theta)).
 *
 * @param params - Rate, mean and volatility.
 * @returns The diffusion.
 * @concept ornstein-uhlenbeck
 * @example
 * ```ts
 * import { ornsteinUhlenbeck } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(ornsteinUhlenbeck({ theta: 0.5, sigma: 1 }).stationary().variance()) // => 1
 * ```
 */
export function ornsteinUhlenbeck(params: OuParams): ErgodicDiffusion {
  const [theta, mu, sigma] = promoteFloat(params.theta, params.mu ?? 0, params.sigma) as [Value, Value, Value]
  const mean = (x0: TensorLike, t: TensorLike): Value => add(mu, mul(sub(x0, mu), exp(neg(mul(theta, t)))))
  const variance = (_x0: TensorLike, t: TensorLike): Value =>
    div(mul(square(sigma), neg(expm1(neg(mul(mul(theta, t), 2))))), mul(theta, 2))
  return {
    name: 'ornstein-uhlenbeck',
    params: { theta, mu, sigma },
    drift: (_t, x) => mul(theta, sub(mu, x)),
    diffusion: (_t, x) => add(zeros(x.shape), sigma),
    transition: (x, dt) => normal(mean(x, dt), sqrt(variance(x, dt))),
    mean,
    variance,
    stationary: () => normal(mu, div(sigma, sqrt(mul(theta, 2)))),
  }
}

/** Parameters of {@link vasicek}. */
export interface VasicekParams {
  /** The speed of reversion a (positive). */
  readonly a: TensorLike
  /** The long-run rate b. */
  readonly b: TensorLike
  /** The volatility (positive). */
  readonly sigma: TensorLike
}

/**
 * The Vasicek short-rate model dr = a (b - r) dt + sigma dW: an Ornstein-Uhlenbeck process, with the zero-coupon bond
 * price of {@link vasicekBondPrice}.
 *
 * @param params - Speed, long-run rate and volatility.
 * @returns The diffusion.
 * @concept short-rate-models
 * @example
 * ```ts
 * import { vasicek } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(vasicek({ a: 1, b: 0.05, sigma: 0.01 }).stationary().mean()) // => 0.05
 * ```
 */
export function vasicek(params: VasicekParams): ErgodicDiffusion {
  const ou = ornsteinUhlenbeck({ theta: params.a, mu: params.b, sigma: params.sigma })
  return {
    ...ou,
    name: 'vasicek',
    params: { a: ou.params['theta'] as Value, b: ou.params['mu'] as Value, sigma: ou.params['sigma'] as Value },
  }
}

/**
 * The Vasicek zero-coupon bond price P(0, T) = A e^{-B r0}, B = (1 - e^{-a T}) / a,
 * log A = (b - sigma^2 / (2 a^2))(B - T) - sigma^2 B^2 / (4 a): the Feynman-Kac solution E[exp(-int r dt)].
 * Differentiable in every argument.
 *
 * @param r0 - The current short rate.
 * @param maturity - The maturity T.
 * @param params - Speed, long-run rate and volatility.
 * @returns The price.
 * @concept short-rate-models
 * @example
 * ```ts
 * import { vasicekBondPrice } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(vasicekBondPrice(0.05, 1, { a: 1, b: 0.05, sigma: 1e-8 })) - Math.exp(-0.05)) < 1e-12 // => true
 * ```
 */
export function vasicekBondPrice(r0: TensorLike, maturity: TensorLike, params: VasicekParams): Value {
  const [r, t, a, b, sigma] = promoteFloat(r0, maturity, params.a, params.b, params.sigma) as [
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  const big = div(neg(expm1(neg(mul(a, t)))), a)
  const s2 = square(sigma)
  const logA = sub(mul(sub(b, div(s2, mul(square(a), 2))), sub(big, t)), div(mul(s2, square(big)), mul(a, 4)))
  return exp(sub(logA, mul(big, r)))
}

/** Parameters of {@link geometricBrownianMotion}. */
export interface GbmParams {
  /** The drift mu. */
  readonly mu: TensorLike
  /** The volatility (positive). */
  readonly sigma: TensorLike
}

/**
 * Geometric Brownian motion dS = mu S dt + sigma S dW: log-normal transitions, S_{t+dt} = S_t exp((mu - sigma^2 / 2) dt
 * + sigma sqrt(dt) Z), mean S_0 e^{mu t} and variance S_0^2 e^{2 mu t}(e^{sigma^2 t} - 1).
 *
 * @param params - Drift and volatility.
 * @returns The diffusion (no stationary law).
 * @concept geometric-brownian-motion
 * @example
 * ```ts
 * import { geometricBrownianMotion } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(geometricBrownianMotion({ mu: 0, sigma: 0.2 }).mean(3, 1)) // => 3
 * ```
 */
export function geometricBrownianMotion(params: GbmParams): ExactDiffusion {
  const [mu, sigma] = promoteFloat(params.mu, params.sigma) as [Value, Value]
  return {
    name: 'geometric-brownian-motion',
    params: { mu, sigma },
    drift: (_t, x) => mul(mu, x),
    diffusion: (_t, x) => mul(sigma, x),
    transition: (x, dt) => logNormal(add(log(x), mul(sub(mu, mul(square(sigma), 0.5)), dt)), mul(sigma, Math.sqrt(dt))),
    mean: (x0, t) => mul(x0, exp(mul(mu, t))),
    variance: (x0, t) => mul(mul(square(asTensor(x0)), exp(mul(mul(mu, t), 2))), expm1(mul(square(sigma), t))),
  }
}

/** Parameters of {@link cir}. */
export interface CirParams {
  /** The speed of reversion kappa (positive). */
  readonly kappa: TensorLike
  /** The long-run mean theta (positive). */
  readonly theta: TensorLike
  /** The volatility (positive). */
  readonly sigma: TensorLike
}

/**
 * The Cox-Ingersoll-Ross process dX = kappa (theta - X) dt + sigma sqrt(X) dW. Its transition is a scaled noncentral
 * chi-square, X_{t+dt} = c Y with c = sigma^2 (1 - e^{-kappa dt}) / (4 kappa), Y ~ chi'^2(4 kappa theta / sigma^2,
 * x e^{-kappa dt} / c), sampled exactly (reparameterised implicitly); the stationary law is
 * Gamma(2 kappa theta / sigma^2, rate 2 kappa / sigma^2). The process stays positive when 2 kappa theta >= sigma^2
 * (Feller). The diffusion coefficient uses sqrt(max(X, 0)) so schemes survive overshoots.
 *
 * @param params - Speed, mean and volatility.
 * @returns The diffusion.
 * @concept cox-ingersoll-ross
 * @example
 * ```ts
 * import { cir } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * Math.round(item(cir({ kappa: 2, theta: 0.04, sigma: 0.2 }).stationary().mean()) * 1e12) / 1e12 // => 0.04
 * ```
 */
export function cir(params: CirParams): ErgodicDiffusion {
  const [kappa, theta, sigma] = promoteFloat(params.kappa, params.theta, params.sigma) as [Value, Value, Value]
  const decay = (t: TensorLike): Value => exp(neg(mul(kappa, t)))
  const scale = (t: TensorLike): Value => div(mul(square(sigma), neg(expm1(neg(mul(kappa, t))))), mul(kappa, 4))
  const df = div(mul(mul(kappa, theta), 4), square(sigma))
  return {
    name: 'cir',
    params: { kappa, theta, sigma },
    drift: (_t, x) => mul(kappa, sub(theta, x)),
    diffusion: (_t, x) => mul(sigma, sqrt(maximum(x, 0))),
    transition: (x, dt) => {
      const c = scale(dt)
      return compose.transformed(
        noncentralChi2(add(zeros(x.shape), df), div(mul(x, decay(dt)), c)),
        bijectors.affine(0, c),
      )
    },
    mean: (x0, t) => add(theta, mul(sub(x0, theta), decay(t))),
    variance: (x0, t) => {
      const e = decay(t)
      const s2k = div(square(sigma), kappa)
      return add(mul(mul(x0, s2k), sub(e, square(e))), mul(mul(theta, div(s2k, 2)), square(sub(1, e))))
    },
    stationary: () => gamma(div(mul(mul(kappa, theta), 2), square(sigma)), div(mul(kappa, 2), square(sigma))),
  }
}

/**
 * The CIR zero-coupon bond price P(0, T) = A e^{-B r0} with h = sqrt(kappa^2 + 2 sigma^2),
 * B = 2 (e^{hT} - 1) / (2h + (kappa + h)(e^{hT} - 1)), A = (2h e^{(kappa + h) T / 2} / (2h + (kappa + h)(e^{hT} - 1)))^{2
 * kappa theta / sigma^2}. Differentiable in every argument.
 *
 * @param r0 - The current short rate.
 * @param maturity - The maturity T.
 * @param params - Speed, mean and volatility.
 * @returns The price.
 * @concept short-rate-models
 * @example
 * ```ts
 * import { cirBondPrice } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(cirBondPrice(0.05, 1, { kappa: 1, theta: 0.05, sigma: 0.1 })) < Math.exp(-0.05) + 1e-3 // => true
 * ```
 */
export function cirBondPrice(r0: TensorLike, maturity: TensorLike, params: CirParams): Value {
  const [r, t, kappa, theta, sigma] = promoteFloat(r0, maturity, params.kappa, params.theta, params.sigma) as [
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  const s2 = square(sigma)
  const h = sqrt(add(square(kappa), mul(s2, 2)))
  const em1 = expm1(mul(h, t))
  const denom = add(mul(h, 2), mul(add(kappa, h), em1))
  const big = div(mul(em1, 2), denom)
  const logA = mul(
    div(mul(mul(kappa, theta), 2), s2),
    sub(add(log(mul(h, 2)), mul(mul(add(kappa, h), t), 0.5)), log(denom)),
  )
  return exp(sub(logA, mul(big, r)))
}

/**
 * The Bessel process of dimension delta, dR = (delta - 1) / (2 R) dt + dW: the norm of a delta-dimensional Brownian
 * motion for integer delta. Its exact transition is R_{t+dt}^2 / dt ~ chi'^2(delta, R_t^2 / dt) (a squared Bessel
 * process is a scaled noncentral chi-square), sampled exactly.
 *
 * @param params - The dimension `delta` (positive; the origin is polar for delta >= 2).
 * @returns The diffusion on (0, infinity), with E[R_t^2 | R_0 = r0] = r0^2 + delta t as `meanSquare`.
 * @concept bessel-process
 * @example
 * ```ts
 * import { besselProcess } from 'omai-engine/stochastic'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(besselProcess({ delta: 3 }).drift(0, tensor(2))) // => 0.5
 * ```
 */
export function besselProcess(params: { readonly delta: TensorLike }): Diffusion & {
  readonly transition: (x: Value, dt: number) => Distribution
  readonly meanSquare: (x0: TensorLike, t: TensorLike) => Value
} {
  const [delta] = promoteFloat(params.delta) as [Value]
  const square2 = (x: Value, dt: number): Distribution =>
    compose.transformed(noncentralChi2(add(zeros(x.shape), delta), div(square(x), dt)), bijectors.affine(0, dt))
  return {
    name: 'bessel',
    params: { delta },
    drift: (_t, x) => div(mul(sub(delta, 1), 0.5), x),
    diffusion: (_t, x) => add(zeros(x.shape), 1),
    transition: (x, dt) => compose.transformed(square2(x, dt), bijectors.power(0.5)),
    // E[R_t^2] = r0^2 + delta t (the mean of R itself needs a confluent hypergeometric function).
    meanSquare: (x0, t) => add(square(asTensor(x0)), mul(delta, t)),
  }
}

/** Parameters of {@link langevinDiffusion}. */
export interface LangevinParams {
  /** The potential U(x): states `[paths, d]` to energies `[paths]`. */
  readonly potential: (x: Value) => Value
  /** The inverse temperature beta (default 1). */
  readonly beta?: TensorLike
}

function gradientOf(potential: (x: Value) => Value): (x: Value) => Value {
  return (x) => grad((z: Value) => sum(potential(z)))(x)
}

/**
 * Overdamped Langevin dynamics dX = -grad U(X) dt + sqrt(2 / beta) dW on states `[paths, d]`, whose stationary density
 * is proportional to exp(-beta U) (the gradient comes from autodiff). For a quadratic U it is an Ornstein-Uhlenbeck
 * process; otherwise simulate it with a scheme.
 *
 * @param params - The potential and inverse temperature.
 * @returns The diffusion and its unnormalised stationary log density.
 * @concept langevin-dynamics
 * @example
 * ```ts
 * import { langevinDiffusion } from 'omai-engine/stochastic'
 * import { mul, sum, square, tensor, toArray } from 'omai-engine/tensor'
 * const l = langevinDiffusion({ potential: (x) => mul(sum(square(x), 1), 0.5) })
 * toArray(l.drift(0, tensor([[1, -2]]))) // => [[-1, 2]]
 * ```
 */
export function langevinDiffusion(params: LangevinParams): Diffusion & { logStationary: (x: TensorLike) => Value } {
  const [beta] = promoteFloat(params.beta ?? 1) as [Value]
  const gradient = gradientOf(params.potential)
  return {
    name: 'langevin',
    params: { beta },
    drift: (_t, x) => neg(gradient(x)),
    diffusion: (_t, x) => add(zeros(x.shape), sqrt(div(2, beta))),
    logStationary: (x) => neg(mul(beta, params.potential(asFloat(asTensor(x))))),
  }
}

/** Parameters of {@link underdampedLangevin}. */
export interface UnderdampedParams extends LangevinParams {
  /** The friction gamma (positive, default 1). */
  readonly friction?: TensorLike
}

/**
 * Underdamped (kinetic) Langevin dynamics on states `[paths, 2 d]` (positions then velocities): dx = v dt,
 * dv = (-gamma v - grad U(x)) dt + sqrt(2 gamma / beta) dW, with stationary density proportional to
 * exp(-beta (U(x) + |v|^2 / 2)).
 *
 * @param params - The potential, inverse temperature and friction.
 * @returns The diffusion (diagonal noise, zero on the positions).
 * @concept langevin-dynamics
 * @example
 * ```ts
 * import { underdampedLangevin } from 'omai-engine/stochastic'
 * import { mul, sum, square, tensor, toArray } from 'omai-engine/tensor'
 * const l = underdampedLangevin({ potential: (x) => mul(sum(square(x), 1), 0.5), friction: 2 })
 * toArray(l.drift(0, tensor([[1, 3]]))) // => [[3, -7]]
 * ```
 */
export function underdampedLangevin(params: UnderdampedParams): Diffusion {
  const [beta, gamma] = promoteFloat(params.beta ?? 1, params.friction ?? 1) as [Value, Value]
  const gradient = gradientOf(params.potential)
  const halves = (x: Value): [Value, Value] => {
    const d2 = x.shape[x.shape.length - 1] ?? 0
    if (d2 % 2 !== 0) throw new OmaiError('shape', 'underdampedLangevin: states must be [paths, 2 d]')
    return [sliceAxis(x, -1, 0, d2 / 2), sliceAxis(x, -1, d2 / 2, d2)]
  }
  return {
    name: 'underdamped-langevin',
    params: { beta, friction: gamma },
    drift: (_t, x) => {
      const [q, v] = halves(x)
      return concat([v, sub(neg(mul(gamma, v)), gradient(q))], -1)
    },
    diffusion: (_t, x) => {
      const [q, v] = halves(x)
      return concat([zeros(q.shape), add(zeros(v.shape), sqrt(div(mul(gamma, 2), beta)))], -1)
    },
  }
}

/**
 * A jump diffusion: a diffusion plus jumps at the times of a Poisson process, each moving the state by a draw of
 * `jumpSize` (added, or multiplied by its exponential when `multiplicative`).
 */
export interface JumpDiffusion {
  readonly name: string
  /** The continuous part. */
  readonly sde: Sde
  /** The jump intensity per path. */
  readonly rate: number
  /** The law of one jump size (scalar). */
  readonly jumpSize: Distribution
  /** Whether a jump J maps x to x e^J (default false: x + J). */
  readonly multiplicative?: boolean
  /** An exact transition sampler (key, x, dt) => X_{t + dt}, used when present. */
  readonly exactStep?: (key: Key, x: Value, dt: number) => Value
}

function jumpTotals(key: Key, jd: JumpDiffusion, shape: readonly number[], dt: number): Value {
  const counts = poisson(foldIn(key, 0), jd.rate * dt, shape).data
  const size = counts.reduce((a, b) => a + b, 0)
  const out = new Float64Array(counts.length)
  if (size > 0) {
    const draws = primalNumbers(jd.jumpSize.sample(foldIn(key, 1), [size]))
    let at = 0
    for (let i = 0; i < counts.length; i++) {
      for (let k = 0; k < (counts[i] ?? 0); k++) out[i] = (out[i] ?? 0) + (draws[at++] ?? 0)
    }
  }
  return fromStorage(out, [...shape], 'float64')
}

/**
 * Simulate a jump diffusion as a traced path simulation: per step the diffusion moves by an SDE scheme step, then the
 * jumps that arrived in the step (a Poisson count per path and coordinate) are applied; with `exactStep` the exact
 * transition is used instead.
 *
 * @param key - The key.
 * @param jd - The process.
 * @param x0 - The start, broadcast to `[paths, ...state]`.
 * @param times - The grid.
 * @param options - Paths, scheme (default `euler`) and tracing.
 * @returns The paths.
 * @concept jump-diffusion
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { simulateJumpDiffusion, timeGrid } from 'omai-engine/stochastic'
 * import { zeros } from 'omai-engine/tensor'
 * const still = { drift: (_t, y) => zeros(y.shape), diffusion: (_t, y) => zeros(y.shape) }
 * const jd = { name: 'jumps', sde: still, rate: 2, jumpSize: normal(1, 0.1) }
 * simulateJumpDiffusion(key(0), jd, 0, timeGrid(1, 4), { paths: 2 }).values.shape // => [2, 5]
 * ```
 */
export function simulateJumpDiffusion(
  key: Key,
  jd: JumpDiffusion,
  x0: TensorLike,
  times: TimesLike,
  options: { readonly paths?: number; readonly method?: SdeMethod } & TraceOptions = {},
): Paths {
  const t = asTimes(times, 'simulateJumpDiffusion')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'simulateJumpDiffusion')
  if (!(jd.rate >= 0)) throw new OmaiError('value', 'simulateJumpDiffusion: the rate must be non-negative')
  const one = asFloat(asTensor(x0))
  const start = add(zeros([paths, ...one.shape]), one)
  const method = options.method ?? 'euler'
  const exactStep = jd.exactStep
  return simulatePaths(
    key,
    {
      name: jd.name,
      config: { paths, steps: t.length - 1, rate: jd.rate },
      times: t,
      x0: start,
      step: (x, c) => {
        if (exactStep !== undefined) return exactStep(c.key, x, c.dt)
        const dw = mul(normalDraws(foldIn(c.key, 0), x.shape), Math.sqrt(c.dt))
        const y = sdeStep(jd.sde, method, c.t, x, c.dt, dw)
        const jumps = jumpTotals(foldIn(c.key, 1), jd, x.shape, c.dt)
        return jd.multiplicative === true ? mul(y, exp(jumps)) : add(y, jumps)
      },
    },
    traceOf(options),
  )
}

/** Parameters of {@link mertonJumpDiffusion}. */
export interface MertonParams {
  /** The drift mu (the expected return, jumps compensated). */
  readonly mu: number
  /** The diffusion volatility. */
  readonly sigma: number
  /** The jump intensity lambda. */
  readonly rate: number
  /** The mean m of the log jump size. */
  readonly jumpMean: number
  /** The standard deviation delta of the log jump size. */
  readonly jumpStd: number
}

/**
 * Merton's jump diffusion dS / S = (mu - lambda k) dt + sigma dW + (e^J - 1) dN with J ~ N(m, delta^2) and
 * k = E[e^J - 1], simulated exactly in log space (a normal for the diffusion plus N(n m, n delta^2) for n ~ Poisson(lambda
 * dt) jumps), with E[S_t] = S_0 e^{mu t} and the log-return density of {@link mertonLogReturnDensity}.
 *
 * @param params - Drift, volatility, intensity and log-jump mean and standard deviation.
 * @returns The jump diffusion.
 * @concept jump-diffusion
 * @example
 * ```ts
 * import { mertonJumpDiffusion } from 'omai-engine/stochastic'
 * mertonJumpDiffusion({ mu: 0.05, sigma: 0.2, rate: 1, jumpMean: -0.1, jumpStd: 0.1 }).multiplicative // => true
 * ```
 */
export function mertonJumpDiffusion(params: MertonParams): JumpDiffusion {
  const { mu, sigma, rate, jumpMean, jumpStd } = params
  if (!(sigma >= 0 && rate >= 0 && jumpStd >= 0)) {
    throw new OmaiError('value', 'mertonJumpDiffusion: sigma, rate and jumpStd must be non-negative')
  }
  const k = Math.exp(jumpMean + jumpStd ** 2 / 2) - 1
  const driftLog = mu - rate * k - sigma ** 2 / 2
  return {
    name: 'merton-jump-diffusion',
    sde: { drift: (_t, y) => mul(y, mu - rate * k), diffusion: (_t, y) => mul(y, sigma) },
    rate,
    jumpSize: normal(jumpMean, Math.max(jumpStd, Number.MIN_VALUE)),
    multiplicative: true,
    exactStep: (key, x, dt) => {
      const counts = poisson(foldIn(key, 0), rate * dt, x.shape).data
      const z = normalDraws(foldIn(key, 1), x.shape).data
      const u = normalDraws(foldIn(key, 2), x.shape).data
      const out = new Float64Array(z.length)
      for (let i = 0; i < z.length; i++) {
        const n = counts[i] ?? 0
        out[i] =
          driftLog * dt + sigma * Math.sqrt(dt) * (z[i] ?? 0) + n * jumpMean + Math.sqrt(n) * jumpStd * (u[i] ?? 0)
      }
      return mul(x, exp(fromStorage(out, [...x.shape], 'float64')))
    },
  }
}

function mertonTerms(rate: number, t: number): number {
  return Math.ceil(rate * t + 12 * Math.sqrt(rate * t + 1) + 20)
}

/**
 * The density of the Merton log return log(S_t / S_0) over a horizon t: the Poisson(lambda t) mixture of normals
 * N((mu - lambda k - sigma^2 / 2) t + n m, sigma^2 t + n delta^2), summed over the terms that carry the weight.
 * Differentiable in the return.
 *
 * @param r - Log returns.
 * @param t - The horizon (positive number).
 * @param params - The Merton parameters.
 * @returns The densities.
 * @concept jump-diffusion
 * @example
 * ```ts
 * import { mertonLogReturnDensity } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * const p = { mu: 0, sigma: 1, rate: 0, jumpMean: 0, jumpStd: 0.1 }
 * Math.abs(item(mertonLogReturnDensity(-0.5, 1, p)) - 1 / Math.sqrt(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function mertonLogReturnDensity(r: TensorLike, t: number, params: MertonParams): Value {
  const { mu, sigma, rate, jumpMean, jumpStd } = params
  const k = Math.exp(jumpMean + jumpStd ** 2 / 2) - 1
  const x = asFloat(asTensor(r))
  const terms = rate === 0 ? 1 : mertonTerms(rate, t)
  const parts: Value[] = []
  for (let n = 0; n < terms; n++) {
    const logW = rate === 0 ? 0 : n * Math.log(rate * t) - rate * t - (lgammaOf(n + 1) ?? 0)
    const m = (mu - rate * k - sigma ** 2 / 2) * t + n * jumpMean
    const v = sigma ** 2 * t + n * jumpStd ** 2
    parts.push(add(sub(mul(square(sub(x, m)), -0.5 / v), 0.5 * Math.log(2 * Math.PI * v)), logW))
  }
  return exp(logsumexp(stack(parts, 0), 0))
}

function lgammaOf(x: number): number | undefined {
  return primalNumbers(lgamma(x))[0]
}

/**
 * Merton's price of a European option under the jump diffusion: the Poisson(lambda' T) mixture of Black-Scholes prices,
 * lambda' = lambda (1 + k), with rate r - lambda k + n log(1 + k) / T and variance sigma^2 + n delta^2 / T for n jumps.
 *
 * @param spot - The spot price.
 * @param strike - The strike.
 * @param maturity - The maturity (positive number).
 * @param rate - The risk-free rate.
 * @param params - Volatility, intensity and log-jump moments (the drift is ignored: pricing is risk-neutral).
 * @param kind - `call` (default) or `put`.
 * @returns The price.
 * @concept jump-diffusion
 * @example
 * ```ts
 * import { blackScholes, mertonPrice } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * const p = { mu: 0, sigma: 0.2, rate: 0, jumpMean: 0, jumpStd: 0 }
 * Math.abs(item(mertonPrice(100, 100, 1, 0.05, p)) - item(blackScholes(100, 100, 1, 0.05, 0.2))) < 1e-12 // => true
 * ```
 */
export function mertonPrice(
  spot: TensorLike,
  strike: TensorLike,
  maturity: number,
  rate: number,
  params: MertonParams,
  kind: 'call' | 'put' = 'call',
): Value {
  const { sigma, rate: lambda, jumpMean, jumpStd } = params
  const k = Math.exp(jumpMean + jumpStd ** 2 / 2) - 1
  const lp = lambda * (1 + k)
  const terms = lambda === 0 ? 1 : mertonTerms(lp, maturity)
  let total: Value = mul(asFloat(asTensor(spot)), 0)
  for (let n = 0; n < terms; n++) {
    const w = lambda === 0 ? 1 : Math.exp(n * Math.log(lp * maturity) - lp * maturity - (lgammaOf(n + 1) ?? 0))
    const rn = rate - lambda * k + (n * Math.log(1 + k)) / maturity
    const vol = Math.sqrt(sigma ** 2 + (n * jumpStd ** 2) / maturity)
    total = add(total, mul(blackScholes(spot, strike, maturity, rn, vol, { kind }), w))
  }
  return total
}
