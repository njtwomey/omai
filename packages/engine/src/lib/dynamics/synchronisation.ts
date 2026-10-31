/**
 * Synchronisation of coupled oscillators: the Kuramoto model (phases with their own frequencies pulled towards one
 * another with strength K, all-to-all or along the edges of a graph), the order parameter that measures how far they
 * agree, the frequency populations (Lorentzian, Gaussian) whose critical coupling and ordered state are known in
 * closed form, a stepped integration that logs the order parameter and the locked fraction, and consensus, the linear
 * cousin whose agreement rate is the graph's algebraic connectivity.
 */

import { MdmlError } from '../base/index.ts'
import { fixedQuad } from '../numerics/quadrature/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { cauchy, type Key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  cos,
  exp,
  expandDims,
  item,
  mul,
  sin,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, type Contract, defineContract, scalar, tensor as tensorSource } from '../trace/index.ts'
import { type FlowState, flowAlgorithm } from './ode/flow.ts'
import type { VectorField } from './ode/problem.ts'

/** How the coupling is scaled. */
export type KuramotoNormalisation = 'n' | 'degree' | 'none'

/** The Kuramoto model: natural frequencies, the coupling strength, and the network (all-to-all when absent). */
export interface KuramotoOptions {
  /** The natural frequencies, one per oscillator. */
  readonly omega: readonly number[]
  /** The coupling strength K. */
  readonly K: number
  /** The adjacency matrix `[n, n]` (default: every pair coupled). */
  readonly adjacency?: TensorLike
  /** `n` divides by the number of oscillators (the classical mean field, default for all-to-all), `degree` by each oscillator's degree, `none` leaves K as the strength of each edge (default on a given network). */
  readonly normalisation?: KuramotoNormalisation
}

/**
 * The Kuramoto equations theta_i' = omega_i + (K / n) sum_j A_ij sin(theta_j - theta_i): each oscillator turns at its own
 * frequency and is pulled towards the phases of those it is coupled to. Above a critical coupling a cluster of oscillators
 * locks together and the order parameter leaves zero.
 *
 * @param options - Frequencies, coupling, network and normalisation.
 * @returns The field on the phases `[n]`.
 * @throws MdmlError with code `shape` when the adjacency is not `[n, n]`.
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { kuramotoField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * // Two oscillators a quarter turn apart, K = 2: each is pulled towards the other at K / 2 = 1.
 * toFlat(kuramotoField({ omega: [0, 0], K: 2 })(0, tensor([0, Math.PI / 2]))) // => [1, -1]
 * ```
 */
export function kuramotoField(options: KuramotoOptions): VectorField {
  const n = options.omega.length
  const omega = tensor([...options.omega])
  const adjacency = options.adjacency === undefined ? undefined : (asTensor(options.adjacency) as Tensor)
  if (
    adjacency !== undefined &&
    (adjacency.shape.length !== 2 || adjacency.shape[0] !== n || adjacency.shape[1] !== n)
  ) {
    throw new MdmlError('shape', `kuramotoField: the adjacency must be [${n}, ${n}]`)
  }
  const normalisation = options.normalisation ?? (adjacency === undefined ? 'n' : 'none')
  const degrees =
    adjacency === undefined
      ? undefined
      : Array.from({ length: n }, (_, i) =>
          Array.from(adjacency.data.slice(i * n, (i + 1) * n)).reduce((a, b) => a + b, 0),
        )
  const scale = tensor(
    Array.from({ length: n }, (_, i) =>
      normalisation === 'n'
        ? options.K / n
        : normalisation === 'degree'
          ? options.K / Math.max(1, degrees?.[i] ?? n)
          : options.K,
    ),
  )
  return (_t, theta) => {
    // sum_j sin(theta_j - theta_i). All-to-all it is the mean-field form cos(theta_i) S - sin(theta_i) C with
    // S = sum sin(theta_j) and C = sum cos(theta_j), linear in n; on a network it needs every pair.
    const pull =
      adjacency === undefined
        ? sub(mul(cos(theta), sum(sin(theta))), mul(sin(theta), sum(cos(theta))))
        : sum(mul(adjacency, sin(sub(expandDims(theta, 0), expandDims(theta, 1)))), 1)
    return add(omega, mul(pull, scale))
  }
}

/** The order parameter r exp(i psi) = (1 / n) sum exp(i theta_j): its length and phase. */
export interface OrderParameter {
  /** 0 for phases spread evenly round the circle, 1 for phases all equal. */
  readonly r: number
  /** The mean phase. */
  readonly psi: number
}

/**
 * The Kuramoto order parameter of a set of phases, the centroid of the points exp(i theta) on the unit circle.
 *
 * @param theta - The phases.
 * @returns Its length r (the coherence) and angle psi (the mean phase).
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { orderParameter } from 'mdml-engine/dynamics'
 * orderParameter([0, Math.PI]).r < 1e-15 // => true
 * ```
 */
export function orderParameter(theta: ArrayLike<number>): OrderParameter {
  let c = 0
  let s = 0
  for (let j = 0; j < theta.length; j++) {
    c += Math.cos(theta[j] ?? 0)
    s += Math.sin(theta[j] ?? 0)
  }
  return { r: Math.hypot(c, s) / theta.length, psi: Math.atan2(s, c) }
}

/** The distribution of natural frequencies, centred at 0. */
export type FrequencyDistribution = 'lorentzian' | 'gaussian'

/**
 * A population of natural frequencies: the quantiles of the distribution at (i + 1/2) / n when no key is given (a
 * deterministic, symmetric, ordered population), or independent draws from it when a key is.
 *
 * @param n - The number of oscillators.
 * @param distribution - `lorentzian` (half-width `spread`) or `gaussian` (standard deviation `spread`).
 * @param spread - The width.
 * @param key - A random key for independent draws.
 * @returns The frequencies.
 * @throws MdmlError with code `value` for a non-positive width or count.
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { kuramotoFrequencies } from 'mdml-engine/dynamics'
 * kuramotoFrequencies(3, 'lorentzian', 1).map((w) => Math.round(w * 1e6) / 1e6) // => [-1.732051, 0, 1.732051]
 * ```
 */
export function kuramotoFrequencies(
  n: number,
  distribution: FrequencyDistribution,
  spread: number,
  key?: Key,
): number[] {
  if (!(n >= 1) || !(spread > 0)) throw new MdmlError('value', 'kuramotoFrequencies: need n >= 1 and a positive spread')
  if (key !== undefined) {
    const draws = distribution === 'lorentzian' ? cauchy(key, [n], 0, spread) : normal(key, [n], { scale: spread })
    return Array.from(draws.data)
  }
  return Array.from({ length: n }, (_, i) => {
    const u = (i + 0.5) / n
    return distribution === 'lorentzian' ? spread * Math.tan(Math.PI * (u - 0.5)) : spread * item(ndtri(u))
  })
}

const density = (distribution: FrequencyDistribution, spread: number, w: number): number =>
  distribution === 'lorentzian'
    ? spread / (Math.PI * (spread * spread + w * w))
    : Math.exp(-(w * w) / (2 * spread * spread)) / (spread * Math.sqrt(2 * Math.PI))

/**
 * The critical coupling K_c = 2 / (pi g(0)) of an infinite population with frequency density g, symmetric and unimodal:
 * 2 gamma for the Lorentzian of half-width gamma, sigma sqrt(8 / pi) for the Gaussian. Below it the incoherent state is
 * stable (r = 0); above it a cluster locks.
 *
 * @param distribution - The family.
 * @param spread - Half-width or standard deviation.
 * @returns K_c.
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { kuramotoCriticalCoupling } from 'mdml-engine/dynamics'
 * kuramotoCriticalCoupling('lorentzian', 0.5) // => 1
 * ```
 */
export function kuramotoCriticalCoupling(distribution: FrequencyDistribution, spread: number): number {
  return 2 / (Math.PI * density(distribution, spread, 0))
}

/**
 * The order parameter of the stationary state of an infinite population at coupling K, from the self-consistency
 * condition r = K r integral cos^2(theta) g(K r sin theta) d theta over (-pi/2, pi/2): zero up to K_c, then
 * sqrt(1 - K_c / K) exactly for the Lorentzian (and a square-root rise near K_c for any smooth g). The Gaussian's
 * condition is solved by bisection on r.
 *
 * @param K - The coupling.
 * @param distribution - The family.
 * @param spread - Half-width or standard deviation.
 * @returns r in [0, 1).
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { kuramotoTheoreticalOrder } from 'mdml-engine/dynamics'
 * kuramotoTheoreticalOrder(4, 'lorentzian', 1) // => 0.7071067811865476
 * ```
 */
export function kuramotoTheoreticalOrder(K: number, distribution: FrequencyDistribution, spread: number): number {
  const critical = kuramotoCriticalCoupling(distribution, spread)
  if (K <= critical) return 0
  if (distribution === 'lorentzian') return Math.sqrt(1 - critical / K)
  // 1 = K * integral cos^2 g(K r sin) : decreasing in r, K * integral(r = 0) = K / K_c > 1, so there is a root in (0, 1).
  const integral = (r: number): number =>
    item(
      fixedQuad(
        (x) => mul(mul(cos(x), cos(x)), gaussianDensity(mul(sin(x), K * r), spread)),
        -Math.PI / 2,
        Math.PI / 2,
        40,
      ),
    )
  let lo = 1e-12
  let hi = 1
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi)
    if (K * integral(mid) > 1) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}

const gaussianDensity = (w: Value, spread: number): Value =>
  mul(exp(mul(mul(w, w), -1 / (2 * spread * spread))), 1 / (spread * Math.sqrt(2 * Math.PI)))

/** The state of {@link kuramotoAlgorithm}. */
export type KuramotoState = FlowState

/** The paths of {@link kuramotoAlgorithm}. */
export type KuramotoPath =
  | 'oscillators/t'
  | 'oscillators/phases'
  | 'oscillators/order'
  | 'oscillators/meanPhase'
  | 'oscillators/locked'

/** What a Kuramoto run logs. */
export const kuramotoContract: Contract<FlowState, KuramotoPath> = defineContract({
  'oscillators/t': scalar({ description: 'time' }),
  'oscillators/phases': tensorSource({ role: 'state', shape: ['n'], description: 'the phases, wrapped to (-pi, pi]' }),
  'oscillators/order': scalar({ description: 'the order parameter r, the coherence of the phases' }),
  'oscillators/meanPhase': scalar({ description: 'the order parameter’s angle psi' }),
  'oscillators/locked': scalar({ description: 'the share of oscillators turning at the mean-field frequency' }),
})

/** Options of {@link kuramotoAlgorithm}. */
export interface KuramotoRunOptions {
  /** The step. */
  readonly dt: number
  /** The steps to take. */
  readonly steps: number
  /** An oscillator counts as locked when its velocity is within this of the mean field's (default 1e-3). */
  readonly lockTolerance?: number
}

const wrap = (x: number): number => x - 2 * Math.PI * Math.round(x / (2 * Math.PI))

/**
 * Kuramoto oscillators as a stepped algorithm (fixed-step Runge-Kutta on plain numbers, so the state serialises),
 * logging the time, the wrapped phases, the order parameter and its angle, and the share of oscillators locked to the
 * mean field: those whose velocity equals the rotation rate of the order parameter's phase to within a tolerance.
 *
 * @param options - Frequencies, coupling and network.
 * @param theta0 - The initial phases.
 * @param run - Step, step count and the locking tolerance.
 * @returns The algorithm.
 * @concept kuramoto-model
 * @example
 * ```ts
 * import { kuramotoAlgorithm } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // Identical oscillators with identical frequencies synchronise from any start when coupled.
 * const alg = kuramotoAlgorithm({ omega: [1, 1, 1], K: 1 }, [0, 1, 2], { dt: 0.05, steps: 600 })
 * (runAlgorithm(alg, { maxSteps: 1000 }).state.t).toFixed(1) // => '30.0'
 * ```
 */
export function kuramotoAlgorithm(
  options: KuramotoOptions,
  theta0: TensorLike,
  run: KuramotoRunOptions,
): Algorithm<FlowState, KuramotoPath> {
  const field = kuramotoField(options)
  const tol = run.lockTolerance ?? 1e-3
  return flowAlgorithm<KuramotoPath>(field, theta0, {
    dt: run.dt,
    steps: run.steps,
    name: 'kuramoto',
    config: {
      omega: [...options.omega],
      K: options.K,
      normalisation: options.normalisation ?? null,
      networked: options.adjacency !== undefined,
    },
    observe: (t, y, ctx) => {
      const { r, psi } = orderParameter(y)
      // The mean field's rotation rate: d psi / dt = Im( Z' / Z ) with Z' = (i / n) sum theta_j' exp(i theta_j).
      const velocity = Array.from((field(t, tensor(Array.from(y))) as Tensor).data)
      let re = 0
      let im = 0
      for (let j = 0; j < y.length; j++) {
        re -= (velocity[j] ?? 0) * Math.sin(y[j] ?? 0)
        im += (velocity[j] ?? 0) * Math.cos(y[j] ?? 0)
      }
      const rate = r > 1e-12 ? (im * Math.cos(psi) - re * Math.sin(psi)) / (y.length * r) : 0
      const locked = velocity.filter((v) => Math.abs(v - rate) < tol).length / y.length
      ctx.log('oscillators/t', t)
      ctx.log('oscillators/phases', y.map(wrap))
      ctx.log('oscillators/order', r)
      ctx.log('oscillators/meanPhase', psi)
      ctx.log('oscillators/locked', locked)
    },
  })
}

/**
 * The consensus flow x' = -L x on a graph: each node moves towards the average of its neighbours, so values converge to
 * the mean of the initial ones at a rate set by the smallest non-zero eigenvalue of the Laplacian L (the algebraic
 * connectivity): the linear cousin of Kuramoto synchronisation.
 *
 * @param adjacency - The (symmetric, weighted) adjacency matrix `[n, n]`.
 * @returns The field.
 * @concept consensus
 * @example
 * ```ts
 * import { consensusField } from 'mdml-engine/dynamics'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * // Two nodes joined by an edge of weight 1: each moves towards the other at the rate of their difference.
 * toFlat(consensusField([[0, 1], [1, 0]])(0, tensor([1, 0]))) // => [-1, 1]
 * ```
 */
export function consensusField(adjacency: TensorLike): VectorField {
  const a = asTensor(adjacency)
  return (_t, x) => {
    // (A x)_i - degree_i x_i = sum_j A_ij (x_j - x_i)
    const diff = sub(expandDims(x, 0), expandDims(x, 1))
    return sum(mul(a, diff), 1)
  }
}
