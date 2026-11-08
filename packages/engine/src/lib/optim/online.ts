/**
 * Online convex optimisation, where a learner plays a point each round, then sees that round's loss: online gradient
 * descent, follow the regularised leader, Hedge (exponential weights over experts), exponentiated gradient and the
 * online Newton step, with a traced runner that records the regret against the best fixed point in hindsight.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { type Json, minOf, OmaiError } from '../base/index.ts'
import { daxpy, ddot, dgemv, dnrm2, dscal, eigh } from '../numerics/linalg/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import {
  asTensor,
  fromStorage,
  item,
  mul,
  primalOf,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { flatOf, vectorOf } from './_vector.ts'

/** A step-size rule for online learners: a constant or a function of the round t (from 1). */
export type OnlineRate = number | ((t: number) => number)

const rateAt = (r: OnlineRate, t: number): number => (typeof r === 'number' ? r : r(t))

/**
 * An online learner (online convex optimisation): it plays a point, sees the gradient of that round's loss at the
 * point, and updates. Learners are plain functions over explicit states, like every algorithm here. The contract is a
 * kernel: `play` and `update` run once per round on typed arrays (D-009 keeps them native; the point is read from `play`,
 * and `onlineAlgorithm` puts it on the tape as a tensor).
 */
export interface OnlineLearner<S> {
  readonly name: string
  init(dim: number): S
  /** The point played in the next round. */
  play(state: S): Float64Array
  /** Fold in the round's loss gradient at the played point (t counts rounds from 1). */
  update(state: S, gradient: Float64Array, t: number): S
}

/** Euclidean projection onto the ball of radius r (the identity inside). */
function projectBall(x: Float64Array, radius: number | undefined): Float64Array {
  if (radius === undefined) return x
  const norm = dnrm2(x)
  return norm <= radius ? x : dscal(radius / norm, x)
}

/** Options of {@link onlineGradientDescent}. */
export interface OgdOptions {
  /** A constant or a schedule eta_t (default 1 / sqrt(t)). */
  readonly learningRate?: OnlineRate
  /** Project onto the L2 ball of this radius. */
  readonly radius?: number
  /** A custom projection onto the feasible set (applied after the ball, if both are given). */
  readonly project?: (x: Float64Array) => Float64Array
}

/**
 * Online (projected) gradient descent (Zinkevich 2003): x_{t+1} = P(x_t - eta_t g_t), with regret O(D G sqrt(T)) for
 * eta_t = D / (G sqrt(t)).
 *
 * @param options - Step sizes, the radius of the feasible ball and a custom projection.
 * @returns The learner.
 * @concept online-learning
 * @example
 * ```ts
 * import { onlineGradientDescent } from 'omai-engine/optim'
 * const l = onlineGradientDescent({ learningRate: 0.5, radius: 1 })
 * Array.from(l.play(l.update(l.init(2), new Float64Array([4, 0]), 1))) // => [-1, 0]
 * ```
 */
export function onlineGradientDescent(options: OgdOptions = {}): OnlineLearner<{ readonly x: Float64Array }> {
  const lr = options.learningRate ?? ((t: number) => 1 / Math.sqrt(t))
  const project = (x: Float64Array): Float64Array => {
    const b = projectBall(x, options.radius)
    return options.project === undefined ? b : Float64Array.from(options.project(b))
  }
  return {
    name: 'onlineGradientDescent',
    init: (dim) => ({ x: new Float64Array(dim) }),
    play: (s) => s.x,
    update: (s, g, t) => ({ x: project(daxpy(-rateAt(lr, t), g, s.x)) }),
  }
}

/** Options of {@link ftrl}. */
export interface FtrlOptions {
  /** The regularisation scale eta: R(x) = |x|^2 / (2 eta) (a constant or a schedule; default 1 / sqrt(t)). */
  readonly learningRate?: OnlineRate
  /** An L1 term lambda |x|_1 in the regulariser (sparse plays). */
  readonly l1?: number
  /** Restrict plays to the L2 ball of this radius. */
  readonly radius?: number
}

/**
 * Follow the regularised leader on linearised losses (Shalev-Shwartz 2012): x_{t+1} = argmin_x <G_t, x> + |x|^2 /
 * (2 eta_t) + l1 |x|_1 over the ball, with G_t the sum of the gradients so far; in closed form
 * -eta_t soft(G_t, l1), projected.
 *
 * @param options - Regularisation scale, L1 strength and radius.
 * @returns The learner.
 * @concept online-learning
 * @example
 * ```ts
 * import { ftrl } from 'omai-engine/optim'
 * const l = ftrl({ learningRate: 0.5, l1: 1 })
 * Array.from(l.play(l.update(l.init(2), new Float64Array([3, 0.5]), 1))) // => [-1, 0]
 * ```
 */
export function ftrl(
  options: FtrlOptions = {},
): OnlineLearner<{ readonly sum: Float64Array; readonly x: Float64Array }> {
  const lr = options.learningRate ?? ((t: number) => 1 / Math.sqrt(t))
  const l1 = options.l1 ?? 0
  return {
    name: 'ftrl',
    init: (dim) => ({ sum: new Float64Array(dim), x: new Float64Array(dim) }),
    play: (s) => s.x,
    update: (s, g, t) => {
      const sumG = daxpy(1, g, s.sum)
      const eta = rateAt(lr, t + 1)
      const x = Float64Array.from(sumG, (v) => -eta * Math.sign(v) * Math.max(Math.abs(v) - l1, 0) + 0)
      return { sum: sumG, x: projectBall(x, options.radius) }
    },
  }
}

/** Options of {@link hedge}. */
export interface HedgeOptions {
  /** A constant or a schedule eta_t (default the anytime rate sqrt(8 ln n / t)). */
  readonly learningRate?: OnlineRate
}

/**
 * Hedge, the exponential-weights algorithm for prediction with expert advice (Freund and Schapire 1997): play
 * p_i proportional to exp(-eta L_i) with L the cumulative losses; for losses in [0, 1] the regret against the best
 * expert is at most ln n / eta + eta T / 8. The gradient of the round's loss p . l is the loss vector l.
 *
 * @param options - The learning rate.
 * @returns The learner over the probability simplex of n experts.
 * @concept online-learning
 * @example
 * ```ts
 * import { hedge } from 'omai-engine/optim'
 * const l = hedge({ learningRate: Math.log(3) })
 * Array.from(l.play(l.update(l.init(2), new Float64Array([1, 0]), 1)), (v) => Math.round(v * 1e12) / 1e12) // => [0.25, 0.75]
 * ```
 */
export function hedge(
  options: HedgeOptions = {},
): OnlineLearner<{ readonly losses: Float64Array; readonly p: Float64Array }> {
  const weights = (losses: Float64Array, eta: number): Float64Array => {
    const lo = minOf(losses)
    const w = Float64Array.from(losses, (l) => Math.exp(-eta * (l - lo)))
    const total = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / total)
  }
  return {
    name: 'hedge',
    init: (n) => ({ losses: new Float64Array(n), p: new Float64Array(n).fill(1 / n) }),
    play: (s) => s.p,
    update: (s, l, t) => {
      const losses = daxpy(1, l, s.losses)
      const n = losses.length
      const eta =
        options.learningRate === undefined
          ? Math.sqrt((8 * Math.log(n)) / (t + 1))
          : rateAt(options.learningRate, t + 1)
      return { losses, p: weights(losses, eta) }
    },
  }
}

/** Options of {@link exponentiatedGradient}. */
export interface ExponentiatedGradientOptions {
  readonly learningRate: OnlineRate
  /** Play on the simplex scaled to this total (default 1). */
  readonly scale?: number
}

/**
 * Exponentiated gradient (Kivinen and Warmuth 1997): multiplicative updates w_i <- w_i exp(-eta g_i), renormalised to
 * the (scaled) simplex: mirror descent with the entropy, for convex losses over distributions.
 *
 * @param options - The learning rate and the total mass.
 * @returns The learner.
 * @concept online-learning
 * @example
 * ```ts
 * import { exponentiatedGradient } from 'omai-engine/optim'
 * const l = exponentiatedGradient({ learningRate: Math.log(3) })
 * Array.from(l.play(l.update(l.init(2), new Float64Array([1, 0]), 1)), (v) => Math.round(v * 1e12) / 1e12) // => [0.25, 0.75]
 * ```
 */
export function exponentiatedGradient(
  options: ExponentiatedGradientOptions,
): OnlineLearner<{ readonly w: Float64Array }> {
  const total = options.scale ?? 1
  return {
    name: 'exponentiatedGradient',
    init: (n) => ({ w: new Float64Array(n).fill(total / n) }),
    play: (s) => s.w,
    update: (s, g, t) => {
      const eta = rateAt(options.learningRate, t)
      // Subtracting the smallest gradient entry leaves the normalised weights unchanged and avoids overflow.
      const shift = minOf(g)
      const w = Float64Array.from(s.w, (v, i) => v * Math.exp(-eta * ((g[i] ?? 0) - shift)))
      const z = w.reduce((a, b) => a + b, 0)
      return { w: w.map((v) => (total * v) / z) }
    },
  }
}

/** Options of {@link onlineNewtonStep}. */
export interface OnlineNewtonOptions {
  /** The step parameter gamma (Hazan et al. take (1/2) min(1 / (4 G D), alpha) for alpha-exp-concave losses). */
  readonly gamma: number
  /** The initial curvature epsilon I (default 1). */
  readonly epsilon?: number
  /** Restrict plays to the L2 ball of this radius (projected in the A-norm). */
  readonly radius?: number
}

/** State of {@link onlineNewtonStep}: the play, A = eps I + sum g g^T and its inverse. */
export interface OnlineNewtonState {
  readonly x: Float64Array
  readonly a: Float64Array
  readonly aInv: Float64Array
}

/** argmin_{|z| <= r} (z - y)^T A (z - y), through A's eigendecomposition and a root in the multiplier. */
function projectInNorm(y: Float64Array, a: Float64Array, radius: number): Float64Array {
  if (dnrm2(y) <= radius) return y
  const n = y.length
  const { eigenvalues, eigenvectors } = eigh(fromStorage(Float64Array.from(a), [n, n], 'float64'))
  const lam = flatOf(eigenvalues)
  const q = flatOf(eigenvectors)
  const c = dgemv(q, n, n, y, true)
  const at = (mu: number): Float64Array => Float64Array.from(c, (ci, i) => ((lam[i] ?? 0) * ci) / ((lam[i] ?? 0) + mu))
  let hi = 1
  while (dnrm2(at(hi)) > radius) hi *= 2
  const mu = brentq((m) => dnrm2(at(m)) - radius, 0, hi, { xtol: 1e-14 }).root
  return dgemv(q, n, n, at(mu))
}

/**
 * Online Newton step (Hazan, Agarwal and Kale 2007): A_t = eps I + sum g g^T, x_{t+1} = P_A(x_t - A_t^-1 g_t / gamma),
 * with A^-1 kept by Sherman-Morrison updates and the projection taken in the A-norm; logarithmic regret for
 * exp-concave losses.
 *
 * @param options - gamma, epsilon and the radius of the feasible ball.
 * @returns The learner.
 * @concept online-learning
 * @example
 * ```ts
 * import { onlineNewtonStep } from 'omai-engine/optim'
 * const l = onlineNewtonStep({ gamma: 1, epsilon: 1 })
 * Array.from(l.play(l.update(l.init(1), new Float64Array([1]), 1))) // => [-0.5]
 * ```
 */
export function onlineNewtonStep(options: OnlineNewtonOptions): OnlineLearner<OnlineNewtonState> {
  const eps = options.epsilon ?? 1
  return {
    name: 'onlineNewtonStep',
    init: (dim) => {
      const a = new Float64Array(dim * dim)
      const aInv = new Float64Array(dim * dim)
      for (let i = 0; i < dim; i++) {
        a[i * dim + i] = eps
        aInv[i * dim + i] = 1 / eps
      }
      return { x: new Float64Array(dim), a, aInv }
    },
    play: (s) => s.x,
    update: (s, g) => {
      const n = g.length
      const a = Float64Array.from(s.a)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) a[i * n + j] = (a[i * n + j] ?? 0) + (g[i] ?? 0) * (g[j] ?? 0)
      const u = dgemv(s.aInv, n, n, g)
      const denom = 1 + ddot(g, u)
      const aInv = Float64Array.from(s.aInv, (v, k) => v - ((u[Math.floor(k / n)] ?? 0) * (u[k % n] ?? 0)) / denom)
      const y = daxpy(-1 / options.gamma, dgemv(aInv, n, n, g), s.x)
      return { x: options.radius === undefined ? y : projectInNorm(y, a, options.radius), a, aInv }
    },
  }
}

/** A round of online learning: the loss as a scalar function of the played point. */
export type OnlineLoss = (x: Value) => Value

/**
 * The linear losses p . l_t of prediction with expert advice, one per row of a loss matrix.
 *
 * @param lossVectors - T x n losses (one row per round).
 * @returns The rounds.
 * @example
 * ```ts
 * import { linearLosses } from 'omai-engine/optim'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(linearLosses([[1, 0]])[0](tensor([0.25, 0.75]))) // => 0.25
 * ```
 */
export function linearLosses(lossVectors: TensorLike): OnlineLoss[] {
  const m = asTensor(lossVectors)
  const [rows = 0, cols = 0] = m.shape
  const flat = toFlat(primalOf(m))
  return Array.from({ length: rows }, (_, t) => {
    const l = fromStorage(Float64Array.from(flat.slice(t * cols, (t + 1) * cols)), [cols], 'float64')
    return (x: Value) => sum(mul(x, l))
  })
}

/** State of an online run: the learner's state and the regret record so far. */
export interface OnlineState<S> {
  readonly learner: S
  readonly t: number
  readonly play: Float64Array
  readonly loss: number
  readonly cumulativeLoss: number
  readonly comparatorLoss: number
  /** Cumulative losses of each expert (only for `{ experts: true }`). */
  readonly expertLosses?: Float64Array
}

/** Options of {@link onlineAlgorithm} and {@link runOnline}. */
export interface OnlineOptions {
  /** The dimension of the plays (the number of experts for simplex learners). */
  readonly dim: number
  /** A fixed comparator u, a vector (a tensor-like; migration, D-009: was `ArrayLike<number>`): regret is against sum_t f_t(u). */
  readonly comparator?: TensorLike
  /** Linear losses over experts: regret is against the best single expert in hindsight (the default comparator). */
  readonly experts?: boolean
}

const SCHEMA = channels({
  metrics: {
    loss: { description: 'loss of the round', direction: 'lower' },
    cumulativeLoss: { description: 'sum of the learner losses so far', direction: 'lower' },
    regret: { description: 'cumulative loss minus the comparator loss', direction: 'lower' },
  },
  tensors: { play: { role: 'param', description: 'the point played' } },
})

/**
 * An online learner facing a sequence of losses, as a traced algorithm: each step plays, suffers the round's loss,
 * computes its gradient by autodiff and updates. Records the round loss, the cumulative loss and the regret against a
 * fixed comparator or (for experts) the best expert in hindsight.
 *
 * @param learner - The learner.
 * @param losses - The losses, one per round.
 * @param options - Dimension and comparator.
 * @returns The algorithm.
 * @concept online-learning
 * @example
 * ```ts
 * import { hedge, linearLosses, onlineAlgorithm } from 'omai-engine/optim'
 * onlineAlgorithm(hedge(), linearLosses([[1, 0]]), { dim: 2, experts: true }).name // => 'online(hedge)'
 * ```
 */
export function onlineAlgorithm<S>(
  learner: OnlineLearner<S>,
  losses: readonly OnlineLoss[],
  options: OnlineOptions,
): Algorithm<OnlineState<S>> {
  const u =
    options.comparator === undefined
      ? undefined
      : fromStorage(Float64Array.from(toVector(options.comparator, 'onlineAlgorithm')), [options.dim], 'float64')
  const basis = (i: number): Value => {
    const e = new Float64Array(options.dim)
    e[i] = 1
    return fromStorage(e, [options.dim], 'float64')
  }
  return {
    name: `online(${learner.name})`,
    config: {
      learner: learner.name,
      dim: options.dim,
      rounds: losses.length,
      experts: options.experts ?? false,
    } as Json,
    schema: SCHEMA,
    init: () => {
      const s = learner.init(options.dim)
      return {
        learner: s,
        t: 0,
        play: learner.play(s),
        loss: 0,
        cumulativeLoss: 0,
        comparatorLoss: 0,
        ...(options.experts === true ? { expertLosses: new Float64Array(options.dim) } : {}),
      }
    },
    step: (state) => {
      const f = losses[state.t]
      if (f === undefined) throw new OmaiError('value', 'online: no loss for this round')
      const x = fromStorage(Float64Array.from(state.play), [options.dim], 'float64')
      const { value, grad } = valueAndGrad(f)(x)
      const loss = item(value)
      const t = state.t + 1
      const next = learner.update(state.learner, Float64Array.from(toFlat(grad as Value)), t)
      const expertLosses =
        state.expertLosses === undefined
          ? undefined
          : Float64Array.from(state.expertLosses, (l, i) => l + item(f(basis(i))))
      const comparatorLoss = u === undefined ? state.comparatorLoss : state.comparatorLoss + item(f(u))
      return {
        learner: next,
        t,
        play: learner.play(next),
        loss,
        cumulativeLoss: state.cumulativeLoss + loss,
        comparatorLoss: expertLosses === undefined ? comparatorLoss : minOf(expertLosses),
        ...(expertLosses === undefined ? {} : { expertLosses }),
      }
    },
    done: (state) => (state.t >= losses.length ? 'completed' : undefined),
    observe: (state) => ({
      metrics: {
        loss: state.loss,
        cumulativeLoss: state.cumulativeLoss,
        ...(u !== undefined || state.expertLosses !== undefined
          ? { regret: state.cumulativeLoss - state.comparatorLoss }
          : {}),
      },
      tensors: { play: { role: 'param', value: state.play } },
    }),
  }
}

/** The record of an online run. Holds tensors (migration, D-009: `plays` was a list of `Float64Array`s, the others `readonly number[]`). */
export interface OnlineResult {
  /** The point played in each round, the rows of a `[T, dim]` matrix. */
  readonly plays: Tensor<'float64'>
  /** The loss of each round, `[T]`. */
  readonly losses: Tensor<'float64'>
  readonly cumulativeLoss: number
  /** The comparator's cumulative loss (the best expert's, for experts). */
  readonly comparatorLoss: number
  /** Regret after each round, `[T]` (NaN without a comparator). */
  readonly regret: Tensor<'float64'>
}

/**
 * Run an online learner over a sequence of losses and return its regret record (see {@link onlineAlgorithm}).
 *
 * @param learner - The learner.
 * @param losses - The losses, one per round.
 * @param options - Dimension, comparator and the run options (tracer, policy).
 * @returns Plays, losses and regret, as tensors (they were plain arrays).
 * @concept online-learning
 * @example
 * ```ts
 * import { hedge, linearLosses, runOnline } from 'omai-engine/optim'
 * const r = runOnline(hedge({ learningRate: 1 }), linearLosses([[1, 0], [1, 0]]), { dim: 2, experts: true })
 * r.comparatorLoss // => 0
 * ```
 */
export function runOnline<S>(
  learner: OnlineLearner<S>,
  losses: readonly OnlineLoss[],
  options: OnlineOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): OnlineResult {
  const { dim, comparator, experts, ...run } = options
  const alg = onlineAlgorithm(learner, losses, {
    dim,
    ...(comparator === undefined ? {} : { comparator }),
    ...(experts === undefined ? {} : { experts }),
  })
  const plays: Float64Array[] = []
  const roundLosses: number[] = []
  const regret: number[] = []
  const hasComparator = comparator !== undefined || experts === true
  const recording: Algorithm<OnlineState<S>> = {
    ...alg,
    step: (s, ctx) => {
      plays.push(s.play)
      const next = alg.step(s, ctx)
      roundLosses.push(next.loss)
      regret.push(hasComparator ? next.cumulativeLoss - next.comparatorLoss : Number.NaN)
      return next
    },
  }
  const result = runAlgorithm(recording, { ...run, maxSteps: losses.length })
  const dims = plays[0]?.length ?? dim
  const stacked = new Float64Array(plays.length * dims)
  plays.forEach((p, t) => {
    stacked.set(p, t * dims)
  })
  return {
    plays: fromStorage(stacked, [plays.length, dims], 'float64'),
    losses: vectorOf(roundLosses),
    cumulativeLoss: result.state.cumulativeLoss,
    comparatorLoss: result.state.comparatorLoss,
    regret: vectorOf(regret),
  }
}
