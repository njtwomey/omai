/**
 * Adaptive gradient optimisers as gradient transformations, matching torch.optim and optax: Adam, AdamW and AMSGrad,
 * Adamax, NAdam, RAdam, Lion, LAMB, Adafactor with factored second moments for large matrices, and Sophia with
 * Hutchinson's diagonal Hessian estimator. Each has its `scaleBy...` building block for composing custom optimisers
 * with `chain`.
 */

import { MdmlError } from '../base/index.ts'
import { bernoulli, foldIn, type Key } from '../random/index.ts'
import {
  abs,
  add,
  div,
  item,
  maximum,
  mean,
  minimum,
  mul,
  sign,
  sqrt,
  square,
  sub,
  treeFlatten,
  treeLeaves,
  treeMap,
  treeUnflatten,
  treeZeros,
  type Value,
  zeros,
  zerosLike,
} from '../tensor/index.ts'
import { flatObjective, type Objective } from './minimize/objective.ts'
import {
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  scaleByAdam,
  scaleByLearningRate,
  scaleByTrustRatio,
  type Tree,
} from './transform.ts'

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/** Options of {@link adam}, {@link adamw} and {@link amsgrad}. */
export interface AdamOptions {
  readonly learningRate: LearningRate
  readonly b1?: number
  readonly b2?: number
  readonly eps?: number
  readonly amsgrad?: boolean
  readonly weightDecay?: number
}

/**
 * Adam (torch.optim.Adam): bias-corrected moment estimates; `weightDecay` is an L2 penalty added to the gradient.
 *
 * @param options - Learning rate, betas, eps, AMSGrad and weight decay.
 * @returns The optimiser.
 * @concept optimisers
 * @index function Adam as a gradient transformation, like torch.optim.Adam.
 * @example
 * ```ts
 * import { adam } from 'mdml-engine/optim'
 * adam({ learningRate: 1e-3 }).name // => 'chain(addDecayedWeights, scaleByAdam, scaleByLearningRate)'
 * ```
 */
export function adam(options: AdamOptions): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8, options.amsgrad ?? false),
    scaleByLearningRate(options.learningRate),
  )
}

/**
 * AdamW (torch.optim.AdamW, Loshchilov and Hutter): Adam with decoupled weight decay (default 0.01), applied to the
 * parameters rather than added to the gradient.
 *
 * @param options - Learning rate, betas, eps, AMSGrad and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adamw } from 'mdml-engine/optim'
 * adamw({ learningRate: 1e-3 }).name // => 'chain(scaleByAdam, addDecayedWeights, scaleByLearningRate)'
 * ```
 */
export function adamw(options: AdamOptions): GradientTransformation<unknown[]> {
  return chain(
    scaleByAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8, options.amsgrad ?? false),
    addDecayedWeights(options.weightDecay ?? 0.01),
    scaleByLearningRate(options.learningRate),
  )
}

/**
 * AMSGrad (Reddi et al. 2018; torch.optim.Adam with amsgrad=True): Adam normalised by the running maximum of the
 * second-moment estimate, so effective step sizes never grow.
 *
 * @param options - Learning rate, betas, eps and L2 weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { amsgrad } from 'mdml-engine/optim'
 * amsgrad({ learningRate: 1e-3 }).name // => 'chain(addDecayedWeights, scaleByAdam, scaleByLearningRate)'
 * ```
 */
export function amsgrad(options: Omit<AdamOptions, 'amsgrad'>): GradientTransformation<unknown[]> {
  return adam({ ...options, amsgrad: true })
}

/** State of {@link scaleByAdamax}. */
export interface AdamaxState {
  readonly count: number
  readonly mu: Tree
  readonly nu: Tree
}

/**
 * Adamax's scaling (Kingma and Ba 2015, section 7; torch.optim.Adamax): u = max(b2 u, |g| + eps), updates =
 * m / ((1 - b1^t) u).
 *
 * @param b1 - First-moment decay.
 * @param b2 - Decay of the infinity norm.
 * @param eps - Added to |g|.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByAdamax } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = scaleByAdamax(0.5, 0.9, 0)
 * toArray(t.update(tensor([-2]), t.init(tensor([0]))).updates as never) // => [-1]
 * ```
 */
export function scaleByAdamax(b1 = 0.9, b2 = 0.999, eps = 1e-8): GradientTransformation<AdamaxState> {
  return {
    name: 'scaleByAdamax',
    init: (params) => ({ count: 0, mu: treeZeros(params), nu: treeZeros(params) }),
    update: (grads, state) => {
      const count = state.count + 1
      const mu = treeMap((g, m) => add(mul(m, b1), mul(g, 1 - b1)), grads, state.mu)
      const nu = treeMap((g, u) => maximum(mul(u, b2), add(abs(g), eps)), grads, state.nu)
      const c = 1 - b1 ** count
      return { updates: treeMap((m, u) => div(m, mul(u, c)), mu, nu), state: { count, mu, nu } }
    },
  }
}

/**
 * Adamax (torch.optim.Adamax): Adam with the infinity norm in place of the second moment.
 *
 * @param options - Learning rate (default 0.002), betas, eps and L2 weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { adamax } from 'mdml-engine/optim'
 * adamax({ learningRate: 2e-3 }).name // => 'chain(addDecayedWeights, scaleByAdamax, scaleByLearningRate)'
 * ```
 */
export function adamax(options: Omit<AdamOptions, 'amsgrad'>): GradientTransformation<unknown[]> {
  return chain(
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByAdamax(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8),
    scaleByLearningRate(options.learningRate),
  )
}

/** State of {@link scaleByNadam}. */
export interface NadamState {
  readonly count: number
  readonly muProduct: number
  readonly mu: Tree
  readonly nu: Tree
}

/**
 * NAdam's scaling (Dozat 2016; torch.optim.NAdam): Adam with Nesterov momentum and the momentum schedule
 * mu_t = b1 (1 - 0.5 * 0.96^(t psi)).
 *
 * @param b1 - First-moment decay.
 * @param b2 - Second-moment decay.
 * @param eps - Added to the denominator.
 * @param momentumDecay - psi (default 0.004).
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByNadam } from 'mdml-engine/optim'
 * scaleByNadam().name // => 'scaleByNadam'
 * ```
 */
export function scaleByNadam(
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
  momentumDecay = 4e-3,
): GradientTransformation<NadamState> {
  return {
    name: 'scaleByNadam',
    init: (params) => ({ count: 0, muProduct: 1, mu: treeZeros(params), nu: treeZeros(params) }),
    update: (grads, state) => {
      const t = state.count + 1
      const c2 = 1 - b2 ** t
      const mut = b1 * (1 - 0.5 * 0.96 ** (t * momentumDecay))
      const mutNext = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentumDecay))
      const muProduct = state.muProduct * mut
      const mu = treeMap((g, m) => add(m, mul(sub(g, m), 1 - b1)), grads, state.mu)
      const nu = treeMap((g, v) => add(mul(v, b2), mul(square(g), 1 - b2)), grads, state.nu)
      const a = (1 - mut) / (1 - muProduct)
      const b = mutNext / (1 - muProduct * mutNext)
      const updates = treeMap(
        (g, m, v) => {
          const denom = add(sqrt(div(v, c2)), eps)
          return add(mul(div(g, denom), a), mul(div(m, denom), b))
        },
        grads,
        mu,
        nu,
      )
      return { updates, state: { count: t, muProduct, mu, nu } }
    },
  }
}

/** Options of {@link nadam} and {@link radam}. */
export interface NadamOptions extends Omit<AdamOptions, 'amsgrad'> {
  /** NAdam's psi (default 0.004). */
  readonly momentumDecay?: number
  /** Apply the weight decay to the parameters (as AdamW) instead of adding it to the gradient. */
  readonly decoupledWeightDecay?: boolean
}

const withDecay = (
  options: NadamOptions,
  scaling: GradientTransformation,
  lrDefault: number,
): GradientTransformation<unknown[]> => {
  const wd = options.weightDecay ?? 0
  const lr = options.learningRate ?? lrDefault
  return options.decoupledWeightDecay === true
    ? chain(scaling, addDecayedWeights(wd), scaleByLearningRate(lr))
    : chain(addDecayedWeights(wd), scaling, scaleByLearningRate(lr))
}

/**
 * NAdam (torch.optim.NAdam): Adam with Nesterov momentum, optionally with decoupled weight decay (NAdamW).
 *
 * @param options - Learning rate (torch's default 0.002), betas, eps, momentum decay and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { nadam } from 'mdml-engine/optim'
 * nadam({ learningRate: 2e-3 }).name // => 'chain(addDecayedWeights, scaleByNadam, scaleByLearningRate)'
 * ```
 */
export function nadam(options: NadamOptions): GradientTransformation<unknown[]> {
  const scaling = scaleByNadam(
    options.b1 ?? 0.9,
    options.b2 ?? 0.999,
    options.eps ?? 1e-8,
    options.momentumDecay ?? 4e-3,
  )
  return withDecay(options, scaling, 2e-3)
}

/** State of {@link scaleByRadam}. */
export interface RadamState {
  readonly count: number
  readonly mu: Tree
  readonly nu: Tree
}

/**
 * RAdam's scaling (Liu et al. 2020; torch.optim.RAdam): bias-corrected momentum, rectified by the variance of the
 * adaptive step once the approximated SMA length rho_t exceeds 5, plain momentum before.
 *
 * @param b1 - First-moment decay.
 * @param b2 - Second-moment decay.
 * @param eps - Added to the denominator.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByRadam } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = scaleByRadam()
 * toArray(t.update(tensor([2]), t.init(tensor([0]))).updates as never) // => [2]
 * ```
 */
export function scaleByRadam(b1 = 0.9, b2 = 0.999, eps = 1e-8): GradientTransformation<RadamState> {
  const rhoInf = 2 / (1 - b2) - 1
  return {
    name: 'scaleByRadam',
    init: (params) => ({ count: 0, mu: treeZeros(params), nu: treeZeros(params) }),
    update: (grads, state) => {
      const t = state.count + 1
      const mu = treeMap((g, m) => add(m, mul(sub(g, m), 1 - b1)), grads, state.mu)
      const nu = treeMap((g, v) => add(mul(v, b2), mul(square(g), 1 - b2)), grads, state.nu)
      const c1 = 1 - b1 ** t
      const c2 = 1 - b2 ** t
      const rhoT = rhoInf - (2 * t * b2 ** t) / c2
      const rect = rhoT > 5 ? Math.sqrt(((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)) : 0
      const updates = treeMap(
        (m, v) => {
          const mHat = div(m, c1)
          return rhoT > 5 ? mul(mul(mHat, div(Math.sqrt(c2), add(sqrt(v), eps))), rect) : mHat
        },
        mu,
        nu,
      )
      return { updates, state: { count: t, mu, nu } }
    },
  }
}

/**
 * RAdam (torch.optim.RAdam): Adam with the variance of the adaptive learning rate rectified, removing the need for
 * warm-up; optionally with decoupled weight decay.
 *
 * @param options - Learning rate (torch's default 0.001), betas, eps and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { radam } from 'mdml-engine/optim'
 * radam({ learningRate: 1e-3 }).name // => 'chain(addDecayedWeights, scaleByRadam, scaleByLearningRate)'
 * ```
 */
export function radam(options: NadamOptions): GradientTransformation<unknown[]> {
  return withDecay(options, scaleByRadam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-8), 1e-3)
}

/**
 * Lion's scaling (Chen et al. 2023; optax.scale_by_lion): updates = sign(b1 m + (1 - b1) g), then m = b2 m + (1 - b2) g.
 *
 * @param b1 - Interpolation weight for the update direction.
 * @param b2 - Decay of the momentum.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleByLion } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = scaleByLion()
 * toArray(t.update(tensor([0.3, -2]), t.init(tensor([0, 0]))).updates as never) // => [1, -1]
 * ```
 */
export function scaleByLion(b1 = 0.9, b2 = 0.99): GradientTransformation<{ readonly mu: Tree }> {
  return {
    name: 'scaleByLion',
    init: (params) => ({ mu: treeZeros(params) }),
    update: (grads, state) => {
      const updates = treeMap((g, m) => sign(add(mul(g, 1 - b1), mul(m, b1))), grads, state.mu)
      const mu = treeMap((g, m) => add(mul(m, b2), mul(g, 1 - b2)), grads, state.mu)
      return { updates, state: { mu } }
    },
  }
}

/** Options of {@link lion}. */
export interface LionOptions {
  readonly learningRate: LearningRate
  readonly b1?: number
  readonly b2?: number
  /** Decoupled weight decay (optax's default 0.001). */
  readonly weightDecay?: number
}

/**
 * Lion (optax.lion): the sign of an interpolated momentum, with decoupled weight decay.
 *
 * @param options - Learning rate, betas and weight decay.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { lion } from 'mdml-engine/optim'
 * lion({ learningRate: 1e-4 }).name // => 'chain(scaleByLion, addDecayedWeights, scaleByLearningRate)'
 * ```
 */
export function lion(options: LionOptions): GradientTransformation<unknown[]> {
  return chain(
    scaleByLion(options.b1 ?? 0.9, options.b2 ?? 0.99),
    addDecayedWeights(options.weightDecay ?? 1e-3),
    scaleByLearningRate(options.learningRate),
  )
}

/**
 * LAMB (You et al. 2020; optax.lamb): Adam updates with decoupled weight decay, each layer rescaled by its trust ratio
 * |param| / |update|.
 *
 * @param options - Learning rate, betas, eps (optax's default 1e-6) and weight decay.
 * @returns The optimiser (needs `params` in `update`).
 * @example
 * ```ts
 * import { lamb } from 'mdml-engine/optim'
 * lamb({ learningRate: 1e-3 }).name // => 'chain(scaleByAdam, addDecayedWeights, scaleByTrustRatio, scaleByLearningRate)'
 * ```
 */
export function lamb(options: Omit<AdamOptions, 'amsgrad'>): GradientTransformation<unknown[]> {
  return chain(
    scaleByAdam(options.b1 ?? 0.9, options.b2 ?? 0.999, options.eps ?? 1e-6),
    addDecayedWeights(options.weightDecay ?? 0),
    scaleByTrustRatio(),
    scaleByLearningRate(options.learningRate),
  )
}

/** Options of {@link adafactor}. */
export interface AdafactorOptions {
  /** The cap on the relative step size: rho_t = min(lr, 1 / sqrt(t)) (torch's default 0.01). */
  readonly learningRate?: LearningRate
  /** The second-moment decay is 1 - t^beta2Decay (default -0.8). */
  readonly beta2Decay?: number
  /** Floor on the second-moment estimate (default machine epsilon) and on the parameter RMS (default 1e-3). */
  readonly eps?: readonly [number | undefined, number]
  /** Clipping threshold on the RMS of the unscaled update (default 1). */
  readonly d?: number
  /** Decoupled weight decay, scaled by the learning rate. */
  readonly weightDecay?: number
}

/** State of {@link adafactor}: row and column statistics for matrices, a full second moment for vectors. */
export interface AdafactorState {
  readonly count: number
  /** Per leaf: `[rowVar, colVar]` for leaves with two or more dimensions, `[variance]` otherwise. */
  readonly stats: Tree
}

const rms = (x: Value): number => Math.sqrt(item(mean(square(x))))

/**
 * Adafactor (Shazeer and Stern 2018; torch.optim.Adafactor): factored second moments (row and column means of g^2
 * for matrices, the last two axes of higher tensors), relative step sizes alpha_t = max(eps2, RMS(param)) rho_t,
 * update clipping by RMS and a decay schedule 1 - t^-0.8, without a first moment.
 *
 * @param options - Learning rate, decay exponent, epsilons, clipping threshold and weight decay.
 * @returns The optimiser (its updates are final; it needs `params` in `update`).
 * @example
 * ```ts
 * import { adafactor } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const opt = adafactor({ eps: [undefined, 1] })
 * toArray(opt.update(tensor([3, -4]), opt.init(tensor([0, 0])), tensor([0, 0])).updates as never) // => [-0.01, 0.01]
 * ```
 */
export function adafactor(options: AdafactorOptions = {}): GradientTransformation<AdafactorState> {
  const lr = options.learningRate ?? 0.01
  const decay = options.beta2Decay ?? -0.8
  const eps1 = options.eps?.[0] ?? 2.220446049250313e-16
  const eps2 = options.eps?.[1] ?? 1e-3
  const d = options.d ?? 1
  const wd = options.weightDecay ?? 0
  const lerp = (a: Value, b: Value, w: number): Value => add(a, mul(sub(b, a), w))
  return {
    name: 'adafactor',
    init: (params) => ({
      count: 0,
      stats: treeLeaves(params).map((p) =>
        p.shape.length > 1
          ? [zeros([...p.shape.slice(0, -1), 1]), zeros([...p.shape.slice(0, -2), 1, p.shape.at(-1) ?? 1])]
          : [zerosLike(p)],
      ),
    }),
    update: (grads, state, params) => {
      if (params === undefined) throw new MdmlError('value', 'adafactor needs the parameters')
      const t = state.count + 1
      const w = t ** decay
      const lrT = rate(lr, t - 1)
      const rhoT = Math.min(lrT, 1 / Math.sqrt(t))
      const { leaves, def } = treeFlatten(grads)
      const ps = treeLeaves(params)
      const prev = state.stats as readonly (readonly Value[])[]
      const stats: Value[][] = []
      const updates = leaves.map((g, i) => {
        const p = ps[i] as Value
        const old = prev[i] ?? []
        const alpha = Math.max(eps2, rms(p)) * rhoT
        let estimate: Value
        if (g.shape.length > 1) {
          const rowVar = lerp(old[0] as Value, mean(square(g), -1, true), w)
          const colVar = lerp(old[1] as Value, mean(square(g), -2, true), w)
          stats.push([rowVar, colVar])
          estimate = div(mul(rowVar, colVar), maximum(mean(rowVar, -2, true), eps1))
        } else {
          const variance = lerp(old[0] as Value, square(g), w)
          stats.push([variance])
          estimate = variance
        }
        const u = div(g, sqrt(maximum(estimate, eps1 * eps1)))
        const step = mul(u, -alpha / Math.max(1, rms(u) / d))
        return wd === 0 ? step : sub(step, mul(p, lrT * wd))
      })
      return { updates: treeUnflatten(def, updates), state: { count: t, stats } }
    },
  }
}

/** A diagonal Hessian estimator for {@link sophiaLite}: given the gradients, parameters and step, an estimate tree. */
export type HessianDiagonalEstimator = (grads: Tree, params: Tree | undefined, step: number) => Tree

/** Options of {@link scaleBySophia}. */
export interface SophiaOptions {
  readonly b1?: number
  readonly b2?: number
  /** The clipping scale gamma (official default 0.04). */
  readonly rho?: number
  readonly eps?: number
  /** Refresh the Hessian estimate every k steps (default 10). */
  readonly hessianEvery?: number
  /** The diagonal Hessian estimator (default the Gauss-Newton proxy g * g). */
  readonly hessian?: HessianDiagonalEstimator
}

/** State of {@link scaleBySophia}. */
export interface SophiaState {
  readonly count: number
  readonly mu: Tree
  readonly h: Tree
}

/**
 * Sophia's scaling (Liu et al. 2023), the lightweight variant: momentum m, an exponential average h of a diagonal
 * Hessian estimate refreshed every k steps, and the clipped preconditioned step sign(m) min(|m| / max(rho h, eps), 1).
 * The default estimate is the Gauss-Newton proxy g * g (the empirical Fisher diagonal); pass
 * {@link hutchinsonDiagonal} for Sophia-H.
 *
 * @param options - Betas, rho, eps, refresh period and the estimator.
 * @returns The transformation.
 * @example
 * ```ts
 * import { scaleBySophia } from 'mdml-engine/optim'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const t = scaleBySophia({ b1: 0, b2: 0, rho: 1 })
 * toArray(t.update(tensor([0.5, 4]), t.init(tensor([0, 0]))).updates as never) // => [1, 0.25]
 * ```
 */
export function scaleBySophia(options: SophiaOptions = {}): GradientTransformation<SophiaState> {
  const b1 = options.b1 ?? 0.965
  const b2 = options.b2 ?? 0.99
  const rho = options.rho ?? 0.04
  const eps = options.eps ?? 1e-12
  const every = options.hessianEvery ?? 10
  const estimate: HessianDiagonalEstimator = options.hessian ?? ((g) => treeMap((x) => square(x), g))
  return {
    name: 'scaleBySophia',
    init: (params) => ({ count: 0, mu: treeZeros(params), h: treeZeros(params) }),
    update: (grads, state, params) => {
      const mu = treeMap((g, m) => add(mul(m, b1), mul(g, 1 - b1)), grads, state.mu)
      const h =
        state.count % every === 0
          ? treeMap((e, hh) => add(mul(hh, b2), mul(e, 1 - b2)), estimate(grads, params, state.count), state.h)
          : state.h
      const updates = treeMap((m, hh) => mul(sign(m), minimum(div(abs(m), maximum(mul(hh, rho), eps)), 1)), mu, h)
      return { updates, state: { count: state.count + 1, mu, h } }
    },
  }
}

/** Options of {@link sophiaLite}. */
export interface SophiaLiteOptions extends SophiaOptions {
  readonly learningRate: LearningRate
  /** Decoupled weight decay (official default 0.1). */
  readonly weightDecay?: number
}

/**
 * Sophia, lightweight (Liu et al. 2023): a second-order clipped optimiser with a cheap diagonal Hessian estimate and
 * decoupled weight decay.
 *
 * @param options - Learning rate, weight decay and the {@link scaleBySophia} options.
 * @returns The optimiser.
 * @example
 * ```ts
 * import { sophiaLite } from 'mdml-engine/optim'
 * sophiaLite({ learningRate: 1e-3 }).name // => 'chain(scaleBySophia, addDecayedWeights, scaleByLearningRate)'
 * ```
 */
export function sophiaLite(options: SophiaLiteOptions): GradientTransformation<unknown[]> {
  const { learningRate, weightDecay, ...rest } = options
  return chain(scaleBySophia(rest), addDecayedWeights(weightDecay ?? 0.1), scaleByLearningRate(learningRate))
}

/**
 * Hutchinson's diagonal Hessian estimator u * (H u) with a Rademacher vector u (Sophia-H, AdaHessian): unbiased for the
 * diagonal, one Hessian-vector product per call. The vector at step t is drawn from `foldIn(key, t)`.
 *
 * @param f - The objective of the parameter tree.
 * @param key - The key of the probe vectors.
 * @returns An estimator for {@link scaleBySophia}.
 * @example
 * ```ts
 * import { hutchinsonDiagonal } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { mul, sum, tensor, toArray, type Value } from 'mdml-engine/tensor'
 * const est = hutchinsonDiagonal((x) => sum(mul(tensor([1, 3]), mul(x as Value, x as Value))), key(0))
 * toArray(est(tensor([0, 0]), tensor([1, 1]), 0) as never) // => [2, 6]
 * ```
 */
export function hutchinsonDiagonal(f: Objective, key: Key): HessianDiagonalEstimator {
  return (_grads, params, step) => {
    if (params === undefined) throw new MdmlError('value', 'hutchinsonDiagonal needs the parameters')
    const obj = flatObjective(f, params)
    const probe = bernoulli(foldIn(key, step), 0.5, [obj.n])
    const v = Float64Array.from(probe.data, (b) => (b ? 1 : -1))
    const hu = obj.hvp(obj.x0, v)
    return obj.unravel(hu.map((h, i) => h * (v[i] ?? 0)))
  }
}
