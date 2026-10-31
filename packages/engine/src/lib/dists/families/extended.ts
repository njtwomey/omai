/**
 * Extended families: a response whose log-likelihood is given with its derivatives in the linear predictors, one
 * predictor for most families and several for the families with more than one parameter that depends on covariates
 * (a mean and a scale, the count and the inflation of a zero-inflated Poisson, the K - 1 logits of a categorical
 * response). This is what Newton's method on a penalised additive model needs when the family is not an exponential
 * dispersion family (Wood, Pya and Saefken 2016): the gradient and the (observed) Hessian of the log-likelihood in the
 * predictors, and optionally the third derivatives that the smoothing-parameter score differentiates. A family either
 * writes its derivatives by hand (`gaussianLocationScale`, with the third derivatives) or states its per-observation
 * log-likelihood over tensor operations and gets them by automatic differentiation (`tensorExtendedFamily`: beta
 * regression, the negative binomial with its dispersion estimated, zero-inflated Poisson, ordered categorical,
 * multinomial, Tweedie with its power estimated). Parameters that are not predictors (the beta precision, the negative
 * binomial dispersion, the ordinal cut points, the Tweedie power) are `aux` parameters on an unconstrained scale,
 * estimated together with the smoothing parameters.
 */

import { hvp, valueAndGrad } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import { lgamma, ndtr, ndtri } from '../../numerics/special/index.ts'
import {
  add,
  exp,
  fromStorage,
  item,
  log,
  maximum,
  mul,
  neg,
  sigmoid,
  sliceAxis,
  softplus,
  stopGradient,
  sub,
  sum,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Distribution } from '../distribution.ts'
import { tweedieDistribution } from './dispersion.ts'
import { identity, type Link, logit, log as logLink } from './links.ts'

/** A parameter of an extended family that is not a predictor, on an unconstrained scale. */
export interface AuxParameter {
  readonly name: string
  /** The starting value. */
  readonly start: number
  /** The bounds the estimate is kept in (the unconstrained scale: a log, a logit). */
  readonly lower: number
  readonly upper: number
}

/** The log-likelihood of an extended family at predictors, with its derivatives in them. */
export interface ExtendedDerivatives {
  /** The weighted log-likelihood `sum w_i l_i`; `-Infinity` where the predictors are infeasible. */
  readonly logLik: number
  /** `d l / d eta_k` per observation: `K` arrays of `n`. */
  readonly grad: readonly Float64Array[]
  /** `d^2 l / d eta_k d eta_l` per observation for `k <= l`, packed by {@link hessIndex}: `K (K + 1) / 2` arrays of `n`. */
  readonly hess: readonly Float64Array[]
  /** Third derivatives `d^3 l / d eta_k d eta_l d eta_m`, full `K^3` arrays of `n` indexed `(k K + l) K + m`, if the family has them. */
  readonly third?: readonly Float64Array[]
}

/**
 * A family given by the derivatives of its log-likelihood in the linear predictors.
 *
 * @concept generalized-additive-models
 */
export interface ExtendedFamily {
  readonly name: string
  /** The names of the predictors, one per parameter that depends on covariates. */
  readonly predictors: readonly string[]
  /** The link of each predictor (its inverse gives the parameter on its natural scale). */
  readonly links: readonly Link[]
  /** The non-predictor parameters, with starting values from the data. */
  aux(y: Float64Array, weights: Float64Array): readonly AuxParameter[]
  /** A starting value of each predictor (a constant, taken by the intercept). */
  start(y: Float64Array, weights: Float64Array): readonly number[]
  /**
   * The log-likelihood and its derivatives at predictors `eta` (`K` arrays of `n`), with prior weights and the aux
   * parameters; `order` 3 asks for the third derivatives too (a family without them leaves `third` out).
   */
  derivatives(
    y: Float64Array,
    weights: Float64Array,
    eta: readonly Float64Array[],
    aux: readonly number[],
    order: 2 | 3,
  ): ExtendedDerivatives
  /** Whether `derivatives` supplies the third derivatives. */
  readonly hasThird: boolean
  /** The fitted distribution of each observation at predictors `eta` (`K` arrays of `n`), for residuals and simulation. */
  readonly distribution?: (eta: readonly Float64Array[], aux: readonly number[]) => Distribution
}

/**
 * The position of the pair `(k, l)`, `k <= l`, in the packed Hessian of `K` predictors.
 *
 * @param k - First predictor.
 * @param l - Second predictor.
 * @param K - The number of predictors.
 * @returns The index into `hess`.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * [families.hessIndex(0, 0, 2), families.hessIndex(0, 1, 2), families.hessIndex(1, 1, 2)] // => [0, 1, 2]
 * ```
 */
export function hessIndex(k: number, l: number, K: number): number {
  const a = Math.min(k, l)
  const b = Math.max(k, l)
  return a * K - (a * (a - 1)) / 2 + (b - a)
}

/** The specification of a family whose derivatives come from automatic differentiation. */
export interface TensorFamilySpec {
  readonly name: string
  readonly predictors: readonly string[]
  readonly links: readonly Link[]
  readonly aux?: (y: Float64Array, weights: Float64Array) => readonly AuxParameter[]
  readonly start: (y: Float64Array, weights: Float64Array) => readonly number[]
  /** The log-likelihood of each observation `[n]` at the predictors (one `[n]` value each) and the aux parameters. */
  readonly logProb: (y: Float64Array, eta: readonly Value[], aux: readonly number[]) => Value
}

const vec = (a: ArrayLike<number>): Value => fromStorage(Float64Array.from(a), [a.length], 'float64')

/**
 * An extended family from the per-observation log-likelihood over tensor operations: the gradient is reverse-mode
 * differentiation of the weighted sum, and the Hessian blocks are K Hessian-vector products, since observations are
 * independent (the product with the indicator of predictor `l` returns `d^2 l / d eta_k d eta_l` for every `k` and
 * observation at once). There are no third derivatives; the smoothing-parameter score differences the Hessian instead.
 *
 * @param spec - The names, links, starting values and per-observation log-likelihood.
 * @returns The family.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * import { exp, mul, sub } from 'mdml-engine/tensor'
 * // Poisson with a log link: l = y eta - exp(eta) (up to a constant)
 * const f = families.tensorExtendedFamily({
 *   name: 'po', predictors: ['mean'], links: [families.log()],
 *   start: () => [0],
 *   logProb: (y, [eta]) => sub(mul(y, eta), exp(eta)),
 * })
 * const d = f.derivatives(Float64Array.of(2), Float64Array.of(1), [Float64Array.of(0)], [], 2)
 * [d.grad[0]?.[0], d.hess[0]?.[0]] // => [1, -1]
 * ```
 */
export function tensorExtendedFamily(spec: TensorFamilySpec): ExtendedFamily {
  const K = spec.predictors.length
  return {
    name: spec.name,
    predictors: spec.predictors,
    links: spec.links,
    aux: spec.aux ?? (() => []),
    start: spec.start,
    hasThird: false,
    derivatives: (y, weights, eta, aux) => {
      const n = y.length
      const w = vec(weights)
      const f = (e: Value): Value => {
        const parts = Array.from({ length: K }, (_, k) => sliceAxis(e, 0, k * n, (k + 1) * n) as Value)
        return sum(mul(w, spec.logProb(y, parts, aux)))
      }
      const flat = new Float64Array(K * n)
      eta.forEach((e, k) => {
        flat.set(e, k * n)
      })
      const x = vec(flat)
      const hess = Array.from({ length: (K * (K + 1)) / 2 }, () => new Float64Array(n))
      const empty = (logLik: number): ExtendedDerivatives => ({
        logLik,
        grad: Array.from({ length: K }, () => new Float64Array(n)),
        hess,
      })
      let res: { value: Value; grad: unknown }
      try {
        res = valueAndGrad(f)(x) as { value: Value; grad: unknown }
      } catch {
        return empty(Number.NEGATIVE_INFINITY)
      }
      const logLik = item(res.value)
      if (!Number.isFinite(logLik)) return empty(Number.NEGATIVE_INFINITY)
      const g = toFlat(res.grad as Value)
      for (let l = 0; l < K; l++) {
        const indicator = new Float64Array(K * n)
        indicator.fill(1, l * n, (l + 1) * n)
        const col = toFlat(hvp(f, x, vec(indicator)))
        for (let k = 0; k <= l; k++) hess[hessIndex(k, l, K)]?.set(Array.from(col).slice(k * n, (k + 1) * n))
      }
      return {
        logLik,
        grad: Array.from({ length: K }, (_, k) => Float64Array.from(Array.from(g).slice(k * n, (k + 1) * n))),
        hess,
      }
    },
  }
}

const setThird = (third: Float64Array[], at: number, i: number, v: number): void => {
  const a = third[at]
  if (a !== undefined) a[i] = v
}

const mean = (y: ArrayLike<number>, w: ArrayLike<number>): number => {
  let s = 0
  let t = 0
  for (let i = 0; i < y.length; i++) {
    s += (w[i] ?? 1) * (y[i] ?? 0)
    t += w[i] ?? 1
  }
  return s / Math.max(t, 1e-300)
}

const variance = (y: ArrayLike<number>, w: ArrayLike<number>): number => {
  const m = mean(y, w)
  let s = 0
  let t = 0
  for (let i = 0; i < y.length; i++) {
    s += (w[i] ?? 1) * ((y[i] ?? 0) - m) ** 2
    t += w[i] ?? 1
  }
  return s / Math.max(t, 1e-300)
}

/**
 * The Gaussian location-scale family (mgcv's `gaulss`): the mean and the log standard deviation are each an additive
 * predictor. The derivatives are written out, third derivatives included.
 *
 * @returns The family with predictors `mean` (identity link) and `logsd`.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * const d = families.gaussianLocationScale().derivatives(Float64Array.of(1), Float64Array.of(1), [Float64Array.of(0), Float64Array.of(0)], [], 2)
 * [d.grad[0]?.[0], d.grad[1]?.[0], d.hess[0]?.[0], d.hess[2]?.[0]] // => [1, 0, -1, -2]
 * ```
 */
export function gaussianLocationScale(): ExtendedFamily {
  const LOG_2PI = Math.log(2 * Math.PI)
  return {
    name: 'gaussianLocationScale',
    predictors: ['mean', 'logsd'],
    links: [identity(), logLink()],
    hasThird: true,
    aux: () => [],
    start: (y, w) => [mean(y, w), 0.5 * Math.log(Math.max(variance(y, w), 1e-12))],
    derivatives: (y, w, eta, _aux, order) => {
      const n = y.length
      const [e1, e2] = eta as [Float64Array, Float64Array]
      const mk = (): Float64Array => new Float64Array(n)
      const g1 = mk()
      const g2 = mk()
      const h11 = mk()
      const h12 = mk()
      const h22 = mk()
      const third = order === 3 ? Array.from({ length: 8 }, mk) : undefined
      let logLik = 0
      for (let i = 0; i < n; i++) {
        const wi = w[i] ?? 1
        const s = e2[i] ?? 0
        const sd = Math.exp(s)
        const r = (y[i] ?? 0) - (e1[i] ?? 0)
        const z = r / sd
        logLik += wi * (-s - 0.5 * z * z - 0.5 * LOG_2PI)
        g1[i] = (wi * z) / sd
        g2[i] = wi * (z * z - 1)
        h11[i] = -wi / (sd * sd)
        h12[i] = (-2 * wi * z) / sd
        h22[i] = -2 * wi * z * z
        if (third !== undefined) {
          // index (k * 2 + l) * 2 + m: 000, 001, 010, 011, 100, 101, 110, 111
          const t001 = (2 * wi) / (sd * sd)
          const t011 = (4 * wi * z) / sd
          const t111 = 4 * wi * z * z
          setThird(third, 1, i, t001)
          setThird(third, 2, i, t001)
          setThird(third, 4, i, t001)
          setThird(third, 3, i, t011)
          setThird(third, 5, i, t011)
          setThird(third, 6, i, t011)
          setThird(third, 7, i, t111)
        }
      }
      return { logLik, grad: [g1, g2], hess: [h11, h12, h22], ...(third === undefined ? {} : { third }) }
    },
  }
}

const logistic = (v: number): number => 1 / (1 + Math.exp(-v))
const logitOf = (p: number): number => Math.log(p / (1 - p))

/**
 * Beta regression (mgcv's `betar`): the mean `mu = logit^-1(eta)` of a response in (0, 1) is an additive predictor and
 * the precision `phi` (variance `mu (1 - mu) / (1 + phi)`) is estimated as an aux parameter on the log scale.
 *
 * @returns The family.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * const f = families.betaRegression()
 * f.aux(Float64Array.of(0.2, 0.5, 0.8), Float64Array.of(1, 1, 1))[0]?.name // => 'logPrecision'
 * ```
 */
export function betaRegression(): ExtendedFamily {
  return tensorExtendedFamily({
    name: 'beta',
    predictors: ['mean'],
    links: [logit()],
    aux: (y, w) => {
      const m = mean(y, w)
      const v = Math.max(variance(y, w), 1e-6)
      return [{ name: 'logPrecision', start: Math.log(Math.max((m * (1 - m)) / v - 1, 0.5)), lower: -6, upper: 12 }]
    },
    start: (y, w) => [logitOf(Math.min(Math.max(mean(y, w), 1e-3), 1 - 1e-3))],
    logProb: (y, [eta], [lphi]) => {
      const phi = Math.exp(lphi ?? 0)
      const mu = sigmoid(eta as Value)
      const a = mul(mu, phi)
      const b = mul(sub(1, mu), phi)
      const ly = vec(Array.from(y, (v) => Math.log(v)))
      const l1y = vec(Array.from(y, (v) => Math.log1p(-v)))
      return add(sub(sub(lgamma(phi), lgamma(a)), lgamma(b)), add(mul(sub(a, 1), ly), mul(sub(b, 1), l1y)))
    },
  })
}

/**
 * The negative binomial with its dispersion estimated together with the smooths (mgcv's `nb`): variance `mu + mu^2 /
 * theta`, log link, `log theta` an aux parameter.
 *
 * @returns The family.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.negativeBinomialTheta().predictors // => ['mean']
 * ```
 */
export function negativeBinomialTheta(): ExtendedFamily {
  return tensorExtendedFamily({
    name: 'negativeBinomial',
    predictors: ['mean'],
    links: [logLink()],
    aux: (y, w) => {
      const m = mean(y, w)
      const excess = Math.max(variance(y, w) - m, 0.05 * m)
      return [
        { name: 'logTheta', start: Math.log(Math.min(Math.max((m * m) / excess, 0.05), 1e3)), lower: -6, upper: 9 },
      ]
    },
    start: (y, w) => [Math.log(Math.max(mean(y, w), 0.1))],
    logProb: (y, [eta], [ltheta]) => {
      const theta = Math.exp(ltheta ?? 0)
      const e = eta as Value
      const yv = vec(y)
      const lg1 = vec(Array.from(y, (v) => lgammaNumber(v + 1)))
      const lgy = lgamma(add(yv, theta))
      const base = sub(sub(lgy, lgamma(theta)), lg1)
      // theta log(theta / (theta + mu)) + y log(mu / (theta + mu)) with log(theta + mu) = softplus(eta - log theta) + log theta
      const logDen = add(softplus(sub(e, ltheta ?? 0)), ltheta ?? 0)
      return add(base, add(mul(theta, sub(ltheta ?? 0, logDen)), mul(yv, sub(e, logDen))))
    },
  })
}

function lgammaNumber(x: number): number {
  return item(lgamma(x))
}

/**
 * The zero-inflated Poisson (mgcv's `ziP`, with the inflation probability its own additive predictor): predictor 1 is
 * the log of the Poisson mean, predictor 2 the logit of the probability of a structural zero.
 *
 * @returns The family.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.zeroInflatedPoissonLogit().predictors // => ['count', 'zero']
 * ```
 */
export function zeroInflatedPoissonLogit(): ExtendedFamily {
  return tensorExtendedFamily({
    name: 'zeroInflatedPoisson',
    predictors: ['count', 'zero'],
    links: [logLink(), logit()],
    start: (y, w) => {
      let pos = 0
      let sum = 0
      let zero = 0
      let tot = 0
      for (let i = 0; i < y.length; i++) {
        const wi = w[i] ?? 1
        tot += wi
        if ((y[i] ?? 0) > 0) {
          pos += wi
          sum += wi * (y[i] ?? 0)
        } else zero += wi
      }
      const lambda = Math.max(sum / Math.max(pos, 1e-300), 0.5)
      const pi = Math.min(Math.max(zero / tot - Math.exp(-lambda), 0.05), 0.9)
      return [Math.log(lambda), logitOf(pi)]
    },
    logProb: (y, [e1, e2]) => {
      const lam = exp(e1 as Value)
      const isZero = vec(Array.from(y, (v) => (v === 0 ? 1 : 0)))
      const yv = vec(y)
      const lg1 = vec(Array.from(y, (v) => lgammaNumber(v + 1)))
      const logPi = neg(softplus(neg(e2 as Value)))
      const log1mPi = neg(softplus(e2 as Value))
      const positive = sub(add(log1mPi, sub(mul(yv, e1 as Value), lam)), lg1)
      // log(pi + (1 - pi) exp(-lam)) as a log-sum-exp of the two parts
      const a = logPi
      const b = sub(log1mPi, lam)
      const top = add(a, log(add(1, exp(sub(b, a)))))
      return add(mul(isZero, top), mul(sub(1, isZero), positive))
    },
  })
}

/**
 * A cumulative-link ordered categorical response with `K` categories (mgcv's `ocat`): `P(Y <= j) = F(c_j - eta)` with the
 * first cut point fixed at -1 (the additive predictor has an intercept) and the other `K - 2` estimated as aux
 * parameters, the logs of the increments between successive cut points.
 *
 * @param categories - The number of categories `K >= 3`; the response holds the codes `0 .. K - 1`.
 * @param link - `logit` (default) or `probit`.
 * @returns The family.
 * @throws MdmlError `value` for fewer than three categories.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.orderedCategorical(4).aux(Float64Array.of(0, 1, 2, 3, 1), Float64Array.of(1, 1, 1, 1, 1)).length // => 2
 * ```
 */
export function orderedCategorical(categories: number, link: 'logit' | 'probit' = 'logit'): ExtendedFamily {
  if (!Number.isInteger(categories) || categories < 3) {
    throw new MdmlError('value', 'orderedCategorical: needs at least three categories')
  }
  const cdf = (x: Value): Value => (link === 'logit' ? sigmoid(x) : ndtr(x))
  const inv = link === 'logit' ? logitOf : (p: number): number => item(ndtri(p))
  const cuts = (aux: readonly number[]): number[] => {
    const c = [-1]
    for (let j = 0; j < categories - 2; j++) c.push((c[j] ?? -1) + Math.exp(aux[j] ?? 0))
    return c
  }
  const proportions = (y: Float64Array, w: Float64Array): number[] => {
    const counts = new Array<number>(categories).fill(0)
    let tot = 0
    for (let i = 0; i < y.length; i++) {
      counts[Math.round(y[i] ?? 0)] = (counts[Math.round(y[i] ?? 0)] ?? 0) + (w[i] ?? 1)
      tot += w[i] ?? 1
    }
    let acc = 0
    return counts.slice(0, categories - 1).map((c) => {
      acc += c / tot
      return Math.min(Math.max(acc, 0.01), 0.99)
    })
  }
  return tensorExtendedFamily({
    name: `orderedCategorical(${link})`,
    predictors: ['location'],
    links: [identity()],
    aux: (y, w) => {
      const c = proportions(y, w).map((p) => inv(p))
      return Array.from({ length: categories - 2 }, (_, j) => ({
        name: `logIncrement${j + 1}`,
        start: Math.log(Math.max((c[j + 1] ?? 1) - (c[j] ?? 0), 0.05)),
        lower: -5,
        upper: 4,
      }))
    },
    start: (y, w) => [-1 - inv(proportions(y, w)[0] ?? 0.5)],
    logProb: (y, [eta], aux) => {
      const c = cuts(aux)
      const upper = vec(Array.from(y, (v) => (Math.round(v) >= categories - 1 ? 1e30 : (c[Math.round(v)] ?? 0))))
      const lower = vec(Array.from(y, (v) => (Math.round(v) <= 0 ? -1e30 : (c[Math.round(v) - 1] ?? 0))))
      return log(sub(cdf(sub(upper, eta as Value)), cdf(sub(lower, eta as Value))))
    },
  })
}

/**
 * The multinomial (categorical) logit model with `K` classes as `K - 1` additive predictors against class 0.
 *
 * @param classes - The number of classes `K >= 3` (use the binomial family for two); the response holds `0 .. K - 1`.
 * @returns The family, with predictors `class1 .. class(K-1)`.
 * @throws MdmlError `value` for fewer than three classes.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.multinomialLogit(3).predictors // => ['class1', 'class2']
 * ```
 */
export function multinomialLogit(classes: number): ExtendedFamily {
  if (!Number.isInteger(classes) || classes < 3)
    throw new MdmlError('value', 'multinomialLogit: needs at least three classes')
  const K = classes - 1
  return tensorExtendedFamily({
    name: 'multinomial',
    predictors: Array.from({ length: K }, (_, j) => `class${j + 1}`),
    links: Array.from({ length: K }, () => identity()),
    start: (y, w) => {
      const counts = new Array<number>(classes).fill(0.5)
      for (let i = 0; i < y.length; i++)
        counts[Math.round(y[i] ?? 0)] = (counts[Math.round(y[i] ?? 0)] ?? 0) + (w[i] ?? 1)
      return Array.from({ length: K }, (_, j) => Math.log((counts[j + 1] ?? 1) / (counts[0] ?? 1)))
    },
    logProb: (y, eta) => {
      // log p_y = eta_y - log(1 + sum_j exp eta_j), eta_0 = 0; the shift is a constant (any constant gives the same value)
      const n = y.length
      let top: Value = vec(new Float64Array(n))
      for (const e of eta) top = maximum(top, e)
      const shift = stopGradient(top)
      let acc: Value = exp(neg(shift))
      for (const e of eta) acc = add(acc, exp(sub(e, shift)))
      const pick = eta.map((_, j) => vec(Array.from(y, (v) => (Math.round(v) === j + 1 ? 1 : 0))))
      let own: Value = vec(new Float64Array(n))
      eta.forEach((e, j) => {
        own = add(own, mul(pick[j] as Value, e))
      })
      return sub(own, add(log(acc), shift))
    },
  })
}

/**
 * The Tweedie compound Poisson-gamma family with both the power `1 < p < 2` and the dispersion estimated (mgcv's `tw`):
 * log link, `log phi` and the logit of `p - 1` aux parameters, the density by Dunn and Smyth's series.
 *
 * @returns The family.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.tweedieEstimated().aux(Float64Array.of(0, 1, 2), Float64Array.of(1, 1, 1)).length // => 2
 * ```
 */
export function tweedieEstimated(): ExtendedFamily {
  return tensorExtendedFamily({
    name: 'tweedie',
    predictors: ['mean'],
    links: [logLink()],
    aux: (y, w) => {
      const m = mean(y, w)
      return [
        {
          name: 'logDispersion',
          start: Math.log(Math.max(variance(y, w) / Math.max(m ** 1.5, 1e-6), 0.1)),
          lower: -5,
          upper: 6,
        },
        { name: 'logitPower', start: 0, lower: -4, upper: 4 },
      ]
    },
    start: (y, w) => [Math.log(Math.max(mean(y, w), 0.05))],
    logProb: (y, [eta], [lphi, lp]) => {
      const power = 1 + logistic(lp ?? 0)
      return tweedieDistribution(exp(eta as Value), Math.exp(lphi ?? 0), power).logProb(vec(y))
    },
  })
}
