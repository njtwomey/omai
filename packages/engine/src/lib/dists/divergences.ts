/**
 * Divergences and distances between distributions: the Kullback-Leibler divergence in closed form for many pairs of
 * families (as torch.distributions.kl_divergence) with a differentiable Monte Carlo fallback, the Bhattacharyya
 * coefficient and distance, the Hellinger distance, and the 2-Wasserstein distance between Gaussians.
 *
 * @index function KL and other divergences between distributions.
 */

import { EULER_GAMMA, OmaiError } from '../base/index.ts'
import { cholesky, funmSymmetric, solveTriangular } from '../numerics/linalg/index.ts'
import { digamma, gamma as gammaFn, ive, lbeta, lgamma, logIv, logSigmoid } from '../numerics/special/index.ts'
import type { Key } from '../random/index.ts'
import {
  abs,
  add,
  cos,
  diagonal,
  div,
  exp,
  expandDims,
  greater,
  less,
  log,
  logicalOr,
  matmul,
  maximum,
  mean,
  mul,
  neg,
  pow,
  primalOf,
  sigmoid,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  swapaxes,
  toFlat,
  trace,
  type Value,
  where,
} from '../tensor/index.ts'
import type { Independent, Transformed } from './compose/structure.ts'
import { type Distribution, sumRightmost } from './distribution.ts'
import { exponentialFamily } from './exponential-family.ts'

const P = (d: Distribution, name: string): Value => {
  const v = d.params[name]
  if (v === undefined) throw new OmaiError('value', `divergences: ${d.name} has no parameter ${name}`)
  return v
}
const transposeLast = (m: Value): Value => swapaxes(m, -1, -2)

/** Location and covariance factor of a (possibly low-rank) multivariate normal. */
function gaussianParts(d: Distribution): { readonly loc: Value; readonly l: Value } | undefined {
  if (d.name === 'multivariateNormal') return { loc: P(d, 'loc'), l: P(d, 'scaleTril') }
  if (d.name === 'lowRankMultivariateNormal') {
    const cov = (d as unknown as { covariance(): Value }).covariance()
    return { loc: P(d, 'loc'), l: cholesky(cov) }
  }
  return undefined
}

function klGaussian(p: { loc: Value; l: Value }, q: { loc: Value; l: Value }): Value {
  const dim = p.loc.shape[p.loc.shape.length - 1] ?? 0
  // KL = (tr(Sq^-1 Sp) + |Lq^-1 (mq - mp)|^2 - d + log |Sq| - log |Sp|) / 2.
  const m = solveTriangular(q.l, p.l, { lower: true })
  const tr = sum(sum(square(m), -1), -1)
  const z = squeeze(solveTriangular(q.l, expandDims(sub(q.loc, p.loc), -1), { lower: true }), -1)
  const logDet = (l: Value): Value => mul(sum(log(diagonal(l, 0, -2, -1)), -1), 2)
  return mul(add(sub(add(tr, sum(square(z), -1)), dim), sub(logDet(q.l), logDet(p.l))), 0.5)
}

type Rule = (p: Distribution, q: Distribution) => Value

// prob * (log p - log q) with 0 * (-Infinity) taken as 0 (a zero-probability outcome, as a masked logit makes): the
// arguments of the masked branch are replaced as well so that no NaN reaches a derivative.
const klTerm = (prob: Value, logP: Value, logQ: Value): Value => {
  const positive = greater(prob, 0)
  return where(positive, mul(prob, sub(where(positive, logP, 0), where(positive, logQ, 0))), 0)
}
const klBernoulliLogits = (lp: Value, lq: Value): Value => {
  const pp = sigmoid(lp)
  return add(klTerm(pp, logSigmoid(lp), logSigmoid(lq)), klTerm(sub(1, pp), logSigmoid(neg(lp)), logSigmoid(neg(lq))))
}
const klCategoricalLogits = (lp: Value, lq: Value): Value => sum(klTerm(exp(lp), lp, lq), -1)
const klGamma = (ap: Value, bp: Value, aq: Value, bq: Value): Value =>
  add(
    add(sub(mul(sub(ap, aq), digamma(ap)), sub(lgamma(ap), lgamma(aq))), mul(aq, sub(log(bp), log(bq)))),
    div(mul(ap, sub(bq, bp)), bp),
  )
const klNormal = (mp: Value, sp: Value, mq: Value, sq: Value): Value =>
  sub(add(log(div(sq, sp)), div(add(square(sp), square(sub(mp, mq))), mul(square(sq), 2))), 0.5)

const RULES: Readonly<Record<string, Rule>> = {
  'normal/normal': (p, q) => klNormal(P(p, 'loc'), P(p, 'scale'), P(q, 'loc'), P(q, 'scale')),
  'logNormal/logNormal': (p, q) => klNormal(P(p, 'mu'), P(p, 'sigma'), P(q, 'mu'), P(q, 'sigma')),
  'halfNormal/halfNormal': (p, q) =>
    klNormal(mul(P(p, 'scale'), 0), P(p, 'scale'), mul(P(q, 'scale'), 0), P(q, 'scale')),
  'gamma/gamma': (p, q) => klGamma(P(p, 'concentration'), P(p, 'rate'), P(q, 'concentration'), P(q, 'rate')),
  // 1 / X of an inverse gamma is a gamma with rate beta, and KL is invariant under the bijection.
  'inverseGamma/inverseGamma': (p, q) =>
    klGamma(P(p, 'concentration'), P(p, 'scale'), P(q, 'concentration'), P(q, 'scale')),
  // Chi-square(k) is Gamma(k / 2, rate 1 / 2).
  'chi2/chi2': (p, q) => {
    const half = (d: Distribution): Value => mul(P(d, 'df'), 0.5)
    return klGamma(half(p), add(mul(half(p), 0), 0.5), half(q), add(mul(half(q), 0), 0.5))
  },
  'exponential/exponential': (p, q) => {
    const ratio = div(P(q, 'rate'), P(p, 'rate'))
    return sub(sub(ratio, log(ratio)), 1)
  },
  'exponential/gamma': (p, q) =>
    klGamma(add(mul(P(p, 'rate'), 0), 1), P(p, 'rate'), P(q, 'concentration'), P(q, 'rate')),
  'beta/beta': (p, q) => {
    const [ap, bp, aq, bq] = [P(p, 'a'), P(p, 'b'), P(q, 'a'), P(q, 'b')]
    return add(
      add(add(sub(lbeta(aq, bq), lbeta(ap, bp)), mul(sub(ap, aq), digamma(ap))), mul(sub(bp, bq), digamma(bp))),
      mul(add(sub(aq, ap), sub(bq, bp)), digamma(add(ap, bp))),
    )
  },
  'dirichlet/dirichlet': (p, q) => {
    const ap = P(p, 'concentration')
    const aq = P(q, 'concentration')
    const sp = sum(ap, -1)
    return add(
      sub(sub(lgamma(sp), lgamma(sum(aq, -1))), sum(sub(lgamma(ap), lgamma(aq)), -1)),
      sum(mul(sub(ap, aq), sub(digamma(ap), expandDims(digamma(sp), -1))), -1),
    )
  },
  'bernoulli/bernoulli': (p, q) => klBernoulliLogits(P(p, 'logits'), P(q, 'logits')),
  'binomial/binomial': (p, q) => {
    const toLogit = (x: Value): Value => sub(log(x), log(sub(1, x)))
    return mul(P(p, 'n'), klBernoulliLogits(toLogit(P(p, 'probs')), toLogit(P(q, 'probs'))))
  },
  'categorical/categorical': (p, q) => klCategoricalLogits(P(p, 'logits'), P(q, 'logits')),
  'oneHotCategorical/oneHotCategorical': (p, q) => klCategoricalLogits(P(p, 'logits'), P(q, 'logits')),
  'multinomial/multinomial': (p, q) => {
    const n = p.support.kind === 'counts' ? p.support.total : 0
    return mul(klCategoricalLogits(log(P(p, 'probs')), log(P(q, 'probs'))), n)
  },
  'poisson/poisson': (p, q) => {
    const [lp, lq] = [P(p, 'rate'), P(q, 'rate')]
    return add(sub(mul(lp, sub(log(lp), log(lq))), lp), lq)
  },
  'geometric/geometric': (p, q) => {
    const [pp, pq] = [P(p, 'probs'), P(q, 'probs')]
    // (E[X] - 1) (log(1 - p) - log(1 - q)) + log p - log q with E[X] = 1 / p.
    return add(mul(div(sub(1, pp), pp), sub(log(sub(1, pp)), log(sub(1, pq)))), sub(log(pp), log(pq)))
  },
  'laplace/laplace': (p, q) => {
    const [mp, bp, mq, bq] = [P(p, 'loc'), P(p, 'scale'), P(q, 'loc'), P(q, 'scale')]
    const d = abs(sub(mp, mq))
    return sub(add(add(log(div(bq, bp)), div(d, bq)), mul(div(bp, bq), exp(neg(div(d, bp))))), 1)
  },
  'gumbel/gumbel': (p, q) => {
    const ct1 = div(P(p, 'scale'), P(q, 'scale'))
    const ct2 = div(P(q, 'loc'), P(q, 'scale'))
    const ct3 = div(P(p, 'loc'), P(q, 'scale'))
    const t1 = add(sub(neg(log(ct1)), ct2), ct3)
    const t3 = exp(sub(add(ct2, lgamma(add(ct1, 1))), ct3))
    return sub(add(add(t1, mul(ct1, EULER_GAMMA)), t3), 1 + EULER_GAMMA)
  },
  'uniform/uniform': (p, q) => {
    const [ap, bp, aq, bq] = [P(p, 'low'), P(p, 'high'), P(q, 'low'), P(q, 'high')]
    const outside = logicalOr(less(ap, aq), greater(bp, bq))
    return where(outside, Number.POSITIVE_INFINITY, log(div(sub(bq, aq), sub(bp, ap))))
  },
  'pareto/pareto': (p, q) => {
    const scaleRatio = div(P(p, 'scale'), P(q, 'scale'))
    const alphaRatio = div(P(q, 'b'), P(p, 'b'))
    const value = sub(add(sub(mul(P(q, 'b'), log(scaleRatio)), log(alphaRatio)), alphaRatio), 1)
    return where(less(P(p, 'scale'), P(q, 'scale')), Number.POSITIVE_INFINITY, value)
  },
  'weibull/weibull': (p, q) => {
    const [kp, lp, kq, lq] = [P(p, 'concentration'), P(p, 'scale'), P(q, 'concentration'), P(q, 'scale')]
    // log(kp / lp^kp) - log(kq / lq^kq) + (kp - kq)(log lp - gamma / kp) + (lp / lq)^kq Gamma(kq / kp + 1) - 1.
    const first = sub(sub(log(kp), mul(kp, log(lp))), sub(log(kq), mul(kq, log(lq))))
    const second = mul(sub(kp, kq), sub(log(lp), div(EULER_GAMMA, kp)))
    const third = mul(pow(div(lp, lq), kq), gammaFn(add(div(kq, kp), 1)))
    return sub(add(add(first, second), third), 1)
  },
  'vonMises/vonMises': (p, q) => {
    const [mp, kp, mq, kq] = [P(p, 'loc'), P(p, 'concentration'), P(q, 'loc'), P(q, 'concentration')]
    const a = div(ive(1, kp), ive(0, kp))
    return add(sub(logIv(0, kq), logIv(0, kp)), mul(a, sub(kp, mul(kq, cos(sub(mp, mq))))))
  },
}

/** The KL divergence between two members of the same exponential family with no fixed parameters. */
function klExponentialFamily(p: Distribution, q: Distribution): Value | undefined {
  if (p.name !== q.name || !FREE_FAMILIES.has(p.name)) return undefined
  const ep = exponentialFamily(p)
  const eq = exponentialFamily(q)
  const means = ep.meanParameters()
  // A(eta_q) - A(eta_p) - (eta_q - eta_p) . grad A(eta_p).
  let total = sub(eq.logPartition(), ep.logPartition())
  for (const [i, m] of means.entries()) {
    const diff = sub(eq.natural[i] as Value, ep.natural[i] as Value)
    total = sub(total, sumRightmost(mul(diff, m), ep.partRanks[i] ?? 0))
  }
  return total
}

// Families whose exponential-family form has no fixed parameters, so any two members are comparable.
const FREE_FAMILIES = new Set([
  'normal',
  'logNormal',
  'halfNormal',
  'exponential',
  'gamma',
  'inverseGamma',
  'inverseGaussian',
  'beta',
  'bernoulli',
  'poisson',
  'geometric',
  'categorical',
  'oneHotCategorical',
  'dirichlet',
  'multivariateNormal',
  'wishart',
  'vonMises',
])

/** Options for Monte Carlo fallbacks. */
export interface DivergenceOptions {
  /** A key enabling a Monte Carlo estimate when there is no closed form. */
  readonly key?: Key
  /** The number of Monte Carlo draws (default 1000). */
  readonly samples?: number
}

/**
 * The Monte Carlo estimate of KL(p || q) = E_p[log p(X) - log q(X)], from reparameterised draws when p has them (so
 * the estimate differentiates in both distributions' parameters).
 *
 * @param p - The first distribution.
 * @param q - The second.
 * @param key - The random key.
 * @param samples - The number of draws (default 1000).
 * @returns The estimate, with the batch shape.
 * @example
 * ```ts
 * import { klMonteCarlo, normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(klMonteCarlo(normal(0, 1), normal(1, 1), key(0), 20000)) - 0.5) < 0.05 // => true
 * ```
 */
export function klMonteCarlo(p: Distribution, q: Distribution, key: Key, samples = 1000): Value {
  const x = p.hasRsample ? p.rsample(key, [samples]) : p.sample(key, [samples])
  return mean(sub(p.logProb(x), q.logProb(x)), 0)
}

/**
 * The Kullback-Leibler divergence KL(p || q) (torch.distributions.kl_divergence): closed forms for the normal,
 * log-normal, half-normal, multivariate normal (full or low-rank), gamma, inverse gamma, chi-square, exponential
 * (also against a gamma), beta, Dirichlet, Bernoulli, binomial, categorical, one-hot categorical, multinomial,
 * Poisson, geometric, Laplace, Gumbel, uniform, Pareto, Weibull and von Mises pairs; sums for `independent`
 * distributions and invariance for `transformed` ones with the same bijector; the exponential-family identity
 * A(eta_q) - A(eta_p) - (eta_q - eta_p) . grad A(eta_p) for other members of one family; and a Monte Carlo estimate
 * when a key is given.
 *
 * @param p - The first distribution.
 * @param q - The second.
 * @param options - A key (and sample count) for the Monte Carlo fallback.
 * @returns KL(p || q) with the batch shape (Infinity where p is not absolutely continuous with respect to q).
 * @throws OmaiError with code `not-implemented` when there is no closed form and no key.
 * @example
 * ```ts
 * import { klDivergence, normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(klDivergence(normal(0, 1), normal(1, 1))) // => 0.5
 * ```
 */
export function klDivergence(p: Distribution, q: Distribution, options: DivergenceOptions = {}): Value {
  const rule = RULES[`${p.name}/${q.name}`]
  if (rule) return rule(p, q)
  const gp = gaussianParts(p)
  const gq = gaussianParts(q)
  if (gp && gq) return klGaussian(gp, gq)
  if (p.name === 'independent' && q.name === 'independent') {
    const ip = p as Independent
    const iq = q as Independent
    if (ip.reinterpretedBatchRank === iq.reinterpretedBatchRank)
      return sumRightmost(klDivergence(ip.base, iq.base, options), ip.reinterpretedBatchRank)
  }
  if (p.name === 'transformed' && q.name === 'transformed') {
    const tp = p as Transformed
    const tq = q as Transformed
    if (tp.bijector === tq.bijector) return klDivergence(tp.base, tq.base, options)
  }
  const ef = klExponentialFamily(p, q)
  if (ef) return ef
  if (options.key !== undefined) return klMonteCarlo(p, q, options.key, options.samples ?? 1000)
  throw new OmaiError(
    'not-implemented',
    `klDivergence: no closed form for ${p.name} and ${q.name}; pass a key for a Monte Carlo estimate`,
  )
}

/** Whether two distributions' fixed exponential-family parameters (trials, successes) agree. */
function sameFixed(p: Distribution, q: Distribution, names: readonly string[]): boolean {
  return names.every((n) => {
    const a = p.params[n]
    const b = q.params[n]
    if (a === undefined || b === undefined) return false
    const fa = toFlat(primalOf(a))
    const fb = toFlat(primalOf(b))
    return fa.length === fb.length && fa.every((v, i) => v === fb[i])
  })
}

/**
 * The Bhattacharyya coefficient BC(p, q) = integral of sqrt(p q): closed forms for normals and multivariate normals,
 * sums over categories for discrete distributions, the exponential-family identity
 * exp(A((eta_p + eta_q) / 2) - (A(eta_p) + A(eta_q)) / 2) for two members of one family, and a Monte Carlo estimate
 * E_p[sqrt(q / p)] when a key is given.
 *
 * @param p - The first distribution.
 * @param q - The second.
 * @param options - A key (and sample count) for the Monte Carlo fallback.
 * @returns BC in [0, 1].
 * @throws OmaiError with code `not-implemented` when there is no closed form and no key.
 * @example
 * ```ts
 * import { bhattacharyyaCoefficient, normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(bhattacharyyaCoefficient(normal(0, 1), normal(0, 1))) // => 1
 * ```
 */
export function bhattacharyyaCoefficient(p: Distribution, q: Distribution, options: DivergenceOptions = {}): Value {
  if (p.name === 'normal' && q.name === 'normal') {
    const [mp, sp, mq, sq] = [P(p, 'loc'), P(p, 'scale'), P(q, 'loc'), P(q, 'scale')]
    const v = add(square(sp), square(sq))
    return mul(sqrt(div(mul(mul(sp, sq), 2), v)), exp(neg(div(square(sub(mp, mq)), mul(v, 4)))))
  }
  const gp = gaussianParts(p)
  const gq = gaussianParts(q)
  if (gp && gq) {
    // D_B = |L^-1 (mp - mq)|^2 / 8 + (log |S| - (log |Sp| + log |Sq|) / 2) / 2, S = (Sp + Sq) / 2.
    const sp = matmul(gp.l, transposeLast(gp.l))
    const sq = matmul(gq.l, transposeLast(gq.l))
    const l = cholesky(mul(add(sp, sq), 0.5))
    const z = squeeze(solveTriangular(l, expandDims(sub(gp.loc, gq.loc), -1), { lower: true }), -1)
    const logDet = (m: Value): Value => mul(sum(log(diagonal(m, 0, -2, -1)), -1), 2)
    const distance = add(
      mul(sum(square(z), -1), 0.125),
      mul(sub(logDet(l), mul(add(logDet(gp.l), logDet(gq.l)), 0.5)), 0.5),
    )
    return exp(neg(distance))
  }
  if ((p.name === 'categorical' || p.name === 'oneHotCategorical') && p.name === q.name)
    return sum(exp(mul(add(P(p, 'logits'), P(q, 'logits')), 0.5)), -1)
  const fixedNames: Readonly<Record<string, readonly string[]>> = {
    binomial: ['n'],
    negativeBinomial: ['n'],
    multinomial: [],
  }
  const comparable =
    p.name === q.name &&
    (FREE_FAMILIES.has(p.name) || (fixedNames[p.name] !== undefined && sameFixed(p, q, fixedNames[p.name] ?? [])))
  if (
    comparable &&
    (p.name !== 'multinomial' ||
      (p.support.kind === 'counts' && q.support.kind === 'counts' && p.support.total === q.support.total))
  ) {
    const ep = exponentialFamily(p)
    const eq = exponentialFamily(q)
    const middle = ep.natural.map((e, i) => mul(add(e, eq.natural[i] as Value), 0.5))
    return exp(sub(ep.logPartition(middle), mul(add(ep.logPartition(), eq.logPartition()), 0.5)))
  }
  if (options.key !== undefined) {
    const x = p.hasRsample
      ? p.rsample(options.key, [options.samples ?? 1000])
      : p.sample(options.key, [options.samples ?? 1000])
    return mean(exp(mul(sub(q.logProb(x), p.logProb(x)), 0.5)), 0)
  }
  throw new OmaiError(
    'not-implemented',
    `bhattacharyyaCoefficient: no closed form for ${p.name} and ${q.name}; pass a key for a Monte Carlo estimate`,
  )
}

/**
 * The Bhattacharyya distance -log BC(p, q).
 *
 * @param p - The first distribution.
 * @param q - The second.
 * @param options - As for {@link bhattacharyyaCoefficient}.
 * @returns The distance (non-negative).
 * @example
 * ```ts
 * import { bhattacharyyaDistance, normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * // Equal scales: (mu_p - mu_q)^2 / (8 sigma^2).
 * Math.abs(item(bhattacharyyaDistance(normal(0, 1), normal(2, 1))) - 0.5) < 1e-15 // => true
 * ```
 */
export function bhattacharyyaDistance(p: Distribution, q: Distribution, options: DivergenceOptions = {}): Value {
  return neg(log(bhattacharyyaCoefficient(p, q, options)))
}

/**
 * The Hellinger distance sqrt(1 - BC(p, q)), a metric on distributions bounded by 1.
 *
 * @param p - The first distribution.
 * @param q - The second.
 * @param options - As for {@link bhattacharyyaCoefficient}.
 * @returns The distance in [0, 1].
 * @example
 * ```ts
 * import { hellinger, normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(hellinger(normal(0, 1), normal(0, 1))) // => 0
 * ```
 */
export function hellinger(p: Distribution, q: Distribution, options: DivergenceOptions = {}): Value {
  return sqrt(maximum(sub(1, bhattacharyyaCoefficient(p, q, options)), 0))
}

/**
 * The 2-Wasserstein distance between Gaussians: sqrt((mu_p - mu_q)^2 + (sigma_p - sigma_q)^2) for normals and
 * sqrt(|mu_p - mu_q|^2 + tr(S_p + S_q - 2 (S_q^(1/2) S_p S_q^(1/2))^(1/2))) for multivariate normals, the matrix roots
 * through symmetric eigendecompositions (differentiable).
 *
 * @param p - A normal or multivariate normal.
 * @param q - Another of the same kind.
 * @returns W_2(p, q).
 * @throws OmaiError with code `value` for other distributions.
 * @example
 * ```ts
 * import { normal, wasserstein2Gaussian } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(wasserstein2Gaussian(normal(0, 1), normal(3, 5))) // => 5
 * ```
 */
export function wasserstein2Gaussian(p: Distribution, q: Distribution): Value {
  if (p.name === 'normal' && q.name === 'normal')
    return sqrt(add(square(sub(P(p, 'loc'), P(q, 'loc'))), square(sub(P(p, 'scale'), P(q, 'scale')))))
  const gp = gaussianParts(p)
  const gq = gaussianParts(q)
  if (!gp || !gq) throw new OmaiError('value', 'wasserstein2Gaussian: needs two (multivariate) normals')
  const sp = matmul(gp.l, transposeLast(gp.l))
  const sq = matmul(gq.l, transposeLast(gq.l))
  const root = (m: Value): Value => funmSymmetric(m, (lam) => sqrt(maximum(lam, 0)))
  const rq = root(sq)
  const cross = root(matmul(matmul(rq, sp), rq))
  const traceTerm = sub(add(trace(sp), trace(sq)), mul(trace(cross), 2))
  return sqrt(maximum(add(sum(square(sub(gp.loc, gq.loc)), -1), traceTerm), 0))
}
