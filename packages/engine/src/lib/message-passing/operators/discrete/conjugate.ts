/**
 * The conjugate probability operators: a Bernoulli outcome with a beta probability, a discrete variable with a
 * Dirichlet probability vector, a Poisson count with a gamma rate, a binomial, geometric or negative binomial count with a
 * beta probability and an exponential or gamma value with a gamma rate. The
 * variational messages are the usual expected log densities, exact; the EP messages are exact marginals to the
 * outcome and a moment-matched mixture of two conjugate updates to the probability.
 */

import { arg } from '../../core/_messages.ts'
import { allProper, etaOf, massOf, numberParam, rawSum, scalarOf } from '../../core/args.ts'
import { bernoulli, beta } from '../../core/families.ts'
import { dirichlet, discreteEta, discreteProbs } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

/** The probability of true of a Bernoulli argument (its message, or its observed 0 or 1). */
function trueProbability(a: Arg): number {
  if (a.value !== undefined) return scalarOf(a) > 0 ? 1 : 0
  return bernoulli.moments(etaOf(a))[0] ?? 0.5
}

/**
 * b ~ Bernoulli(p) with p beta distributed: the conjugate pair of a coin and its bias. Variational messages are exact
 * (`E log p - E log(1 - p)` to the coin, `[P(b), 1 - P(b)]` to the bias); EP gives the logit of the mean to the coin and
 * to the bias the moment-matched mixture of the two conjugate updates (exact for an observed coin).
 *
 * @concept factor-operators
 * @index structure A Bernoulli outcome with a beta probability (conjugate).
 * @example
 * ```ts
 * import { bernoulliBeta, beta, betaEta, messageArg, valueArg } from 'omai-engine/message-passing'
 * import { bernoulli } from 'omai-engine/message-passing'
 * const args = [valueArg(bernoulli, 1), messageArg(beta, betaEta(2, 3))]
 * bernoulliBeta.vmp?.({ mode: 'vmp', to: 1, args, params: {}, power: 1 }).message // => [1, 0]
 * ```
 */
export const bernoulliBeta: FactorOperator = {
  name: 'bernoulli',
  description: 'b ~ Bernoulli(p) with p ~ Beta',
  ports: [
    { name: 'b', accepts: ['bernoulli'], constant: true },
    { name: 'p', accepts: ['beta'] },
  ],
  params: {},
  support: { ep: 'exact', vmp: 'exact' },
  readsTarget: (to) => to === 1,
  ep: (c): OperatorResult => {
    const pArg = arg(c, 1)
    const eta = etaOf(pArg)
    const a = (eta[0] ?? 0) + 1
    const b = (eta[1] ?? 0) + 1
    if (c.to === 0) {
      const mean = a / (a + b)
      return { message: [Math.log(mean) - Math.log1p(-mean)] }
    }
    const pi = trueProbability(arg(c, 0))
    if (pi === 1) return { message: [1, 0] }
    if (pi === 0) return { message: [0, 1] }
    const w1 = (pi * a) / (a + b)
    const w0 = ((1 - pi) * b) / (a + b)
    const m1 = beta.moments([a, b - 1])
    const m0 = beta.moments([a - 1, b])
    const total = w0 + w1
    const mixed = m1.map((v, i) => (w1 * v + w0 * (m0[i] ?? 0)) / total)
    return { message: beta.fromMoments(mixed).map((v, i) => v - (eta[i] ?? 0)) }
  },
  vmp: (c): OperatorResult => {
    const pArg = arg(c, 1)
    if (c.to === 0) {
      const mo = beta.moments(etaOf(pArg))
      return { message: [(mo[0] ?? 0) - (mo[1] ?? 0)] }
    }
    const pi = trueProbability(arg(c, 0))
    return { message: [pi, 1 - pi] }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const eta = etaOf(arg(c, 1))
    const mean = ((eta[0] ?? 0) + 1) / ((eta[0] ?? 0) + (eta[1] ?? 0) + 2)
    const b = arg(c, 0)
    if (b.value !== undefined) return massOf(arg(c, 1)) + Math.log(scalarOf(b) > 0 ? mean : 1 - mean)
    const pi = trueProbability(b)
    return massOf(arg(c, 1)) + massOf(b) + Math.log((1 - pi) * (1 - mean) + pi * mean)
  },
  expectedLog: (c) => {
    const mo = beta.moments(etaOf(arg(c, 1)))
    const pi = trueProbability(arg(c, 0))
    return pi * (mo[0] ?? 0) + (1 - pi) * (mo[1] ?? 0)
  },
}

/** The probabilities of a discrete argument: its message's, or the indicator of its observed state. */
function stateProbabilities(a: Arg, k: number): number[] {
  if (a.value !== undefined) {
    const out = new Array<number>(k).fill(0)
    out[Math.trunc(scalarOf(a))] = 1
    return out
  }
  return discreteProbs(etaOf(a))
}

/**
 * z ~ Discrete(pi) with pi Dirichlet distributed: a categorical variable and its probability vector (a mixture's
 * assignment and weights, a topic model's word draw). Variational messages are exact (`E log pi` to z, the
 * probabilities of z to pi); EP gives the log mean to z and to pi the Dirichlet moment-matched to the mixture of the
 * conjugate updates.
 *
 * @concept factor-operators
 * @index structure A discrete variable with a Dirichlet probability vector (conjugate).
 * @example
 * ```ts
 * import { dirichlet, discrete, discreteEta, discreteDirichlet, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(discrete(3), discreteEta([0.2, 0.3, 0.5])), messageArg(dirichlet(3), [1, 1, 1])]
 * discreteDirichlet.vmp?.({ mode: 'vmp', to: 1, args, params: {}, power: 1 }).message // => [0.2, 0.3, 0.5]
 * ```
 */
export const discreteDirichlet: FactorOperator = {
  name: 'discrete',
  description: 'z ~ Discrete(pi) with pi ~ Dirichlet',
  ports: [
    { name: 'z', accepts: ['discrete'], constant: true },
    { name: 'pi', accepts: ['dirichlet'] },
  ],
  params: {},
  support: { ep: 'exact', vmp: 'exact' },
  readsTarget: (to) => to === 1,
  ep: (c): OperatorResult => {
    const piArg = arg(c, 1)
    const eta = etaOf(piArg)
    const k = eta.length
    const alpha = eta.map((v) => v + 1)
    const total = alpha.reduce((a, b) => a + b, 0)
    if (c.to === 0) return { message: discreteEta(alpha.map((a) => a / total)) }
    const q = stateProbabilities(arg(c, 0), k)
    const family = dirichlet(k)
    if (q.some((v) => v === 1)) {
      const out = new Array<number>(k).fill(0)
      out[q.indexOf(1)] = 1
      return { message: out }
    }
    const weights = q.map((v, i) => v * (alpha[i] ?? 0))
    const norm = weights.reduce((a, b) => a + b, 0)
    const mixed = new Array<number>(k).fill(0)
    for (let i = 0; i < k; i++) {
      if ((weights[i] ?? 0) === 0) continue
      const comp = family.moments(eta.map((v, j) => (j === i ? v + 1 : v)))
      for (let j = 0; j < k; j++) mixed[j] = (mixed[j] ?? 0) + ((weights[i] ?? 0) / norm) * (comp[j] ?? 0)
    }
    return { message: family.fromMoments(mixed).map((v, i) => v - (eta[i] ?? 0)) }
  },
  vmp: (c): OperatorResult => {
    const piArg = arg(c, 1)
    const eta = etaOf(piArg)
    if (c.to === 0) return { message: dirichlet(eta.length).moments(eta) }
    return { message: stateProbabilities(arg(c, 0), eta.length) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const eta = etaOf(arg(c, 1))
    const alpha = eta.map((v) => v + 1)
    const total = alpha.reduce((a, b) => a + b, 0)
    const q = stateProbabilities(arg(c, 0), eta.length)
    const mean = q.reduce((s, v, i) => s + v * ((alpha[i] ?? 0) / total), 0)
    return massOf(arg(c, 1)) + massOf(arg(c, 0)) + Math.log(mean)
  },
  expectedLog: (c) => {
    const eta = etaOf(arg(c, 1))
    const mo = dirichlet(eta.length).moments(eta)
    const q = stateProbabilities(arg(c, 0), eta.length)
    return q.reduce((s, v, i) => (v > 0 ? s + v * (mo[i] ?? 0) : s), 0)
  },
}

/**
 * n ~ Poisson(exposure * rate) for an observed count n and a gamma-distributed rate: the message to the rate is the
 * likelihood `rate^n exp(-exposure rate)`, a gamma, exact under every rule.
 *
 * @concept factor-operators
 * @index structure An observed Poisson count with a gamma rate (conjugate).
 * @example
 * ```ts
 * import { gamma, gammaEta, gaussian, messageArg, poisson, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 4), messageArg(gamma, gammaEta(1, 1))]
 * poisson.ep?.({ mode: 'ep', to: 1, args, params: { exposure: 2 }, power: 1 }).message // => [4, -2]
 * ```
 */
export const poisson: FactorOperator = {
  name: 'poisson',
  description: 'n ~ Poisson(exposure * rate), n observed, rate ~ Gamma',
  ports: [
    { name: 'n', accepts: ['gaussian'], constant: true, observed: true },
    { name: 'rate', accepts: ['gamma'] },
  ],
  params: { exposure: param.positive({ optional: true }) },
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c) => ({ message: [scalarOf(arg(c, 0)), -numberParam(c.params, 'exposure', 1)] }),
  vmp: (c) => ({ message: [scalarOf(arg(c, 0)), -numberParam(c.params, 'exposure', 1)] }),
  logEvidence: (c) => {
    const n = scalarOf(arg(c, 0))
    const e = numberParam(c.params, 'exposure', 1)
    const gammaFamily = arg(c, 1).family
    return n * Math.log(e) - lgammaOf(n + 1) + gammaFamily.logPartition(rawSum(etaOf(arg(c, 1)), [n, -e]))
  },
  expectedLog: (c) => {
    const n = scalarOf(arg(c, 0))
    const e = numberParam(c.params, 'exposure', 1)
    const mo = arg(c, 1).family.moments(etaOf(arg(c, 1)))
    return n * (Math.log(e) + (mo[0] ?? 0)) - e * (mo[1] ?? 0) - lgammaOf(n + 1)
  },
}

/**
 * k ~ Binomial(n, p) for an observed count k and a beta-distributed probability p (`n` in the parameters): the message
 * to p is the likelihood `p^k (1 - p)^(n - k)`, a beta, exact under every rule.
 *
 * @concept factor-operators
 * @index structure An observed binomial count with a beta probability (conjugate).
 * @example
 * ```ts
 * import { beta, betaEta, binomial, gaussian, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 3), messageArg(beta, betaEta(1, 1))]
 * binomial.ep?.({ mode: 'ep', to: 1, args, params: { n: 10 }, power: 1 }).message // => [3, 7]
 * ```
 */
export const binomial: FactorOperator = {
  name: 'binomial',
  description: 'k ~ Binomial(n, p), k observed, p ~ Beta',
  ports: [
    { name: 'k', accepts: ['gaussian'], constant: true, observed: true },
    { name: 'p', accepts: ['beta'] },
  ],
  params: { n: param.integer({ min: 0 }) },
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c) => ({ message: [scalarOf(arg(c, 0)), numberParam(c.params, 'n') - scalarOf(arg(c, 0))] }),
  vmp: (c) => ({ message: [scalarOf(arg(c, 0)), numberParam(c.params, 'n') - scalarOf(arg(c, 0))] }),
  logEvidence: (c) => {
    const n = numberParam(c.params, 'n')
    const k = scalarOf(arg(c, 0))
    const logChoose = lgammaOf(n + 1) - lgammaOf(k + 1) - lgammaOf(n - k + 1)
    return logChoose + arg(c, 1).family.logPartition(rawSum(etaOf(arg(c, 1)), [k, n - k]))
  },
  expectedLog: (c) => {
    const n = numberParam(c.params, 'n')
    const k = scalarOf(arg(c, 0))
    const mo = beta.moments(etaOf(arg(c, 1)))
    return lgammaOf(n + 1) - lgammaOf(k + 1) - lgammaOf(n - k + 1) + k * (mo[0] ?? 0) + (n - k) * (mo[1] ?? 0)
  },
}

/**
 * x ~ Gamma(shape, c r) for an observed positive value x and a gamma-distributed rate r (the `shape` and the multiple `c`, as `scale`, in the
 * parameters, `c` default 1): the exponential waiting time (`shape` 1) and the gamma observation with a known shape, whose
 * likelihood `r^shape exp(-c x r)` is a gamma in `r`, so the message is exact under every rule.
 *
 * @concept factor-operators
 * @index structure An observed gamma or exponential value with a gamma rate (conjugate).
 * @example
 * ```ts
 * import { gamma, gammaEta, gammaObserved, gaussian, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 0.8), messageArg(gamma, gammaEta(2, 1))]
 * gammaObserved.ep?.({ mode: 'ep', to: 1, args, params: { shape: 1 }, power: 1 }).message // => [1, -0.8]
 * ```
 */
export const gammaObserved: FactorOperator = {
  name: 'gamma-observed',
  description: 'x ~ Gamma(shape, c r), x observed, r ~ Gamma (exponential when shape = 1)',
  ports: [
    { name: 'x', accepts: ['gaussian'], constant: true, observed: true },
    { name: 'r', accepts: ['gamma'] },
  ],
  params: { shape: param.positive({ optional: true }), scale: param.positive({ optional: true }) },
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c) => ({
    message: [numberParam(c.params, 'shape', 1), -numberParam(c.params, 'scale', 1) * scalarOf(arg(c, 0))],
  }),
  vmp: (c) => ({
    message: [numberParam(c.params, 'shape', 1), -numberParam(c.params, 'scale', 1) * scalarOf(arg(c, 0))],
  }),
  logEvidence: (c) => {
    const k = numberParam(c.params, 'shape', 1)
    const s = numberParam(c.params, 'scale', 1)
    const x = scalarOf(arg(c, 0))
    const constant = k * Math.log(s) + (k - 1) * Math.log(x) - lgammaOf(k)
    return constant + arg(c, 1).family.logPartition(rawSum(etaOf(arg(c, 1)), [k, -s * x]))
  },
  expectedLog: (c) => {
    const k = numberParam(c.params, 'shape', 1)
    const s = numberParam(c.params, 'scale', 1)
    const x = scalarOf(arg(c, 0))
    const mo = arg(c, 1).family.moments(etaOf(arg(c, 1)))
    return k * (Math.log(s) + (mo[0] ?? 0)) + (k - 1) * Math.log(x) - s * x * (mo[1] ?? 0) - lgammaOf(k)
  },
}

/**
 * k ~ NegativeBinomial(r, p) for an observed number of failures k before the `r`-th success (`successes` in the parameters,
 * default 1: the geometric distribution, with `k` the trials minus one) and a beta-distributed success probability p: the
 * likelihood `p^r (1 - p)^k` is a beta in `p`, so the message is exact under every rule.
 *
 * @concept factor-operators
 * @index structure An observed geometric or negative binomial count with a beta probability (conjugate).
 * @example
 * ```ts
 * import { beta, betaEta, gaussian, messageArg, negativeBinomialBeta, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 4), messageArg(beta, betaEta(1, 1))]
 * negativeBinomialBeta.ep?.({ mode: 'ep', to: 1, args, params: { successes: 1 }, power: 1 }).message // => [1, 4]
 * ```
 */
export const negativeBinomialBeta: FactorOperator = {
  name: 'negative-binomial-beta',
  description:
    'k ~ NegativeBinomial(r, p) failures before the r-th success, k observed, p ~ Beta (geometric when r = 1)',
  ports: [
    { name: 'k', accepts: ['gaussian'], constant: true, observed: true },
    { name: 'p', accepts: ['beta'] },
  ],
  params: { successes: param.positive({ optional: true }) },
  support: { ep: 'exact', vmp: 'exact' },
  ep: (c) => ({ message: [numberParam(c.params, 'successes', 1), scalarOf(arg(c, 0))] }),
  vmp: (c) => ({ message: [numberParam(c.params, 'successes', 1), scalarOf(arg(c, 0))] }),
  logEvidence: (c) => {
    const r = numberParam(c.params, 'successes', 1)
    const k = scalarOf(arg(c, 0))
    const logChoose = lgammaOf(k + r) - lgammaOf(k + 1) - lgammaOf(r)
    return logChoose + arg(c, 1).family.logPartition(rawSum(etaOf(arg(c, 1)), [r, k]))
  },
  expectedLog: (c) => {
    const r = numberParam(c.params, 'successes', 1)
    const k = scalarOf(arg(c, 0))
    const mo = beta.moments(etaOf(arg(c, 1)))
    return lgammaOf(k + r) - lgammaOf(k + 1) - lgammaOf(r) + r * (mo[0] ?? 0) + k * (mo[1] ?? 0)
  },
}

/** log(l1 p + l0 (1 - p)) at the coordinate p. */
const mixtureWeightLog =
  (c: OperatorContext) =>
  (x: readonly number[]): number =>
    Math.log(numberParam(c.params, 'l1') * (x[0] as number) + numberParam(c.params, 'l0') * (1 - (x[0] as number)))

/**
 * An observation of a two-component mixture with an unknown weight, the switch summed out: `l1 p + l0 (1 - p)` for a beta
 * probability `p`, where `l1` and `l0` (parameters) are the likelihoods of the observation under the second and the first
 * component. The cavity of `p` times it is the mixture of the two conjugate updates, weighted by `l1 E[p]` and `l0 E[1 - p]`; EP
 * matches it with a beta. It is what the explicit gate `z ~ Bernoulli(p)` with an observed pair of likelihoods sends to `p` when
 * the switch `z` has no other factor, so the two routes agree on `p`.
 *
 * @concept factor-operators
 * @index structure A two-component mixture likelihood with a beta weight: the switch summed out.
 * @example
 * ```ts
 * import { beta, betaEta, messageArg, mixtureWeight } from 'omai-engine/message-passing'
 * const args = [messageArg(beta, betaEta(2, 2))]
 * mixtureWeight.ep?.({ mode: 'ep', to: 0, args, params: { l1: 3, l0: 1 }, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.543', '0.082']
 * ```
 */
export const mixtureWeight: FactorOperator = {
  name: 'mixture-weight',
  description: 'f(p) = l1 p + l0 (1 - p): an observation of a two-component mixture with a beta weight',
  ports: [{ name: 'p', accepts: ['beta'] }],
  params: { l1: param.nonNegative(), l0: param.nonNegative() },
  support: { ep: 'exact', vmp: 'numeric' },
  readsTarget: () => true,
  // The mean field of the switch summed out: the expected log of `l1 p + l0 (1 - p)` is not linear in the beta's statistics, so the
  // message is the non-conjugate one (the explicit switch `z ~ Bernoulli(p)` with a gate has exact messages and a different bound).
  vmp: (c): OperatorResult => ({ message: numericMessage(quadraturePorts(c.args), 0, mixtureWeightLog(c)) }),
  expectedLog: (c) => numericExpectedLog(quadraturePorts(c.args), mixtureWeightLog(c)),
  ep: (c): OperatorResult => {
    const eta = etaOf(arg(c, 0))
    const a = (eta[0] ?? 0) + 1
    const b = (eta[1] ?? 0) + 1
    const l1 = numberParam(c.params, 'l1')
    const l0 = numberParam(c.params, 'l0')
    const w1 = (l1 * a) / (a + b)
    const w0 = (l0 * b) / (a + b)
    const m1 = beta.moments([a, b - 1])
    const m0 = beta.moments([a - 1, b])
    const total = w0 + w1
    const mixed = m1.map((v, i) => (w1 * v + w0 * (m0[i] ?? 0)) / total)
    return { message: beta.fromMoments(mixed).map((v, i) => v - (eta[i] ?? 0)) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const eta = etaOf(arg(c, 0))
    const mean = ((eta[0] ?? 0) + 1) / ((eta[0] ?? 0) + (eta[1] ?? 0) + 2)
    return massOf(arg(c, 0)) + Math.log(numberParam(c.params, 'l1') * mean + numberParam(c.params, 'l0') * (1 - mean))
  },
}
