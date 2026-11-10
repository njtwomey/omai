/**
 * The conjugate probability operators: a Bernoulli outcome with a beta probability, a discrete variable with a
 * Dirichlet probability vector, a Poisson count with a gamma rate and a binomial count with a beta probability. The
 * variational messages are the usual expected log densities, exact; the EP messages are exact marginals to the
 * outcome and a moment-matched mixture of two conjugate updates to the probability.
 */

import { lgammaOf } from '../_scalar.ts'
import { bernoulli, beta } from '../families.ts'
import { dirichlet, discreteEta, discreteProbs } from '../families-vector.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { allProper, etaOf, massOf, numberParam, rawSum, scalarOf } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

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
