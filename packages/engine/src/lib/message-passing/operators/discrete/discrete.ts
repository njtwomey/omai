/**
 * The operators of discrete structure: a table over discrete variables (sum-product belief propagation, and its
 * mean-field variational counterpart), the logic tables built on it, the gate that selects one of several variables
 * by a discrete index (the mixture), and the Gaussian mixture observation for variational inference. The table reuses
 * the discrete factors of `pgm` for the sums.
 */

import { logSumExpOf, OmaiError } from '../../../base/index.ts'
import { factor, factorProductAll, logFactor, logTotal, marginalise, reduceFactor } from '../../../pgm/factor.ts'
import { allProper, etaOf, ImproperMessageError, massOf, scalarOf, vectorParam } from '../../core/args.ts'
import { discreteEta, discreteProbs } from '../../core/families-vector.ts'
import type {
  Arg,
  Eta,
  FactorOperator,
  Family,
  OperatorContext,
  OperatorResult,
  Params,
  Port,
} from '../../core/types.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

const statesOf = (params: Params): readonly number[] => vectorParam(params, 'states')

/** The log probabilities of the states of a discrete or Bernoulli argument (an observed one is a point mass). */
function logProbabilities(a: Arg, k: number): number[] {
  if (a.value !== undefined)
    return Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(a)) ? 0 : Number.NEGATIVE_INFINITY))
  const lp = a.family.logProbs
  if (lp === undefined) throw new OmaiError('value', 'a table argument must be a Bernoulli or discrete variable')
  return [...lp(etaOf(a))]
}

const tableVariable = (i: number): string => `x${i}`

/** The table's message to one port from the other ports' incoming messages, by the sum-product of `pgm/factor`. */
function sumProduct(c: OperatorContext, to: number | undefined): { readonly logs: number[]; readonly logZ: number } {
  const states = statesOf(c.params)
  const values = vectorParam(c.params, 'values')
  const names = states.map((_, i) => tableVariable(i))
  let f = factor(names, [...states], values)
  const fixed: Record<string, number> = {}
  const parts = [] as ReturnType<typeof logFactor>[]
  for (const [i, k] of states.entries()) {
    const a = arg(c, i)
    if (a.value !== undefined) fixed[tableVariable(i)] = Math.trunc(scalarOf(a))
    else if (i !== to) parts.push(logFactor([tableVariable(i)], [k], logProbabilities(a, k)))
  }
  f = reduceFactor(f, fixed)
  const joint = factorProductAll([f, ...parts])
  if (to === undefined) return { logs: [], logZ: logTotal(joint) }
  const out = marginalise(
    joint,
    joint.variables.filter((v) => v !== tableVariable(to)),
  )
  return { logs: Array.from(out.logValues), logZ: logTotal(out) }
}

/** The discrete message from log weights, in the argument's family (Bernoulli or discrete). */
function toMessage(family: Family, logs: readonly number[]): Eta {
  if (family.fromLogProbs === undefined)
    throw new OmaiError('value', 'a table port must be a Bernoulli or discrete variable')
  return family.fromLogProbs(logs)
}

/** E_q[log f] over the product of the arguments' beliefs, summed over the states of every port but `to` (all when undefined). */
function expectedTable(c: OperatorContext, to: number | undefined): number[] {
  const states = statesOf(c.params)
  const values = vectorParam(c.params, 'values')
  const n = states.length
  const probs = states.map((k, i) => (i === to ? [] : stateProbabilitiesOf(arg(c, i), k)))
  const out = new Array<number>(to === undefined ? 1 : (states[to] ?? 1)).fill(0)
  const index = new Array<number>(n).fill(0)
  const total = states.reduce((a, b) => a * b, 1)
  for (let flat = 0; flat < total; flat++) {
    let r = flat
    for (let i = n - 1; i >= 0; i--) {
      index[i] = r % (states[i] ?? 1)
      r = Math.floor(r / (states[i] ?? 1))
    }
    let w = 1
    for (let i = 0; i < n; i++) if (i !== to) w *= probs[i]?.[index[i] ?? 0] ?? 0
    if (w === 0) continue
    const v = values[flat] ?? 0
    const slot = to === undefined ? 0 : (index[to] ?? 0)
    out[slot] = (out[slot] ?? 0) + w * (v > 0 ? Math.log(v) : Number.NEGATIVE_INFINITY)
  }
  return out
}

function stateProbabilitiesOf(a: Arg, k: number): number[] {
  if (a.value !== undefined) return Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(a)) ? 1 : 0))
  return logProbabilities(a, k).map((v, _, all) => Math.exp(v - logSumExpOf(all)))
}

function tablePorts(params: Params): readonly Port[] {
  return statesOf(params).map((_, i) => ({ name: `x${i}`, accepts: ['bernoulli', 'discrete'], constant: true }))
}

/**
 * A table over discrete variables: `values` (non-negative, row-major with the last variable varying fastest) over
 * variables with `states` states each (a Bernoulli variable has two). The message to a variable is the sum-product of
 * the table with the others' messages (exact belief propagation on a tree, loopy BP otherwise; the same rule under EP,
 * which on discrete variables is BP); the variational message is the mean-field expected log table. An observed
 * variable is a state index.
 *
 * @concept factor-operators
 * @index structure A table over discrete variables: sum-product and mean-field messages.
 * @example
 * ```ts
 * import { discrete, discreteEta, messageArg, table } from 'omai-engine/message-passing'
 * const d = discrete(2)
 * const args = [messageArg(d, d.flat), messageArg(d, discreteEta([0.2, 0.8]))]
 * const m = table.bp?.({ mode: 'bp', to: 0, args, params: { states: [2, 2], values: [0.9, 0.1, 0.4, 0.6] }, power: 1 }).message
 * m?.map((v) => Math.exp(v).toFixed(4)) // => ['0.3171', '0.6829']
 * ```
 */
export const table: FactorOperator = {
  name: 'table',
  description: 'a non-negative table over discrete variables',
  ports: tablePorts,
  support: { ep: 'exact', bp: 'exact', vmp: 'exact' },
  check: (params, families) => {
    const states = statesOf(params)
    const size = states.reduce((a, b) => a * b, 1)
    if (vectorParam(params, 'values').length !== size)
      return 'table: values must have the product of the states as its length'
    for (const [i, k] of states.entries()) {
      const f = families[i]
      if (f === undefined || (f.kind !== 'bernoulli' && f.kind !== 'discrete'))
        return `table: port ${i} must be a Bernoulli or discrete variable`
      if ((f.kind === 'bernoulli' ? 2 : f.dim) !== k)
        return `table: port ${i} has ${f.kind === 'bernoulli' ? 2 : f.dim} states, not ${k}`
    }
    return undefined
  },
  ep: (c): OperatorResult => ({ message: toMessage(arg(c, c.to).family, sumProduct(c, c.to).logs) }),
  bp: (c): OperatorResult => ({ message: toMessage(arg(c, c.to).family, sumProduct(c, c.to).logs) }),
  vmp: (c): OperatorResult => ({ message: toMessage(arg(c, c.to).family, expectedTable(c, c.to)) }),
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    let total = sumProduct(c, undefined).logZ
    for (const a of c.args) total += massOf(a)
    return total
  },
  expectedLog: (c) => expectedTable(c, undefined)[0] ?? Number.NaN,
}

/**
 * The values of a logic table over n Bernoulli variables, to give a `table` factor: the output (the first variable) is
 * the conjunction (`and`), disjunction (`or`) or parity (`xor`) of the inputs, or the negation (`not`, one input).
 *
 * @param kind - The logic function.
 * @param inputs - The number of inputs (1 for `not`).
 * @returns The `states` and `values` parameters of a `table` over the output followed by the inputs.
 * @example
 * ```ts
 * import { logicTable } from 'omai-engine/message-passing'
 * logicTable('and', 2).values // => [1, 1, 1, 0, 0, 0, 0, 1]
 * ```
 */
export function logicTable(
  kind: 'and' | 'or' | 'xor' | 'not',
  inputs: number,
): { readonly states: readonly number[]; readonly values: readonly number[] } {
  const n = kind === 'not' ? 1 : inputs
  const values: number[] = []
  for (let out = 0; out < 2; out++) {
    for (let mask = 0; mask < 2 ** n; mask++) {
      const bits = Array.from({ length: n }, (_, i) => (mask >> (n - 1 - i)) & 1)
      const result =
        kind === 'and'
          ? bits.every((b) => b === 1)
            ? 1
            : 0
          : kind === 'or'
            ? bits.some((b) => b === 1)
              ? 1
              : 0
            : kind === 'xor'
              ? bits.reduce((a, b) => a ^ b, 0)
              : 1 - (bits[0] ?? 0)
      values.push(out === result ? 1 : 0)
    }
  }
  return { states: Array.from({ length: n + 1 }, () => 2), values }
}

const selectPorts = (params: Params): readonly Port[] => {
  const k = Math.trunc(Number(params['components']))
  return [
    { name: 'y', accepts: ['gaussian', 'gamma', 'beta', 'bernoulli'] },
    { name: 'z', accepts: ['discrete', 'bernoulli'], constant: true },
    ...Array.from(
      { length: k },
      (_, i): Port => ({ name: `c${i}`, accepts: ['gaussian', 'gamma', 'beta', 'bernoulli'] }),
    ),
  ]
}

/** log of the mass shared by two unnormalised messages of a family: A(a + b) - A(a) - A(b) plus the two masses' own A. */
const logOverlap = (f: Family, a: Eta, b: Eta): number => f.logPartition(a.map((v, i) => v + (b[i] ?? 0)))

/**
 * y = c_z: the variable y is the z-th of several variables c_0, ..., c_{K-1} of one family, chosen by a discrete
 * variable z (the gate or switch of Infer.NET: a mixture, a regime, a lookup). The EP messages are the moment-matched
 * mixtures: to y the mixture of the cavities' overlaps, to z the overlaps themselves, to c_k the mixture of "y
 * constrains c_k" and "c_k is not selected". Gaussian, gamma, beta or Bernoulli y; the gate z is discrete or Bernoulli
 * (state 0 and 1 select the first and the second component). Variational message passing treats `y` as defined by the components (the mean
 * field of the model with `y` replaced by the selected one): its belief is their mixture, matched on the family's statistics.
 *
 * @concept factor-operators
 * @index structure The gate y = c_z selecting one of several variables by a discrete index.
 * @example
 * ```ts
 * import { discrete, discreteEta, gaussian, gaussianEta, messageArg, select } from 'omai-engine/message-passing'
 * const z = messageArg(discrete(2), discreteEta([0.5, 0.5]))
 * const g = (m: number) => messageArg(gaussian, gaussianEta(m, 1))
 * const args = [messageArg(gaussian, gaussian.flat), z, g(-1), g(1)]
 * select.ep?.({ mode: 'ep', to: 0, args, params: { components: 2 }, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.0000', '0.5000']
 * ```
 */
export const select: FactorOperator = {
  name: 'select',
  description: 'y = c_z: select one of several variables by a discrete index',
  ports: selectPorts,
  support: { ep: 'exact', vmp: 'exact' },
  readsTarget: (to) => to >= 2,
  // The output is the selected component, so under the mean field it is defined by the others: its belief is the mixture of the
  // components' beliefs (matched on the family's statistics, which is all a reader of `y` sees), the message to the switch is the
  // expected log of the output's cavity under each component, and the message to a component is that cavity weighted by the
  // probability of selecting it: the mean field of the model with `y` replaced by `c_z`.
  defines: 0,
  vmp: (c): OperatorResult => {
    const f = arg(c, 0).family
    const k = c.args.length - 2
    const moments = (i: number): Eta => f.moments(etaOf(arg(c, i + 2)))
    if (c.to === 0) {
      const weights = selectWeights(arg(c, 1), k)
      const mixed = new Array<number>(f.size).fill(0)
      for (let i = 0; i < k; i++) {
        const m = moments(i)
        for (let j = 0; j < mixed.length; j++) mixed[j] = (mixed[j] ?? 0) + (weights[i] ?? 0) * (m[j] ?? 0)
      }
      return { message: f.fromMoments(mixed) }
    }
    const cavity = etaOf(arg(c, 0))
    if (c.to === 1) {
      const logs = Array.from({ length: k }, (_, i) => f.pair(cavity, moments(i)))
      const z = arg(c, 1).family
      return { message: z.kind === 'bernoulli' ? (z.fromLogProbs?.(logs) as Eta) : logs }
    }
    const weights = selectWeights(arg(c, 1), k)
    const w = weights[c.to - 2] ?? 0
    return { message: cavity.map((v) => w * v) }
  },
  expectedLog: () => 0,
  check: (params, families) => {
    const k = Math.trunc(Number(params['components']))
    const states = families[1]?.kind === 'bernoulli' ? 2 : families[1]?.dim
    if (states !== k) return 'select: z must have as many states as there are components'
    const kind = families[0]?.kind
    for (let i = 2; i < families.length; i++)
      if (families[i]?.kind !== kind || families[i]?.size !== families[0]?.size)
        return 'select: y and every component must be of one family'
    return undefined
  },
  ep: (c): OperatorResult => {
    const f = arg(c, 0).family
    const k = c.args.length - 2
    const pi = (a: Arg): number[] =>
      a.value !== undefined
        ? discreteProbs(discreteEta(Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(a)) ? 1 : 0))))
        : selectProbs(a)
    const weights = pi(arg(c, 1))
    const etaY = arg(c, 0).eta
    const component = (i: number): Eta => etaOf(arg(c, i + 2))
    const total = (eta: Eta, other: Eta): Eta => eta.map((v, i) => v + (other[i] ?? 0))
    if (c.to === 0 || c.to === 1) {
      const y = etaY ?? f.flat
      const joint = Array.from({ length: k }, (_, i) => total(y, component(i)))
      const overlap = joint.map((e, i) =>
        f.isProper(e) ? f.logPartition(e) - f.logPartition(component(i)) : Number.NEGATIVE_INFINITY,
      )
      if (c.to === 1) {
        const top = Math.max(...overlap)
        if (!(top > Number.NEGATIVE_INFINITY))
          throw new ImproperMessageError('select: no component overlaps the output')
        const z = arg(c, 1).family
        return {
          message:
            z.kind === 'bernoulli'
              ? (z.fromLogProbs?.(overlap.map((v) => v - top)) as Eta)
              : discreteEta(overlap.map((v) => Math.exp(v - top))),
        }
      }
      const lw = overlap.map((v, i) => v + Math.log(weights[i] ?? 0))
      const top = Math.max(...lw)
      if (!(top > Number.NEGATIVE_INFINITY)) throw new ImproperMessageError('select: no component overlaps the output')
      const w = lw.map((v) => Math.exp(v - top))
      const norm = w.reduce((a, b) => a + b, 0)
      const mixed = new Array<number>(f.size).fill(0)
      for (const [i, e] of joint.entries()) {
        if ((w[i] ?? 0) === 0) continue
        const mo = f.moments(e)
        for (let j = 0; j < mo.length; j++) mixed[j] = (mixed[j] ?? 0) + ((w[i] ?? 0) / norm) * (mo[j] ?? 0)
      }
      return { message: f.fromMoments(mixed).map((v, i) => v - (y[i] ?? 0)) }
    }
    const kk = c.to - 2
    const y = etaY ?? f.flat
    const own = component(kk)
    const withY = total(own, y)
    const logM1 = Math.log(weights[kk] ?? 0) + f.logPartition(withY)
    const others = Array.from({ length: k }, (_, j) => j)
      .filter((j) => j !== kk && (weights[j] ?? 0) > 0)
      .map((j) => Math.log(weights[j] ?? 0) + f.logPartition(total(y, component(j))) - f.logPartition(component(j)))
    const logM2 = f.logPartition(own) + (others.length === 0 ? Number.NEGATIVE_INFINITY : logSumExpOf(others))
    const top = Math.max(logM1, logM2)
    const w1 = Math.exp(logM1 - top)
    const w2 = Math.exp(logM2 - top)
    const m1 = f.isProper(withY) && w1 > 0 ? f.moments(withY) : new Array<number>(f.size).fill(0)
    const m2 = w2 > 0 ? f.moments(own) : new Array<number>(f.size).fill(0)
    const mixed = m1.map((v, i) => (w1 * v + w2 * (m2[i] ?? 0)) / (w1 + w2))
    return { message: f.fromMoments(mixed).map((v, i) => v - (own[i] ?? 0)) }
  },
  logEvidence: (c) => {
    const f = arg(c, 0).family
    const k = c.args.length - 2
    const weights =
      arg(c, 1).value !== undefined
        ? Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(arg(c, 1))) ? 1 : 0))
        : selectProbs(arg(c, 1))
    const y = etaOf(arg(c, 0))
    const terms = Array.from({ length: k }, (_, i) => {
      const own = etaOf(arg(c, i + 2))
      return Math.log(weights[i] ?? 0) + logOverlap(f, y, own) - f.logPartition(own)
    })
    let total = massOf(arg(c, 1))
    for (let i = 0; i < k; i++) total += massOf(arg(c, i + 2))
    return total + logSumExpOf(terms)
  },
}

/** The probabilities of the switch of a select over `k` states, from its belief or its observed state. */
function selectWeights(a: Arg, k: number): number[] {
  return a.value !== undefined
    ? Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(a)) ? 1 : 0))
    : selectProbs(a)
}

/** The probabilities of the states of the gate argument of a select: a discrete or a Bernoulli message. */
function selectProbs(a: Arg): number[] {
  return a.family.kind === 'bernoulli' ? (a.family.logProbs?.(etaOf(a)) ?? []).map(Math.exp) : discreteProbs(etaOf(a))
}

const mixturePorts = (params: Params): readonly Port[] => {
  const k = Math.trunc(Number(params['components']))
  return [
    { name: 'x', accepts: ['gaussian'], constant: true },
    { name: 'z', accepts: ['discrete'] },
    ...Array.from({ length: k }, (_, i): Port => ({ name: `theta${i}`, accepts: ['normal-gamma'] })),
  ]
}

/**
 * x ~ N(mu_z, 1 / lambda_z) with (mu_k, lambda_k) normal-gamma for each of K components and a discrete assignment z:
 * the observation of a Gaussian mixture, for variational message passing (the Bayesian Gaussian mixture). The message
 * to z is the expected log density of x under each component, the message to a component its responsibility times
 * the normal-gamma likelihood message. Exact; x is observed (or a Gaussian variable read through its moments).
 *
 * @concept factor-operators
 * @index structure A Gaussian mixture observation with normal-gamma components (variational).
 * @example
 * ```ts
 * import { discrete, discreteEta, gaussian, gaussianMixture, messageArg, normalGamma, normalGammaEta, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 0), messageArg(discrete(2), discreteEta([1, 0])), messageArg(normalGamma, normalGammaEta(0, 1, 2, 1)), messageArg(normalGamma, normalGammaEta(5, 1, 2, 1))]
 * gaussianMixture.vmp?.({ mode: 'vmp', to: 2, args, params: { components: 2 }, power: 1 }).message // => [0, -0.5, 0, 0.5]
 * ```
 */
export const gaussianMixture: FactorOperator = {
  name: 'gaussian-mixture',
  description: 'x ~ N(mu_z, 1 / lambda_z) with normal-gamma components',
  ports: mixturePorts,
  support: { vmp: 'exact' },
  check: (params, families) =>
    families[1]?.dim === Math.trunc(Number(params['components']))
      ? undefined
      : 'gaussian-mixture: z must have as many states as there are components',
  vmp: (c): OperatorResult => {
    const x = arg(c, 0)
    const mean = x.value === undefined ? (etaOf(x)[0] ?? 0) / (etaOf(x)[1] ?? 1) : scalarOf(x)
    const variance = x.value === undefined ? 1 / (etaOf(x)[1] ?? 1) : 0
    const k = c.args.length - 2
    const zArg = arg(c, 1)
    const r =
      zArg.value !== undefined
        ? Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(zArg)) ? 1 : 0))
        : discreteProbs(etaOf(zArg))
    const second = variance + mean * mean
    if (c.to >= 2) {
      const w = r[c.to - 2] ?? 0
      return { message: [w * mean, -0.5 * w, -0.5 * w * second, 0.5 * w] }
    }
    const logs = Array.from({ length: k }, (_, i) => {
      const t = arg(c, i + 2)
      const mo = t.family.moments(etaOf(t))
      return (
        0.5 * ((mo[3] ?? 0) - Math.log(2 * Math.PI)) -
        0.5 * (second * (mo[2] ?? 0) - 2 * mean * (mo[0] ?? 0) + (mo[1] ?? 0))
      )
    })
    return { message: logs }
  },
  expectedLog: (c) => {
    const x = arg(c, 0)
    const mean = x.value === undefined ? (etaOf(x)[0] ?? 0) / (etaOf(x)[1] ?? 1) : scalarOf(x)
    const variance = x.value === undefined ? 1 / (etaOf(x)[1] ?? 1) : 0
    const k = c.args.length - 2
    const zArg = arg(c, 1)
    const r =
      zArg.value !== undefined
        ? Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(zArg)) ? 1 : 0))
        : discreteProbs(etaOf(zArg))
    const second = variance + mean * mean
    let total = 0
    for (let i = 0; i < k; i++) {
      if ((r[i] ?? 0) === 0) continue
      const t = arg(c, i + 2)
      const mo = t.family.moments(etaOf(t))
      total +=
        (r[i] ?? 0) *
        (0.5 * ((mo[3] ?? 0) - Math.log(2 * Math.PI)) -
          0.5 * (second * (mo[2] ?? 0) - 2 * mean * (mo[0] ?? 0) + (mo[1] ?? 0)))
    }
    return total
  },
}
