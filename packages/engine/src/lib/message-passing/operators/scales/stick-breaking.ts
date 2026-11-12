/**
 * The stick-breaking factor: a discrete variable whose probabilities are the stick-breaking weights of beta variables, the
 * assignment of a (truncated) Dirichlet-process mixture. State `k` has probability `v_k (1 - v_0) ... (1 - v_{k-1})` and the last
 * state takes what is left, so `K` states have `K - 1` breaks. Under expectation propagation the cavities of the breaks are
 * independent, which makes every message a short sum: exact to the discrete variable, the moment-matched mixture of three
 * betas to a break, and exact under variational message passing.
 */

import { arg } from '../../core/_messages.ts'
import { allProper, etaOf, massOf, numberParam, scalarOf } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { ParameterError } from '../../core/errors.ts'
import { beta } from '../../core/families.ts'
import { discreteEta, discreteProbs } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult, Params, Port } from '../../core/types.ts'
import { unitArray } from '../../numerics/_arrays.ts'

const sticksOf = (params: Params): number => {
  const k = Math.trunc(numberParam(params, 'sticks'))
  if (!(k >= 1)) throw new ParameterError('stick-breaking: sticks must be at least 1')
  return k
}

function ports(params: Params): readonly Port[] {
  return [
    { name: 'z', accepts: ['discrete'], constant: true },
    ...Array.from({ length: sticksOf(params) }, (_, j): Port => ({ name: `v${j}`, accepts: ['beta'] })),
  ]
}

/** The state probabilities of the assignment argument: its message, or the indicator of its observed state. */
function stateProbabilities(a: Arg, k: number): number[] {
  if (a.value !== undefined) {
    return unitArray(k, Math.trunc(scalarOf(a)))
  }
  return discreteProbs(etaOf(a))
}

/** The shapes `(a, b)` of each break's (cavity) beta, and the mean of each. */
function breaks(c: OperatorContext, sticks: number): { a: number[]; b: number[]; mean: number[] } {
  const a: number[] = []
  const b: number[] = []
  for (let j = 0; j < sticks; j++) {
    const eta = etaOf(arg(c, 1 + j))
    a.push((eta[0] ?? 0) + 1)
    b.push((eta[1] ?? 0) + 1)
  }
  return { a, b, mean: a.map((v, j) => v / (v + (b[j] as number))) }
}

/** `E[p(z = k | v)]` over independent betas of the given means, for every state. */
function stateMeans(mean: readonly number[]): number[] {
  const sticks = mean.length
  const out: number[] = []
  let rest = 1
  for (let k = 0; k < sticks; k++) {
    out.push(rest * (mean[k] as number))
    rest *= 1 - (mean[k] as number)
  }
  out.push(rest)
  return out
}

/**
 * z ~ Discrete(pi) with `pi` the stick-breaking weights of beta breaks `v_0, ..., v_{K-2}`: `pi_k = v_k prod_{j<k} (1 - v_j)` and
 * `pi_{K-1} = prod_j (1 - v_j)` (parameter `sticks` = `K - 1`). The EP message to `z` is the mean weights under the breaks'
 * cavities, the message to a break the beta that matches (on the log moments, as the conjugate operators do) the cavity times
 * `r_j v + (sum of r_k over the later states) (1 - v) + (the earlier states)`, a mixture of three betas, where `r` is the cavity of
 * `z`; the variational messages are the expected logs. Exact for a tree, and the building block of a Dirichlet-process mixture.
 *
 * @concept factor-operators
 * @index structure Discrete variable with stick-breaking weights of beta breaks (truncated Dirichlet process).
 * @example
 * ```ts
 * import { beta, betaEta, discrete, discreteEta, messageArg, stickBreaking } from 'omai-engine/message-passing'
 * const args = [messageArg(discrete(3), discreteEta([0.2, 0.3, 0.5])), messageArg(beta, betaEta(1, 1)), messageArg(beta, betaEta(1, 1))]
 * stickBreaking.vmp?.({ mode: 'vmp', to: 1, args, params: { sticks: 2 }, power: 1 }).message // => [0.2, 0.8]
 * ```
 */
export const stickBreaking: FactorOperator = defineOperator({
  name: 'stick-breaking',
  description: 'z ~ Discrete(pi) with pi the stick-breaking weights of beta breaks',
  ports: ports,
  params: { sticks: param.integer({ min: 1 }) },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const sticks = sticksOf(c.params)
      const { a, b, mean } = breaks(c, sticks)
      const weights = stateMeans(mean)
      if (c.to === 0) return { message: discreteEta(weights) }
      const j = c.to - 1
      const r = stateProbabilities(arg(c, 0), sticks + 1)
      // The expectation of p(z = k | v) over the other breaks, for each state k, as the part of it that does not depend on v_j.
      const others = mean.map((m, i) => (i === j ? 1 : m))
      const gone = mean.map((m, i) => (i === j ? 1 : 1 - m))
      let before = 0 // states below j: independent of v_j
      let rest = 1
      for (let k = 0; k < j; k++) {
        before += (r[k] as number) * rest * (others[k] as number)
        rest *= gone[k] as number
      }
      // State j takes v_j times the earlier complements, the later states (1 - v_j) times the other breaks and complements.
      const atJ = (r[j] as number) * rest
      let later = 0
      let run = rest
      for (let k = j + 1; k <= sticks; k++) {
        later += (r[k] as number) * run * (k < sticks ? (others[k] as number) : 1)
        run *= k < sticks ? (gone[k] as number) : 1
      }
      const aj = a[j] as number
      const bj = b[j] as number
      const mj = mean[j] as number
      const weight = [before, atJ * mj, later * (1 - mj)]
      const total = (weight[0] as number) + (weight[1] as number) + (weight[2] as number)
      const eta = etaOf(arg(c, c.to))
      if (!(total > 0)) return { message: [0, 0] }
      const parts = [beta.moments([aj - 1, bj - 1]), beta.moments([aj, bj - 1]), beta.moments([aj - 1, bj])]
      const mixed = [0, 1].map(
        (i) => weight.reduce((s, w, m) => s + w * ((parts[m] as number[])[i] as number), 0) / total,
      )
      return { message: beta.fromMoments(mixed).map((v, i) => v - (eta[i] ?? 0)) }
    }),
    vmp: messageRule((c): OperatorResult => {
      const sticks = sticksOf(c.params)
      const r = stateProbabilities(arg(c, 0), sticks + 1)
      if (c.to === 0) {
        const out: number[] = []
        let rest = 0
        for (let k = 0; k <= sticks; k++) {
          const mo = k < sticks ? beta.moments(etaOf(arg(c, 1 + k))) : [0, 0]
          out.push(rest + (k < sticks ? (mo[0] as number) : 0))
          rest += k < sticks ? (mo[1] as number) : 0
        }
        return { message: out }
      }
      const j = c.to - 1
      let later = 0
      for (let k = j + 1; k <= sticks; k++) later += r[k] as number
      return { message: [r[j] as number, later] }
    }),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const sticks = sticksOf(c.params)
      const { mean } = breaks(c, sticks)
      const r = stateProbabilities(arg(c, 0), sticks + 1)
      const weights = stateMeans(mean)
      let total = massOf(arg(c, 0))
      for (let j = 0; j < sticks; j++) total += massOf(arg(c, 1 + j))
      return total + Math.log(weights.reduce((s, w, k) => s + w * (r[k] as number), 0))
    },
    vmp: (c) => {
      const sticks = sticksOf(c.params)
      const r = stateProbabilities(arg(c, 0), sticks + 1)
      let out = 0
      let rest = 0
      for (let k = 0; k <= sticks; k++) {
        const mo = k < sticks ? beta.moments(etaOf(arg(c, 1 + k))) : [0, 0]
        if ((r[k] as number) > 0) out += (r[k] as number) * (rest + (k < sticks ? (mo[0] as number) : 0))
        rest += k < sticks ? (mo[1] as number) : 0
      }
      return out
    },
  },
  traits: { readsTarget: (to) => to > 0 },
})
