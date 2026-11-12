/**
 * The spin factor: a Gaussian variable pinned to plus or minus one, `f(x) = exp(field x) (delta(x - 1) + delta(x + 1))`, the
 * non-Gaussian site of the Ising model in the expectation consistent (adaptive TAP) approximation. Against a Gaussian
 * cavity the tilted distribution has mass on the two points only, so its mean is `tanh(field + cavity mean / variance)`
 * and its variance one minus the square of that: a closed form that EP and expectation consistent inference use as the
 * moment match.
 */

import { logAddExp } from '../../../base/index.ts'
import { allProper, etaOf, gaussianSiteMessage, numberParam } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { param } from '../../core/params.ts'
import type { FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'

/** log of the weights of the two spin values against a Gaussian message `exp(h x - j x^2 / 2)`: the points x = +1 and x = -1. */
function weights(c: OperatorContext): { plus: number; minus: number } {
  const eta = etaOf(c.args[0] as never)
  const field = numberParam(c.params, 'field', 0)
  const h = eta[0] ?? 0
  const j = eta[1] ?? 0
  return { plus: field + h - 0.5 * j, minus: -field - h - 0.5 * j }
}

/**
 * A spin `x in {-1, +1}` with a field: `exp(field x)` on the two points. EP is exact in closed form (the mean of the tilted
 * distribution is a hyperbolic tangent, its variance `1 - mean^2`), and the factor's mass is the sum of the two weights.
 *
 * @concept factor-operators
 * @index structure The spin factor x in {-1, +1} with a field, exact EP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, spin } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(0, 1))]
 * spin.ep?.({ mode: 'ep', to: 0, args, params: { field: 0.5 }, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.5876', '0.2715']
 * ```
 */
export const spin: FactorOperator = defineOperator({
  name: 'spin',
  description: 'x in {-1, +1} with weight exp(field x)',
  ports: [{ name: 'x', accepts: ['gaussian'] }],
  params: { field: param.number({ optional: true }) },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const { plus, minus } = weights(c)
      const mean = Math.tanh(0.5 * (plus - minus))
      return { message: gaussianSiteMessage(etaOf(c.args[0] as never), mean, Math.max(1 - mean * mean, 1e-12)) }
    }),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const { plus, minus } = weights(c)
      return logAddExp(plus, minus)
    },
  },
  traits: { readsTarget: true },
  noRule: {
    vmp: 'the spin is two point masses at plus and minus one: KL(q || p) is infinite for every Gaussian q (use expectation propagation, or a binary variable with a table)',
  },
})
