/**
 * The setups of expectation propagation for counts with structure: a hurdle regression, a count that is zero with probability `zero`
 * and otherwise a zero-truncated Poisson of a log-linear rate. Given the count the state of the hurdle is known, so the model
 * splits into a conjugate Bernoulli part on the zero probability (exact) and a zero-truncated Poisson site on the two coefficients
 * (a numeric rule). The exact answers are a Beta for the zero probability and a grid for the coefficients.
 */

import { dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type AtomProblem, defineAtom, type EpAtom } from './ep-atoms.ts'
import { gridMarginals, quadratureMarginal } from './ep-atoms-quad.ts'

const XS = [-1, -0.6, -0.2, 0.1, 0.5, 0.9, 1.3, -0.8, 0.3, 1.1, -0.4, 0.7]
const YS = [0, 0, 1, 0, 2, 4, 6, 1, 3, 0, 1, 5]

function hurdleProblem(k: Config): AtomProblem {
  const priorSd = configNumber(k, 'priorSd')
  // region hurdle-poisson
  const definition = defineModel('hurdle-poisson', (m) => {
    const row = m.range('row')
    const x = m.data('x', { dims: [row] })
    const y = m.data('y', { dims: [row], role: 'response' })
    // the probability that a count is a zero of the hurdle, and the log-linear rate of the positive counts
    const zero = m.latent('zero', d.beta(1, 1))
    const b0 = m.latent('b0', d.normal(0, priorSd))
    const b1 = m.latent('b1', d.normal(0, priorSd))
    const rate = ops.exp(ops.add(b0, ops.mul(b1, x)))
    m.observe('y', d.hurdlePoisson(rate, zero), y)
  })
  // endregion
  const zeros = YS.filter((v) => v === 0).length
  const [a, b] = [1 + zeros, 1 + YS.length - zeros]
  const logBeta = (p: number): number => (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p)
  // The coefficients see the positive counts only, through the zero-truncated Poisson.
  const g = gridMarginals(
    (c) => {
      const [c0, c1] = [c[0] ?? 0, c[1] ?? 0]
      let lp = -(c0 * c0 + c1 * c1) / (2 * priorSd ** 2)
      for (const [r, y] of YS.entries()) {
        if (y === 0) continue
        const mu = Math.exp(c0 + c1 * (XS[r] as number))
        lp += y * Math.log(mu) - mu - Math.log(-Math.expm1(-mu))
      }
      return lp
    },
    [
      [-3, 3],
      [-3, 3],
    ],
    161,
  )
  return {
    definition,
    data: { x: XS, y: YS },
    equations: [
      'zero ~ Beta(1, 1)   (the probability that a count is a zero of the hurdle)',
      `b_0, b_1 ~ N(0, ${priorSd}²),  rate_i = exp(b_0 + b_1 x_i)`,
      'y_i = 0 with probability zero, else y_i ~ Poisson(rate_i) given y_i > 0   (a hurdle count)',
      `observe the twelve counts drawn in the figure (${zeros} zeros)`,
    ],
    quantities: [
      { name: 'zero', label: 'zero (probability of a zero)', support: 'unit' },
      { name: 'b0', label: 'b_0 (log rate at x = 0)', support: 'real' },
      { name: 'b1', label: 'b_1 (slope of the log rate)', support: 'real' },
    ],
    pair: ['b0', 'b1'],
    exact: {
      zero: quadratureMarginal(logBeta, 0, 1, 4000),
      b0: g.marginal(0),
      b1: g.marginal(1),
    },
    exactJoint: { a: 'b0', b: 'b1', ...g.joint(0, 1) },
    referenceSource: 'a Beta for the zero probability and a grid over the coefficients with the positive counts',
    figure: {
      caption: 'The twelve counts and the rate of the positive part, exp(b_0 + b_1 x), at the means of the beliefs.',
      xLabel: 'x',
      yLabel: 'count',
      xLimits: [-1.5, 1.5],
      yLimits: [-0.5, 8],
      points: [{ name: 'counts', x: XS, y: YS }],
      answer: (mean) => {
        const grid = Array.from({ length: 31 }, (_, i) => -1.5 + (3 * i) / 30)
        return [
          {
            name: 'rate',
            x: grid,
            y: grid.map((x) => Math.exp((mean['b0'] ?? Number.NaN) + (mean['b1'] ?? Number.NaN) * x)),
          },
        ]
      },
    },
  }
}

/**
 * A hurdle regression on twelve counts: a Beta probability of a zero and a log-linear rate of the positive counts, which are
 * zero-truncated Poisson. The zero part is a conjugate Bernoulli observation, the positive part a truncated-Poisson site.
 *
 * @index estimator EP atom: a hurdle count (a Beta zero probability, a zero-truncated Poisson on a log-linear rate).
 * @example
 * ```ts
 * import { hurdlePoissonAtom } from 'omai-engine/probprog/ep'
 * hurdlePoissonAtom.name // => 'hurdle-poisson'
 * ```
 */
export const hurdlePoissonAtom: EpAtom = defineAtom({
  name: 'hurdle-poisson',
  title: 'Hurdle count (a zero, or a zero-truncated Poisson)',
  what: 'Twelve counts that are zero with an unknown probability and otherwise zero-truncated Poisson with a log-linear rate: a conjugate Bernoulli factor on the zero probability and a zero-truncated-Poisson site on the two coefficients.',
  factors: [
    'prior (beta, gaussian x2)',
    'bernoulli (the hurdle, conjugate)',
    'likelihood-site (poisson-log-truncated)',
  ],
  family: 'counts',
  reference: 'exact',
  tolerance: {
    meanZ: 0.05,
    sdRatio: [0.92, 1.08],
    ks: 0.05,
    reason:
      'the zero probability is exact (a Beta of the counts); the coefficients are the truncated-Poisson likelihood matched by one Gaussian each: measured within 0.004 reference sds and 3 percent of the spread',
  },
  knobs: dict({ priorSd: real(0.5, 4, { default: 1.5, step: 0.1, label: 'prior sd of the coefficients' }) }),
  problem: hurdleProblem,
})
