/**
 * The setups of expectation propagation with a hard comparison: two Gaussians with the observation that the first is larger
 * (a truncation) and two means constrained to be ordered against data that prefer the opposite order. The exact answers are a
 * truncated Gaussian and a grid of the constrained density.
 */

import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { dict, real } from '../space/index.ts'
import type { Config } from '../trace/index.ts'
import { type AtomProblem, defineAtom, type EpAtom, type RefJoint } from './ep-atoms.ts'
import { gridMarginals, normalCdf, normalPdf, quadratureMarginal } from './ep-atoms-quad.ts'
import { exampleNumber } from './example-method.ts'

function orderedProblem(k: Config): AtomProblem {
  const gap = exampleNumber(k, 'gap')
  const [ma, mb, s] = [0, gap, 1]
  const definition = defineModel('ordered-pair', (m) => {
    const a = m.latent('a', d.normal(ma, s))
    const b = m.latent('b', d.normal(mb, s))
    m.observe('ordered', d.bernoulli({ probs: ops.gt(ops.sub(a, b), 0) }), m.data('ordered', { support: 'binary' }))
  })
  // a > b is observed: the density is the product of the two priors on the half plane a > b.
  const za = 1 - normalCdf((mb - ma) / Math.SQRT2 / s)
  const logA = (x: number): number =>
    Math.log(normalPdf(x, ma, s)) + Math.log(Math.max(normalCdf((x - mb) / s), 1e-300))
  const logB = (x: number): number =>
    Math.log(normalPdf(x, mb, s)) + Math.log(Math.max(1 - normalCdf((x - ma) / s), 1e-300))
  const N = 70
  const lo = -4.5 + Math.min(ma, mb)
  const hi = 4.5 + Math.max(ma, mb)
  const xs = Array.from({ length: N }, (_, i) => lo + ((hi - lo) * i) / (N - 1))
  const joint: RefJoint = {
    a: 'a',
    b: 'b',
    x: xs,
    y: xs,
    p: xs.map((x) => xs.map((y) => (x > y ? (normalPdf(x, ma, s) * normalPdf(y, mb, s)) / za : 0))),
  }
  return {
    definition,
    data: { ordered: 1 },
    equations: [`a ~ N(${ma}, ${s}²)`, `b ~ N(${mb}, ${s}²)`, 'observe a > b   (a hard comparison: a truncation)'],
    quantities: [
      { name: 'a', label: 'a', support: 'real' },
      { name: 'b', label: 'b', support: 'real' },
    ],
    pair: ['a', 'b'],
    exact: { a: quadratureMarginal(logA, lo, hi, 4000), b: quadratureMarginal(logB, lo, hi, 4000) },
    exactJoint: joint,
  }
}

/**
 * Two Gaussians and the observation that the first is larger: a hard comparison, whose tilted moments are those of a
 * truncated Gaussian in closed form (the greater-than factor).
 *
 * @index estimator EP atom: an ordered pair, a hard comparison (greater-than, truncation).
 * @example
 * ```ts
 * import { orderedPairAtom } from 'omai-engine/bayesian-examples'
 * orderedPairAtom.name // => 'ordered-pair'
 * ```
 */
export const orderedPairAtom: EpAtom = defineAtom({
  name: 'ordered-pair',
  title: 'Ordered pair (truncation)',
  what: 'Two independent Gaussians a and b with the observation that a > b: the greater-than factor, a truncation of their difference.',
  factors: ['prior (gaussian) x2', 'greater-than (truncation)'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.7, 1.2],
    ks: 0.12,
    reason:
      'the exact beliefs are skew-normal; EP matches their mean and variance with Gaussians, and the joint leans where the two beliefs cannot',
  },
  knobs: dict({ gap: real(-2, 2, { default: 0.5, step: 0.1, label: 'mean of b (a has mean 0)' }) }),
  problem: orderedProblem,
})

function orderedMeansProblem(k: Config): AtomProblem {
  const priorSd = exampleNumber(k, 'priorSd')
  const y1 = [0.8, 1.2]
  const y2 = [0.3, 0.6]
  const definition = defineModel('ordered-means', (m) => {
    const group = m.range('group', { size: 2 })
    const obs = m.range('obs')
    const mu = m.latent('mu', d.normal(0, priorSd), { dims: [group], constraint: { kind: 'ordered' } })
    const g = m.data('g', { dims: [obs], support: 'index', indexes: group })
    m.observe('y', d.normal(mu.at(g), 1), m.data('y', { dims: [obs], role: 'response' }))
  })
  const logDensity = (x: readonly number[]): number => {
    const [a, b] = [x[0] ?? 0, x[1] ?? 0]
    if (!(a < b)) return Number.NEGATIVE_INFINITY
    const lp = (v: number): number => -0.5 * (v / priorSd) ** 2
    const ll = (m0: number, ys: readonly number[]): number => ys.reduce((acc, v) => acc - 0.5 * (v - m0) ** 2, 0)
    return lp(a) + lp(b) + ll(a, y1) + ll(b, y2)
  }
  const g = gridMarginals(
    logDensity,
    [
      [-4, 4],
      [-4, 4],
    ],
    161,
  )
  return {
    definition,
    data: { g: [0, 0, 1, 1], y: [...y1, ...y2] },
    bindOptions: { sizes: { group: 2 } },
    equations: [
      `mu_1, mu_2 ~ N(0, ${priorSd}²), constrained to mu_1 < mu_2   (an ordered vector)`,
      'y ~ N(mu_group, 1²), two observations per group',
      `observe group 1: y = (${y1.join(', ')}),  group 2: y = (${y2.join(', ')})   (the data disagree with the order)`,
    ],
    quantities: [
      { name: 'mu[0]', label: 'mu_1', support: 'real' },
      { name: 'mu[1]', label: 'mu_2', support: 'real' },
    ],
    pair: ['mu[0]', 'mu[1]'],
    exact: { 'mu[0]': g.marginals[0] as never, 'mu[1]': g.marginals[1] as never },
    exactJoint: { a: 'mu[0]', b: 'mu[1]', ...g.joint(0, 1) },
  }
}

/**
 * Two means constrained to be ordered, whose data say the opposite: a constraint written as an ordered vector, which is a
 * chain of greater-than factors, so the posterior is the unconstrained one cut along the diagonal.
 *
 * @index estimator EP atom: two means with an ordered constraint, against data that disagree.
 * @example
 * ```ts
 * import { orderedMeansAtom } from 'omai-engine/bayesian-examples'
 * orderedMeansAtom.name // => 'ordered-means'
 * ```
 */
export const orderedMeansAtom: EpAtom = defineAtom({
  name: 'ordered-means',
  title: 'Ordered means',
  what: 'Two means with an ordered-vector constraint mu_1 < mu_2 and data that prefer the opposite order: the constraint is a greater-than factor between neighbours, and the posterior is cut along the diagonal.',
  factors: ['prior (gaussian) x2', 'ordered constraint (greater-than between neighbours)', 'gaussian likelihood'],
  family: 'gates',
  expected: 'runs',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.7, 1.2],
    ks: 0.12,
    reason: 'the truncated posterior is skewed; EP matches the mean and variance of each margin with a Gaussian',
  },
  knobs: dict({ priorSd: real(0.5, 6, { default: 3, step: 0.1, label: 'prior sd of the means' }) }),
  problem: orderedMeansProblem,
})
