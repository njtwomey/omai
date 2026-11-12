/**
 * A truncated stick-breaking mixture, the weights of a Dirichlet-process mixture cut at four components: `v_k ~ Beta(1, alpha)`
 * for the first three sticks, weights `pi_k = v_k prod_{j<k} (1 - v_j)` (the last takes what is left), a component label per
 * row and unit-noise normal components with known means. With the means known the unknowns are the three stick breaks, so
 * the posterior is three-dimensional and the labels sum out: the truth is a grid over the sticks with the labels
 * enumerated, exact to the quadrature, which is the "exact for a fixed number of components" case of the plan. The stick-breaking
 * map is the model layer's `stickBreaking` operator, which message passing reads as a `stick-breaking` factor on each label.
 */

import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { add, arange, div, toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type MarginalOracle, weightedOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan } from './plans.ts'

const MEANS = [-3, -1, 1, 3]
const STICKS = MEANS.length - 1
const CONCENTRATION = 1
const WEIGHTS = [0.5, 0.25, 0.2, 0.05]

function generateMixture(key: Key, size = 30): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  const label = Array.from(u, (v) => {
    let acc = 0
    for (let k = 0; k < WEIGHTS.length; k++) {
      acc += WEIGHTS[k] as number
      if (v < acc) return k
    }
    return WEIGHTS.length - 1
  })
  return {
    data: { y: label.map((l, i) => (MEANS[l] as number) + (e[i] ?? 0)) },
    generating: { weights: WEIGHTS, means: MEANS, label },
  }
}

/**
 * The posterior of the stick breaks by a midpoint rule on the unit cube, with the labels summed out: the three breaks are
 * bounded and have a smooth density there, so a rule of `n` nodes per axis converges fast (60 nodes agree with 120 to
 * 1e-4 in the log evidence). The oracles are those of the breaks `v[k]` and of the weights `pi[k]`, as weighted node values.
 *
 * @param y - The observations.
 * @param n - Nodes per axis (default 60).
 * @returns The truth: `v[0..2]` and `pi[0..3]` and the log evidence.
 * @concept mixture-models
 * @example
 * ```ts
 * import { stickBreakingTruth } from 'omai-engine/probprog/models'
 * Object.keys(stickBreakingTruth([-3, 3, 1], 8).marginals) // => ['v[0]', 'v[1]', 'v[2]', 'pi[0]', 'pi[1]', 'pi[2]', 'pi[3]']
 * ```
 */
export function stickBreakingTruth(y: readonly number[], n = 60): BenchmarkTruth {
  const nodes = toFlat(div(add(arange(n), 0.5), n))
  const total = n ** 3
  // The tables of the n^3 grid (mass, coordinates, component weights, log density) are filled in place by the sweep below.
  // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
  const mass = new Float64Array(total)
  const coords = [new Float64Array(total), new Float64Array(total), new Float64Array(total)]
  const weights = Array.from({ length: MEANS.length }, () => new Float64Array(total))
  const normal = (v: number, m: number): number => Math.exp(-0.5 * (v - m) ** 2) / Math.sqrt(2 * Math.PI)
  let max = Number.NEGATIVE_INFINITY
  // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
  const logs = new Float64Array(total)
  let at = 0
  for (const v0 of nodes)
    for (const v1 of nodes)
      for (const v2 of nodes) {
        const pi = [v0, v1 * (1 - v0), v2 * (1 - v0) * (1 - v1), (1 - v0) * (1 - v1) * (1 - v2)]
        let logp = 0
        for (const yi of y) logp += Math.log(pi.reduce((s, p, k) => s + p * normal(yi, MEANS[k] as number), 0))
        // The Beta(1, alpha) prior of each break.
        for (const v of [v0, v1, v2]) logp += Math.log(CONCENTRATION) + (CONCENTRATION - 1) * Math.log1p(-v)
        logs[at] = logp
        max = Math.max(max, logp)
        ;(coords[0] as Float64Array)[at] = v0
        ;(coords[1] as Float64Array)[at] = v1
        ;(coords[2] as Float64Array)[at] = v2
        pi.forEach((p, k) => {
          ;(weights[k] as Float64Array)[at] = p
        })
        at++
      }
  let z = 0
  for (let i = 0; i < total; i++) {
    mass[i] = Math.exp((logs[i] as number) - max)
    z += mass[i] as number
  }
  const marginals: Record<string, MarginalOracle> = {}
  // A break takes `n` values: the mass of each, summed over the other two breaks.
  coords.forEach((c, k) => {
    // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
    const byNode = new Float64Array(n)
    for (let i = 0; i < total; i++) {
      const node = Math.min(n - 1, Math.floor((c[i] as number) * n))
      byNode[node] = (byNode[node] as number) + (mass[i] as number)
    }
    marginals[`v[${k}]`] = weightedOracle(`v[${k}]`, nodes, byNode)
  })
  weights.forEach((w, k) => {
    marginals[`pi[${k}]`] = weightedOracle(`pi[${k}]`, w, mass)
  })
  return {
    kind: 'quadrature',
    source: `a midpoint rule with ${n} nodes per axis on the unit cube for the three breaks, the component labels summed out exactly (the log evidence agrees with a rule of twice the nodes to 1e-4); the weights pi[k] are the nodes' weights under the posterior`,
    logEvidence: max + Math.log(z / total),
    marginals,
  }
}

/**
 * A stick-breaking mixture truncated at four components with known means `(-3, -1, 1, 3)`, unit noise and `Beta(1, 1)` breaks, thirty
 * rows drawn from weights `(0.5, 0.25, 0.2, 0.05)`. The posterior of the three breaks is exact to a three-dimensional
 * quadrature with the labels summed out; NUTS sums them out too, and the mean-field guide and Laplace fit see a posterior on
 * the unit cube whose breaks are strongly coupled through the weights.
 *
 * @concept mixture-models
 * @index dataset Truncated stick-breaking (Dirichlet-process style) mixture with known components; a 3-D grid truth.
 * @example
 * ```ts
 * import { stickBreakingMixture } from 'omai-engine/probprog/models'
 * [stickBreakingMixture.bound().sizes, stickBreakingMixture.truth(undefined, 'fast').kind] // => [{ row: 30, stick: 3, component: 4 }, 'quadrature']
 * ```
 */
export const stickBreakingMixture: BenchmarkModel = defineBenchmark({
  name: 'stick-breaking-mixture',
  title: 'Stick-breaking mixture (truncated Dirichlet process)',
  description:
    'The weights of a Dirichlet-process mixture truncated at four known normal components: three Beta stick breaks and a label per row, whose posterior is a three-dimensional quadrature with the labels summed out.',
  tags: ['mixture', 'nonparametric', 'dirichlet-process', 'stick-breaking', 'discrete-latent', 'quadrature'],
  references: [
    'Sethuraman (1994), A constructive definition of Dirichlet priors',
    'Ishwaran and James (2001), Gibbs sampling methods for stick-breaking priors',
    'Blei and Jordan (2006), Variational inference for Dirichlet process mixtures',
  ],
  definition: defineModel('stick-breaking-mixture', (b) => {
    const row = b.range('row')
    const stick = b.range('stick', { size: STICKS })
    const component = b.range('component', { size: MEANS.length })
    const y = b.data('y', { dims: [row], role: 'response' })
    const means = b.const(MEANS, [component])
    const v = b.latent('v', d.beta(1, CONCENTRATION), { dims: [stick], doc: 'the stick-breaking fractions' })
    const pi = b.derived('pi', ops.stickBreaking(v), { doc: 'the mixture weights' })
    const z = b.latent('z', d.categorical({ probs: pi }), { dims: [row], doc: 'the component of each row' })
    b.observe('y', d.normal(means.at(z), 1), y)
  }),
  generate: (key, size) => generateMixture(key, size),
  truth: (data, quality) => stickBreakingTruth(data.data['y'] as number[], quality === 'fast' ? 30 : 60),
  plans: (scale) => [
    nutsPlan(scale, {
      note: 'NUTS on the logit-scale breaks with the labels summed out: consistent',
      tolerances: { meanError: 0.2, meanZ: 6, sdRatio: [0.75, 1.25], ks: 0.2 },
    }),
    laplacePlan(scale, {
      note: 'a Gaussian at the mode on the logit scale: the breaks are bounded and skewed, so the tails are mis-shaped',
    }),
    sviPlan(scale, {
      note: 'a mean-field normal on the logit breaks ignores their coupling through the weights: spreads too small',
    }),
  ],
})
