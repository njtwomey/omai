/**
 * Gates composed with other pieces of the model language, each a small setup that shows several elements working together: a
 * switch per observation that chooses between likelihoods of different families (a Poisson and a negative binomial) whose means
 * share a log-linear rate on a covariate, with a Beta prior on the switch and a learned dispersion. The reference is the grid
 * posterior of the same model with the switches summed out; EP reports the probability of each switch besides the parameters.
 */

import { foldIn, gamma, key, normal, poisson, uniform } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { add, arange, div, mul, toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { type ExactMarginal, gridMarginals, lg, probabilityMarginal, type RefJoint } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import type { AtomParts, EpAtom } from '../atom-types.ts'

/** The mean of the negative-binomial branch, as a multiple of the rate of the Poisson branch (the 4 written in the shown model). */
const MULTIPLE = 4
const TRUE_B0 = 1
const TRUE_B1 = 0.5
const POINTS = 25
/** The box of the reference grid: w, b_0, b_1 and the dispersion. */
const BOX: readonly (readonly [number, number])[] = [
  [0.002, 0.998],
  [-1.5, 3],
  [-1.5, 2.5],
  [0.02, 3],
]

/** The covariate, the switch of each observation and the counts of one simulated series. */
function simulateCounts(seed: number, n: number, share: number, dispersion: number) {
  const k = key(seed)
  const x = Array.from(toFlat(normal(foldIn(k, 1), [n])))
  const u = toFlat(uniform(foldIn(k, 2), [n]))
  const z = Array.from(u, (v) => (v < share ? 1 : 0))
  const y = x.map((xi, i) => {
    const rate = Math.exp(TRUE_B0 + TRUE_B1 * xi)
    if (z[i] === 0) return Number(toFlat(poisson(foldIn(k, 100 + i), rate, [1]))[0])
    // a negative binomial is a Poisson whose rate is gamma distributed: shape 1 / dispersion, mean MULTIPLE * rate
    const lambda = (toFlat(gamma(foldIn(k, 300 + i), [1], 1 / dispersion))[0] as number) * dispersion * MULTIPLE * rate
    return Number(toFlat(poisson(foldIn(k, 500 + i), lambda, [1]))[0])
  })
  return { x, y, z }
}

interface GridReference {
  readonly exact: Record<string, ExactMarginal>
  readonly joint: RefJoint
  readonly logEvidence: number
}

/**
 * The posterior of the model on a grid over its four continuous coordinates, with each observation's switch summed out (the
 * density is the mixture of its two branches): the marginals, the joint of the rate's coefficients, the log evidence with every
 * constant in it, and the probability that each observation is from the second branch. Computed once, when first read.
 */
function referenceOf(data: { readonly x: readonly number[]; readonly y: readonly number[] }): () => GridReference {
  let memo: GridReference | undefined
  const lgY = data.y.map((v) => lg(v + 1))
  // the negative binomial's normalising terms depend on the dispersion and the count only: one table per dispersion value
  const tables = new Map<number, number[]>()
  const table = (a: number): number[] => {
    const cached = tables.get(a)
    if (cached !== undefined) return cached
    const made = data.y.map((v, i) => lg(v + 1 / a) - lg(1 / a) - (lgY[i] as number))
    tables.set(a, made)
    return made
  }
  // the log of each branch of observation i, weighted by its probability, at a point (w, b_0, b_1, dispersion)
  const branches = (point: readonly number[], i: number): { readonly first: number; readonly second: number } => {
    const [w, b0, b1, a] = point as [number, number, number, number]
    const v = data.y[i] as number
    const rate = Math.exp(b0 + b1 * (data.x[i] as number))
    const r = 1 / a
    const mean = MULTIPLE * rate
    return {
      first: Math.log(1 - w) + v * Math.log(rate) - rate - (lgY[i] as number),
      second: Math.log(w) + (table(a)[i] as number) + r * Math.log(r / (r + mean)) + v * Math.log(mean / (r + mean)),
    }
  }
  return () => {
    if (memo !== undefined) return memo
    const logNormal = (v: number): number => -0.5 * Math.log(8 * Math.PI) - (v * v) / 8
    const grid = gridMarginals(
      (point) => {
        const [w, b0, b1, a] = point as [number, number, number, number]
        // Beta(1, 3), N(0, 2²) twice and a half-normal with scale one
        let lp = Math.log(3) + 2 * Math.log(1 - w) + logNormal(b0) + logNormal(b1)
        lp += 0.5 * Math.log(2 / Math.PI) - (a * a) / 2
        for (let i = 0; i < data.y.length; i++) {
          const { first, second } = branches(point, i)
          const top = Math.max(first, second)
          lp += top + Math.log(Math.exp(first - top) + Math.exp(second - top))
        }
        return lp
      },
      BOX,
      POINTS,
    )
    const exact: Record<string, ExactMarginal> = {
      w: grid.marginal(0),
      b0: grid.marginal(1),
      b1: grid.marginal(2),
      dispersion: grid.marginal(3),
    }
    for (let i = 0; i < data.y.length; i++)
      exact[`z[${i}]`] = probabilityMarginal(
        grid.expect((point) => {
          const { first, second } = branches(point, i)
          return 1 / (1 + Math.exp(first - second))
        }),
      )
    memo = { exact, joint: { a: 'b0', b: 'b1', ...grid.joint(1, 2) }, logEvidence: grid.logEvidence }
    return memo
  }
}

function gatePoissonNegativeBinomialProblem(k: Config): AtomParts {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [share, dispersion] = [configNumber(k, 'trueShare'), configNumber(k, 'trueDispersion')]
  const data = simulateCounts(seed, n, share, dispersion)
  // region gate-poisson-negative-binomial
  const definition = defineModel('gate-poisson-negative-binomial', (m) => {
    const row = m.range('i')
    const x = m.data('x', { dims: [row], role: 'covariate' })
    const y = m.data('y', { dims: [row], role: 'response', support: 'count' })
    // the log-linear rate on the covariate, shared by the means of both branches
    const b0 = m.latent('b0', d.normal(0, 2))
    const b1 = m.latent('b1', d.normal(0, 2))
    // the dispersion of the negative binomial, and the probability of its branch
    const dispersion = m.latent('dispersion', d.halfNormal(1))
    const w = m.latent('w', d.beta(1, 3))
    // the switch of each observation
    const z = m.latent('z', d.bernoulli({ probs: w }), { dims: [row] })
    const rate = ops.exp(ops.add(b0, ops.mul(b1, x)))
    // the second branch has a mean four times the rate, and a dispersion of its own
    const mean = ops.mul(4, rate)
    const first = d.poisson(rate)
    const successes = ops.div(1, dispersion)
    const probs = ops.div(1, ops.add(1, ops.mul(dispersion, mean)))
    const second = d.negativeBinomial(successes, probs)
    // the switch chooses the family of the likelihood
    m.observe('y', d.gate(z, [first, second]), y)
  })
  // endregion
  const ref = referenceOf(data)
  const at = data.y.map((_, i) => i)
  return {
    build: {
      definition,
      data: { x: data.x, y: data.y },
      quantities: [
        { name: 'w', label: 'w (probability of the second branch)', support: 'unit', truth: share },
        { name: 'b0', label: 'b_0 (log rate at x = 0)', support: 'real', truth: TRUE_B0 },
        { name: 'b1', label: 'b_1 (slope of the log rate)', support: 'real', truth: TRUE_B1 },
        { name: 'dispersion', label: 'dispersion of the negative binomial', support: 'positive', truth: dispersion },
        ...at.map((i) => ({
          name: `z[${i}]`,
          label: `z_${i + 1} is the second branch`,
          support: 'probability' as const,
          hidden: true,
        })),
      ],
    },
    reference: {
      // lazy: the grid is summed when a reference is first asked for
      get exact() {
        return ref().exact
      },
      get exactJoint() {
        return ref().joint
      },
      get exactLogEvidence() {
        return ref().logEvidence
      },
      referenceSource: `quadrature of the posterior over (w, b_0, b_1, dispersion) on a ${POINTS}-point grid per coordinate, the switches summed out`,
    },
    present: {
      equations: [
        `truth: w = ${share}, b = (${TRUE_B0}, ${TRUE_B1}), dispersion ${dispersion}; ${n} observations, seed ${seed}`,
        'b_0, b_1 ~ N(0, 2²),  rate_i = exp(b_0 + b_1 x_i)',
        'dispersion ~ HalfNormal(1),  w ~ Beta(1, 3)',
        'z_i ~ Bernoulli(w)',
        `y_i ~ z_i = 0: Poisson(rate_i);  z_i = 1: NegativeBinomial(mean ${MULTIPLE} rate_i, dispersion)   (a gate between two families)`,
      ],
      pair: ['b0', 'b1'],
      figure: {
        caption:
          'The counts, marked by the branch each was drawn from, with the mean of each branch at the means of the beliefs.',
        xLabel: 'x',
        yLabel: 'y',
        points: [
          {
            name: 'counts',
            x: data.x,
            y: data.y,
            group: data.z,
            groupNames: { 0: 'Poisson branch', 1: 'negative binomial branch' },
          },
        ],
        answer: (mean) => {
          const grid = toFlat(add(div(mul(arange(41), 5), 40), -2.5))
          const rate = grid.map((x) => Math.exp((mean['b0'] ?? Number.NaN) + (mean['b1'] ?? Number.NaN) * x))
          return [
            { name: 'Poisson mean', x: grid, y: rate },
            { name: 'negative binomial mean', colour: 'tab1', x: grid, y: rate.map((r) => MULTIPLE * r) },
          ]
        },
      },
    },
  }
}

/**
 * A gate between a Poisson and a negative binomial: each observation has a switch that chooses the family of its likelihood, the
 * two means share a log-linear rate on a covariate, the switch has a Beta probability and the negative binomial a learned
 * dispersion. The data are simulated from stated values. Against the grid posterior with the switches summed out.
 */
const gatePoissonNegativeBinomialAtom: EpAtom = defineAtom({
  name: 'gate-poisson-negative-binomial',
  title: 'A gate between a Poisson and a negative binomial',
  what: 'A switch per observation chooses between two count likelihoods of different families, a Poisson and a negative binomial whose mean is a multiple of the Poisson rate, with a log-linear rate on a covariate, a Beta prior on the probability of the switch and a half-normal dispersion; EP reports the probability of each switch.',
  factors: [
    'prior (gaussian x2, half-normal, beta)',
    'bernoulli (a switch with a beta probability)',
    'gate per observation (poisson-log and negative-binomial-log branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'measured on five seeds of thirty observations: the means are within 0.15 reference sds, the spreads 0.7 to 1.0 of the reference for the rate and the switch probability (the four parameters meet through the rate that the branches share, so one Gaussian each is a little narrow) and up to 1.5 for the dispersion, whose posterior is skewed; the probability of each switch is within 0.06 of the reference',
    { meanZ: 0.3, sdRatio: [0.6, 1.6], ks: 0.2 },
  ),
  knobs: dict({
    seed: int(1, 9999, { default: 1, label: 'seed (the simulated data)' }),
    n: int(8, 40, { default: 30, label: 'observations' }),
    trueShare: real(0.05, 0.6, { default: 0.25, step: 0.01, label: 'true probability of the second branch' }),
    trueDispersion: real(0.1, 2, { default: 0.5, step: 0.05, label: 'true dispersion' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  parts: gatePoissonNegativeBinomialProblem,
})

/** The atoms of the `composed-gates` file. */
export const composedGatesAtoms: readonly EpAtom[] = [gatePoissonNegativeBinomialAtom]
