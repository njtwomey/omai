/**
 * The computation of 'Logistic regression': the library's logistic model (a chance of yes that is an S-shaped curve of a score,
 * the score being an offset plus a weight for each of two measurements) is fitted to forty marked points by expectation
 * propagation. The three numbers are one vector, so the method keeps how they move together. Its belief about them is a
 * bell curve in three dimensions, with a mean and a covariance. Nothing is drawn from it: the chance of yes at a place is the
 * S-shaped curve averaged over the bell curve of the score there (Gauss-Hermite quadrature), and the densities of the weights
 * are read from the bell curve and its marginals. It is a pure function of the seed, which only decides the order the points
 * are handed over in, and it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { type MultivariateNormal, normal } from 'omai-engine/dists'
import { gaussHermite } from 'omai-engine/numerics/quadrature'
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, uniform } from 'omai-engine/random'
import {
  add,
  argsort,
  div,
  exp,
  linspace,
  matmul,
  meshgrid,
  mul,
  ones,
  reshape,
  sigmoid,
  sqrt,
  stack,
  sum,
  take,
  tensor,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import { toArray, toFlat, type Value } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { designOf, drawPoints, GAP_ROWS, LIMIT, type Points, TRUTH } from './data.ts'

/** A mean and a 90% interval. */
export interface Interval {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The grid the maps are drawn on: the values of the first and the second measurement. */
  readonly xs: readonly number[]
  readonly ys: readonly number[]
  /** The expected chance of yes in each cell, by row (the second measurement) and then column. */
  readonly chance: readonly (readonly number[])[]
  /** The standard deviation of the score in each cell, in the same layout: how unsure the model is about the score. */
  readonly doubt: readonly (readonly number[])[]
  /** The offset and the two weights, each with its mean and 90% interval from the belief. */
  readonly weights: { readonly offset: Interval; readonly first: Interval; readonly second: Interval }
  /** The correlation of the two weights in the belief. */
  readonly correlation: number
  /** The joint density of the two weights on a grid: the values of the first weight, of the second, and the density by row. */
  readonly joint: {
    readonly first: readonly number[]
    readonly second: readonly number[]
    readonly density: readonly (readonly number[])[]
  }
  /** The density of the offset and of each weight along one shared axis. */
  readonly curves: {
    readonly axis: readonly number[]
    readonly offset: readonly number[]
    readonly first: readonly number[]
    readonly second: readonly number[]
  }
  /** The distance between the 0.2 and 0.8 lines along a row of the map in the middle of the points and along one near the edge. */
  readonly gaps: { readonly centre: number; readonly edge: number }
  /** The offset and the two weights the marks were drawn from. */
  readonly truth: { readonly offset: number; readonly first: number; readonly second: number }
  /** The belief about the offset and the two weights: its mean and its covariance. */
  readonly belief: { readonly mean: readonly number[]; readonly covariance: readonly (readonly number[])[] }
  /** How many of the points lie on the side of the one-half line that matches their mark, and how many points there are. */
  readonly right: number
  readonly count: number
  /** How many sweeps the method took and whether it settled. */
  readonly sweeps: number
  readonly settled: boolean
}

/** How much to do. */
export interface Size {
  readonly grid: number
  readonly density: number
}

/** The size of the real run: maps of 61 by 61 cells, and densities on a grid of 80 by 80 (and 200 for the curves). */
export const DEFAULT_SIZE: Size = { grid: 61, density: 80 }

// region quadrature
/**
 * The expected chance of yes at some places: the S-shaped curve averaged over the bell curve
 * of the score there. A place is a row `[1, first, second]`, so its score is the row times
 * the weights (offset included): a bell curve with mean `row . centre` and variance
 * `row' covariance row`, which uses every entry of the covariance. The average is a
 * Gauss-Hermite rule of 100 points.
 *
 * @param rows - One row per place, `[places, 3]`.
 * @param centre - The mean of the belief about the offset and the two weights.
 * @param covariance - Its covariance, `[3, 3]`.
 * @returns The expected chance of yes, and the spread of the score, at each place.
 */
type Chance = { chance: Value; doubt: Value }
export function expectedChance(rows: Value, centre: Value, covariance: Value): Chance {
  // the Gauss-Hermite nodes and weights: they average a function over a bell curve
  const rule = gaussHermite(100)
  const mean = matmul(rows, centre)
  const doubt = sqrt(sum(mul(matmul(rows, covariance), rows), 1))
  // the S-shaped curve at the nodes of that bell curve, weighted and added up
  const at = add(reshape(mean, [-1, 1]), mul(reshape(doubt, [-1, 1]), mul(rule.nodes, Math.SQRT2)))
  return { chance: div(sum(mul(sigmoid(at), rule.weights), 1), Math.sqrt(Math.PI)), doubt }
}
// endregion

/** The 90% interval of a bell curve: its mean and the 5% and 95% points. */
function intervalOf(centre: number, spread: number): Interval {
  const bell = normal(centre, spread)
  return { mean: centre, lo: Number(toFlat(bell.ppf(0.05))[0]), hi: Number(toFlat(bell.ppf(0.95))[0]) }
}

/**
 * The inference, from a seed: write the model, run expectation propagation on the (fixed) points, and read the chance of yes
 * and the densities of the weights from the belief it settles on.
 *
 * @param seed - Decides the order the points are handed over in (the points do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const points: Points = drawPoints()
  say('Writing the model: a chance of yes that rises across a line')
  // region model
  const classifier = defineModel('logistic', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    // one row per point: a 1 for the offset, then the two measurements
    const design = m.data('design', { dims: [row, feature] })
    const mark = m.data('mark', { dims: [row], support: 'binary' })
    // the offset and the two weights are one vector, so their ups and downs stay tied together
    const weights = m.latent('weights', d.normal(0, 2), { dims: [feature] })
    // the chance of yes is an S-shaped curve of the score: offset plus the weights times the measurements
    m.observe('mark', d.bernoulli({ logits: design.matmul(weights) }), mark)
  })
  // endregion
  say('Passing messages from the points to the weights until none changes')
  // region infer
  // the order the points are handed over in is the seed's; the answer should not depend on it
  const order = argsort(uniform(foldIn(key(seed), 0), [points.y.length]), 0)
  const design = take(tensor(designOf(points)), order, 0)
  const bound = classifier.bind({ design, mark: take(tensor([...points.y]), order, 0) })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: every point sends a new message, one at a time, until none changes
  const run = session(route.method, { steps: 200, mode: 'eager', config: {} })
  // the belief the messages settled on: one bell curve over the three numbers together
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Working out the chance of yes at every place')
  // region chance
  const belief = beliefs.marginal('weights') as MultivariateNormal
  const covariance = belief.covariance()
  // a grid over the plane, one row [1, first, second] per cell
  const axis = linspace(-LIMIT, LIMIT, size.grid)
  const flat = meshgrid([axis, axis]).map((g) => reshape(g, [-1]))
  const cells = stack([ones([size.grid ** 2]), ...flat], 1)
  const field = expectedChance(cells, belief.loc, covariance)
  // endregion
  say('Reading the density of each weight')
  // region densities
  const centre = toFlat(belief.loc)
  const spread = toFlat(belief.stddev())
  const lowest = (j: number): number => (centre[j] as number) - 3.5 * (spread[j] as number)
  const highest = (j: number): number => (centre[j] as number) + 3.5 * (spread[j] as number)
  // the joint density of the two weights: the belief's marginal on coordinates 1 and 2, on a grid
  const along = [1, 2].map((j) => linspace(lowest(j), highest(j), size.density))
  const pairs = stack(
    meshgrid(along).map((g) => reshape(g, [-1])),
    1,
  )
  const density = exp(belief.marginal([1, 2]).logProb(pairs))
  // the density of each single weight, along one shared axis
  const floor = Math.min(lowest(0), lowest(1), lowest(2))
  const ceiling = Math.max(highest(0), highest(1), highest(2))
  const shared = linspace(floor, ceiling, 200)
  const each = [0, 1, 2].map((j) => exp(belief.marginal([j]).logProb(reshape(shared, [-1, 1]))))
  // endregion
  // how many of the points lie on the side of the one-half line that matches their mark
  const own = toFlat(expectedChance(tensor(designOf(points)), belief.loc, covariance).chance)
  const right = own.filter((p, i) => (p > 0.5 ? 1 : 0) === points.y[i]).length
  const [offset, first, second] = [0, 1, 2].map((j) => intervalOf(centre[j] as number, spread[j] as number)) as [
    Interval,
    Interval,
    Interval,
  ]
  // how far apart the 0.2 and 0.8 lines are along a row of the map, found on a fine line of places that runs past the figure
  const across = linspace(-12, 12, 4801)
  const gapAlong = (height: number): number => {
    const row = stack([ones([4801]), across, mul(ones([4801]), height)], 1)
    const chances = toFlat(expectedChance(row, belief.loc, covariance).chance)
    const where = toFlat(across)
    const crossing = (level: number): number => {
      const i = chances.findIndex((c, k) => k > 0 && ((chances[k - 1] as number) - level) * (c - level) <= 0)
      const [a, b] = [chances[i - 1] as number, chances[i] as number]
      return i < 1
        ? Number.NaN
        : (where[i - 1] as number) + ((level - a) / (b - a)) * ((where[i] as number) - (where[i - 1] as number))
    }
    return Math.abs(crossing(0.8) - crossing(0.2))
  }
  const cov = toArray(covariance) as number[][]
  return {
    xs: toFlat(axis),
    ys: toFlat(axis),
    chance: toArray(reshape(field.chance, [size.grid, size.grid])) as number[][],
    doubt: toArray(reshape(field.doubt, [size.grid, size.grid])) as number[][],
    weights: { offset, first, second },
    correlation: (cov[1]?.[2] as number) / Math.sqrt((cov[1]?.[1] as number) * (cov[2]?.[2] as number)),
    joint: {
      first: toFlat(along[0] as Value),
      second: toFlat(along[1] as Value),
      density: toArray(reshape(density, [size.density, size.density])) as number[][],
    },
    curves: {
      axis: toFlat(shared),
      offset: toFlat(each[0] as Value),
      first: toFlat(each[1] as Value),
      second: toFlat(each[2] as Value),
    },
    gaps: { centre: gapAlong(GAP_ROWS[0]), edge: gapAlong(GAP_ROWS[1]) },
    truth: { offset: TRUTH[0], first: TRUTH[1], second: TRUTH[2] },
    belief: { mean: [...centre], covariance: cov },
    right,
    count: points.y.length,
    sweeps: run.t,
    settled: run.status === 'converged',
  }
}

/**
 * The run of the story: the inference at its full size.
 *
 * @param seed - Decides the order of the points.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
