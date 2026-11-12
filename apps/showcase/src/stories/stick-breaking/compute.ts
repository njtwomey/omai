/**
 * The computation of 'Breaking a stick': first a few draws of weights from the stick-breaking prior (break a stick of length 1:
 * each piece takes a random share of what is left), then the library's stick-breaking mixture, cut off at eight pieces, fitted
 * to 80 numbers by expectation propagation. Each piece has a weight and a centre; the model is told the width of every group. The
 * method settles on weights where a few pieces are large and the rest are small, and the run turns them into bell curves over
 * the data. It is a pure function of the seed, which decides the prior draws and the order the numbers are handed over in, and
 * it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { beta, foldIn, key, uniform } from 'omai-engine/random'
import {
  add,
  argsort,
  concat,
  cumprod,
  div,
  exp,
  linspace,
  maximum,
  mul,
  ones,
  reshape,
  sqrt,
  square,
  sub,
  sum,
  take,
  tensor,
  transpose,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import { toArray, toFlat, type Value } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { drawNumbers, PIECES, RANGE, WIDTH } from './data.ts'

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** Some draws of weights from the prior: one row per draw, one column per piece. */
  readonly prior: readonly (readonly number[])[]
  /** The weight and the centre the method found for each piece. */
  readonly weights: readonly number[]
  readonly centres: readonly number[]
  /** The spread (standard deviation) of the belief about each weight and about each centre. */
  readonly spread: readonly number[]
  readonly centreSpread: readonly number[]
  /** The grid the curves are drawn on, the curve of each piece (its weight times its bell curve) on it, and their sum. */
  readonly grid: readonly number[]
  readonly curves: readonly (readonly number[])[]
  readonly total: readonly number[]
  /** How many numbers were used, how many sweeps the method took and whether it settled. */
  readonly rows: number
  readonly sweeps: number
  readonly settled: boolean
}

/** How much to do. */
export interface Size {
  readonly rows: number
  readonly priorDraws: number
  readonly grid: number
}

/** The size of the real run: all 80 numbers, five prior draws. */
export const DEFAULT_SIZE: Size = { rows: 80, priorDraws: 5, grid: 180 }

/**
 * The inference, from a seed: draw weights from the prior, write the model, run expectation propagation on the (fixed) numbers
 * and turn the settled beliefs into weights, centres and curves.
 *
 * @param seed - Decides the prior draws and the order the numbers are handed over in (the numbers do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Breaking sticks from the prior')
  // region sticks
  // piece k is its break times what the earlier pieces left; the last piece takes what is left
  const product = (took: Value, left: Value): Value => {
    const end = ones([...took.shape.slice(0, -1), 1])
    return mul(concat([took, end], -1), concat([end, cumprod(left, -1)], -1))
  }
  const stickWeights = (breaks: Value): Value => product(breaks, sub(1, breaks))
  // each break takes a share of what is left: Beta(1, 1) is any share, equally likely
  const breaks = beta(foldIn(key(seed), 0), [size.priorDraws, PIECES - 1], 1, 1)
  const prior = stickWeights(breaks)
  // endregion
  say('Writing the model: pieces with weights, centres and a piece for each number')
  // region model
  const mixture = defineModel('stick-breaking mixture', (m) => {
    const row = m.range('row')
    const piece = m.range('piece', { size: PIECES })
    const cut = m.range('cut', { size: PIECES - 1 })
    const y = m.data('y', { dims: [row] })
    // each break takes a share of what is left of the stick
    const shares = m.latent('shares', d.beta(1, 1), { dims: [cut] })
    // the weight of each piece
    const weights = m.derived('weights', ops.stickBreaking(shares))
    // the centre of the group that each piece stands for
    const centres = m.latent('centres', d.normal(0, 4), { dims: [piece] })
    // the piece each number belongs to
    const which = m.latent('which', d.categorical({ probs: weights }), { dims: [row] })
    m.observe('y', d.normal(ops.at(centres, which), WIDTH), y)
  })
  // endregion
  say('Passing messages between the numbers and the pieces until none changes')
  // region infer
  // the order the numbers are handed over in is the seed's; the answer should hardly depend on it
  const order = argsort(uniform(foldIn(key(seed), 1), [size.rows]), 0)
  const bound = mixture.bind({ y: take(tensor(drawNumbers(size.rows)), order, 0) })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep of messages from every number to the pieces and back
  const run = session(route.method, { steps: 200, mode: 'eager', config: {} })
  // the beliefs the messages settled on
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Reading off the weights and the groups')
  // region reading
  const shares = beliefs.mean('shares')
  const doubts = beliefs.variance('shares')
  // the breaks are independent in the belief, so the mean of a weight is a product of means
  const weights = stickWeights(shares)
  // and its second moment is a product of second moments, which gives its spread
  const second = product(add(doubts, square(shares)), add(doubts, square(sub(1, shares))))
  const spread = sqrt(maximum(sub(second, square(weights)), 0))
  const centres = beliefs.mean('centres')
  const centreSpread = beliefs.stddev('centres')
  // each piece is its weight times a bell curve at its centre, evaluated along a grid
  const grid = linspace(RANGE[0], RANGE[1], size.grid)
  const z = div(sub(reshape(grid, [-1, 1]), reshape(centres, [1, -1])), WIDTH)
  const bells = div(exp(mul(-0.5, square(z))), WIDTH * Math.sqrt(2 * Math.PI))
  const curves = mul(bells, reshape(weights, [1, -1]))
  const total = sum(curves, 1)
  // endregion
  return {
    prior: toArray(prior) as number[][],
    weights: toFlat(weights),
    spread: toFlat(spread),
    centres: toFlat(centres),
    centreSpread: toFlat(centreSpread),
    grid: toFlat(grid),
    curves: toArray(transpose(curves)) as number[][],
    total: toFlat(total),
    rows: size.rows,
    sweeps: run.t,
    settled: run.status === 'converged',
  }
}

/**
 * The run of the story: the inference at its full size.
 *
 * @param seed - Decides the prior draws and the order of the numbers.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
