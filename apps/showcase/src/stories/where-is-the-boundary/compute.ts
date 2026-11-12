/**
 * The computation of 'Where is the boundary?': the library's Gaussian-process classifier, fitted by expectation propagation.
 * Each point has a hidden score; the score of a point is tied to the scores of the points near it (a smooth function), and
 * a point's class says only whether its score is above or below zero. Expectation propagation gives every point a site: a
 * bell curve for that point's score, built from its class alone. It then passes the sites' bell curves through the smoothness
 * tie, updates one site at a time against what the others say, and repeats until no site changes. The result is the
 * probability of the second class over a grid of the plane (from each cell's score and its uncertainty). It is a pure
 * function of the seed (which decides the order the points are handed over in), so it runs in the story's worker. The regions
 * are the code shown.
 */

// region imports
import { gpClassifier } from 'omai-engine/gp'
import { rbfKernel, scaleKernel } from 'omai-engine/kernels'
import { key } from 'omai-engine/random'
// endregion

import { hyperparameters, type Kernel } from 'omai-engine/kernels'
import { linspace, meshgrid, reshape, sqrt, stack, take, toArray, toFlat } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { drawPoints, EXTENT, type Points, shuffled } from './data.ts'

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The points and their classes. */
  readonly points: Points
  /** The grid the map is drawn on: the values of the first and the second coordinate. */
  readonly xs: readonly number[]
  readonly ys: readonly number[]
  /** The probability of the second class in each cell, by row (the second coordinate) and then column. */
  readonly probability: readonly (readonly number[])[]
  /** The standard deviation of the hidden score in each cell, in the same layout: how unsure the model is about the score. */
  readonly doubt: readonly (readonly number[])[]
  /** The share of the points that the classifier puts on the side of their own class. */
  readonly accuracy: number
  /** How many site updates the run took, whether it settled, and the log marginal likelihood it ended with. */
  readonly updates: number
  readonly settled: boolean
  /** The fitted kernel's length scale and variance, and the size of the grid. */
  readonly kernel: { readonly lengthScale: number; readonly variance: number }
}

/** How much to do. */
export interface Size {
  readonly points: number
  readonly grid: number
}

/** The size of the real run: forty points and a map of 41 by 41 cells. */
export const DEFAULT_SIZE: Size = { points: 40, grid: 41 }

/** The length scale and the output variance of a fitted scaled kernel. */
function kernelValues(kernel: Kernel): [number, number] {
  const found = hyperparameters(kernel)
  const read = (word: string): number => found.find((h) => h.name.includes(word))?.value[0] ?? Number.NaN
  return [read('lengthScale'), read('outputScale')]
}

/**
 * The inference of the story, from a seed.
 *
 * @param seed - Decides the order the points are handed over in.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Setting out the points')
  const points = shuffled(drawPoints(size.points), seed)
  say('Passing messages')
  // region infer
  // a smooth function: points close together get similar scores, and the length scale says how close
  const smooth = scaleKernel(rbfKernel({ lengthScale: 1 }), 4)
  // a point's class is its score through a probit link, and the classifier is run by expectation propagation
  const classifier = gpClassifier.with({
    method: 'ep',
    likelihood: 'probit',
    kernel: smooth,
    optimise: true,
  })
  const data = { x: points.x.map((p) => [...p]), y: [...points.y] }
  const run = classifier.session(data, { key: key(seed), logs: 'all' })
  // run the site updates until none of them changes
  run.advance()
  const fitted = run.fittedAt(run.t)
  // endregion
  say('Drawing the map')
  // region map
  // a grid over the plane, one row per cell
  const xs = linspace(EXTENT.x[0], EXTENT.x[1], size.grid)
  const ys = linspace(EXTENT.y[0], EXTENT.y[1], size.grid)
  const cells = stack(
    meshgrid([xs, ys]).map((g) => reshape(g, [-1])),
    1,
  )
  // the probability of the second class in each cell
  const probability = reshape(take(fitted.predictProba(cells), 1, 1), [size.grid, size.grid])
  // how unsure the score is in each cell, once the sites have settled
  const doubt = sqrt(reshape(fitted.latent(cells).variance, [size.grid, size.grid]))
  // endregion
  // how many of the points land on the side of their own class
  const own = toFlat(fitted.predict(points.x.map((p) => [...p])))
  const rightSide = own.filter((c, i) => c === points.y[i]).length / points.y.length
  const [lengthScale, variance] = kernelValues(fitted.kernel)
  return {
    points,
    xs: toFlat(xs),
    ys: toFlat(ys),
    probability: toArray(probability) as number[][],
    doubt: toArray(doubt) as number[][],
    accuracy: rightSide,
    updates: run.t,
    settled: run.status === 'converged',
    kernel: { lengthScale, variance },
  }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the order the points are handed over in.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
