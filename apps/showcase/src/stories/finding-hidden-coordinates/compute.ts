/**
 * The computation of 'Finding hidden coordinates': twenty points are each measured five ways, and every measurement is a smooth
 * function of one hidden coordinate per point. The library's latent variable model writes each function as a short sum of
 * cosines with fixed random frequencies and unknown weights, gives every point an unknown coordinate, and expectation propagation
 * passes messages between the coordinates and the weights until it stops (or until the sweeps run out). The recovered coordinates
 * are then lined up with the true ones, which the model cannot know, and the learned functions are read along a grid. The result
 * is plain data and a pure function of the seed (which draws the points and the random waves), so it runs in the story's worker.
 * The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { alignLatents, fourierFeatures, gplvmFourierModel, simulateGplvm } from 'omai-engine/probprog/models'
import { foldIn, key } from 'omai-engine/random'
import {
  add,
  cos,
  div,
  linspace,
  matmul,
  mean,
  mul,
  reshape,
  sqrt,
  square,
  sub,
  toArray,
  toFlat,
  transpose,
  type Value,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import type { Say } from '../../kit/protocol.ts'
import { DAMPING, FEATURES, NOISE, OUTPUTS, POINTS, SWEEPS } from './data.ts'

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  readonly seed: number
  /** The true coordinate of each point, and the recovered one after it is lined up with the true ones, with its spread. */
  readonly truth: readonly number[]
  readonly recovered: readonly number[]
  readonly spread: readonly number[]
  /** The measurements, and what they would be without noise, one row per point. */
  readonly y: readonly (readonly number[])[]
  readonly clean: readonly (readonly number[])[]
  /** The learned functions: the grid of coordinates (in the frame of the true ones) and, for each measurement, its values on it. */
  readonly grid: readonly number[]
  readonly curves: readonly (readonly number[])[]
  /** How far the learned functions are from the measurements at the points (root mean square), and the noise that was put in. */
  readonly residual: number
  readonly noise: number
  /** The correlation of the recovered coordinate with the true one, and the root mean square distance over the coordinates' own size. */
  readonly correlation: number
  readonly error: number
  /** How many sweeps the run took, and whether the messages settled before the sweeps ran out. */
  readonly sweeps: number
  readonly settled: boolean
}

/** How much to do: a test uses fewer points, outputs and sweeps. */
export interface Size {
  readonly points: number
  readonly outputs: number
  readonly features: number
  readonly sweeps: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { points: POINTS, outputs: OUTPUTS, features: FEATURES, sweeps: SWEEPS }

/**
 * The inference, from a seed: make the points and their measurements, write the model, pass messages, then line the recovered
 * coordinates up with the true ones and read the learned functions.
 *
 * @param seed - Decides the points, their measurements and the random waves of the model.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Making the points and their measurements')
  const { points, outputs, features, sweeps } = size
  // region data
  const k = key(seed)
  // each measurement is a smooth function of one hidden coordinate per point, plus noise
  const data = simulateGplvm(k, { points, coords: 1, outputs, noise: NOISE })
  // the waves the model builds its functions from: random frequencies and phases, drawn once
  const waves = fourierFeatures(foldIn(k, 10), { features, coords: 1 })
  // endregion
  // region model
  // a coordinate for each point, a weight for each measurement and wave, and noise of a known size
  const model = gplvmFourierModel({ features, coords: 1, outputs, noise: NOISE })
  const bound = model.bind({ omega: waves.frequencies, phase: waves.phases, y: data.y })
  // endregion
  say('Passing messages between the coordinates and the weights')
  // region infer
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep; each message is mixed with the old one, which steadies the loops
  const config = { damping: DAMPING, restarts: 0, tol: 1e-6 }
  const run = session(route.method, { steps: sweeps, mode: 'eager', config })
  // the belief about every coordinate and weight when the messages stopped
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Lining the coordinates up with the true ones')
  // region reading
  // the model cannot say which way a coordinate points or how long it is,
  // so turn, stretch and shift it onto the true one
  const aligned = alignLatents(beliefs.mean('x'), data.latent)
  const spread = mul(beliefs.stddev('x'), aligned.scale)
  // a function of a coordinate is a weighted sum of the waves
  const wavesAt = (x: Value): Value => cos(add(matmul(x, waves.frequencies), waves.phases))
  const weights = transpose(beliefs.mean('W'))
  // what the learned functions say at the points, and how far that is from the measurements
  const residual = sqrt(mean(square(sub(data.y, matmul(wavesAt(beliefs.mean('x')), weights)))))
  // the learned functions along a grid of true-frame coordinates, taken back to the model's own coordinates
  const grid = linspace(-1.9, 1.9, 61)
  const inTrueFrame = div(sub(reshape(grid, [-1, 1]), aligned.to), aligned.scale)
  const own = add(matmul(inTrueFrame, transpose(aligned.rotation)), aligned.from)
  const curves = transpose(matmul(wavesAt(own), weights))
  // endregion
  return {
    seed,
    truth: toFlat(data.latent),
    recovered: toFlat(aligned.aligned),
    spread: toFlat(spread),
    y: toArray(data.y) as number[][],
    clean: toArray(data.clean) as number[][],
    grid: toFlat(grid),
    curves: toArray(curves) as number[][],
    residual: toFlat(residual)[0] as number,
    noise: NOISE,
    correlation: toFlat(aligned.correlation)[0] as number,
    error: toFlat(aligned.error)[0] as number,
    sweeps: run.t,
    settled: run.status === 'converged',
  }
}

/** The story's computation: the real size. */
export const run = (seed: number, say: Say = () => {}): Result => fit(seed, say)
