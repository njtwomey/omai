/**
 * The computation of 'Two humps': the library's two-component Gaussian mixture (two bell curves with unknown centres, one shared
 * width and an unknown share of the eruptions each, plus a hidden label per eruption) is fitted to the Old Faithful waiting times by
 * expectation propagation, which passes messages between the eruptions and the components until none of them changes. The
 * result is plain data: the fitted curves, each eruption's chance of belonging to the upper component, and how the two centres
 * settled sweep by sweep. It is a pure function of the seed, which only decides the order in which the eruptions are handed to
 * the method. It runs in the story's worker, so the engine is bundled there and not in the page. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { gaussianMixtureModel } from 'omai-engine/probprog/models'
import { foldIn, key, uniform } from 'omai-engine/random'
import {
  add,
  argsort,
  div,
  exp,
  linspace,
  mean,
  mul,
  reshape,
  sliceAxis,
  square,
  std,
  sub,
  sum,
  take,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import { toFlat, type Value } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { WAIT_RANGE, waitingTimes } from './data.ts'

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The waiting times (minutes) the curves are drawn at. */
  readonly grid: readonly number[]
  /** The density of the lower and the upper component (its share times its bell curve) and their sum, on the grid. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  readonly total: readonly number[]
  /** For each observation, in the order of the data, the chance (0 to 1) that it belongs to the upper component. */
  readonly chance: readonly number[]
  /** The fitted centres (minutes) of the lower and upper component, their common spread (minutes) and their shares. */
  readonly means: readonly [number, number]
  readonly spread: number
  readonly shares: readonly [number, number]
  /** The two centres after some of the sweeps of the method's last pass (sweep 0 is before any note has been passed), in minutes. */
  readonly trace: {
    readonly sweep: readonly number[]
    readonly lower: readonly number[]
    readonly upper: readonly number[]
  }
  /** How many sweeps the last pass of the method took, and whether the method settled (it should). */
  readonly sweeps: number
  readonly settled: boolean
  /** Whether the method started over more gently after a first pass that could not place some eruptions (the trace is of the last pass). */
  readonly restarted: boolean
  /** How many observations were used. */
  readonly rows: number
}

/** How much to use: the run uses every waiting time, a test the first few. */
export interface Size {
  readonly rows: number
  readonly grid: number
}

/** The size of the real run: all 272 waiting times. */
export const DEFAULT_SIZE: Size = { rows: 272, grid: 160 }

/**
 * The step the last pass of a method that restarts begins at: 0 for a run that never restarted, else the step after the one
 * the last restart happened at, and never later than the last step (so there is always at least one step to read).
 *
 * @param restartAt - The sweeps completed when the last restart began (`restartAt` of the method's state, 0 for none).
 * @param steps - The steps the run took.
 * @returns The first step of the last pass.
 */
export function lastPassStart(restartAt: number, steps: number): number {
  return Math.min(restartAt === 0 ? 0 : restartAt + 1, Math.max(0, steps))
}

/**
 * The inference, from a seed: put the waiting times on a scale of about one, fit the mixture by expectation propagation and
 * turn the fitted beliefs into curves, chances and a trace of the sweeps.
 *
 * @param seed - Decides the order in which the eruptions are handed to the method (the data do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much of the data to use.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Writing the model: two bell curves and a hidden label for each eruption')
  const raw = sliceAxis(waitingTimes(), 0, 0, size.rows)
  // region model
  // two bell curves with unknown centres, one shared width and unknown shares of the eruptions;
  // the centres are kept in order, so "the first curve" is always the lower one
  const mixture = gaussianMixtureModel({ components: 2, ordered: true })
  // the model is written for data of about unit size: centre and scale the minutes
  const centre = mean(raw)
  const scale = std(raw)
  const standard = div(sub(raw, centre), scale)
  // the order the eruptions are handed over in is the seed's; the answer should not depend on it
  const order = argsort(uniform(foldIn(key(seed), 0), [size.rows]), 0)
  const shuffled = take(standard, order, 0)
  // endregion
  say('Passing notes between the eruptions and the two curves')
  // region infer
  const bound = mixture.bind({ y: shuffled })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep of notes from every eruption to the curves and back
  const run = session(route.method, { steps: 200, mode: 'eager', config: { tol: 1e-6 } })
  // the beliefs the notes settled on
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Reading off the curves')
  // region reading
  const minutes = (z: Value): Value => add(mul(z, scale), centre)
  const means = minutes(beliefs.mean('mu'))
  const spread = mul(beliefs.mean('sigma'), scale)
  const shares = reshape(beliefs.mean('weights'), [1, 2])
  // each curve is its share times a bell curve, evaluated along a grid of waiting times
  const grid = linspace(WAIT_RANGE[0], WAIT_RANGE[1], size.grid)
  const z = div(sub(reshape(grid, [-1, 1]), reshape(means, [1, 2])), spread)
  const bells = div(exp(mul(-0.5, square(z))), mul(spread, Math.sqrt(2 * Math.PI)))
  const curves = mul(bells, shares)
  const total = sum(curves, 1)
  // the chance that each eruption belongs to the upper curve, back in the order of the data
  const chance = take(beliefs.mean('z'), argsort(order, 0), 0)
  // endregion
  // The method may start again more gently when its first pass stalls on the eruptions between the humps; the figure of the sweeps
  // shows the last pass only, from where its restart count last changed.
  const began = run.state.restartAt
  const finalPass = lastPassStart(began, run.t)
  // A few sweeps are read, the first ones closely and then every fourth, because each reading costs a moment.
  const kept = toFlat(linspace(finalPass, run.t, run.t - finalPass + 1)).filter(
    (t) => t - finalPass < 8 || (t - finalPass) % 4 === 0 || t === run.t,
  )
  const at = (t: number): readonly number[] => {
    const early = route.method.posterior?.(run.stateAt(t), run.config)
    return early === undefined ? [Number.NaN, Number.NaN] : toFlat(minutes(early.mean('mu')))
  }
  const settling = kept.map(at)
  const centres = toFlat(means)
  const weights = toFlat(shares)
  return {
    grid: toFlat(grid),
    lower: toFlat(sliceAxis(curves, 1, 0, 1)),
    upper: toFlat(sliceAxis(curves, 1, 1, 2)),
    total: toFlat(total),
    chance: toFlat(chance),
    means: [centres[0] as number, centres[1] as number],
    spread: toFlat(spread)[0] as number,
    shares: [weights[0] as number, weights[1] as number],
    trace: {
      sweep: kept.map((t) => t - finalPass),
      lower: settling.map((m) => m[0] as number),
      upper: settling.map((m) => m[1] as number),
    },
    sweeps: run.t - finalPass,
    settled: run.status === 'converged',
    restarted: began > 0,
    rows: size.rows,
  }
}

/**
 * The run of the story: the inference on all the waiting times.
 *
 * @param seed - Decides the order of the eruptions.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
