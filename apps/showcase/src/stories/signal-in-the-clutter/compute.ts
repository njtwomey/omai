/**
 * The computation of 'A signal in the clutter': Minka's clutter problem from the library. A hidden value is read several
 * times; each reading is either a measurement of the value with a spread of one, or clutter, a draw from a wide bell curve
 * around zero that says nothing about the value. A hidden switch for each reading says which it is. Expectation propagation
 * sends bell curves between the value and the readings and reports one bell curve for the value, plus a chance for each
 * reading that it is clutter. The exact posterior of the value, which the library works out by summing over a fine grid of
 * values, is returned beside it as a curve. The model is run on two sets of readings: one with a clear signal, and the same
 * readings with a second group added that could pass for the signal. The result is plain data and a pure function of the seed
 * (which draws the readings), so it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { clutterProblem, clutterTruth } from 'omai-engine/probprog/models'
import { session } from 'omai-engine/trace'
// endregion

import type { ModelRun } from 'omai-engine/exposition'
import { linspace, toFlat } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { replayOfModel } from '../../kit/replay.ts'
import { clearReadings, RANGE, withDecoys } from './data.ts'

/** The curve the exact posterior is drawn on: values of the hidden value, and the density at each. */
export interface Curve {
  readonly x: readonly number[]
  readonly density: readonly number[]
}

/** What the model concludes about one set of readings. */
export interface Case {
  /** The readings, in the order they were taken. */
  readonly readings: readonly number[]
  /** EP's bell curve for the value: its mean and standard deviation. */
  readonly bell: { readonly mean: number; readonly sd: number }
  /** For each reading, the chance (0 to 1) that it is clutter. */
  readonly clutter: readonly number[]
  /** The exact posterior of the value: its mean and standard deviation, the density on a grid, and where its highest point is. */
  readonly exact: { readonly mean: number; readonly sd: number; readonly curve: Curve; readonly peak: number }
  /** How many sweeps of messages it took, and whether they settled. */
  readonly sweeps: number
  readonly settled: boolean
  /** The messages of the run as the library records them, for a replay (undefined when it was not kept). */
  readonly record: ModelRun | undefined
}

/** What the figures and the prose read: the readings with a clear signal, and the same with a decoy group added. */
export interface Result {
  readonly clear: Case
  readonly ambiguous: Case
}

/** How much to do. */
export interface Size {
  readonly record: boolean
  readonly grid: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { record: true, grid: 241 }

/**
 * The inference on one set of readings: pass messages until they settle, read the bell curve and the clutter chances, and
 * work out the exact curve to put beside it.
 *
 * @param readings - The readings.
 * @param say - Says what is happening.
 * @param size - How much to do.
 * @returns What the model concludes.
 */
export function solve(readings: readonly number[], say: Say = () => {}, size: Size = DEFAULT_SIZE): Case {
  say('Sending bell curves between the value and the readings')
  // region infer
  // the library's clutter problem, with these readings as the data
  const bound = clutterProblem.bound({ data: { y: [...readings] }, generating: {} })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: every reading sends a message to the value, which sends one back to each
  const run = session(route.method, {
    steps: 300,
    mode: 'eager',
    logs: 'all',
    config: { tol: 1e-8, damping: 0.5 },
  })
  // the bell curve for the value and the chance of clutter for each reading
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Working out the exact curve, to see what the bell curve left out')
  // region exact
  // the exact posterior of the value: the library sums it over a fine grid of values
  const truth = clutterTruth([...readings], 4001)
  const oracle = truth.marginals['theta']
  // endregion
  if (oracle === undefined) throw new Error('the library gave no exact posterior')
  // The density at a point is the exact probability of the cell around it, divided by its width: the slope of the cumulative
  // distribution across half a grid step on each side. A fixed wider window would blur the narrow humps the figure is about.
  const xs = toFlat(linspace(RANGE[0], RANGE[1], size.grid))
  const h = (RANGE[1] - RANGE[0]) / (size.grid - 1) / 2
  const density = xs.map((x) => Math.max(0, (oracle.cdf(x + h) - oracle.cdf(x - h)) / (2 * h)))
  const peak = xs[density.indexOf(Math.max(...density))] as number
  return {
    readings,
    bell: { mean: toFlat(beliefs.mean('theta'))[0] as number, sd: toFlat(beliefs.stddev('theta'))[0] as number },
    clutter: toFlat(beliefs.mean('z')),
    exact: { mean: oracle.mean, sd: oracle.sd, curve: { x: xs, density }, peak },
    sweeps: run.t,
    settled: run.status === 'converged',
    record: size.record ? replayOfModel(bound, run.run()) : undefined,
  }
}

/**
 * The run of the story: the readings with a clear signal, then the same readings with a decoy group added.
 *
 * @param seed - Draws the readings.
 * @param say - Says what is happening.
 * @returns The result.
 */
export function run(seed: number, say: Say = () => {}): Result {
  say('Taking the readings')
  const clear = solve(clearReadings().values, say)
  say('Adding a second group of readings')
  const ambiguous = solve(withDecoys(seed).values, say)
  return { clear, ambiguous }
}
