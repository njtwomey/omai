/**
 * The computation of 'Where is it going?': a position and a velocity that change a little at each step are tracked from
 * noisy readings of the position. The model is a chain: each step's state (position and velocity) follows from the step
 * before, and each reading sees the position only. Expectation propagation sends a bell curve along the chain in both
 * directions: forward, what the earlier readings say about each step, and backward, what the later ones say. On a chain
 * of this kind the messages settle to the exact answer. The model is also given further steps with no readings, and the
 * messages that reach those steps are the prediction. The result is plain data and a pure function of the seed, so it runs in
 * the story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { session } from 'omai-engine/trace'
// endregion

import type { ModelRun } from 'omai-engine/exposition'
import { concat, full, tensor, zeros } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { replayOfModel } from '../../kit/replay.ts'
import { AHEAD, DT, READING_SD, READINGS, START, simulate, stepCovariance } from './data.ts'

/** A belief about the position at one step: its mean and the ends of a 90 percent interval. */
export interface Belief {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The readings (one per step), and the true positions (readings plus the steps ahead). */
  readonly readings: readonly number[]
  readonly truth: readonly number[]
  /** The belief about the position at every step, readings first and then the steps ahead. */
  readonly beliefs: readonly Belief[]
  /** The belief about the velocity at the last step with a reading, and at the first step. */
  readonly velocity: { readonly first: Belief; readonly last: Belief; readonly truthLast: number }
  /** How many sweeps of messages it took, and whether they settled. */
  readonly sweeps: number
  readonly settled: boolean
  /** The messages of the run as the library records them, for a replay (undefined when it was not kept). */
  readonly record: ModelRun | undefined
}

/** How much to do. */
export interface Size {
  readonly steps: number
  readonly ahead: number
  readonly record: boolean
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { steps: READINGS, ahead: AHEAD, record: true }

/**
 * The inference of the story, from a seed.
 *
 * @param seed - Decides the track and its noise.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Moving the object and reading its position')
  const track = simulate(seed, size.steps, size.ahead)
  say('Sending bell curves along the track')
  // region model
  // each step the position moves by the velocity, and the velocity stays, up to a small random push
  const step = tensor([
    [1, DT],
    [0, 1],
  ])
  // a reading sees the position only
  const sees = tensor([1, 0])
  const tracker = defineModel('tracker', (m) => {
    const y = m.data('y', { dims: ['time'] })
    // how far each reading can be from the position: a little for a real reading, a great deal for a step ahead
    const noise = m.data('noise', { dims: ['time'] })
    const move = m.const(step)
    const position = m.const(sees)
    // the first state: a position and a velocity, each a bell curve
    const start = d.multivariateNormal(START.mean, {
      covariance: [
        [START.sd[0] ** 2, 0],
        [0, START.sd[1] ** 2],
      ],
    })
    m.scan('x', {
      over: 'time',
      init: start,
      step: ({ prev }) => d.multivariateNormal(ops.matvec(move, prev), { covariance: stepCovariance() }),
      emit: ({ state, observe }) => observe('y', d.normal(ops.dot(position, state), noise), y),
    })
  })
  // endregion
  // region infer
  // the readings, followed by steps that have none: a reading of zero that could be off by ten thousand says nothing
  const readings = concat([tensor([...track.readings]), zeros([size.ahead])])
  const noise = concat([full([size.steps], READING_SD), full([size.ahead], 10000)])
  const bound = tracker.bind({ y: readings, noise })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: messages go along the chain forward and then back
  const run = session(route.method, { steps: 300, mode: 'eager', logs: 'all', config: { tol: 1e-10 } })
  // the position and the velocity at every step, with 90 percent intervals
  const rows = route.method.posterior?.(run.state, run.config)?.summary({ prob: 0.9 }) ?? []
  // endregion
  if (rows.length === 0) throw new Error('the method returned no beliefs')
  const coordinate = (k: 0 | 1): Belief[] =>
    rows.filter((r) => r.name.endsWith(`][${k}]`)).map((r) => ({ mean: r.mean, lo: r.lower, hi: r.upper }))
  const positions = coordinate(0)
  const velocities = coordinate(1)
  const lastRead = size.steps - 1
  return {
    readings: track.readings,
    truth: track.position,
    beliefs: positions,
    velocity: {
      first: velocities[0] as Belief,
      last: velocities[lastRead] as Belief,
      truthLast: track.velocity[lastRead] as number,
    },
    sweeps: run.t,
    settled: run.status === 'converged',
    record: size.record ? replayOfModel(bound, run.run()) : undefined,
  }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the track.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
