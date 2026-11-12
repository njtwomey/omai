/**
 * The computation of 'Combining two measurements': a room has a temperature drawn from what we believed beforehand, and two
 * thermometers (a precise digital one and a rough dial) each read it with an error of their own size. The library's expectation
 * propagation is run on the model of that (a prior factor and one factor per reading, all attached to the temperature) and
 * everything it sent is kept: the three messages of the first sweep with their numbers, the belief it ends with, how much each
 * sweep changed, and the whole run as a record the page can replay. A pure function of the seed, which decides the temperature and
 * the two errors. It runs in the story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal } from 'omai-engine/random'
import { item } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import type { ModelRun } from 'omai-engine/exposition'
import type { Say } from '../../kit/protocol.ts'
import { replayOfModel } from '../../kit/replay.ts'
import { DIAL_SD, DIGITAL_SD, PRIOR, SOURCES } from './data.ts'

/** A message one factor sent to the temperature in the first sweep: a bell curve, in the numbers EP keeps it by. */
export interface Message {
  /** Which factor sent it. */
  readonly from: (typeof SOURCES)[number]
  /** The mean and variance of the bell curve. */
  readonly mean: number
  readonly variance: number
  /** One divided by the variance: how sharply the curve points at its mean. */
  readonly precision: number
}

/** What the figures and the prose read. All of it is plain data. */
export interface Result {
  /** The temperature the room really had, which no part of the inference was told. */
  readonly truth: number
  /** The two readings. */
  readonly digital: number
  readonly dial: number
  /** The messages of the first sweep, in the order of {@link SOURCES}. */
  readonly messages: readonly Message[]
  /** The belief about the temperature EP ends with: a bell curve with this mean and spread. */
  readonly belief: { readonly mean: number; readonly sd: number }
  /** How many sweeps EP made, and whether it ended because a sweep changed nothing. */
  readonly sweeps: number
  readonly settled: boolean
  /** The largest change of a belief in each sweep, from the first. */
  readonly changes: readonly number[]
  /** The run as a model-run record (the graph, and the messages and beliefs at each step), at most 40 steps. */
  readonly record: ModelRun
}

/** The part of a tape event this reads. */
interface MessageEvent {
  readonly type: string
  readonly kind?: string
  readonly step?: number
  readonly from?: string
  readonly payload?: unknown
}

/** The ids of the three factors in the factor graph of the model. */
const FACTOR: Readonly<Record<Message['from'], string>> = {
  prior: 'factor:temperature',
  digital: 'factor:digital',
  dial: 'factor:dial',
}

/** The message of a tape event as mean and variance, from its natural parameters `[h, J]` (mean h / J, variance 1 / J). */
function messageOf(from: Message['from'], event: MessageEvent | undefined): Message {
  if (event === undefined) throw new Error(`the run has no message from the ${from} factor in its first sweep`)
  const eta = (event.payload as { readonly eta: readonly [number, number] }).eta
  return { from, mean: eta[0] / eta[1], variance: 1 / eta[1], precision: eta[1] }
}

/**
 * The story, from a seed: make the temperature and the two readings, write the model, run expectation propagation on it and keep
 * the messages it sent.
 *
 * @param seed - Decides the temperature and the errors of the two readings.
 * @param say - Says what is happening, for the play button.
 * @returns The result.
 */
export function combine(seed: number, say: Say = () => {}): Result {
  say('Making a room and two readings')
  // region readings
  const base = key(seed)
  // the room's temperature, drawn from what we believed beforehand
  const truth = item(normal(foldIn(base, 0), [], { loc: PRIOR.mean, scale: PRIOR.sd }))
  // each reading is the temperature plus an error of its instrument's size
  const digital = item(normal(foldIn(base, 1), [], { loc: truth, scale: DIGITAL_SD }))
  const dial = item(normal(foldIn(base, 2), [], { loc: truth, scale: DIAL_SD }))
  // endregion
  say('Setting up the model')
  // region model
  const model = defineModel('two-measurements', (m) => {
    // a bell curve for what we believed before measuring
    const temperature = m.latent('temperature', d.normal(PRIOR.mean, PRIOR.sd))
    // each reading is a bell curve around the temperature, as wide as its instrument's error
    m.observe('digital', d.normal(temperature, DIGITAL_SD), m.data('digital', { role: 'response' }))
    m.observe('dial', d.normal(temperature, DIAL_SD), m.data('dial', { role: 'response' }))
  })
  // endregion
  say('Passing messages')
  // region infer
  const bound = model.bind({ digital, dial })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: every factor sends its message, then the belief is updated
  const sweeps = session(route.method, { steps: 20, mode: 'eager', config: { tol: 1e-9 }, logs: 'all' })
  // the belief the messages settled on
  const beliefs = route.method.posterior?.(sweeps.state, sweeps.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  say('Reading the messages')
  const tape = sweeps.run()
  const events: readonly MessageEvent[] = tape.events
  const first = events.filter((e) => e.type === 'message' && e.kind === 'mp/message' && e.step === 1)
  const record = replayOfModel(bound, tape)
  const change = record.series.find((s) => s.name === 'change')
  return {
    truth,
    digital,
    dial,
    messages: SOURCES.map((from) =>
      messageOf(
        from,
        first.find((e) => e.from === FACTOR[from]),
      ),
    ),
    belief: { mean: item(beliefs.mean('temperature')), sd: item(beliefs.stddev('temperature')) },
    sweeps: sweeps.t,
    settled: sweeps.status === 'converged',
    changes: change?.values ?? [],
    record,
  }
}

/**
 * The run of the story.
 *
 * @param seed - The seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => combine(seed, say)
