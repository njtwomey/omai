/**
 * Flocking: Reynolds' boids run on the dynamics core and recorded frame by frame (positions and headings) with the
 * flock's polarisation, so a view can play order emerging from three local rules.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoidsOptions, type BoidsState, boidsAlgorithm, polarisation } from '../../lib/dynamics/index.ts'
import { type AgentFrame, agentSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'

/** Options of {@link flocking}: the flock's, and how long to run and record. */
export interface FlockingOptions extends BoidsOptions {
  /** Time steps to run (default 600). */
  readonly steps?: number
  /** Record every this many steps (default 2). */
  readonly every?: number
  readonly key?: Key
}

/** The outcome of {@link flocking}: the final flock and the polarisation at each recorded frame. */
export interface FlockingResult {
  readonly flock: BoidsState
  readonly polarisation: readonly number[]
}

/**
 * Run a flock of boids and record it: every recorded frame's positions and headings, and the polarisation (1 when the
 * flock is aligned) as a series beneath.
 *
 * @param options - The flock (size, world, sight, the three rule weights, speeds, noise), the steps, how often to
 *   record, and the key.
 * @returns The final flock and the polarisation per frame, with an `agentSteps` record.
 * @throws MdmlError with code `value` for non-positive steps or recording interval.
 */
export function flocking(options: FlockingOptions = {}): Explained<FlockingResult> {
  const { steps = 600, every = 2, key = makeKey(0), ...boids } = options
  if (!(Number.isInteger(steps) && steps >= 1 && Number.isInteger(every) && every >= 1))
    throw new MdmlError('value', 'flocking: steps and every must be positive integers')
  const algorithm = boidsAlgorithm(boids)
  const frames: AgentFrame[] = []
  const pol: number[] = []
  const record = (s: BoidsState, t: number): void => {
    frames.push({
      t,
      x: Array.from(s.x),
      y: Array.from(s.y),
      heading: Array.from(s.vx, (vx, i) => Math.atan2(s.vy[i] ?? 0, vx)),
    })
    pol.push(polarisation(s))
  }
  let state = algorithm.init(key)
  record(state, 0)
  for (let t = 0; t < steps; t += every) {
    const n = Math.min(every, steps - t)
    state = runAlgorithm(algorithm, { key, maxSteps: n, start: { state, step: t } }).state
    record(state, t + n)
  }
  return {
    flock: state,
    polarisation: pol,
    exposition: [
      agentSteps({
        method: 'boids',
        world: { width: options.width ?? 100, height: options.height ?? 100, wrap: true },
        frames,
        series: [{ name: 'polarisation', values: pol }],
        seriesName: 'polarisation',
        timeName: 'step',
      }),
    ],
  }
}
