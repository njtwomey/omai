/**
 * The exposition record for agents moving in a rectangular world, frame by frame: every agent's position, heading and
 * state (flocking boids, people in an epidemic), with series over the frames (the flock's alignment, the counts in each
 * state), so a view can play the swarm beside its summary.
 */

import { MdmlError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** One frame: every agent's position, and its heading (radians) and state when the record has them. */
export interface AgentFrame {
  /** The time of the frame (a step, or a day). */
  readonly t: number
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly heading?: readonly number[]
  /** Each agent's state, an index into the record's `states`. */
  readonly state?: readonly number[]
}

/**
 * Agents in a world, frame by frame: positions with headings (drawn as arrows) or states (drawn in their colours), and
 * series with one value per frame drawn beneath.
 */
export interface AgentSteps {
  readonly kind: 'agentSteps'
  readonly method: string
  /** The world's size, and whether it wraps round (a torus). */
  readonly world: { readonly width: number; readonly height: number; readonly wrap: boolean }
  /** The names of the agents' states, in index order (S, E, I, R). */
  readonly states?: readonly string[]
  /** A CSS colour per state (default the categorical palette). */
  readonly stateColours?: readonly string[]
  readonly frames: readonly AgentFrame[]
  /** Series over the frames, one value per frame, drawn beneath with the current frame marked. */
  readonly series?: readonly { readonly name: string; readonly values: readonly number[] }[]
  /** What the series measure (their axis), and what the time is (default 'step'). */
  readonly seriesName?: string
  readonly timeName?: string
}

/**
 * Agent steps, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a frame's arrays or a series differ in length; `value` for a world that is not
 *   positive.
 * @example
 * ```ts
 * import { agentSteps } from 'mdml-engine/exposition'
 * const r = agentSteps({
 *   method: 'boids',
 *   world: { width: 10, height: 10, wrap: true },
 *   frames: [{ t: 0, x: [1, 2], y: [3, 4], heading: [0, 1.5] }],
 *   series: [{ name: 'polarisation', values: [0.2] }],
 * })
 * r.frames.length // => 1
 * ```
 */
export function agentSteps(record: Omit<AgentSteps, 'kind'>): AgentSteps {
  if (!(record.world.width > 0 && record.world.height > 0))
    throw new MdmlError('value', 'agentSteps: the world must have a positive size')
  for (const [f, frame] of record.frames.entries()) {
    sameLength('agentSteps', `frames[${f}].y`, frame.y.length, frame.x.length)
    if (frame.heading !== undefined)
      sameLength('agentSteps', `frames[${f}].heading`, frame.heading.length, frame.x.length)
    if (frame.state !== undefined) sameLength('agentSteps', `frames[${f}].state`, frame.state.length, frame.x.length)
  }
  for (const s of record.series ?? [])
    sameLength('agentSteps', `series '${s.name}'`, s.values.length, record.frames.length)
  return { kind: 'agentSteps', ...record }
}
