// Running an embedding algorithm while keeping every k-th configuration for the embeddingSteps record (private).
import { collecting } from '../../lib/learn/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../../lib/trace/index.ts'

/** What a recorded run returns: the final state, the step count and the recorded configurations. */
export interface RecordedRun<S> {
  readonly state: S
  readonly steps: number
  readonly positions: number[][][]
  readonly iterations: number[]
  readonly objective: number[]
}

/**
 * Run an algorithm, keeping the configuration and objective at step 0, every `every` steps and at the end.
 *
 * @param algorithm - The algorithm.
 * @param options - Run options.
 * @param every - The recording interval.
 * @param positionsOf - The configuration of a state.
 * @param objectiveOf - The objective of a state.
 * @returns The final state and the recorded steps.
 */
export function recordedRun<S>(
  algorithm: Algorithm<S>,
  options: RunOptions,
  every: number,
  positionsOf: (s: S) => number[][],
  objectiveOf: (s: S) => number,
): RecordedRun<S> {
  const step = Math.max(1, Math.floor(every))
  const c = collecting(algorithm, (s: S, t: number) =>
    t % step === 0 ? { t, positions: positionsOf(s), objective: objectiveOf(s) } : { t },
  )
  const out = runAlgorithm(c.algorithm, options)
  const kept = c.history.filter((h): h is { t: number; positions: number[][]; objective: number } => 'positions' in h)
  if (kept.at(-1)?.t !== out.steps)
    kept.push({ t: out.steps, positions: positionsOf(out.state), objective: objectiveOf(out.state) })
  return {
    state: out.state,
    steps: out.steps,
    positions: kept.map((h) => h.positions),
    iterations: kept.map((h) => h.t),
    objective: kept.map((h) => h.objective),
  }
}
