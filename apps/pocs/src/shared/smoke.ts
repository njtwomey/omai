import type { Config } from 'omai-engine/trace'
import { session } from 'omai-engine/trace'
import type { AnytimeRunSpec } from './useAnytimeRuns.ts'

/**
 * Runs one of a page's runs for a few steps in a test and reads it as the page would at a snapshot: the session of the
 * spec's method with its knobs (over which a test may shrink the budget of a sampler), then `derive` of its state.
 *
 * @param spec - The run.
 * @param over - Knobs that replace the spec's, to make the run small.
 * @param steps - The most steps to take.
 * @returns What the page keeps.
 */
export function snapshotOf<T>(spec: AnytimeRunSpec<T>, over: Config = {}, steps = 40): T {
  const s = session(spec.method, {
    config: { ...spec.config, ...over },
    seed: spec.seed ?? 1,
    steps,
    mode: 'eager',
    ...(spec.logs === undefined ? {} : { logs: spec.logs }),
  })
  return spec.derive(s, { t: s.t, elapsedMs: 0, state: s.state, status: s.status, final: true }, undefined)
}
