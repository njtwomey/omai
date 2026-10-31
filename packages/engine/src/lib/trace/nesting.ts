/**
 * Nested runs: `subRun` runs an inner algorithm inside a step of an outer one, so the inner trace nests under the outer
 * run with the outer tracer and a policy derived from the outer one (a solver inside an optimiser, trials inside a
 * study), and `mountContext`, the step context of an inner method mounted under a prefix of the outer one's paths (its
 * contract is mounted with `prefixContract`; the types follow, `inner/` + each path).
 */

import { foldIn } from '../random/index.ts'
import { type Algorithm, type RunOptions, type RunResult, runAlgorithm, type StepContext } from './algorithm.ts'
import type { Concrete, Prefixed } from './contract.ts'

/**
 * Run an inner algorithm inside a step of an outer one, as a child run: its trace nests under the outer run, it uses the
 * outer run's tracer and the policy {@link childPolicy} derives from the outer one (unless `policy` is given), and its
 * key is derived from the outer step's key (so nesting stays reproducible). Read run trees back with `runTree`.
 *
 * @param ctx - The outer step's context.
 * @param algorithm - The inner algorithm.
 * @param options - Its step budget, and an index to tell several inner runs in one step apart.
 * @returns The inner run's result.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { recorder, runAlgorithm, subRun, type Algorithm } from 'mdml-engine/trace'
 * const inner: Algorithm<number> = { name: 'inner', config: {}, init: () => 0, step: (s) => s + 1 }
 * const outer: Algorithm<number> = {
 *   name: 'outer', config: {}, init: () => 0,
 *   step: (s, ctx) => s + subRun(ctx, inner, { maxSteps: 3 }).state,
 * }
 * const rec = recorder()
 * runAlgorithm(outer, { seed: 0, maxSteps: 2, tracer: rec }).state // => 6
 * ```
 */
export function subRun<S>(
  ctx: Pick<StepContext, 't' | 'key' | 'run'>,
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'maxSteps' | 'policy' | 'runId'> & { readonly index?: number },
): RunResult<S> {
  const { index = 0, ...rest } = options
  return runAlgorithm(algorithm, { ...rest, key: foldIn(ctx.key, index), parent: ctx.run })
}

/**
 * The step context of an inner method mounted under a prefix: whatever the inner algorithm logs, asks `wants` of,
 * emits or taps goes to the outer context under `prefix/`. The outer context's path type must hold the mounted paths
 * (`Prefixed<P, K>`, the type of {@link prefixContract}'s result); where the outer paths are computed at run time
 * (a variant's name in the prefix), they are `string` and the mount is unchecked.
 *
 * @param ctx - The outer step's context.
 * @param prefix - The prefix, without a trailing `/`.
 * @returns A context for the inner algorithm, whose paths are the inner ones.
 * @concept log-contract
 * @example
 * ```ts
 * import { mountContext, recorder, runAlgorithm, startRun, stepContext } from 'mdml-engine/trace'
 * import { key } from 'mdml-engine/random'
 * const rec = recorder()
 * const run = startRun(rec, { name: 'x', seed: 0, config: {} })
 * mountContext(stepContext(2, key(0), run), 'inner').log('loss', 0.5)
 * rec.events.at(-1) // => { type: 'metric', runId: 'x:0', step: 2, name: 'inner/loss', value: 0.5 }
 * ```
 */
export function mountContext<P extends string, K extends string>(
  ctx: StepContext<Prefixed<P, K>>,
  prefix: P,
): StepContext<K> {
  const at = (path: string): Concrete<Prefixed<P, K>> => `${prefix}/${path}` as Concrete<Prefixed<P, K>>
  return {
    ...ctx,
    log: (path, value) => ctx.log(at(path), value),
    wants: (path) => ctx.wants(at(path)),
    emit: (path, context) => ctx.emit(at(path), context),
    taps: (name) => ctx.taps(`${prefix}/${name}`),
  }
}
