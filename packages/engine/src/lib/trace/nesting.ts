/**
 * Nested runs: `subRun` runs an inner algorithm inside a step of an outer one, so the inner trace nests under the outer
 * run with the outer tracer and a policy derived from the outer one (a solver inside an optimiser, trials inside a
 * study).
 */

import { foldIn } from '../random/index.ts'
import { type Algorithm, type RunOptions, type RunResult, runAlgorithm, type StepContext } from './algorithm.ts'

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
  ctx: StepContext,
  algorithm: Algorithm<S>,
  options: Pick<RunOptions, 'maxSteps' | 'policy' | 'runId'> & { readonly index?: number },
): RunResult<S> {
  const { index = 0, ...rest } = options
  return runAlgorithm(algorithm, { ...rest, key: foldIn(ctx.key, index), parent: ctx.run })
}
