import type { Key } from '../../random/index.ts'
import type { RunHandle, TracePolicy, Tracer } from '../../trace/index.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import type { Posterior, PosteriorForm } from '../posterior.ts'
import type { Model } from '../primitives.ts'

/**
 * How good an engine's answer is, which the planner ranks before cost: exact, asymptotically exact by gradient-based
 * or conditional Markov chains, asymptotically exact by weighting, an approximation, or likelihood-free.
 */
export type EngineQuality = 'exact' | 'mcmc' | 'weighted' | 'approximate' | 'likelihood-free'

/** The order of {@link EngineQuality} (lower is better). */
export const QUALITY_RANK: Readonly<Record<EngineQuality, number>> = Object.freeze({
  exact: 0,
  mcmc: 1,
  weighted: 2,
  approximate: 3,
  'likelihood-free': 4,
})

/** Whether an engine applies to an analysed program, its estimated cost and the reason. */
export interface Feasibility {
  readonly applies: boolean
  /** A rough count of elementary operations (log-density terms, factor entries, gradient evaluations). */
  readonly cost: number
  readonly quality: EngineQuality
  readonly reason: string
}

/** What every engine run receives. */
export interface EngineRunOptions {
  readonly key: Key
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  /** Run as a child of another traced run. */
  readonly parent?: RunHandle
}

/**
 * The inference-engine contract (section 3.6 of the design): a name, the posterior form it produces, `applies`
 * (feasibility, estimated cost and a reason, from the program's analysis) and `run` (a traced algorithm returning a
 * {@link Posterior}). A new algorithm is a new engine; the planner ranks every engine that applies.
 *
 * @concept inference-engines
 */
export interface InferenceEngine {
  readonly name: string
  readonly form: PosteriorForm
  readonly description: string
  applies(analysis: ProgramAnalysis): Feasibility
  run<D>(model: Model<D>, data: D, analysis: ProgramAnalysis, options: EngineRunOptions): Posterior
}

/**
 * A feasibility verdict.
 *
 * @param applies - Whether the engine applies.
 * @param quality - The quality of its answer.
 * @param cost - The estimated cost (Infinity when it does not apply).
 * @param reason - Why.
 * @returns The verdict.
 * @example
 * ```ts
 * import { feasibility } from 'whitebox-engine/probprog'
 * feasibility(false, 'exact', 0, 'no discrete sites').cost // => Infinity
 * ```
 */
export function feasibility(applies: boolean, quality: EngineQuality, cost: number, reason: string): Feasibility {
  return { applies, quality, cost: applies ? cost : Number.POSITIVE_INFINITY, reason }
}

/**
 * The trace and parent options of a run, for passing on to inner algorithms.
 *
 * @param options - The engine run options.
 * @returns The tracer, policy and parent that are set.
 * @example
 * ```ts
 * import { runTracing } from 'whitebox-engine/probprog'
 * import { key } from 'whitebox-engine/random'
 * Object.keys(runTracing({ key: key(0) })) // => []
 * ```
 */
export function runTracing(options: EngineRunOptions): {
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  readonly parent?: RunHandle
} {
  return {
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }
}
