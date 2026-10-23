import type { Json } from '../base/index.ts'
import type { TracePolicy } from './policy.ts'
import type { TraceSchema } from './schema.ts'
import { nullTracer, type RunHandle, startRun, type Tracer } from './tracer.ts'

/**
 * Trace one call of a non-iterative computation (an elimination, a factorisation, a pipeline step): the function
 * receives a run handle and emits whatever steps it wants; the call is wrapped in `run` and `end` events.
 *
 * @param name - The run name.
 * @param fn - The computation, given the run handle.
 * @param options - Configuration, seed, tracer, policy and schema.
 * @returns The function's result.
 * @concept tracing
 * @example
 * ```ts
 * import { traceCall, recorder } from 'whitebox-engine/trace'
 * const rec = recorder()
 * traceCall('sum', (run) => { run.metric(0, 'partial', 1); return 1 }, { tracer: rec }) // => 1
 * rec.events.length // => 3
 * ```
 */
export function traceCall<T>(
  name: string,
  fn: (run: RunHandle) => T,
  options: {
    readonly config?: Json
    readonly seed?: number
    readonly tracer?: Tracer
    readonly policy?: TracePolicy
    readonly schema?: TraceSchema
  } = {},
): T {
  const run = startRun(options.tracer ?? nullTracer, {
    name,
    seed: options.seed ?? null,
    config: options.config ?? {},
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.schema === undefined ? {} : { schema: options.schema }),
  })
  try {
    const result = fn(run)
    run.end('completed', 1)
    return result
  } catch (error) {
    run.end('failed', 0)
    throw error
  }
}
