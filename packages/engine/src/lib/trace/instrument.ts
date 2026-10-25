/**
 * Tracing for code that is not an `Algorithm`: `instrument` wraps a function so every call becomes a run recording its
 * numeric inputs and output, and `traceCall` traces a single non-iterative computation (an elimination, a
 * factorisation, a pipeline step) that emits its own steps.
 */

import type { Json } from '../base/index.ts'
import { isTensor } from '../tensor/index.ts'
import type { TensorRole } from './events.ts'
import { policies, type TracePolicy } from './policy.ts'
import { channels, type TraceSchema } from './schema.ts'
import { nullTracer, type RunHandle, startRun, type Tracer } from './tracer.ts'

/** Options for {@link instrument}. */
export interface InstrumentOptions {
  /** The run name (default the function's name, else `call`). */
  readonly name?: string
  readonly tracer?: Tracer
  /** What to record (default: metrics and `state` and `output` tensor summaries, every call). */
  readonly policy?: TracePolicy
  readonly config?: Json
  /**
   * A clock in milliseconds, injected by the caller. When given, each call's duration is logged at `debug` level; it
   * is never part of a run otherwise, since published code reads no clocks and traces stay deterministic.
   */
  readonly clock?: () => number
}

function recordValue(run: RunHandle, name: string, role: TensorRole, value: unknown): void {
  if (typeof value === 'number') run.metric(0, name, value)
  else if (typeof value === 'boolean') run.metric(0, name, value ? 1 : 0)
  else if (isTensor(value)) run.tensor(0, name, role, value)
  else if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    run.tensor(0, name, role, value as unknown as ArrayLike<number>)
  } else if (Array.isArray(value) && value.every((v) => typeof v === 'number')) {
    run.tensor(0, name, role, value as number[])
  }
}

/**
 * Trace every call of a function: each call is a run (named after the function, with ids `name#1`, `name#2`, ...)
 * that records its numeric inputs and output, numbers as metrics (`arg0`, `arg1`, ..., `output`) and tensors or
 * numeric arrays as `state` (inputs) and `output` tensors, summarised or in full as the policy says. Other arguments
 * are not recorded. A throwing call ends its run as `failed` and rethrows. Timing is excluded unless a clock is
 * injected (then it is a debug log line), so traces of the same calls are identical.
 *
 * @param fn - The function.
 * @param options - Run name, tracer, policy, configuration and an optional clock.
 * @returns A function with the same signature that records each call.
 * @concept tracing
 * @example
 * ```ts
 * import { instrument, metricSeries, recorder } from 'mdml-engine/trace'
 * const rec = recorder()
 * const hyp = instrument((a: number, b: number) => Math.hypot(a, b), { name: 'hypot', tracer: rec })
 * hyp(3, 4) // => 5
 * metricSeries(rec.run(), 'output').values // => [5]
 * rec.events[0].runId // => 'hypot#1'
 * ```
 */
export function instrument<A extends readonly unknown[], R>(
  fn: (...args: A) => R,
  options: InstrumentOptions = {},
): (...args: A) => R {
  const name = options.name ?? (fn.name === '' ? 'call' : fn.name)
  const arity = Math.max(1, fn.length)
  const metrics: Record<string, { readonly description: string }> = {}
  const tensors: Record<string, { readonly role: TensorRole; readonly description: string }> = {}
  for (let i = 0; i < arity; i++) {
    metrics[`arg${i}`] = { description: `argument ${i}` }
    tensors[`arg${i}`] = { role: 'state', description: `argument ${i}` }
  }
  metrics['output'] = { description: 'return value' }
  tensors['output'] = { role: 'output', description: 'return value' }
  const schema = channels({ metrics, tensors })
  const policy = options.policy ?? { ...policies.minimal, tensors: true, roles: ['state', 'output'] }
  let calls = 0
  return (...args: A): R => {
    calls += 1
    const run = startRun(options.tracer ?? nullTracer, {
      name,
      runId: `${name}#${calls}`,
      seed: null,
      config: options.config ?? {},
      schema,
      policy,
    })
    const started = options.clock?.()
    if (run.enabled) for (const [i, a] of args.entries()) recordValue(run, `arg${i}`, 'state', a)
    try {
      const result = fn(...args)
      if (run.enabled) recordValue(run, 'output', 'output', result)
      if (options.clock !== undefined && started !== undefined) {
        run.log(0, 'debug', `duration ${options.clock() - started} ms`)
      }
      run.end('completed', 1)
      return result
    } catch (error) {
      run.end('failed', 0)
      throw error
    }
  }
}

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
 * import { traceCall, recorder } from 'mdml-engine/trace'
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
