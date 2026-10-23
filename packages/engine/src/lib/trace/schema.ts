import type { Direction, Json } from '../base/index.ts'
import type { TensorRole, TraceEvent } from './events.ts'

/** A metric channel an algorithm can emit. */
export interface MetricChannel {
  readonly description: string
  readonly unit?: string
  readonly direction?: Direction
}

/** A tensor channel an algorithm can emit. */
export interface TensorChannel {
  readonly role: TensorRole
  readonly description: string
}

/** A message kind an algorithm can emit. */
export interface MessageChannel {
  readonly description: string
}

/**
 * What an algorithm can emit: the contract between an algorithm and anything that records, plots or checks its trace.
 * A {@link TracePolicy} chooses among these channels.
 *
 * @concept tracing
 */
export interface TraceSchema {
  readonly metrics: Readonly<Record<string, MetricChannel>>
  readonly tensors: Readonly<Record<string, TensorChannel>>
  readonly messages: Readonly<Record<string, MessageChannel>>
  readonly episodes: boolean
  readonly frames: boolean
}

/**
 * Declare an algorithm's channels; omitted parts are empty.
 *
 * @param spec - Metrics, tensors, messages, and whether episodes and frames are emitted.
 * @returns The schema.
 * @concept tracing
 * @example
 * ```ts
 * import { channels } from 'whitebox-engine/trace'
 * const s = channels({ metrics: { loss: { description: 'training loss', direction: 'lower' } } })
 * Object.keys(s.metrics) // => ['loss']
 * ```
 */
export function channels(spec: Partial<TraceSchema>): TraceSchema {
  return {
    metrics: spec.metrics ?? {},
    tensors: spec.tensors ?? {},
    messages: spec.messages ?? {},
    episodes: spec.episodes ?? false,
    frames: spec.frames ?? false,
  }
}

/**
 * The schema as JSON, for the `channels` field of a run event.
 *
 * @param schema - The schema.
 * @returns A JSON description.
 * @example
 * ```ts
 * import { channels, schemaToJson } from 'whitebox-engine/trace'
 * (schemaToJson(channels({})) as { episodes: boolean }).episodes // => false
 * ```
 */
export function schemaToJson(schema: TraceSchema): Json {
  return JSON.parse(JSON.stringify(schema)) as Json
}

/**
 * Check recorded events against a schema: every metric, tensor and message must be a declared channel.
 *
 * @param events - The events.
 * @param schema - The declared channels.
 * @returns One message per undeclared channel (empty when the events conform).
 * @example
 * ```ts
 * import { channels, validateAgainstSchema } from 'whitebox-engine/trace'
 * validateAgainstSchema([{ type: 'metric', runId: 'r', step: 0, name: 'x', value: 1 }], channels({})) // => ['undeclared metric x']
 * ```
 */
export function validateAgainstSchema(events: readonly TraceEvent[], schema: TraceSchema): string[] {
  const problems = new Set<string>()
  for (const e of events) {
    if (e.type === 'metric' && !(e.name in schema.metrics)) problems.add(`undeclared metric ${e.name}`)
    if (e.type === 'tensor' && !(e.name in schema.tensors)) problems.add(`undeclared tensor ${e.name}`)
    if (e.type === 'message' && !(e.kind in schema.messages)) problems.add(`undeclared message ${e.kind}`)
    if (e.type === 'episode' && !schema.episodes) problems.add('undeclared episodes')
    if (e.type === 'frame' && !schema.frames) problems.add('undeclared frames')
  }
  return [...problems]
}
