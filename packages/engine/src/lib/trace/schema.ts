/**
 * Trace schemas, the algorithm's half of the trace contract: the metric, tensor and message channels an algorithm can
 * emit (tensors with their role and shape), declared with `channels`, serialised into the run event, and checked
 * against recorded events by `validateAgainstSchema`.
 */

import type { Direction, Json } from '../base/index.ts'
import type { TensorRole, TraceEvent } from './events.ts'

/** A metric channel an algorithm can emit. */
export interface MetricChannel {
  readonly description: string
  readonly unit?: string
  readonly direction?: Direction
}

/**
 * A tensor channel an algorithm can emit, with its role and optionally its shape (numbers are fixed sizes, strings name
 * sizes that vary, such as `'n'` or `'batch'`).
 */
export interface TensorChannel {
  readonly role: TensorRole
  readonly description: string
  readonly shape?: readonly (number | string)[]
}

/** A message kind an algorithm can emit, optionally describing its payload fields. */
export interface MessageChannel {
  readonly description: string
  readonly payload?: Readonly<Record<string, string>>
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
 * import { channels } from 'mdml-engine/trace'
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
 * import { channels, schemaToJson } from 'mdml-engine/trace'
 * (schemaToJson(channels({})) as { episodes: boolean }).episodes // => false
 * ```
 */
export function schemaToJson(schema: TraceSchema): Json {
  return JSON.parse(JSON.stringify(schema)) as Json
}

function shapeMatches(declared: readonly (number | string)[], actual: readonly number[]): boolean {
  if (declared.length !== actual.length) return false
  const named = new Map<string, number>()
  for (let i = 0; i < declared.length; i++) {
    const d = declared[i]
    const a = actual[i] as number
    if (typeof d === 'number') {
      if (d !== a) return false
    } else if (d !== undefined) {
      // A named size must be the same wherever it appears in one shape.
      if ((named.get(d) ?? a) !== a) return false
      named.set(d, a)
    }
  }
  return true
}

/**
 * Check recorded events against a schema: every metric, tensor and message must be a declared channel, and tensors
 * must have their declared role and shape (named sizes match any size, but the same size wherever they repeat).
 *
 * @param events - The events.
 * @param schema - The declared channels.
 * @returns One message per problem (empty when the events conform).
 * @example
 * ```ts
 * import { channels, validateAgainstSchema } from 'mdml-engine/trace'
 * validateAgainstSchema([{ type: 'metric', runId: 'r', step: 0, name: 'x', value: 1 }], channels({})) // => ['undeclared metric x']
 * const s = channels({ tensors: { w: { role: 'param', description: 'weights', shape: ['n', 2] } } })
 * validateAgainstSchema([{ type: 'tensor', runId: 'r', step: 0, name: 'w', role: 'param', shape: [3, 3] }], s) // => ['tensor w has shape [3,3], declared [n,2]']
 * ```
 */
export function validateAgainstSchema(events: readonly TraceEvent[], schema: TraceSchema): string[] {
  const problems = new Set<string>()
  for (const e of events) {
    if (e.type === 'metric' && !Object.hasOwn(schema.metrics, e.name)) problems.add(`undeclared metric ${e.name}`)
    if (e.type === 'tensor') {
      const declared = schema.tensors[e.name]
      if (declared === undefined) problems.add(`undeclared tensor ${e.name}`)
      else {
        if (declared.role !== e.role) problems.add(`tensor ${e.name} has role ${e.role}, declared ${declared.role}`)
        if (declared.shape !== undefined && !shapeMatches(declared.shape, e.shape)) {
          problems.add(`tensor ${e.name} has shape [${e.shape}], declared [${declared.shape}]`)
        }
      }
    }
    if (e.type === 'message' && !Object.hasOwn(schema.messages, e.kind)) problems.add(`undeclared message ${e.kind}`)
    if (e.type === 'episode' && !schema.episodes) problems.add('undeclared episodes')
    if (e.type === 'frame' && !schema.frames) problems.add('undeclared frames')
  }
  return [...problems]
}
