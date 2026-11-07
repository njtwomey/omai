/**
 * The trace event vocabulary and its schemas: a run starts with a `run` event (what ran, with which seed, version and
 * configuration) and ends with an `end` event, and between them records metrics, tensors (full or summarised),
 * structured messages, episodes, frames for animation, checkpoints and log lines. A `Run` is the unit that is exported,
 * replayed and visualised.
 */

import { z } from 'zod'
import { type Json, JsonSchema } from '../base/index.ts'

/** How a run ended. */
export type RunStatus = 'completed' | 'converged' | 'diverged' | 'stopped' | 'failed'

/** What a recorded tensor is: a parameter, a gradient, an algorithm state, an activation or an output. */
export type TensorRole = 'param' | 'grad' | 'state' | 'activation' | 'output'

// What the schema's enums read; each is checked against its type (every member is one, and `events.test.ts` checks
// that none is missing), so the type text stays the plain union a reader sees.
const RUN_STATUSES = ['completed', 'converged', 'diverged', 'stopped', 'failed'] as const satisfies readonly RunStatus[]
const TENSOR_ROLES = ['param', 'grad', 'state', 'activation', 'output'] as const satisfies readonly TensorRole[]

/** Summary statistics recorded instead of full values at `summary` detail. */
export interface TensorSummary {
  readonly n: number
  readonly mean: number
  readonly std: number
  readonly min: number
  readonly max: number
  readonly norm: number
  readonly zeros: number
  readonly histogram: { readonly edges: readonly number[]; readonly counts: readonly number[] }
}

/** The first event of every run: what ran, with which seed, version and configuration. */
export interface RunEvent {
  readonly type: 'run'
  readonly runId: string
  readonly name: string
  readonly seed: number | null
  readonly version: string
  readonly config: Json
  readonly parent?: string
  readonly channels?: Json
}

/** A scalar measurement at a step. */
export interface MetricEvent {
  readonly type: 'metric'
  readonly runId: string
  readonly step: number
  readonly name: string
  readonly value: number
}

/** A tensor at a step: full values (row-major) or a summary, with its role. */
export interface TensorEvent {
  readonly type: 'tensor'
  readonly runId: string
  readonly step: number
  readonly name: string
  readonly role: TensorRole
  readonly shape: readonly number[]
  /** Full values (row-major), or with `indices` an evenly spaced subsample of them at those flat positions. */
  readonly values?: readonly number[]
  readonly indices?: readonly number[]
  readonly summary?: TensorSummary
}

/** A structured message: a message passed in inference, an eliminated variable, a split chosen, a solver note. */
export interface MessageEvent {
  readonly type: 'message'
  readonly runId: string
  readonly step: number
  readonly kind: string
  readonly from?: string
  readonly to?: string
  readonly payload: Json
}

/** The summary of one finished episode of an agent in an environment. */
export interface EpisodeEvent {
  readonly type: 'episode'
  readonly runId: string
  readonly episode: number
  readonly return: number
  readonly length: number
  readonly terminated: boolean
}

/** A picture of the state at a step (a scene or structure as JSON), for animations. */
export interface FrameEvent {
  readonly type: 'frame'
  readonly runId: string
  readonly step: number
  readonly frame: Json
}

/** A serialised algorithm state from which the run can resume. */
export interface CheckpointEvent {
  readonly type: 'checkpoint'
  readonly runId: string
  readonly step: number
  readonly state: Json
}

/** A line of text (in place of console output). */
export interface LogEvent {
  readonly type: 'log'
  readonly runId: string
  readonly step: number
  readonly level: 'debug' | 'info' | 'warn'
  readonly text: string
}

/** The last event of every run. */
export interface EndEvent {
  readonly type: 'end'
  readonly runId: string
  readonly status: RunStatus
  readonly steps: number
  readonly summary: Readonly<Record<string, number>>
}

/** Any trace event. */
export type TraceEvent =
  | RunEvent
  | MetricEvent
  | TensorEvent
  | MessageEvent
  | EpisodeEvent
  | FrameEvent
  | CheckpointEvent
  | LogEvent
  | EndEvent

/**
 * A recorded run: a versioned list of events, the unit that is exported, replayed and visualised. A run may contain
 * several runs (children carry their parent's id).
 *
 * @concept tracing
 */
export interface Run {
  readonly schema: 'mdml.run/1'
  readonly events: readonly TraceEvent[]
}

const SummarySchema = z.object({
  n: z.number(),
  mean: z.number(),
  std: z.number(),
  min: z.number(),
  max: z.number(),
  norm: z.number(),
  zeros: z.number(),
  histogram: z.object({ edges: z.array(z.number()), counts: z.array(z.number()) }),
})

const step = z.number().int()
const id = z.string()

/**
 * Schema for one trace event, used to validate runs that cross a boundary (files, workers, network).
 *
 * @returns Parses one event.
 * @concept tracing
 * @example
 * ```ts
 * import { TraceEventSchema } from 'mdml-engine/trace'
 * TraceEventSchema.parse({ type: 'metric', runId: 'r', step: 0, name: 'loss', value: 1 }).type // => 'metric'
 * ```
 */
export const TraceEventSchema: z.ZodType<TraceEvent> = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('run'),
    runId: id,
    name: z.string(),
    seed: z.number().int().nullable(),
    version: z.string(),
    config: JsonSchema,
    parent: id.optional(),
    channels: JsonSchema.optional(),
  }),
  z.object({ type: z.literal('metric'), runId: id, step, name: z.string(), value: z.number() }),
  z.object({
    type: z.literal('tensor'),
    runId: id,
    step,
    name: z.string(),
    role: z.enum(TENSOR_ROLES),
    shape: z.array(z.number().int()),
    values: z.array(z.number()).optional(),
    indices: z.array(z.number().int()).optional(),
    summary: SummarySchema.optional(),
  }),
  z.object({
    type: z.literal('message'),
    runId: id,
    step,
    kind: z.string(),
    from: z.string().optional(),
    to: z.string().optional(),
    payload: JsonSchema,
  }),
  z.object({
    type: z.literal('episode'),
    runId: id,
    episode: z.number().int(),
    return: z.number(),
    length: z.number().int(),
    terminated: z.boolean(),
  }),
  z.object({ type: z.literal('frame'), runId: id, step, frame: JsonSchema }),
  z.object({ type: z.literal('checkpoint'), runId: id, step, state: JsonSchema }),
  z.object({ type: z.literal('log'), runId: id, step, level: z.enum(['debug', 'info', 'warn']), text: z.string() }),
  z.object({
    type: z.literal('end'),
    runId: id,
    status: z.enum(RUN_STATUSES),
    steps: z.number().int(),
    summary: z.record(z.string(), z.number()),
  }),
  // The cast is for `exactOptionalPropertyTypes` (zod reads `parent?: string | undefined`); `events.test.ts` is what
  // checks that this schema and the interfaces above name the same fields, kinds, statuses and roles.
]) as z.ZodType<TraceEvent>

/**
 * Schema for a whole {@link Run}.
 *
 * @returns Parses a run.
 * @example
 * ```ts
 * import { RunSchema } from 'mdml-engine/trace'
 * RunSchema.parse({ schema: 'mdml.run/1', events: [] }).events.length // => 0
 * ```
 */
export const RunSchema: z.ZodType<Run> = z.object({
  schema: z.literal('mdml.run/1'),
  events: z.array(TraceEventSchema),
})
