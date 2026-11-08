/**
 * Moving runs across boundaries and back: `exportRun` and `parseRun` serialise and validate runs as JSON, `replay`
 * feeds a recorded run's events to handlers in order to rebuild a live view, and `mergeRuns` concatenates runs such as
 * several seeds of one experiment.
 */

import { OmaiError } from '../base/index.ts'
import { type Run, RunSchema, type TraceEvent } from './events.ts'

/**
 * Serialise a run to JSON text for files, workers or the network.
 *
 * @param run - The run.
 * @returns JSON text.
 * @example
 * ```ts
 * import { exportRun } from 'omai-engine/trace'
 * exportRun({ schema: 'omai.run/1', events: [] }) // => '{"schema":"omai.run/1","events":[]}'
 * ```
 */
export function exportRun(run: Run): string {
  return JSON.stringify(run)
}

/**
 * Parse and validate a run from JSON text or an unknown value.
 *
 * @param input - JSON text or a parsed value.
 * @returns The validated run.
 * @throws OmaiError with code `schema` when the input is not a valid run.
 * @example
 * ```ts
 * import { parseRun } from 'omai-engine/trace'
 * parseRun('{"schema":"omai.run/1","events":[]}').events.length // => 0
 * ```
 */
export function parseRun(input: unknown): Run {
  const value = typeof input === 'string' ? (JSON.parse(input) as unknown) : input
  const parsed = RunSchema.safeParse(value)
  if (!parsed.success) throw new OmaiError('schema', `invalid run: ${parsed.error.message}`)
  return parsed.data
}

/** Per-type callbacks for {@link replay}. */
export type ReplayHandlers = {
  readonly [T in TraceEvent['type']]?: (event: Extract<TraceEvent, { type: T }>) => void
}

/**
 * Feed a recorded run's events, in order, to handlers (to rebuild a live view step by step).
 *
 * @param run - The run.
 * @param handlers - A callback per event type.
 * @returns Nothing.
 * @example
 * ```ts
 * import { replay } from 'omai-engine/trace'
 * let n = 0
 * replay({ schema: 'omai.run/1', events: [{ type: 'metric', runId: 'r', step: 0, name: 'x', value: 1 }] }, {
 *   metric: () => { n++ },
 * })
 * n // => 1
 * ```
 */
export function replay(run: Run, handlers: ReplayHandlers): void {
  for (const event of run.events) {
    const handler = handlers[event.type] as ((e: TraceEvent) => void) | undefined
    handler?.(event)
  }
}

/**
 * Concatenate runs (for example several seeds of one experiment) into one trace.
 *
 * @param runs - The runs.
 * @returns One run holding all events in order.
 * @example
 * ```ts
 * import { mergeRuns } from 'omai-engine/trace'
 * mergeRuns([{ schema: 'omai.run/1', events: [] }]).events.length // => 0
 * ```
 */
export function mergeRuns(runs: readonly Run[]): Run {
  return { schema: 'omai.run/1', events: runs.flatMap((r) => r.events) }
}

/**
 * Whether a value is a valid {@link Run}; the example runner uses this for `// => trace`.
 *
 * @param value - Any value.
 * @returns `true` for valid runs.
 * @example
 * ```ts
 * import { isRun } from 'omai-engine/trace'
 * isRun({ schema: 'omai.run/1', events: [] }) // => true
 * ```
 */
export function isRun(value: unknown): value is Run {
  return RunSchema.safeParse(value).success
}
