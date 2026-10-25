/**
 * Reading recorded traces: metric and tensor series over steps for plotting, messages, episodes and animation frames,
 * the tree of runs by parent link, per-run summaries, and `diffRuns` to compare the configuration, seeds and results of
 * two runs.
 */

import type { Json } from '../base/index.ts'
import type {
  EndEvent,
  EpisodeEvent,
  FrameEvent,
  MessageEvent,
  Run,
  RunEvent,
  TensorEvent,
  TraceEvent,
} from './events.ts'

/** A time series of one metric. */
export interface Series {
  readonly steps: number[]
  readonly values: number[]
}

function inRun(e: TraceEvent, runId: string | undefined): boolean {
  return runId === undefined || e.runId === runId
}

/**
 * The values of one metric over steps, for plotting.
 *
 * @param run - The trace.
 * @param name - The metric.
 * @param runId - Restrict to one run when the trace holds several.
 * @returns Parallel arrays of steps and values.
 * @example
 * ```ts
 * import { metricSeries } from 'whitebox-engine/trace'
 * const run = { schema: 'whitebox.run/1' as const, events: [
 *   { type: 'metric' as const, runId: 'r', step: 0, name: 'loss', value: 2 },
 *   { type: 'metric' as const, runId: 'r', step: 1, name: 'loss', value: 1 },
 * ] }
 * metricSeries(run, 'loss') // => { steps: [0, 1], values: [2, 1] }
 * ```
 */
export function metricSeries(run: Run, name: string, runId?: string): Series {
  const steps: number[] = []
  const values: number[] = []
  for (const e of run.events) {
    if (e.type === 'metric' && e.name === name && inRun(e, runId)) {
      steps.push(e.step)
      values.push(e.value)
    }
  }
  return { steps, values }
}

/**
 * The recorded values of one tensor channel over steps.
 *
 * @param run - The trace.
 * @param name - The tensor channel.
 * @param runId - Restrict to one run.
 * @returns The tensor events in order.
 * @example
 * ```ts
 * import { tensorSeries } from 'whitebox-engine/trace'
 * tensorSeries({ schema: 'whitebox.run/1', events: [] }, 'w') // => []
 * ```
 */
export function tensorSeries(run: Run, name: string, runId?: string): TensorEvent[] {
  return run.events.filter((e): e is TensorEvent => e.type === 'tensor' && e.name === name && inRun(e, runId))
}

/**
 * The messages of a trace, optionally of one kind.
 *
 * @param run - The trace.
 * @param kind - The message kind.
 * @returns The message events in order.
 * @example
 * ```ts
 * import { messages } from 'whitebox-engine/trace'
 * messages({ schema: 'whitebox.run/1', events: [] }) // => []
 * ```
 */
export function messages(run: Run, kind?: string): MessageEvent[] {
  return run.events.filter((e): e is MessageEvent => e.type === 'message' && (kind === undefined || e.kind === kind))
}

/**
 * The episodes of a trace.
 *
 * @param run - The trace.
 * @param runId - Restrict to one run.
 * @returns The episode events in order.
 * @example
 * ```ts
 * import { episodes } from 'whitebox-engine/trace'
 * episodes({ schema: 'whitebox.run/1', events: [] }) // => []
 * ```
 */
export function episodes(run: Run, runId?: string): EpisodeEvent[] {
  return run.events.filter((e): e is EpisodeEvent => e.type === 'episode' && inRun(e, runId))
}

/**
 * The frames of a trace, as `{ step, frame }` pairs for building animations.
 *
 * @param run - The trace.
 * @param runId - Restrict to one run.
 * @returns The frames in order.
 * @example
 * ```ts
 * import { frames } from 'whitebox-engine/trace'
 * frames({ schema: 'whitebox.run/1', events: [] }) // => []
 * ```
 */
export function frames(run: Run, runId?: string): { readonly step: number; readonly frame: Json }[] {
  return run.events
    .filter((e): e is FrameEvent => e.type === 'frame' && inRun(e, runId))
    .map((e) => ({ step: e.step, frame: e.frame }))
}

/** A node of a run tree. */
export interface RunNode {
  readonly run: RunEvent
  readonly end: EndEvent | undefined
  readonly children: readonly RunNode[]
}

/**
 * The runs of a trace arranged as a tree by their parent links (a study with trials, an outer loop with inner solves).
 *
 * @param run - The trace.
 * @returns The root runs with their children.
 * @example
 * ```ts
 * import { runTree } from 'whitebox-engine/trace'
 * runTree({ schema: 'whitebox.run/1', events: [] }) // => []
 * ```
 */
export function runTree(run: Run): RunNode[] {
  const runs = run.events.filter((e): e is RunEvent => e.type === 'run')
  const ends = new Map(run.events.filter((e): e is EndEvent => e.type === 'end').map((e) => [e.runId, e]))
  const build = (r: RunEvent): RunNode => ({
    run: r,
    end: ends.get(r.runId),
    children: runs.filter((c) => c.parent === r.runId).map(build),
  })
  return runs.filter((r) => r.parent === undefined || !runs.some((p) => p.runId === r.parent)).map(build)
}

/**
 * The final status, step count and summary metrics of each run in a trace.
 *
 * @param run - The trace.
 * @returns One summary per run, keyed by run id.
 * @example
 * ```ts
 * import { summary } from 'whitebox-engine/trace'
 * summary({ schema: 'whitebox.run/1', events: [{ type: 'end', runId: 'r', status: 'converged', steps: 3, summary: { loss: 0 } }] }).r.steps // => 3
 * ```
 */
export function summary(run: Run): Record<string, EndEvent> {
  const out: Record<string, EndEvent> = {}
  for (const e of run.events) if (e.type === 'end') out[e.runId] = e
  return out
}

/**
 * Compare two runs: configuration differences, seeds, versions, statuses and final metrics.
 *
 * @param a - The first trace (its first run is compared).
 * @param b - The second trace.
 * @returns The differing fields.
 * @example
 * ```ts
 * import { diffRuns } from 'whitebox-engine/trace'
 * const r = (lr: number) => ({ schema: 'whitebox.run/1' as const, events: [{ type: 'run' as const, runId: 'x', name: 'sgd', seed: 0, version: '1', config: { lr } }] })
 * diffRuns(r(0.1), r(0.2)).config // => { lr: [0.1, 0.2] }
 * ```
 */
export function diffRuns(
  a: Run,
  b: Run,
): {
  readonly config: Record<string, [Json | undefined, Json | undefined]>
  readonly seed?: [number | null, number | null]
  readonly version?: [string, string]
  readonly status?: [string | undefined, string | undefined]
  readonly metrics: Record<string, [number | undefined, number | undefined]>
} {
  const ra = a.events.find((e): e is RunEvent => e.type === 'run')
  const rb = b.events.find((e): e is RunEvent => e.type === 'run')
  const ea = a.events.find((e): e is EndEvent => e.type === 'end' && e.runId === ra?.runId)
  const eb = b.events.find((e): e is EndEvent => e.type === 'end' && e.runId === rb?.runId)
  const ca = (ra?.config ?? {}) as Record<string, Json>
  const cb = (rb?.config ?? {}) as Record<string, Json>
  const config: Record<string, [Json | undefined, Json | undefined]> = {}
  for (const k of new Set([...Object.keys(ca), ...Object.keys(cb)])) {
    if (JSON.stringify(ca[k]) !== JSON.stringify(cb[k])) config[k] = [ca[k], cb[k]]
  }
  const metrics: Record<string, [number | undefined, number | undefined]> = {}
  for (const k of new Set([...Object.keys(ea?.summary ?? {}), ...Object.keys(eb?.summary ?? {})])) {
    const x = ea?.summary[k]
    const y = eb?.summary[k]
    if (x !== y) metrics[k] = [x, y]
  }
  return {
    config,
    ...(ra?.seed === rb?.seed ? {} : { seed: [ra?.seed ?? null, rb?.seed ?? null] as [number | null, number | null] }),
    ...(ra?.version === rb?.version ? {} : { version: [ra?.version ?? '', rb?.version ?? ''] as [string, string] }),
    ...(ea?.status === eb?.status
      ? {}
      : { status: [ea?.status, eb?.status] as [string | undefined, string | undefined] }),
    metrics,
  }
}
