/**
 * Tracers, where trace events go: `nullTracer` drops everything so algorithms run at full speed, `recorder` keeps
 * events in memory to export, replay or plot, and `tee` and `filtered` fan out or filter them. `startRun` opens a run
 * and returns the handle that applies its policy (selection, on-change filtering, detail, subsampling and budgets).
 */

import { type Json, VERSION } from '../base/index.ts'
import { isTensor, type NestedArray, tensor, type Value } from '../tensor/index.ts'
import type { Run, RunStatus, TensorRole, TraceEvent } from './events.ts'
import { changed, channelSelected, childPolicy, onSchedule, policies, type TracePolicy } from './policy.ts'
import { schemaToJson, type TraceSchema } from './schema.ts'
import { subsample, summarise } from './summaries.ts'

/**
 * Where trace events go. Algorithms emit through a {@link RunHandle}; they never log to the console.
 *
 * @concept tracing
 */
export interface Tracer {
  readonly enabled: boolean
  emit(event: TraceEvent): void
}

/**
 * A tracer that drops everything; with it, algorithms run at full speed.
 *
 * @returns The no-op tracer.
 * @example
 * ```ts
 * import { nullTracer } from 'omai-engine/trace'
 * nullTracer.enabled // => false
 * ```
 */
export const nullTracer: Tracer = { enabled: false, emit: () => {} }

/** A tracer that keeps every event in memory, and can be asked to let one go. */
export interface Recorder extends Tracer {
  readonly events: readonly TraceEvent[]
  run(): Run
  /** Drop one recorded event (the oldest of a path whose subscription keeps a window); an unknown one is ignored. */
  forget(event: TraceEvent): void
}

/**
 * A tracer that records events in memory, to export, replay or plot afterwards. `forget` drops an event it holds, by
 * identity, so the order of the rest is untouched.
 *
 * @returns The recorder.
 * @concept tracing
 * @index function A tracer that keeps events in memory to replay or plot.
 * @example
 * ```ts
 * import { recorder, startRun } from 'omai-engine/trace'
 * const rec = recorder()
 * const run = startRun(rec, { name: 'demo', seed: 0, config: {} })
 * run.metric(0, 'loss', 1.5)
 * run.end('completed', 1)
 * rec.events.length // => 3
 * const metric = rec.events.find((e) => e.type === 'metric')
 * if (metric !== undefined) rec.forget(metric)
 * rec.events.map((e) => e.type) // => ['run', 'end']
 * ```
 */
export function recorder(): Recorder {
  const events: TraceEvent[] = []
  return {
    enabled: true,
    events,
    emit: (event) => {
      events.push(event)
    },
    forget: (event) => {
      // The forgotten event is usually near the end (a path's window of the last few), so search from there.
      const i = events.lastIndexOf(event)
      if (i >= 0) events.splice(i, 1)
    },
    run: () => ({ schema: 'omai.run/1', events: [...events] }),
  }
}

/**
 * Forward each event to several tracers (for example a recorder and a live view).
 *
 * @param tracers - The destinations.
 * @returns The combined tracer.
 * @example
 * ```ts
 * import { tee, recorder } from 'omai-engine/trace'
 * const a = recorder()
 * const b = recorder()
 * tee(a, b).emit({ type: 'metric', runId: 'r', step: 0, name: 'x', value: 1 })
 * a.events.length + b.events.length // => 2
 * ```
 */
export function tee(...tracers: readonly Tracer[]): Tracer {
  return {
    enabled: tracers.some((t) => t.enabled),
    emit: (event) => {
      for (const t of tracers) if (t.enabled) t.emit(event)
    },
  }
}

/**
 * Pass on only the events a predicate accepts.
 *
 * @param tracer - The destination.
 * @param keep - The predicate.
 * @returns The filtering tracer.
 * @example
 * ```ts
 * import { filtered, recorder } from 'omai-engine/trace'
 * const rec = recorder()
 * filtered(rec, (e) => e.type !== 'log').emit({ type: 'log', runId: 'r', step: 0, level: 'info', text: 'hi' })
 * rec.events.length // => 0
 * ```
 */
export function filtered(tracer: Tracer, keep: (event: TraceEvent) => boolean): Tracer {
  return {
    enabled: tracer.enabled,
    emit: (event) => {
      if (keep(event)) tracer.emit(event)
    },
  }
}

/** Describes a run when it starts. */
export interface RunInfo {
  readonly name: string
  readonly seed: number | null
  readonly config: Json
  readonly runId?: string
  readonly parent?: string
  readonly schema?: TraceSchema
  readonly policy?: TracePolicy
}

/**
 * Emits the events of one run, applying its policy: channel selection, on-change filtering, detail and subsampling,
 * and the event and byte budgets. Scheduling (which steps are observed) is the runner's job, through `due`.
 */
export interface RunHandle {
  readonly runId: string
  readonly enabled: boolean
  readonly policy: TracePolicy
  /** Whether a step is on the policy's schedule. */
  due(step: number): boolean
  /** Whether a channel would be recorded (so callers can skip computing it). */
  wants(kind: 'metric' | 'message' | 'frame' | 'checkpoint', name?: string): boolean
  /** Whether tensors of a role and name would be recorded. */
  wantsTensor(name: string, role: TensorRole): boolean
  metric(step: number, name: string, value: number): void
  tensor(
    step: number,
    name: string,
    role: TensorRole,
    value: Value | ArrayLike<number>,
    shape?: readonly number[],
  ): void
  message(step: number, kind: string, payload: Json, route?: { readonly from?: string; readonly to?: string }): void
  episode(episode: number, ret: number, length: number, terminated: boolean): void
  frame(step: number, frame: Json): void
  checkpoint(step: number, state: Json): void
  log(step: number, level: 'debug' | 'info' | 'warn', text: string): void
  end(status: RunStatus, steps: number, summary?: Readonly<Record<string, number>>): void
  /** Start a child run (an inner algorithm), linked to this one. */
  child(info: Omit<RunInfo, 'parent' | 'policy'> & { readonly policy?: TracePolicy }): RunHandle
}

/**
 * Start a run on a tracer: emits the `run` event and returns a handle for the rest.
 *
 * Run ids are derived from the name and seed (children add their name and a counter), so repeated runs with the same
 * inputs produce identical traces: no clocks, no random ids.
 *
 * @param tracer - The destination.
 * @param info - Name, seed, configuration, schema, policy, and an explicit id or parent.
 * @returns The run handle.
 * @concept tracing
 * @example
 * ```ts
 * import { recorder, startRun } from 'omai-engine/trace'
 * startRun(recorder(), { name: 'sgd', seed: 3, config: { lr: 0.1 } }).runId // => 'sgd:3'
 * ```
 */
export function startRun(tracer: Tracer, info: RunInfo): RunHandle {
  const runId = info.runId ?? `${info.name}:${info.seed ?? 'none'}`
  const policy = info.policy ?? policies.minimal
  const enabled = tracer.enabled
  const budgetBytes = Number.isFinite(policy.maxBytes)
  const onChange = 'onChange' in policy.schedule ? policy.schedule : undefined
  const lastMetric = new Map<string, number>()
  const lastTensor = new Map<string, Float64Array>()
  let count = 0
  let bytes = 0
  let children = 0
  const emit = (event: TraceEvent, always = false): void => {
    if (!enabled) return
    if (!always && count >= policy.maxEvents) return
    if (budgetBytes) {
      const size = JSON.stringify(event).length
      if (!always && bytes + size > policy.maxBytes) return
      bytes += size
    }
    count += 1
    tracer.emit(event)
  }
  emit(
    {
      type: 'run',
      runId,
      name: info.name,
      seed: info.seed,
      version: VERSION,
      config: info.config,
      ...(info.parent === undefined ? {} : { parent: info.parent }),
      ...(info.schema === undefined ? {} : { channels: schemaToJson(info.schema) }),
    },
    true,
  )
  const handle: RunHandle = {
    runId,
    enabled,
    policy,
    due: (step) => enabled && onSchedule(policy.schedule, step),
    wants: (kind, name = '') => {
      if (!enabled) return false
      if (kind === 'metric' || kind === 'message') return channelSelected(policy, { kind, name })
      return channelSelected(policy, { kind })
    },
    wantsTensor: (name, role) => enabled && channelSelected(policy, { kind: 'tensor', name, role }),
    metric: (step, name, value) => {
      if (!handle.wants('metric', name)) return
      if (onChange !== undefined) {
        if (!changed(lastMetric.get(name), value, onChange.tolerance ?? 0)) return
        lastMetric.set(name, value)
      }
      emit({ type: 'metric', runId, step, name, value })
    },
    tensor: (step, name, role, value, shape) => {
      if (!handle.wantsTensor(name, role)) return
      let data: ArrayLike<number>
      let dims: readonly number[]
      if (typeof value === 'object' && value !== null && 'kind' in value) {
        if (!isTensor(value)) return
        data = value.data
        dims = value.shape
      } else if (Array.isArray(value) && value.some((v) => Array.isArray(v) || ArrayBuffer.isView(v))) {
        // A nested array (a matrix logged as rows): its row-major values and its own shape.
        const t = tensor(value as NestedArray)
        data = t.data
        dims = shape ?? t.shape
      } else {
        data = value as ArrayLike<number>
        dims = shape ?? [data.length]
      }
      if (onChange !== undefined) {
        if (!changed(lastTensor.get(name), data, onChange.tolerance ?? 0)) return
        lastTensor.set(name, Float64Array.from(data))
      }
      const full = policy.detail === 'full' && data.length <= policy.maxTensorSize
      const sample = !full && policy.detail === 'full' && policy.subsample > 0 ? subsample(data, policy.subsample) : {}
      emit({
        type: 'tensor',
        runId,
        step,
        name,
        role,
        shape: [...dims],
        ...(full ? { values: Array.from(data) } : { summary: summarise(data), ...sample }),
      })
    },
    message: (step, kind, payload, route = {}) => {
      if (handle.wants('message', kind)) emit({ type: 'message', runId, step, kind, payload, ...route })
    },
    episode: (episode, ret, length, terminated) => {
      emit({ type: 'episode', runId, episode, return: ret, length, terminated })
    },
    frame: (step, frame) => {
      if (handle.wants('frame')) emit({ type: 'frame', runId, step, frame })
    },
    checkpoint: (step, state) => {
      if (handle.wants('checkpoint')) emit({ type: 'checkpoint', runId, step, state })
    },
    log: (step, level, text) => {
      emit({ type: 'log', runId, step, level, text })
    },
    end: (status, steps, summary = {}) => {
      emit({ type: 'end', runId, status, steps, summary }, true)
    },
    child: (childInfo) => {
      children += 1
      const inner = childInfo.policy ?? childPolicy(policy)
      return startRun(inner === undefined ? nullTracer : tracer, {
        ...childInfo,
        runId: childInfo.runId ?? `${runId}/${childInfo.name}#${children}`,
        parent: runId,
        policy: inner ?? policy,
      })
    },
  }
  return handle
}
