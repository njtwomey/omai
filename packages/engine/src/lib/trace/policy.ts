/**
 * Trace policies, the caller's half of the trace contract: which channels to record (metrics, tensors by role,
 * messages, frames, checkpoints), how often (every n steps, log-spaced, first and last, on change) and at what detail.
 * Includes the `minimal`, `standard` and `debug` presets and `shouldRecord`, the single recording decision every run
 * makes.
 *
 * @index structure Trace policies: which channels to keep, and how often.
 */

import type { TensorRole } from './events.ts'

/** Which channels of one kind to record: all, none, or a list of names. */
export type ChannelSelection = boolean | readonly string[]

/**
 * How often to record: every n steps, at log-spaced steps (0, 1, 2, 4, 8, ...), only at the first and last step, or
 * whenever a value changes (by more than `tolerance`, default 0) since it was last recorded.
 */
export type RecordSchedule =
  | { readonly every: number }
  | { readonly logSpaced: true }
  | { readonly firstAndLast: true }
  | { readonly onChange: true; readonly tolerance?: number }

/**
 * What to record from a run: the caller's half of the trace contract (the algorithm's half is its
 * {@link TraceSchema}). Policies decide; algorithms never check whether they are being traced. Gradients are recorded
 * when `roles` includes `grad` (and off otherwise).
 *
 * @concept tracing
 */
export interface TracePolicy {
  readonly metrics: ChannelSelection
  readonly tensors: ChannelSelection
  /** Tensor roles to record (for example `['param', 'grad']`). */
  readonly roles: readonly TensorRole[]
  readonly messages: ChannelSelection
  readonly frames: boolean
  readonly checkpoints: boolean
  /** With `checkpoints`, the runner stores a checkpoint every this many steps (0: only when an algorithm asks). */
  readonly checkpointInterval: number
  readonly schedule: RecordSchedule
  /** Record full tensor values, or summaries (norms, moments, histograms). */
  readonly detail: 'summary' | 'full'
  /** Tensors with more elements than this are summarised even at full detail. */
  readonly maxTensorSize: number
  /** When a tensor is summarised at full detail, also keep this many evenly spaced values with their indices. */
  readonly subsample: number
  /** Stop recording (except the run and end events) after this many events. */
  readonly maxEvents: number
  /** Stop recording (except the run and end events) once the JSON size of the events passes this many bytes. */
  readonly maxBytes: number
  /** The policy of child runs (inner algorithms): another policy, `off` to leave them untraced, or the same policy. */
  readonly inner?: TracePolicy | 'off'
}

const base: TracePolicy = {
  metrics: true,
  tensors: false,
  roles: [],
  messages: false,
  frames: false,
  checkpoints: false,
  checkpointInterval: 0,
  schedule: { every: 1 },
  detail: 'summary',
  maxTensorSize: 4096,
  subsample: 0,
  maxEvents: 1_000_000,
  maxBytes: Number.POSITIVE_INFINITY,
}

/**
 * Preset policies: `minimal` (metrics every step), `standard` (metrics, messages, and parameter and gradient summaries
 * every 10 steps), `debug` (everything, full values, every step, checkpoints every 10 steps), `teaching` (metrics,
 * messages, frames and small full tensors every step, which is what interactive figures want).
 *
 * @returns The presets.
 * @concept tracing
 * @example
 * ```ts
 * import { policies } from 'omai-engine/trace'
 * policies.standard.roles // => ['param', 'grad']
 * ```
 */
export const policies: {
  readonly minimal: TracePolicy
  readonly standard: TracePolicy
  readonly debug: TracePolicy
  readonly teaching: TracePolicy
} = {
  minimal: base,
  standard: { ...base, tensors: true, roles: ['param', 'grad'], messages: true, schedule: { every: 10 } },
  debug: {
    ...base,
    tensors: true,
    roles: ['param', 'grad', 'state', 'activation', 'output'],
    messages: true,
    frames: true,
    checkpoints: true,
    checkpointInterval: 10,
    detail: 'full',
    maxTensorSize: 1_000_000,
  },
  teaching: {
    ...base,
    tensors: true,
    roles: ['param', 'grad', 'state', 'output'],
    messages: true,
    frames: true,
    detail: 'full',
    maxTensorSize: 1024,
    subsample: 256,
  },
}

/**
 * Build a policy from a preset (default `minimal`) and overrides.
 *
 * @param overrides - Fields to change.
 * @param preset - The starting preset.
 * @returns The policy.
 * @example
 * ```ts
 * import { tracePolicy } from 'omai-engine/trace'
 * tracePolicy({ roles: ['grad'], tensors: true }).roles // => ['grad']
 * ```
 */
export function tracePolicy(
  overrides: Partial<TracePolicy> = {},
  preset: keyof typeof policies = 'minimal',
): TracePolicy {
  return { ...policies[preset], ...overrides }
}

/**
 * Whether a step is on the schedule (step 0 always is, and every step is for `onChange`, where the values decide; the
 * last step is recorded by the runner).
 *
 * @param schedule - The schedule.
 * @param step - The step.
 * @returns `true` when the step should be recorded.
 * @example
 * ```ts
 * import { onSchedule } from 'omai-engine/trace'
 * [0, 1, 2, 3, 4, 5, 8].filter((s) => onSchedule({ logSpaced: true }, s)) // => [0, 1, 2, 4, 8]
 * ```
 */
export function onSchedule(schedule: RecordSchedule, step: number): boolean {
  if (step === 0) return true
  if ('every' in schedule) return step % Math.max(1, schedule.every) === 0
  if ('logSpaced' in schedule) return (step & (step - 1)) === 0
  if ('onChange' in schedule) return true
  return false
}

/**
 * Whether a channel is selected.
 *
 * @param selection - The selection.
 * @param name - The channel name.
 * @returns `true` when selected.
 * @example
 * ```ts
 * import { selected } from 'omai-engine/trace'
 * selected(['loss'], 'loss') // => true
 * ```
 */
export function selected(selection: ChannelSelection, name: string): boolean {
  return selection === true || (Array.isArray(selection) && selection.includes(name))
}

/** A channel of a run, as a policy sees it. */
export type ChannelRef =
  | { readonly kind: 'metric' | 'message'; readonly name: string }
  | { readonly kind: 'tensor'; readonly name: string; readonly role: TensorRole }
  | { readonly kind: 'frame' | 'checkpoint' }

/**
 * Whether a policy selects a channel at all, whatever the step: the channel lists for metrics, tensors (with their
 * role) and messages, and the frame and checkpoint switches.
 *
 * @param policy - The policy.
 * @param channel - The channel.
 * @returns `true` when the channel is selected.
 * @example
 * ```ts
 * import { channelSelected, policies } from 'omai-engine/trace'
 * channelSelected(policies.standard, { kind: 'tensor', name: 'w', role: 'grad' }) // => true
 * channelSelected(policies.minimal, { kind: 'frame' }) // => false
 * ```
 */
export function channelSelected(policy: TracePolicy, channel: ChannelRef): boolean {
  switch (channel.kind) {
    case 'metric':
      return selected(policy.metrics, channel.name)
    case 'message':
      return selected(policy.messages, channel.name)
    case 'tensor':
      return selected(policy.tensors, channel.name) && policy.roles.includes(channel.role)
    case 'frame':
      return policy.frames
    case 'checkpoint':
      return policy.checkpoints
  }
}

/**
 * Whether a value differs from the one last recorded by more than a tolerance (a missing previous value always
 * differs; arrays differ when their lengths or any element do).
 *
 * @param previous - The last recorded value, if any.
 * @param current - The new value.
 * @param tolerance - The largest difference that still counts as unchanged (default 0).
 * @returns `true` when the value changed.
 * @example
 * ```ts
 * import { changed } from 'omai-engine/trace'
 * changed(1, 1.05, 0.1) // => false
 * changed([1, 2], [1, 3]) // => true
 * ```
 */
export function changed(
  previous: number | ArrayLike<number> | undefined,
  current: number | ArrayLike<number>,
  tolerance = 0,
): boolean {
  if (previous === undefined) return true
  const a = typeof previous === 'number' ? [previous] : previous
  const b = typeof current === 'number' ? [current] : current
  if (a.length !== b.length) return true
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number
    const y = b[i] as number
    if (Number.isNaN(x) !== Number.isNaN(y) || Math.abs(x - y) > tolerance) return true
  }
  return false
}

/**
 * The single recording decision of the trace contract: is this channel recorded at this step? It combines channel
 * selection ({@link channelSelected}), the schedule ({@link onSchedule}, or the checkpoint interval for checkpoints),
 * the last step (always recorded) and, for an `onChange` schedule, whether the value changed since it was last
 * recorded. The run handle and the runner apply exactly this rule.
 *
 * @param policy - The policy.
 * @param channel - The channel.
 * @param step - The step.
 * @param context - Whether this is the last step, and the previous and current values for `onChange` schedules.
 * @returns `true` when the event should be recorded.
 * @concept tracing
 * @example
 * ```ts
 * import { shouldRecord, tracePolicy } from 'omai-engine/trace'
 * const every5 = tracePolicy({ schedule: { every: 5 } })
 * [0, 3, 5, 7, 10].filter((t) => shouldRecord(every5, { kind: 'metric', name: 'loss' }, t)) // => [0, 5, 10]
 * shouldRecord(every5, { kind: 'metric', name: 'loss' }, 7, { last: true }) // => true
 * const changes = tracePolicy({ schedule: { onChange: true } })
 * shouldRecord(changes, { kind: 'metric', name: 'x' }, 3, { previous: 2, current: 2 }) // => false
 * ```
 */
export function shouldRecord(
  policy: TracePolicy,
  channel: ChannelRef,
  step: number,
  context: {
    readonly last?: boolean
    readonly previous?: number | ArrayLike<number> | undefined
    readonly current?: number | ArrayLike<number>
  } = {},
): boolean {
  if (!channelSelected(policy, channel)) return false
  if (channel.kind === 'checkpoint') {
    return policy.checkpointInterval > 0 && step % policy.checkpointInterval === 0
  }
  if (context.last !== true && !onSchedule(policy.schedule, step)) return false
  const s = policy.schedule
  if ('onChange' in s && context.current !== undefined) {
    return changed(context.previous, context.current, s.tolerance ?? 0)
  }
  return true
}

/**
 * The policy a child run gets from its parent's: the parent's `inner` policy when it has one (`off` means the child is
 * not traced, returned here as `undefined`), else the parent's own policy. `subRun` and `RunHandle.child` apply this
 * rule unless a policy is given explicitly, so a caller can trace an outer loop in detail and its inner solvers
 * lightly (or not at all) without touching the algorithm code.
 *
 * @param policy - The parent's policy.
 * @returns The child's policy, or `undefined` for an untraced child.
 * @example
 * ```ts
 * import { childPolicy, policies, tracePolicy } from 'omai-engine/trace'
 * childPolicy(tracePolicy({ inner: policies.minimal }, 'debug')) === policies.minimal // => true
 * childPolicy(tracePolicy({ inner: 'off' })) // => undefined
 * ```
 */
export function childPolicy(policy: TracePolicy): TracePolicy | undefined {
  if (policy.inner === 'off') return undefined
  return policy.inner ?? policy
}
