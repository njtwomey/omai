import type { TensorRole } from './events.ts'

/** Which channels of one kind to record: all, none, or a list of names. */
export type Selection = boolean | readonly string[]

/** How often to record: every n steps, at log-spaced steps, or only at the first and last step. */
export type Schedule = { readonly every: number } | { readonly logSpaced: true } | { readonly firstAndLast: true }

/**
 * What to record from a run: the caller's half of the trace contract (the algorithm's half is its
 * {@link TraceSchema}). Policies decide; algorithms never check whether they are being traced.
 *
 * @concept tracing
 */
export interface TracePolicy {
  readonly metrics: Selection
  readonly tensors: Selection
  /** Tensor roles to record (for example `['param', 'grad']`). */
  readonly roles: readonly TensorRole[]
  readonly messages: Selection
  readonly frames: boolean
  readonly checkpoints: boolean
  readonly schedule: Schedule
  /** Record full tensor values, or summaries (norms, moments, histograms). */
  readonly detail: 'summary' | 'full'
  /** Tensors with more elements than this are summarised even at full detail. */
  readonly maxTensorSize: number
  /** Stop recording (except the end event) after this many events. */
  readonly maxEvents: number
}

const base: TracePolicy = {
  metrics: true,
  tensors: false,
  roles: [],
  messages: false,
  frames: false,
  checkpoints: false,
  schedule: { every: 1 },
  detail: 'summary',
  maxTensorSize: 4096,
  maxEvents: 1_000_000,
}

/**
 * Preset policies: `minimal` (metrics every step), `standard` (metrics, messages, and parameter and gradient summaries
 * every 10 steps), `debug` (everything, full values, every step), `teaching` (metrics, messages, frames and small full
 * tensors every step, which is what interactive figures want).
 *
 * @returns The presets.
 * @concept tracing
 * @example
 * ```ts
 * import { policies } from 'whitebox-engine/trace'
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
 * import { tracePolicy } from 'whitebox-engine/trace'
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
 * Whether a step is on the schedule (step 0 always is; the last step is recorded by the runner).
 *
 * @param schedule - The schedule.
 * @param step - The step.
 * @returns `true` when the step should be recorded.
 * @example
 * ```ts
 * import { onSchedule } from 'whitebox-engine/trace'
 * [0, 1, 2, 3, 4, 5, 8].filter((s) => onSchedule({ logSpaced: true }, s)) // => [0, 1, 2, 4, 8]
 * ```
 */
export function onSchedule(schedule: Schedule, step: number): boolean {
  if (step === 0) return true
  if ('every' in schedule) return step % Math.max(1, schedule.every) === 0
  if ('logSpaced' in schedule) return (step & (step - 1)) === 0
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
 * import { selected } from 'whitebox-engine/trace'
 * selected(['loss'], 'loss') // => true
 * ```
 */
export function selected(selection: Selection, name: string): boolean {
  return selection === true || (Array.isArray(selection) && selection.includes(name))
}
