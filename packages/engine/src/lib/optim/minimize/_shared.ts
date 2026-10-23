// Shared machinery of the vector methods (not exported from the barrel): the common state, trace schema, observation
// and the runner that turns a final state into a MinimizeResult.
import type { Json } from '../../base/index.ts'
import {
  type Algorithm,
  channels,
  type MessageChannel,
  type MetricChannel,
  type Observation,
  runAlgorithm,
  type TensorChannel,
  type TraceSchema,
} from '../../trace/index.ts'
import { type FlatObjective, infNorm, type MinimizeResult, type VectorMethodOptions } from './objective.ts'

/** The state every vector method carries (extended by each method). */
export interface VectorState {
  readonly x: Float64Array
  readonly value: number
  /** The gradient at x (empty for derivative-free methods). */
  readonly grad: Float64Array
  readonly nfev: number
  readonly ngev: number
  readonly nhev: number
  /** The last accepted step length (or trust radius, or simplex size: what the method calls its step). */
  readonly step: number
  /** Set when the method has decided to stop. */
  readonly stop?: { readonly status: 'converged' | 'failed' | 'diverged'; readonly message: string }
}

/** The schema of a vector method: loss, gradient norm and step, the iterate and gradient, plus extra channels. */
export function vectorSchema(
  extra: {
    readonly metrics?: Readonly<Record<string, MetricChannel>>
    readonly tensors?: Readonly<Record<string, TensorChannel>>
    readonly messages?: Readonly<Record<string, MessageChannel>>
  } = {},
): TraceSchema {
  return channels({
    metrics: {
      loss: { description: 'objective value', direction: 'lower' },
      gradNorm: { description: 'infinity norm of the gradient' },
      step: { description: 'length of the last accepted step' },
      nfev: { description: 'objective evaluations so far' },
      ...extra.metrics,
    },
    tensors: {
      x: { role: 'param', description: 'current iterate' },
      grad: { role: 'grad', description: 'gradient at the iterate' },
      ...extra.tensors,
    },
    messages: extra.messages ?? {},
  })
}

/** The common observation of a vector state, with extra metrics, tensors and messages. */
export function observeVector(s: VectorState, extra: Observation = {}): Observation {
  const tensors: Record<string, { role: 'param' | 'grad' | 'state'; value: Float64Array }> = {
    x: { role: 'param', value: s.x },
  }
  if (s.grad.length > 0) tensors['grad'] = { role: 'grad', value: s.grad }
  return {
    metrics: {
      loss: s.value,
      ...(s.grad.length > 0 ? { gradNorm: infNorm(s.grad) } : {}),
      step: s.step,
      nfev: s.nfev,
      ...extra.metrics,
    },
    tensors: { ...tensors, ...extra.tensors },
    ...(extra.messages === undefined ? {} : { messages: extra.messages }),
  }
}

/** The `done` test shared by the vector methods: a decided stop, or a non-finite objective. */
export function vectorDone(s: VectorState): 'converged' | 'failed' | 'diverged' | undefined {
  if (s.stop !== undefined) return s.stop.status
  return Number.isFinite(s.value) ? undefined : 'diverged'
}

/** Run a vector method and shape its final state as a MinimizeResult. */
export function runVector<S extends VectorState>(
  algorithm: Algorithm<S>,
  objective: FlatObjective,
  options: VectorMethodOptions,
  defaultIterations: number,
): MinimizeResult {
  const { maxIterations, gradTol: _g, ...run } = options
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? defaultIterations })
  const s = result.state
  const hasGrad = s.grad.length > 0
  return {
    x: objective.unravel(s.x),
    fun: s.value,
    ...(hasGrad ? { grad: objective.unravel(s.grad) } : {}),
    gradNorm: hasGrad ? infNorm(s.grad) : Number.NaN,
    steps: result.steps,
    nfev: s.nfev,
    ngev: s.ngev,
    nhev: s.nhev,
    status: result.status,
    converged: result.status === 'converged',
    message:
      s.stop?.message ??
      (result.status === 'completed' ? 'maximum number of iterations reached' : `stopped: ${result.status}`),
  }
}

/** A JSON configuration from options, dropping functions, tracers and undefined values. */
export function configOf(options: object): Json {
  const out: Record<string, Json> = {}
  for (const [k, v] of Object.entries(options)) {
    if (v === undefined || typeof v === 'function' || k === 'tracer' || k === 'policy' || k === 'parent' || k === 'key')
      continue
    if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean' || v === null) out[k] = v
    else if (Array.isArray(v) && v.every((e) => typeof e === 'number')) out[k] = v as number[]
  }
  return out
}
