/**
 * The wire protocol between a page and the worker that computes its sessions: the requests a page sends (make the
 * sessions of a method by name, compute ahead, pause, follow the cursor, probe a derived path, dispose), the events the
 * worker answers with (the tape as it grows, one delta per variant; probe results; the producer going idle; errors),
 * and what of a session's options can cross (JSON: the configuration, seed, budget and subscription, with hooks by
 * name). Plain data both ways, so a worker and a test double speak the same language.
 */

import { type Json, OmaiError } from 'omai-engine/base'
import type { Config, RunStatus, SessionOptions, Subscription, TraceEvent } from 'omai-engine/trace'

/** The part of {@link SessionOptions} that crosses to a worker: all of it JSON. */
export interface WireOptions {
  readonly config?: Json
  readonly seed?: number
  readonly steps?: number
  readonly checkpointEvery?: number
  readonly past?: boolean
  /** The subscription: `Log` specs, and hooks by name (`On('name', options)`); a hook given as a function cannot cross. */
  readonly logs?: Json
}

/** One variant of a comparison on the wire: a label and the knob values that set it apart. */
export interface WireVariant {
  readonly label: string
  readonly config: Json
}

/** What a page sends a worker. Every request names the producer it is for (`id`), so one worker may host several. */
export type ProducerRequest =
  /** Make the sessions of the method called `method` in the worker's registry, one per variant, standing at step 0. */
  | {
      readonly type: 'init'
      readonly id: string
      readonly method: string
      readonly options: WireOptions
      readonly variants: readonly WireVariant[]
      /** Ship checkpoint events too (default no: the page cannot use them, and they are the heavy ones). */
      readonly checkpoints: boolean
      /** How many steps the producer may run ahead of the cursor (`null`: to the end of the run). */
      readonly lookahead: number | null
    }
  /** Compute ahead until the end of the run or the lookahead; `eager` ignores the lookahead. */
  | { readonly type: 'run'; readonly id: string; readonly eager: boolean }
  /** Compute exactly this many more steps (every variant in step), then idle. */
  | { readonly type: 'advance'; readonly id: string; readonly steps: number }
  /** Stop computing (the tape stays). */
  | { readonly type: 'pause'; readonly id: string }
  /** The step the consumer is viewing, so the producer knows how far ahead it is. */
  | { readonly type: 'cursor'; readonly id: string; readonly step: number }
  /**
   * A derived path of a variant at a step; answered by `probed` with the same `req`. A `latest` probe is one a view
   * wants only while it is the newest of its path (the page moved on to another step): when a newer one for the same
   * variant, path and arguments is waiting, the older is answered `superseded` without being computed.
   */
  | {
      readonly type: 'probe'
      readonly id: string
      readonly req: number
      readonly latest: boolean
      readonly variant: number
      readonly t: number
      readonly path: string
      readonly args?: Json
    }
  /** Drop the sessions of this producer. */
  | { readonly type: 'dispose'; readonly id: string }

/** What the tape of one variant gained since the last delta: events appended after `truncate` events were kept. */
export interface TapeDelta {
  readonly variant: number
  /** How many of the events already sent are still on the tape (fewer than all when a window forgot some). */
  readonly truncate: number
  readonly events: readonly TraceEvent[]
  /** The step reached and, once the run has ended, its status. */
  readonly t: number
  readonly status: RunStatus | null
  /** Event paths of the contract that fired since the last delta, with the steps at which they did. */
  readonly fired: Readonly<Record<string, readonly number[]>>
}

/** Why a producer stopped computing without ending its run. */
export type IdleReason = 'paused' | 'lookahead' | 'advanced' | 'ended' | 'failed'

/** What a worker sends a page. `seq` increases by one with every event of a producer, so order is checkable. */
export type ProducerEvent =
  /** The sessions are made: what each resolved to (the configuration with defaults, the seed) and its paths' kinds. */
  | {
      readonly type: 'ready'
      readonly id: string
      readonly seq: number
      readonly variants: readonly {
        readonly config: Json
        readonly seed: number
        readonly kinds: Readonly<Record<string, string>>
      }[]
    }
  /** The tape grew: one delta per variant that changed, and the milliseconds the last slice took. */
  | {
      readonly type: 'tape'
      readonly id: string
      readonly seq: number
      readonly deltas: readonly TapeDelta[]
      readonly sliceMs: number
    }
  /** The producer stopped computing, after its last `tape`. */
  | { readonly type: 'idle'; readonly id: string; readonly seq: number; readonly reason: IdleReason }
  /** The answer to a probe: the value, or the message of the error it threw. */
  | {
      readonly type: 'probed'
      readonly id: string
      readonly seq: number
      readonly req: number
      readonly ok: boolean
      readonly value?: unknown
      readonly error?: string
      /** The probe was not computed: a newer one for the same path was waiting. */
      readonly superseded?: boolean
    }
  /** Something failed: making the sessions (unknown method, a configuration outside the space) or computing them. */
  | { readonly type: 'error'; readonly id: string; readonly seq: number; readonly message: string }

/** A handler typed as a method, so a real `Worker`'s `onmessage`/`onerror` (typed on `MessageEvent`/`ErrorEvent`) fits. */
type Handler<E> = { bivarianceHack(event: E): void }['bivarianceHack']

/** The two ends of a producer's wire, as the page sees the worker: the part of a `Worker` it uses. */
export interface WorkerLike {
  postMessage(message: ProducerRequest): void
  terminate(): void
  onmessage: Handler<{ readonly data: unknown }> | null
  onerror: Handler<{ readonly message?: string }> | null
}

/** The worker's side: the part of a dedicated worker's global scope the server uses. */
export interface ScopeLike {
  postMessage(message: ProducerEvent): void
  onmessage: Handler<{ readonly data: unknown }> | null
}

const REQUESTS = new Set(['init', 'run', 'advance', 'pause', 'cursor', 'probe', 'dispose'])
const EVENTS = new Set(['ready', 'tape', 'idle', 'probed', 'error'])

function typed(x: unknown, kinds: ReadonlySet<string>): boolean {
  if (typeof x !== 'object' || x === null) return false
  const o = x as { readonly type?: unknown; readonly id?: unknown }
  return typeof o.type === 'string' && kinds.has(o.type) && typeof o.id === 'string'
}

/**
 * Whether a received value is a request of the protocol.
 *
 * @param x - Anything a worker received.
 * @returns Whether it is a {@link ProducerRequest}.
 */
export function isProducerRequest(x: unknown): x is ProducerRequest {
  return typed(x, REQUESTS)
}

/**
 * Whether a received value is an event of the protocol.
 *
 * @param x - Anything a page received from a worker.
 * @returns Whether it is a {@link ProducerEvent}.
 */
export function isProducerEvent(x: unknown): x is ProducerEvent {
  return typed(x, EVENTS)
}

// The first function found in a value, as a path for the message that names it; `undefined` when it is all data.
function functionIn(x: unknown, path: string, seen: Set<unknown>): string | undefined {
  if (typeof x === 'function') return path
  if (typeof x !== 'object' || x === null || seen.has(x)) return undefined
  seen.add(x)
  for (const [k, v] of Object.entries(x)) {
    const found = functionIn(v, `${path}/${k}`, seen)
    if (found !== undefined) return found
  }
  return undefined
}

/**
 * What of a session's options crosses to a worker: the configuration, seed, budget and subscription, with every hook
 * by name (`On('name', options)`, resolved against the method's `hooks` in the worker). A hook given as a function,
 * a start state or a ready-made key cannot cross.
 *
 * @param options - The options a page gives `useSession`.
 * @returns The wire options.
 * @throws OmaiError with code `value` naming what cannot cross.
 */
export function wireOptions<S, C extends Config>(options: SessionOptions<S, C>): WireOptions {
  if (options.key !== undefined)
    throw new OmaiError('value', 'a worker session is made from a seed: a ready-made key cannot cross')
  if (options.start !== undefined) throw new OmaiError('value', 'a worker session cannot start from a given state')
  const logs = options.logs as Subscription | undefined
  if (logs !== undefined && typeof logs === 'object') {
    for (const [path, spec] of Object.entries(logs)) {
      if (spec.kind === 'hook' && spec.hook !== undefined)
        throw new OmaiError(
          'value',
          `the hook on '${path}' is a function and cannot cross to a worker: export it in Method.hooks and use On('name', options)`,
        )
    }
  }
  const found = functionIn(options.config, 'config', new Set())
  if (found !== undefined) throw new OmaiError('value', `${found} is a function and cannot cross to a worker`)
  return {
    ...(options.config === undefined ? {} : { config: options.config as Json }),
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.steps === undefined ? {} : { steps: options.steps }),
    ...(options.checkpointEvery === undefined ? {} : { checkpointEvery: options.checkpointEvery }),
    ...(options.past === undefined ? {} : { past: options.past }),
    ...(logs === undefined ? {} : { logs: logs as unknown as Json }),
  }
}

/**
 * The session options a worker makes sessions with, from the wire.
 *
 * @param wire - The options as they crossed.
 * @returns Options for `session`, in live mode (the producer decides what to compute and when).
 */
export function sessionOptionsOf(wire: WireOptions): SessionOptions<never, never> {
  return {
    mode: 'live',
    ...(wire.config === undefined ? {} : { config: wire.config as never }),
    ...(wire.seed === undefined ? {} : { seed: wire.seed }),
    ...(wire.steps === undefined ? {} : { steps: wire.steps }),
    ...(wire.checkpointEvery === undefined ? {} : { checkpointEvery: wire.checkpointEvery }),
    ...(wire.past === undefined ? {} : { past: wire.past }),
    ...(wire.logs === undefined ? {} : { logs: wire.logs as never }),
  }
}
