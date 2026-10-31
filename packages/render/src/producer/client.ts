/**
 * The page's side of a worker producer: `workerProducer` makes a `ProducerFactory` whose producers hand a Web Worker
 * the name of a method and its JSON options, ask it to compute ahead of the consumer's cursor, and apply the tape it
 * streams back to `RemoteSession`s the panels read. The worker starts when it is first needed (a play, a probe), and a
 * producer that is disposed terminates it and forgets the tape, so a cancel, a reset and a remake all end the same
 * clean way. Without Workers (a test, a server render), or with options that cannot cross, it makes the main-thread
 * producer instead and says why.
 */

import { type Json, MdmlError } from 'mdml-engine/base'
import { defaults } from 'mdml-engine/space'
import type { Config, Derived, Method, SessionOptions, SliceBudget } from 'mdml-engine/trace'
import { inlineProducer } from './inline.ts'
import type { Variant } from './make.ts'
import type { Producer, ProducerFactory } from './producer.ts'
import {
  isProducerEvent,
  type ProducerEvent,
  type ProducerRequest,
  type WireVariant,
  type WorkerLike,
  wireOptions,
} from './protocol.ts'
import { ProbePending, RemoteSession } from './remote.ts'

/** How a worker producer is made. */
export interface WorkerProducerOptions {
  /** Make the worker (called when first needed, and again after a dispose). */
  readonly createWorker: () => WorkerLike
  /** Whether Workers exist here (default: the global `Worker` is defined); false makes every producer a main-thread one. */
  readonly available?: boolean
  /** How many steps the producer may run ahead of the cursor (default to the end of the run). */
  readonly lookahead?: number
  /** Ship checkpoint events too (default no). */
  readonly checkpoints?: boolean
}

// The probes the page keeps: a bounded cache, oldest dropped first.
const CACHE = 600

type Outcome = { readonly ok: true; readonly value: Derived } | { readonly ok: false; readonly error: string }

let counter = 0

const keyOf = (variant: number, t: number, path: string, args: Json | undefined): string =>
  `${variant}|${t}|${path}|${JSON.stringify(args ?? null)}`

/**
 * A factory of producers that compute in a worker.
 *
 * @param options - How to make the worker, and the lookahead.
 * @returns The factory, to give a hook as its `producer`.
 */
export function workerProducer(options: WorkerProducerOptions): ProducerFactory {
  const available = options.available ?? typeof Worker !== 'undefined'
  return <S, C extends Config>(
    method: Method<S, C>,
    sessionOptions: SessionOptions<S, C>,
    variants: readonly Variant<C>[],
  ): Producer<S, C> => {
    if (!available) return inlineProducer(method, sessionOptions, variants, 'Web Workers are not available here')
    let wire: ReturnType<typeof wireOptions>
    try {
      wire = wireOptions(sessionOptions)
      JSON.stringify(variants.map((v) => v.config))
    } catch (e) {
      return inlineProducer(method, sessionOptions, variants, e instanceof Error ? e.message : String(e))
    }
    return new WorkerSessions<S, C>(method, wire, sessionOptions, variants, options)
  }
}

class WorkerSessions<S, C extends Config> implements Producer<S, C> {
  readonly kind = 'worker'
  readonly background = true
  readonly sessions: readonly RemoteSession<S, C>[]
  readonly labels: readonly string[]
  error: string | undefined
  lastMs = 0
  probed = 0
  private readonly id = `p${++counter}`
  private worker: WorkerLike | undefined
  private lastSeq = 0
  private readonly listeners = new Set<() => void>()
  private readonly cache = new Map<string, Outcome | 'pending'>()
  private readonly waiting = new Map<
    number,
    { key: string; pathKey: string; resolve: (d: Derived) => void; reject: (e: Error) => void }
  >()
  // The last answer to each (variant, path, arguments), whatever its step: shown while a newer one is on its way.
  private readonly latest = new Map<string, Derived>()
  private request = 0
  private cursor = -1
  private readonly method: Method<S, C>
  private readonly wire: ReturnType<typeof wireOptions>
  private readonly variants: readonly Variant<C>[]
  private readonly settings: WorkerProducerOptions

  constructor(
    method: Method<S, C>,
    wire: ReturnType<typeof wireOptions>,
    sessionOptions: SessionOptions<S, C>,
    variants: readonly Variant<C>[],
    settings: WorkerProducerOptions,
  ) {
    this.method = method
    this.wire = wire
    this.variants = variants
    this.settings = settings
    this.labels = variants.map((v) => v.label)
    const seed = sessionOptions.seed ?? 0
    const link = {
      probe: (variant: number, t: number, path: string, args: Json | undefined): Derived =>
        this.probeNow(variant, t, path, args),
      probeAsync: (variant: number, t: number, path: string, args: Json | undefined): Promise<Derived> =>
        this.probeLater(variant, t, path, args),
      advance: (budget: SliceBudget): void => this.advance(budget),
      ensure: (): void => {
        this.spawn()
      },
    }
    this.sessions = variants.map(
      (v, i) =>
        new RemoteSession<S, C>(
          method,
          {
            ...(defaults(method.config) as Config),
            ...(sessionOptions.config as Config | undefined),
            ...(v.config as Config),
          } as C,
          seed,
          link,
          i,
        ),
    )
  }

  // The worker, made and told to make the sessions, the first time it is needed.
  private spawn(): WorkerLike | undefined {
    if (this.worker !== undefined) return this.worker
    if (this.error !== undefined) return undefined
    try {
      const w = this.settings.createWorker()
      this.worker = w
      this.lastSeq = 0
      w.onmessage = (e) => this.receive(e.data)
      w.onerror = (e) => this.fail(e.message ?? 'the worker failed')
      this.send({
        type: 'init',
        id: this.id,
        method: this.method.name,
        options: this.wire,
        variants: this.variants.map((v): WireVariant => ({ label: v.label, config: v.config as Json })),
        checkpoints: this.settings.checkpoints ?? false,
        lookahead: this.settings.lookahead ?? null,
      })
      if (this.cursor >= 0) this.send({ type: 'cursor', id: this.id, step: this.cursor })
      return w
    } catch (e) {
      this.fail(e instanceof Error ? e.message : String(e))
      return undefined
    }
  }

  private send(req: ProducerRequest): void {
    try {
      this.worker?.postMessage(req)
    } catch (e) {
      this.fail(e instanceof Error ? e.message : String(e))
    }
  }

  private notify(): void {
    for (const l of this.listeners) l()
  }

  private fail(message: string): void {
    if (this.error !== undefined) return
    this.error = message
    for (const s of this.sessions) s.fail()
    for (const [, w] of this.waiting) w.reject(new MdmlError('value', message))
    this.waiting.clear()
    this.notify()
  }

  private receive(data: unknown): void {
    if (!isProducerEvent(data) || data.id !== this.id) return
    const ev: ProducerEvent = data
    // The worker numbers its events one by one; one that is late or repeated is an old one and is dropped.
    if (ev.seq !== 0 && ev.seq <= this.lastSeq) return
    this.lastSeq = Math.max(this.lastSeq, ev.seq)
    switch (ev.type) {
      case 'ready':
        for (const [i, v] of ev.variants.entries()) this.sessions[i]?.applyReady(v.config as C, v.seed, v.kinds)
        break
      case 'tape':
        for (const d of ev.deltas) this.sessions[d.variant]?.applyDelta(d)
        this.lastMs = ev.sliceMs
        break
      case 'idle':
        break
      case 'probed': {
        const w = this.waiting.get(ev.req)
        this.waiting.delete(ev.req)
        if (w === undefined) break
        if (ev.superseded) {
          // A newer probe of the same path was waiting; this step was not computed, and may be asked again.
          this.cache.delete(w.key)
          w.reject(new MdmlError('value', 'superseded by a newer probe'))
          break
        }
        const outcome: Outcome = ev.ok
          ? { ok: true, value: ev.value as Derived }
          : { ok: false, error: ev.error ?? 'the probe failed' }
        this.remember(w.key, outcome)
        if (outcome.ok) this.latest.set(w.pathKey, outcome.value)
        this.probed += 1
        if (outcome.ok) w.resolve(outcome.value)
        else w.reject(new MdmlError('value', outcome.error))
        break
      }
      case 'error':
        this.fail(ev.message)
        return
    }
    this.notify()
  }

  private remember(key: string, outcome: Outcome): void {
    this.cache.delete(key)
    this.cache.set(key, outcome)
    if (this.cache.size > CACHE) {
      const oldest = this.cache.keys().next().value
      if (oldest !== undefined) this.cache.delete(oldest)
    }
  }

  private ask(variant: number, t: number, path: string, args: Json | undefined, latest: boolean): Promise<Derived> {
    const key = keyOf(variant, t, path, args)
    const pathKey = keyOf(variant, 0, path, args)
    return new Promise<Derived>((resolve, reject) => {
      const req = ++this.request
      this.waiting.set(req, { key, pathKey, resolve, reject })
      this.send({ type: 'probe', id: this.id, req, latest, variant, t, path, ...(args === undefined ? {} : { args }) })
    })
  }

  private probeNow(variant: number, t: number, path: string, args: Json | undefined): Derived {
    const key = keyOf(variant, t, path, args)
    const hit = this.cache.get(key)
    if (hit !== undefined && hit !== 'pending') {
      if (hit.ok) return hit.value
      throw new MdmlError('value', hit.error)
    }
    if (hit === undefined) {
      if (this.error !== undefined) throw new MdmlError('value', this.error)
      this.cache.set(key, 'pending')
      if (this.spawn() !== undefined)
        this.ask(variant, t, path, args, true).then(
          () => undefined,
          () => undefined,
        )
    }
    // Until the answer for this step comes, the view keeps the last one it had for the path (if it had one).
    const before = this.latest.get(keyOf(variant, 0, path, args))
    if (before !== undefined) return before
    throw new ProbePending(`${path} at step ${t}`)
  }

  private probeLater(variant: number, t: number, path: string, args: Json | undefined): Promise<Derived> {
    const key = keyOf(variant, t, path, args)
    const hit = this.cache.get(key)
    if (hit !== undefined && hit !== 'pending')
      return hit.ok ? Promise.resolve(hit.value) : Promise.reject(new MdmlError('value', hit.error))
    if (this.spawn() === undefined) return Promise.reject(new MdmlError('value', this.error ?? 'no worker'))
    return this.ask(variant, t, path, args, false)
  }

  probeAsync(variant: number, t: number, path: string, args?: Json): Promise<Derived> {
    return this.probeLater(variant, t, path, args)
  }

  advance(budget: SliceBudget): void {
    if (this.spawn() === undefined) return
    const steps = budget.steps ?? Number.POSITIVE_INFINITY
    if (Number.isFinite(steps)) this.send({ type: 'advance', id: this.id, steps })
    else this.send({ type: 'run', id: this.id, eager: true })
  }

  start(options: { readonly eager?: boolean } = {}): void {
    if (this.spawn() === undefined) return
    this.send({ type: 'run', id: this.id, eager: options.eager ?? false })
  }

  stop(): void {
    if (this.worker !== undefined) this.send({ type: 'pause', id: this.id })
  }

  follow(step: number): void {
    if (step === this.cursor) return
    this.cursor = step
    if (this.worker !== undefined) this.send({ type: 'cursor', id: this.id, step })
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  dispose(): void {
    const w = this.worker
    this.worker = undefined
    if (w !== undefined) {
      try {
        w.postMessage({ type: 'dispose', id: this.id })
      } catch {
        // The worker is going away anyway.
      }
      w.onmessage = null
      w.onerror = null
      w.terminate()
    }
    for (const [, p] of this.waiting) p.reject(new MdmlError('value', 'the producer was disposed'))
    this.waiting.clear()
    this.cache.clear()
    this.latest.clear()
    for (const s of this.sessions) s.reset()
    this.error = undefined
    this.lastSeq = 0
    this.lastMs = 0
    this.cursor = -1
  }
}
