/**
 * A session whose run happens somewhere else: `RemoteSession` is a `Session` the panels read as any other, fed by
 * deltas of the tape that a worker streams. Its tape, step and status are whatever has arrived; its derived paths are
 * probed in the worker on request, so a probe that has not been answered yet throws `ProbePending` (the panels show it
 * as waiting and ask again when the answer comes). What needs the live algorithm (the state, a fork, the algorithm
 * itself) is not available on this side of the wire.
 */

import { type Json, OmaiError } from 'omai-engine/base'
import type {
  Algorithm,
  Config,
  Contract,
  Derived,
  Method,
  Run,
  RunStatus,
  Session,
  Slice,
  SliceBudget,
  TraceEvent,
} from 'omai-engine/trace'
import type { TapeDelta } from './protocol.ts'

/** Thrown by a probe whose answer is still on its way from the worker. */
export class ProbePending extends OmaiError {
  constructor(what: string) {
    super('value', `computing ${what} in the worker`)
    this.name = 'ProbePending'
  }
}

/** What a remote session asks of its producer. */
export interface RemoteLink {
  /** The probe at once if answered; else ask and throw {@link ProbePending}. */
  probe(variant: number, t: number, path: string, args: Json | undefined): Derived
  /** The probe, whenever it comes. */
  probeAsync(variant: number, t: number, path: string, args: Json | undefined): Promise<Derived>
  /** Compute more (the producer decides how). */
  advance(budget: SliceBudget): void
  /** Make sure the worker exists (it is started when first needed). */
  ensure(): void
}

/** A session that can also probe asynchronously (a remote one); the tape export prefers it. */
export interface AsyncProbing {
  probeAsync(t: number, path: string, args?: Json): Promise<Derived>
}

/**
 * Whether a session probes asynchronously.
 *
 * @param s - A session.
 * @returns Whether it has `probeAsync`.
 */
export function canProbeAsync(s: unknown): s is AsyncProbing {
  return typeof (s as Partial<AsyncProbing>).probeAsync === 'function'
}

function stubSource(kind: string): Contract<never>[string] {
  switch (kind) {
    case 'derived':
      return {
        kind: 'derived',
        description: '',
        compute: () => {
          throw new OmaiError('value', 'a derived path of a worker session is computed in the worker')
        },
      }
    case 'tensor':
      return { kind: 'tensor', role: 'state', description: '' }
    case 'event':
      return { kind: 'event', description: '' }
    case 'message':
      return { kind: 'message', description: '' }
    case 'frame':
      return { kind: 'frame', description: '' }
    default:
      return { kind: 'scalar', description: '' }
  }
}

/**
 * A contract that has only what the panels ask of one: the kind of each path (a derived path cannot be computed from
 * it). Used by sessions that are not running here, remote ones and sessions replayed from a file.
 *
 * @param kinds - The kind of each path.
 * @returns The contract.
 */
export function contractOfKinds(kinds: Readonly<Record<string, string>>): Contract<never> {
  return Object.fromEntries(Object.entries(kinds).map(([path, kind]) => [path, stubSource(kind)]))
}

const unavailable = (what: string): never => {
  throw new OmaiError('value', `${what} is not available on a session computed in a worker`)
}

/**
 * A session computed in a worker, as the page sees it.
 *
 * @concept log-contract
 */
export class RemoteSession<S, C extends Config = Config, K extends string = string> implements Session<S, C, K> {
  readonly method: Method<S, C, K>
  config: C
  seed: number
  private kinds: Readonly<Record<string, string>> | undefined
  private stubs: Contract<never> | undefined
  private events: TraceEvent[] = []
  private cached: Run | undefined
  private reached = 0
  private ended: RunStatus | undefined
  private fired = new Map<string, number[]>()
  private readonly link: RemoteLink
  private readonly variant: number

  constructor(method: Method<S, C, K>, config: C, seed: number, link: RemoteLink, variant: number) {
    this.method = method
    this.config = config
    this.seed = seed
    this.link = link
    this.variant = variant
  }

  /** The worker made the session: what it resolved to. */
  applyReady(config: C, seed: number, kinds: Readonly<Record<string, string>>): void {
    this.config = config
    this.seed = seed
    this.kinds = kinds
    this.stubs = undefined
  }

  /** A delta of the tape arrived. */
  applyDelta(delta: TapeDelta): void {
    if (delta.truncate < this.events.length) this.events.length = delta.truncate
    for (const e of delta.events) this.events.push(e)
    this.reached = delta.t
    this.ended = delta.status ?? undefined
    for (const [path, steps] of Object.entries(delta.fired)) {
      const have = this.fired.get(path)
      if (have === undefined) this.fired.set(path, [...steps])
      else have.push(...steps)
    }
    this.cached = undefined
  }

  /** The run failed or was cancelled from outside: it ends here. */
  fail(): void {
    if (this.ended === undefined) this.ended = 'failed'
    this.cached = undefined
  }

  /** Forget the tape (the producer was reset): back to step 0 with nothing computed. */
  reset(): void {
    this.events = []
    this.fired = new Map()
    this.cached = undefined
    this.reached = 0
    this.ended = undefined
    this.kinds = undefined
    this.stubs = undefined
  }

  get contract(): Contract<S, K> {
    if (this.kinds === undefined) {
      // The kinds come with the worker's first answer; asking for them starts the worker.
      this.link.ensure()
      return {} as Contract<S, K>
    }
    this.stubs ??= contractOfKinds(this.kinds)
    return this.stubs as unknown as Contract<S, K>
  }

  get algorithm(): Algorithm<S, K> {
    return unavailable('the algorithm')
  }

  get state(): S {
    return unavailable('the state')
  }

  get t(): number {
    return this.reached
  }

  get status(): RunStatus | undefined {
    return this.ended
  }

  advance(budget?: SliceBudget): Slice<S> {
    this.link.advance(budget ?? {})
    return { state: undefined as S, t: this.reached, ...(this.ended === undefined ? {} : { status: this.ended }) }
  }

  stateAt(_t: number): S {
    return unavailable('stateAt')
  }

  fork(): Session<S, C, K> {
    return unavailable('fork')
  }

  probe(t: number, path: K, args?: Json): Derived {
    return this.link.probe(this.variant, t, path, args)
  }

  probeAsync(t: number, path: string, args?: Json): Promise<Derived> {
    return this.link.probeAsync(this.variant, t, path, args)
  }

  project(name: string, options?: Json): Derived | undefined {
    const p = this.method.projections?.[name]
    if (p === undefined) throw new OmaiError('value', `${this.method.name}: no projection '${name}'`)
    return p(this.run(), options)
  }

  run(): Run {
    this.cached ??= { schema: 'omai.run/1', events: [...this.events] }
    return this.cached
  }

  stepsOf(event: K): readonly number[] {
    return this.fired.get(event) ?? []
  }
}
