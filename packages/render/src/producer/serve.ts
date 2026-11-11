/**
 * The worker's side of a producer: `serveProducer` answers the requests of the protocol on a worker's scope. It makes
 * the sessions of a method looked up by name, computes them ahead in short slices (so a pause, a cursor update or a
 * probe is heard between slices), keeps the heads of a comparison in step, stays within the lookahead of the
 * consumer's cursor unless asked to run eagerly, streams the tape as deltas and answers probes on demand. A failure
 * becomes an `error` event, never an uncaught exception. Everything it needs from the outside (clock, scheduling) is
 * injected, so it runs under a test double as well as in a real worker.
 */

import { OmaiError } from 'omai-engine/base'
import type { Run, RunStatus, Session, TraceEvent } from 'omai-engine/trace'
import { makeSessions } from './make.ts'
import {
  type IdleReason,
  isProducerRequest,
  type ProducerEvent,
  type ProducerRequest,
  type ScopeLike,
  sessionOptionsOf,
  type TapeDelta,
} from './protocol.ts'
import type { MethodRegistry } from './registry.ts'
import { advanceInStep, guardBudget, sharedHead, stepsLeft } from './step.ts'

/** What the server takes from its surroundings. */
export interface ServeOptions {
  /** Milliseconds, monotonic (default `performance.now`). */
  readonly clock?: () => number
  /** Run a task soon, after the messages already waiting (default a message channel, else a zero timeout). */
  readonly schedule?: (task: () => void) => void
  /** How long one slice of computation may take (default 20 ms): the latency of a pause or a probe. */
  readonly sliceMs?: number
  /** The least time between two `tape` events while computing (default 40 ms); the end and idling always post. */
  readonly postMs?: number
}

// One producer the worker hosts.
interface Hosted {
  readonly id: string
  // biome-ignore lint/suspicious/noExplicitAny: sessions of any method
  readonly sessions: readonly Session<any, any>[]
  readonly checkpoints: boolean
  readonly lookahead: number
  // The events of each variant as last sent, by identity, so a delta is found even when a window forgot some.
  readonly sent: (readonly TraceEvent[])[]
  readonly firedSent: Map<string, number>[]
  readonly eventPaths: readonly string[][]
  seq: number
  cursor: number
  /** The consumer wants the run computed (it may be held by the lookahead); `running` is whether the loop is going. */
  wanted: boolean
  running: boolean
  eager: boolean
  advanceLeft: number | undefined
  failed: boolean
  lastPost: number
  lastSliceMs: number
  /** Steps per call within a slice, adapted to how long they take (cheap steps run in the thousands). */
  chunk: number
  /** Probes waiting to be answered, oldest first. */
  probes: Extract<ProducerRequest, { type: 'probe' }>[]
  answering: boolean
  /** Milliseconds of probing owed to the computation: probes yield to slices until it is paid (about half and half). */
  debt: number
  /** Milliseconds spent computing the sessions: what the methods' `budgetMs` is held against. */
  spentMs: number
  // Bumped whenever the run loop is restarted or stopped, so a slice scheduled earlier knows it is stale.
  token: number
  readonly lastT: number[]
  readonly lastStatus: (RunStatus | null)[]
}

function defaultSchedule(): (task: () => void) => void {
  if (typeof MessageChannel === 'function') {
    const queue: (() => void)[] = []
    const { port1, port2 } = new MessageChannel()
    port1.onmessage = () => queue.shift()?.()
    return (task) => {
      queue.push(task)
      port2.postMessage(0)
    }
  }
  return (task) => {
    setTimeout(task, 0)
  }
}

const messageOf = (e: unknown): string => (e instanceof Error ? e.message : String(e))

/**
 * Answer the requests of the producer protocol on a worker scope.
 *
 * @param scope - The worker's global scope (or a test double).
 * @param registry - The methods the worker can make, by name.
 * @param options - Clock, scheduling and the slice and post intervals.
 * @returns A handle that stops listening and drops every hosted producer.
 */
export function serveProducer(
  scope: ScopeLike,
  registry: MethodRegistry,
  options: ServeOptions = {},
): { readonly close: () => void } {
  const clock = options.clock ?? ((): number => performance.now()) // allow: pacing of slices, never on the tape
  const schedule = options.schedule ?? defaultSchedule()
  const sliceMs = options.sliceMs ?? 20
  const postMs = options.postMs ?? 40
  const hosted = new Map<string, Hosted>()

  const send = (h: { readonly id: string; seq: number }, make: (seq: number) => ProducerEvent): void => {
    h.seq += 1
    scope.postMessage(make(h.seq))
  }
  const fail = (id: string, message: string): void => {
    const h = hosted.get(id)
    if (h === undefined) {
      scope.postMessage({ type: 'error', id, seq: 0, message })
      return
    }
    h.failed = true
    h.wanted = false
    h.running = false
    h.token += 1
    send(h, (seq) => ({ type: 'error', id, seq, message }))
  }

  // What each variant's tape gained since the last delta, by identity of the events, or nothing when it is unchanged.
  const deltaOf = (h: Hosted, i: number): TapeDelta | undefined => {
    const session = h.sessions[i]
    const prev = h.sent[i]
    if (session === undefined || prev === undefined) return undefined
    // Reading the step first starts a session that has not yet recorded its first state.
    const t = session.t
    const status = session.status ?? null
    const run: Run = session.run()
    const now = h.checkpoints ? run.events : run.events.filter((e) => e.type !== 'checkpoint')
    let k = 0
    const limit = Math.min(prev.length, now.length)
    while (k < limit && prev[k] === now[k]) k++
    const events = now.slice(k)
    const fired: Record<string, number[]> = {}
    const seen = h.firedSent[i]
    for (const path of h.eventPaths[i] ?? []) {
      const steps = session.stepsOf(path)
      const had = seen?.get(path) ?? 0
      if (steps.length > had) {
        fired[path] = steps.slice(had)
        seen?.set(path, steps.length)
      }
    }
    const same = k === prev.length && events.length === 0 && Object.keys(fired).length === 0
    if (same && h.lastT[i] === t && h.lastStatus[i] === status) return undefined
    h.sent[i] = now
    h.lastT[i] = t
    h.lastStatus[i] = status
    return { variant: i, truncate: k, events, t, status, fired }
  }

  const flush = (h: Hosted): void => {
    const deltas: TapeDelta[] = []
    for (let i = 0; i < h.sessions.length; i++) {
      const d = deltaOf(h, i)
      if (d !== undefined) deltas.push(d)
    }
    h.lastPost = clock()
    if (deltas.length > 0) send(h, (seq) => ({ type: 'tape', id: h.id, seq, deltas, sliceMs: h.lastSliceMs }))
  }

  const idle = (h: Hosted, reason: IdleReason): void => {
    h.running = false
    h.token += 1
    // Held by the lookahead, the consumer still wants the run: the cursor moving up resumes it.
    h.wanted = reason === 'lookahead'
    flush(h)
    send(h, (seq) => ({ type: 'idle', id: h.id, seq, reason }))
    if (h.probes.length > 0) scheduleAnswers(h)
  }

  const headOf = (h: Hosted): number => sharedHead(h.sessions.map((s) => ({ head: s.t, done: s.status !== undefined })))

  // How many steps may be computed now: the ones asked for, else those within the lookahead of the cursor.
  const roomOf = (h: Hosted, head: number): number => {
    const budget = stepsLeft(h.sessions as never)
    if (h.advanceLeft !== undefined) return Math.min(h.advanceLeft, budget)
    if (!h.eager && Number.isFinite(h.lookahead)) return Math.min(h.cursor + h.lookahead - head, budget)
    return budget
  }

  const probeKey = (r: { variant: number; path: string; args?: unknown }): string =>
    `${r.variant}|${r.path}|${JSON.stringify(r.args ?? null)}`

  // Answer waiting probes. A `latest` probe with a newer one for the same path behind it is dropped (superseded). While
  // the run is computing, probes take turns with slices: one at a time, and only when the time owed to the computation
  // is paid, so a view that asks at every step cannot starve the run. Idle, everything waiting is answered at once.
  const answerProbes = (h: Hosted, everything: boolean): void => {
    const queue = h.probes
    h.probes = []
    const newest = new Map<string, number>()
    queue.forEach((r, i) => {
      if (r.latest) newest.set(probeKey(r), i)
    })
    const keep: typeof queue = []
    queue.forEach((r, i) => {
      if (r.latest && newest.get(probeKey(r)) !== i)
        send(h, (seq) => ({ type: 'probed', id: h.id, seq, req: r.req, ok: false, superseded: true }))
      else keep.push(r)
    })
    for (const [i, r] of keep.entries()) {
      if (!everything && h.debt > 0) {
        h.probes = [...keep.slice(i), ...h.probes]
        return
      }
      const t0 = clock()
      const session = h.sessions[r.variant]
      try {
        if (session === undefined) throw new OmaiError('value', `no variant ${r.variant}`)
        const value = session.probe(r.t, r.path, r.args)
        send(h, (seq) => ({ type: 'probed', id: h.id, seq, req: r.req, ok: true, value }))
      } catch (e) {
        const error = messageOf(e)
        send(h, (seq) => ({ type: 'probed', id: h.id, seq, req: r.req, ok: false, error }))
      }
      h.debt += clock() - t0
    }
  }

  const scheduleAnswers = (h: Hosted): void => {
    if (h.answering) return
    h.answering = true
    schedule(() => {
      h.answering = false
      if (hosted.get(h.id) === h && !h.running) answerProbes(h, true)
    })
  }

  // One slice: calls of a few steps each (every variant brought up to the slowest's head plus the steps) until the slice's
  // time is spent or the room used; the calls grow while steps are cheap and shrink while they are dear, so a slice
  // ends close to its time whatever a step costs, and never takes more than the room.
  const runSlice = (h: Hosted, room: number): void => {
    const deadline = clock() + sliceMs
    let left = room
    for (;;) {
      const before = headOf(h)
      const t0 = clock()
      advanceInStep(h.sessions, { steps: Math.min(left, h.chunk) })
      const dt = clock() - t0
      const done = headOf(h) - before
      left -= done
      if (dt < sliceMs / 8) h.chunk = Math.min(h.chunk * 2, 4096)
      else if (dt > sliceMs / 2) h.chunk = Math.max(1, h.chunk >> 1)
      if (left <= 0 || done <= 0 || h.sessions.every((s) => s.status !== undefined) || clock() >= deadline) return
    }
  }

  const pump = (h: Hosted, token: number): void => {
    if (hosted.get(h.id) !== h || !h.running || token !== h.token) return
    try {
      if (h.sessions.every((s) => s.status !== undefined)) {
        idle(h, 'ended')
        return
      }
      const head = headOf(h)
      const room = roomOf(h, head)
      if (room <= 0) {
        idle(h, h.advanceLeft !== undefined ? 'advanced' : 'lookahead')
        return
      }
      const t0 = clock()
      runSlice(h, room)
      h.lastSliceMs = clock() - t0
      h.spentMs += h.lastSliceMs
      guardBudget(h.sessions as never, h.spentMs)
      h.debt = Math.max(0, h.debt - h.lastSliceMs)
      if (h.probes.length > 0) answerProbes(h, false)
      if (h.advanceLeft !== undefined) h.advanceLeft = Math.max(0, h.advanceLeft - (headOf(h) - head))
      if (h.sessions.every((s) => s.status !== undefined)) {
        idle(h, 'ended')
        return
      }
      if (clock() - h.lastPost >= postMs) flush(h)
    } catch (e) {
      fail(h.id, messageOf(e))
      return
    }
    const next = h.token
    schedule(() => pump(h, next))
  }

  const resume = (h: Hosted): void => {
    h.wanted = true
    h.running = true
    h.token += 1
    const token = h.token
    schedule(() => pump(h, token))
  }

  const handle = (req: ProducerRequest): void => {
    if (req.type === 'init') {
      hosted.delete(req.id)
      const method = registry.need(req.method)
      const made = makeSessions(method, sessionOptionsOf(req.options), req.variants as never)
      const eventPaths = made.sessions.map((s) =>
        Object.entries(s.contract)
          .filter(([, source]) => source.kind === 'event')
          .map(([path]) => path),
      )
      const h: Hosted = {
        id: req.id,
        sessions: made.sessions,
        checkpoints: req.checkpoints,
        lookahead: req.lookahead ?? Number.POSITIVE_INFINITY,
        sent: made.sessions.map(() => []),
        firedSent: made.sessions.map(() => new Map()),
        eventPaths,
        seq: 0,
        cursor: 0,
        wanted: false,
        running: false,
        eager: false,
        advanceLeft: undefined,
        failed: false,
        lastPost: clock(),
        lastSliceMs: 0,
        chunk: 1,
        probes: [],
        answering: false,
        debt: 0,
        spentMs: 0,
        token: 0,
        lastT: made.sessions.map(() => -1),
        lastStatus: made.sessions.map(() => null),
      }
      hosted.set(req.id, h)
      send(h, (seq) => ({
        type: 'ready',
        id: h.id,
        seq,
        variants: made.sessions.map((s) => ({
          config: s.config as never,
          seed: s.seed,
          kinds: Object.fromEntries(Object.entries(s.contract).map(([path, source]) => [path, source.kind])),
        })),
      }))
      // The first state is on the tape from the start (the session records step 0 when it is first read).
      flush(h)
      return
    }
    const h = hosted.get(req.id)
    if (req.type === 'dispose') {
      if (h !== undefined) h.token += 1
      hosted.delete(req.id)
      return
    }
    if (h === undefined) throw new OmaiError('value', `no producer '${req.id}' (it was disposed, or never made)`)
    switch (req.type) {
      case 'run':
        if (h.failed) return
        h.eager = req.eager
        h.advanceLeft = undefined
        resume(h)
        return
      case 'advance':
        if (h.failed) return
        h.advanceLeft = req.steps
        resume(h)
        return
      case 'pause':
        if (h.running || h.wanted) idle(h, 'paused')
        return
      case 'cursor':
        h.cursor = req.step
        // A producer held by the lookahead goes on when the cursor moves up to it.
        if (h.wanted && !h.running && roomOf(h, headOf(h)) > 0) resume(h)
        return
      case 'probe':
        h.probes.push(req)
        if (!h.running) scheduleAnswers(h)
        return
    }
  }

  scope.onmessage = (event) => {
    const req = event.data
    if (!isProducerRequest(req)) return
    try {
      handle(req)
    } catch (e) {
      fail(req.id, messageOf(e))
    }
  }
  return {
    close: () => {
      scope.onmessage = null
      for (const h of hosted.values()) h.token += 1
      hosted.clear()
    },
  }
}

/**
 * Serve the producer protocol on the current worker's global scope: the whole body of a worker entry file.
 *
 * @param registry - The methods the worker can make, by name.
 * @param options - Slice and post intervals.
 * @returns A handle that stops serving.
 * @index function The whole body of a worker entry file: `serveWorker(registry)`.
 */
export function serveWorker(registry: MethodRegistry, options: ServeOptions = {}): { readonly close: () => void } {
  return serveProducer(globalThis as unknown as ScopeLike, registry, options)
}
