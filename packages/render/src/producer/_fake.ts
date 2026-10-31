/**
 * A worker in memory, for tests: the page's end (`WorkerLike`) and the worker's scope (`ScopeLike`) joined by one
 * queue, with the server's scheduling in the same queue, so a test steps a conversation message by message, or runs it
 * until it goes quiet. Messages are structured-cloned on the way, as a real worker's are.
 */

import type { ProducerEvent, ProducerRequest, ScopeLike, WorkerLike } from './protocol.ts'

/** The two ends and the queue between them. */
export interface FakeWire {
  readonly worker: WorkerLike & { terminated: boolean }
  readonly scope: ScopeLike
  /** Every request the page sent, and every event the worker answered with, in order. */
  readonly requests: ProducerRequest[]
  readonly events: ProducerEvent[]
  /** For the server's `schedule`. */
  readonly schedule: (task: () => void) => void
  /** How many tasks are waiting. */
  pending(): number
  /** Run one waiting task; `false` when there was none. */
  step(): boolean
  /** Run waiting tasks until none is left (or `limit` have run); returns how many ran. */
  drain(limit?: number): number
}

/**
 * An in-memory worker wire.
 *
 * @returns The wire.
 */
export function fakeWire(): FakeWire {
  const queue: (() => void)[] = []
  const requests: ProducerRequest[] = []
  const events: ProducerEvent[] = []
  const worker: FakeWire['worker'] = {
    terminated: false,
    onmessage: null,
    onerror: null,
    postMessage: (m) => {
      if (worker.terminated) return
      requests.push(m)
      const copy = structuredClone(m)
      queue.push(() => {
        if (!worker.terminated) scope.onmessage?.({ data: copy })
      })
    },
    terminate: () => {
      worker.terminated = true
    },
  }
  const scope: ScopeLike = {
    onmessage: null,
    postMessage: (m) => {
      if (worker.terminated) return
      events.push(m)
      const copy = structuredClone(m)
      queue.push(() => {
        if (!worker.terminated) worker.onmessage?.({ data: copy })
      })
    },
  }
  const step = (): boolean => {
    const task = queue.shift()
    if (task === undefined) return false
    task()
    return true
  }
  return {
    worker,
    scope,
    requests,
    events,
    schedule: (task) => {
      queue.push(task)
    },
    pending: () => queue.length,
    step,
    drain: (limit = 100_000) => {
      let n = 0
      while (n < limit && step()) n++
      return n
    },
  }
}

/**
 * A clock that moves by `by` milliseconds every time it is read, so time budgets end deterministically.
 *
 * @param by - The milliseconds per reading.
 * @returns The clock.
 */
export function tickingClock(by = 5): () => number {
  let now = 0
  return () => {
    now += by
    return now
  }
}
