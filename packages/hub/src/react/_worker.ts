// The hub's Web Worker: runs catalogue entries off the main thread for `useHubRun`.
import { catalogue } from '../catalogue/index.ts'
import type { HubProgress } from '../entry/entry.ts'
import { runDemo } from '../entry/run.ts'
import type { HubRunReply, HubRunRequest } from './useHubRun.ts'

const scope = globalThis as unknown as {
  onmessage: ((e: MessageEvent<HubRunRequest>) => void) | null
  postMessage: (reply: HubRunReply) => void
}

scope.onmessage = (e: MessageEvent<HubRunRequest>) => {
  const { token, id, state } = e.data
  const entry = catalogue.byId.get(id)
  if (entry === undefined) {
    scope.postMessage({ token, ok: false, error: `no hub entry '${id}'` })
    return
  }
  const start = performance.now()
  // Progress at most ten times a second.
  let last = Number.NEGATIVE_INFINITY
  const progress = (p: HubProgress): void => {
    const now = performance.now()
    if (now - last < 100 && p.fraction < 1) return
    last = now
    scope.postMessage({ token, progress: p })
  }
  const fail = (err: unknown): void =>
    scope.postMessage({ token, ok: false, error: err instanceof Error ? err.message : String(err) })
  runDemo(entry, state, progress)
    .then((r) =>
      scope.postMessage({ token, ok: true, exposition: r.exposition, rows: r.rows, ms: performance.now() - start }),
    )
    .catch(fail)
}
