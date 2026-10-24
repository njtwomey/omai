// The hub's Web Worker: runs catalogue entries off the main thread for `useHubRun`.
import { catalogue } from '../catalogue/index.ts'
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
  const fail = (err: unknown): void =>
    scope.postMessage({ token, ok: false, error: err instanceof Error ? err.message : String(err) })
  runDemo(entry, state)
    .then((r) =>
      scope.postMessage({ token, ok: true, exposition: r.exposition, rows: r.rows, ms: performance.now() - start }),
    )
    .catch(fail)
}
