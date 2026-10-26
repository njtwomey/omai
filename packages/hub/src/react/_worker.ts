// The hub's Web Worker: runs catalogue entries off the main thread for `useHubRun`, loading the entry's own module when
// the request names it and the whole catalogue only when it does not.
import { type HubEntry, type HubProgress, isHubEntry } from '../entry/entry.ts'
import { runDemo } from '../entry/run.ts'
import { entryModules } from '../lazy/index.ts'
import type { HubRunReply, HubRunRequest } from './useHubRun.ts'

const scope = globalThis as unknown as {
  onmessage: ((e: MessageEvent<HubRunRequest>) => void) | null
  postMessage: (reply: HubRunReply) => void
}

async function find(id: string, module: string | undefined): Promise<HubEntry | undefined> {
  const load = module === undefined ? undefined : entryModules[module]
  if (load !== undefined) {
    const found = Object.values(await load()).find((v): v is HubEntry => isHubEntry(v) && v.id === id)
    if (found !== undefined) return found
  }
  return (await import('../catalogue/index.ts')).catalogue.byId.get(id)
}

scope.onmessage = async (e: MessageEvent<HubRunRequest>) => {
  const { token, id, module, state } = e.data
  const fail = (err: unknown): void =>
    scope.postMessage({ token, ok: false, error: err instanceof Error ? err.message : String(err) })
  let entry: HubEntry | undefined
  try {
    entry = await find(id, module)
  } catch (err) {
    fail(err)
    return
  }
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
  runDemo(entry, state, progress)
    .then((r) =>
      scope.postMessage({ token, ok: true, exposition: r.exposition, rows: r.rows, ms: performance.now() - start }),
    )
    .catch(fail)
}
