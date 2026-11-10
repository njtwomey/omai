/**
 * The lab's worker entry: one request is one method on one problem (`lab.ts`), answered with its result. It speaks the
 * wire of the producer's `serveStage` (a `run` request, then `result` or `error`), with two additions the page's pool
 * uses: `status` messages while it runs, and a `stop` message that ends the run early and returns what it has.
 */

import { serveStage } from 'omai-render/producer'
import { runLab } from './lab.ts'
import type { LabRequest, MethodResult } from './types.ts'

let stopRequested = false

// A message to ourselves, so a `stop` that is waiting is handled between slices of the run.
const yieldToPage = (): Promise<void> =>
  new Promise((resolve) => {
    const channel = new MessageChannel()
    channel.port1.onmessage = () => {
      channel.port1.close()
      resolve()
    }
    channel.port2.postMessage(null)
  })

const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void
  onmessage: ((e: { readonly data: unknown }) => void) | null
}

serveStage<LabRequest, MethodResult>(scope, (request) => {
  stopRequested = false
  return runLab(request, {
    now: () => performance.now(),
    yieldToPage,
    stopped: () => stopRequested,
    status: (status) => scope.postMessage(status),
  })
})

const serve = scope.onmessage
scope.onmessage = (e) => {
  const data = e.data as { readonly type?: unknown } | null
  if (data !== null && typeof data === 'object' && data.type === 'stop') stopRequested = true
  else serve?.(e)
}
