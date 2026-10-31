/**
 * The producer factory the gallery's recipes give their hooks: computing in the gallery's worker. Where Workers do not
 * exist, or a session's options cannot cross, the hook computes on the main thread as before.
 */

import { type WorkerLike, workerProducer } from 'mdml-render/producer'

/** Sessions computed in a worker made from `toys/worker.ts`. */
export const toyWorkerProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})

/** The same, held to 150 steps ahead of the cursor: the worker computes no further than the page is about to show. */
export const toyWorkerAhead = workerProducer({
  lookahead: 150,
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
