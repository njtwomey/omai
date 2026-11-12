/** What the page and its worker share: the library's classifier method and the producer that runs it in a worker. */

import { type WorkerLike, workerProducer } from 'omai-render/producer'
import { classifierMethod } from '../../../examples/index.ts'

export { classifierMethod }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const classifyProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
