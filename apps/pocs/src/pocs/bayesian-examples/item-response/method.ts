/** What the page and its worker share: the library's item response example and the producer that runs it in a worker. */

import { type WorkerLike, workerProducer } from 'omai-render/producer'
import { irtExample, irtMethod } from '../../../examples/index.ts'

export { irtExample, irtMethod }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const irtProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
