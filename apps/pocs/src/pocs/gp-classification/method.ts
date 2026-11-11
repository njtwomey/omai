/** What the page and its worker share: the library's Gaussian process classification method and the producer that runs it in a worker. */

import { gpClassificationMethod } from 'omai-engine/bayesian-examples'
import { type WorkerLike, workerProducer } from 'omai-render/producer'

export { gpClassificationMethod }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const gpClassificationProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
