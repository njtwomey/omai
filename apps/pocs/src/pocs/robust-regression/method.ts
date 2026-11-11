/** What the page and its worker share: the library's robust regression method and the producer that runs it in a worker. */

import { robustRegressionMethod } from 'omai-engine/bayesian-examples'
import { type WorkerLike, workerProducer } from 'omai-render/producer'

export { robustRegressionMethod }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const robustRegressionProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
