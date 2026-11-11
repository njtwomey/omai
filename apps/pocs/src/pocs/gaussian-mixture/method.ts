/** What the page and its worker share: the library's Gaussian mixture example and the producer that runs it in a worker. */

import { mixtureExample, mixtureMethod } from 'omai-engine/bayesian-examples'
import { type WorkerLike, workerProducer } from 'omai-render/producer'

export { mixtureExample, mixtureMethod }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const mixtureProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
