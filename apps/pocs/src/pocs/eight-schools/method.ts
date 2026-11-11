/** What the page and its worker share: the library's eight schools method and the producer that runs it in a worker. */

import { type WorkerLike, workerProducer } from 'omai-render/producer'
import { schoolsExample } from '../../examples/index.ts'

export { schoolsExample }

/** Sessions computed in `worker.ts` (the main thread where Workers do not exist). */
export const schoolsProducer = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
