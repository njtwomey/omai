/** The page's worker: it serves the Gaussian process classification method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { gpClassificationMethod } from './method.ts'

serveWorker(methodRegistry([gpClassificationMethod]))
