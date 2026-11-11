/** The page's worker: it serves the robust regression method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { robustRegressionMethod } from './method.ts'

serveWorker(methodRegistry([robustRegressionMethod]))
