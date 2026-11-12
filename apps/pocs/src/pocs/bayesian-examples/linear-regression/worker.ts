/** The page's worker: it serves the linear regression method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { linearRegressionMethod } from './method.ts'

serveWorker(methodRegistry([linearRegressionMethod]))
