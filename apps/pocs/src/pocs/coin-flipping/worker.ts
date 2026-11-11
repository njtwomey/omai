/** The page's worker: it serves the coin method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { coinMethod } from './method.ts'

serveWorker(methodRegistry([coinMethod]))
