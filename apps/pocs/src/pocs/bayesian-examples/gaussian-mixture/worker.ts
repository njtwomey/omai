/** The page's worker: it serves the mixture method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { mixtureMethod } from './method.ts'

serveWorker(methodRegistry([mixtureMethod]))
