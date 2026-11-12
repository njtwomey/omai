/** The page's worker: it serves the item response method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { irtMethod } from './method.ts'

serveWorker(methodRegistry([irtMethod]))
