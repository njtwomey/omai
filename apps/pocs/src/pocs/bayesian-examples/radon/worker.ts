/** The page's worker: it serves the radon method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { radonExample } from './method.ts'

serveWorker(methodRegistry([radonExample.method]))
