/** The page's worker: it serves the eight schools method, so the run (and its probes) compute off the page's thread. */

import { methodRegistry, serveWorker } from 'omai-render/producer'
import { schoolsExample } from './method.ts'

serveWorker(methodRegistry([schoolsExample.method]))
