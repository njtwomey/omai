/**
 * The mixture page's worker entry: one request is one method on the mixture (`compute.ts`), answered with its draws, on the
 * pocs' shared wire (`serveJob`: a `run` request, `status` messages while it works, a `stop` that returns what it has).
 */

import { serveJob } from '../../shared/workerPool.ts'
import { type MixtureRequest, type MixtureResult, runMixture } from './compute.ts'

serveJob<MixtureRequest, MixtureResult>(self as unknown as Parameters<typeof serveJob>[0], runMixture)
