/** The page's worker entry: one request is one NUTS run on one form (`compute.ts`), on the pocs' shared wire. */

import { serveJob } from '../../../shared/workerPool.ts'
import { type PoolingRequest, type PoolingResult, runPooling } from './compute.ts'

serveJob<PoolingRequest, PoolingResult>(self as unknown as Parameters<typeof serveJob>[0], runPooling)
