/**
 * The lab's worker entry: one request is one method on one problem (`lab.ts`), answered with its result, on the pocs'
 * shared wire (`serveJob`: a `run` request, `status` messages while it works, a `stop` that returns what it has).
 */

import { serveJob } from '../../shared/workerPool.ts'
import { runLab } from './lab.ts'
import type { LabRequest, MethodResult } from './types.ts'

serveJob<LabRequest, MethodResult>(self as unknown as Parameters<typeof serveJob>[0], runLab)
