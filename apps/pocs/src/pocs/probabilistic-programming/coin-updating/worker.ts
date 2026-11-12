/**
 * The coin page's worker entry: one request is one method fitted along the ladder of prefixes (`compute.ts`), answered
 * with its frames, on the pocs' shared wire (`serveJob`: a `run` request, `status` messages while it works, a `stop`
 * that returns what it has).
 */

import { serveJob } from '../../../shared/workerPool.ts'
import { type CoinRequest, type CoinResult, runCoin } from './compute.ts'

serveJob<CoinRequest, CoinResult>(self as unknown as Parameters<typeof serveJob>[0], runCoin)
