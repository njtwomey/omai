import { serveRun } from '../../shared/runWorker.ts'
import { exactSparse } from './compute.ts'

type Request = { readonly features: number; readonly X: number[][]; readonly y: number[] }

// All 2^p subsets are summed in one library call: it cannot report how far it is, so the page shows a timer and an unfilled
// bar, and Stop ends the worker.
serveRun(self, async (request: Request) => exactSparse(request))
