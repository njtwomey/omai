import { serveRun } from '../../../../shared/runWorker.ts'
import { exactZip } from './compute.ts'

type Request = { readonly x: number[]; readonly y: number[] }

// The grid of the exact posterior is one library call: it cannot report how far it is, so the page shows a timer and an
// unfilled bar, and Stop ends the worker.
serveRun(self, async (request: Request) => exactZip(request))
