import { epHooks } from '../../shared/ep/sweeps.ts'
import { serveRun } from '../../shared/runWorker.ts'
import { nutsReference, type Reference } from './reference.ts'

type Request = { counts: number[]; prior: { amplitude: number; lengthScale: number } }

serveRun<Request, Reference>(self, (request, host) =>
  nutsReference(
    request.counts,
    request.prior,
    { chains: 2, warmup: 300, draws: 1000 },
    epHooks(host, 'NUTS reference', 'iteration'),
  ),
)
