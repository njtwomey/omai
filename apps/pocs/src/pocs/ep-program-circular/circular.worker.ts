import { epHooks } from '../../shared/ep/sweeps.ts'
import { serveRun } from '../../shared/runWorker.ts'
import {
  type CalibrationRow,
  type CircularRun,
  type CircularSetup,
  calibrate,
  type Equivalence,
  equivalence,
  runCircular,
} from './compute.ts'

/** What the page asks of the worker: one fit, the comparison of the two forms, or the calibration over seeds. */
export type CircularRequest =
  | { readonly kind: 'fit'; readonly setup: CircularSetup }
  | { readonly kind: 'equivalence'; readonly setup: CircularSetup; readonly current: CircularRun | undefined }
  | { readonly kind: 'calibration'; readonly setup: CircularSetup; readonly seeds: readonly number[] }

/** What it answers with. */
export type CircularReply = CircularRun | Equivalence | CalibrationRow[]

// Each EP sweep of this model takes from a fifth of a second to a second (a gate per reading), so every request runs here and
// the page only draws. A stop ends the sweep loop where it stands, and the fit keeps the sweeps done.
serveRun<CircularRequest, CircularReply>(self, async (request, host) => {
  if (request.kind === 'fit') return runCircular(request.setup, undefined, epHooks(host, 'EP'))
  if (request.kind === 'equivalence')
    return equivalence(request.setup, request.current, (stage) => epHooks(host, stage))
  const rows: CalibrationRow[] = []
  const total = request.seeds.length
  await calibrate(
    request.setup,
    request.seeds,
    (row) => {
      rows.push(row)
      host.partial([...rows])
      host.report({ stage: 'seeds', done: rows.length, total, detail: `seed ${rows.length} of ${total} done` })
    },
    (seed) => ({
      onSweep: ({ sweep, cap, change, tol }) =>
        host.report({
          stage: 'seeds',
          done: rows.length,
          total,
          detail: `seed ${seed} of ${total}, EP sweep ${sweep} of ${cap}${change === undefined || tol === undefined ? '' : `, change ${change.toExponential(1)}, tolerance ${tol.toExponential(1)}`}`,
        }),
      stopped: host.stopped,
      yieldToPage: host.yieldToPage,
    }),
    host.stopped,
  )
  return rows
})
