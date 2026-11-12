import { serveRun } from '../../../../shared/runWorker.ts'
import { runSeed, type SeedRun } from './calibration.ts'

/** What the page asks: the knobs of the data and of EP, and how many seeds. */
export interface CalibrationRequest {
  readonly stay: number
  readonly steps: number
  readonly damping: number
  readonly seeds: number
}

// The calibration, a seed at a time: each is simulated, fitted by EP and by Gibbs, and sent to the page as a result so far.
// A stop ends the seed it is in (its EP and Gibbs stop at their next sweep) and drops it; the seeds done are kept.
serveRun<CalibrationRequest, SeedRun[]>(self, async ({ stay, steps, damping, seeds }, host) => {
  const rows: SeedRun[] = []
  for (let seed = 1; seed <= seeds; seed++) {
    host.report({ stage: 'seeds', done: seed - 1, total: seeds, detail: `seed ${seed} of ${seeds}` })
    const hooks = {
      onSweep: ({ sweep, cap }: { sweep: number; cap: number }) =>
        host.report({
          stage: 'seeds',
          done: seed - 1,
          total: seeds,
          detail: `seed ${seed} of ${seeds}, sweep ${sweep} of ${cap}`,
        }),
      stopped: host.stopped,
      yieldToPage: host.yieldToPage,
    }
    const row = await runSeed(seed, { stay, steps, damping }, hooks)
    if (host.stopped()) break
    rows.push(row)
    host.partial([...rows])
  }
  host.report({ stage: 'seeds', done: rows.length, total: seeds, detail: `${rows.length} of ${seeds} seeds` })
  return rows
})
