import { type CalibrationSample, calibrationSample, epAtom } from 'omai-engine/bayesian-examples'
import type { Config } from 'omai-engine/trace'
import { serveRun } from '../runWorker.ts'

/** What the page asks: the atom by name, the knobs of the page (the seed is replaced) and how many seeds. */
export interface AtomCalibrationRequest {
  readonly name: string
  readonly knobs: Config
  readonly seeds: number
}

// EP over the seeds, one seed at a time (a seed takes up to a second or two for the heavier atoms), each sent to the page as a
// result so far. A stop is heard between seeds and the seeds done are kept.
serveRun<AtomCalibrationRequest, CalibrationSample[]>(self, async ({ name, knobs, seeds }, host) => {
  const atom = epAtom(name)
  const samples: CalibrationSample[] = []
  for (let seed = 1; seed <= seeds; seed++) {
    host.report({ stage: 'seeds', done: samples.length, total: seeds, detail: `seed ${seed} of ${seeds}` })
    await host.yieldToPage()
    if (host.stopped()) break
    samples.push(calibrationSample(atom, seed, knobs))
    host.partial([...samples])
  }
  host.report({ stage: 'seeds', done: samples.length, total: seeds, detail: `${samples.length} of ${seeds} seeds` })
  return samples
})
