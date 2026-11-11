import { runSeed, type SeedRun } from './calibration.ts'

/** What the page asks: the knobs of the data and of EP, and how many seeds. */
export interface CalibrationRequest {
  readonly stay: number
  readonly steps: number
  readonly damping: number
  readonly seeds: number
}

/** A message from the worker: one seed done. */
export type CalibrationMessage = SeedRun

self.onmessage = (e: MessageEvent<CalibrationRequest>): void => {
  const { stay, steps, damping, seeds } = e.data
  for (let seed = 1; seed <= seeds; seed++) self.postMessage(runSeed(seed, { stay, steps, damping }))
}
