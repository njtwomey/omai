import { type CalibrationRow, type CircularSetup, calibrate } from './compute.ts'

type Request = { setup: CircularSetup; seeds: number[] }

self.onmessage = (e: MessageEvent<Request>): void => {
  const rows: CalibrationRow[] = calibrate(e.data.setup, e.data.seeds)
  self.postMessage(rows)
}
