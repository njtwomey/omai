import { useCallback, useEffect, useRef, useState } from 'react'
import type { SeedRun } from './calibration.ts'
import type { CalibrationRequest } from './reference.worker.ts'

const makeWorker = (): Worker => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' })

/**
 * The calibration seeds run in a worker, one result per seed as it finishes, started when the knobs are set and restarted when
 * they change (the page keeps responding meanwhile).
 *
 * @param request - The knobs of the data and of EP, and how many seeds.
 * @returns The seeds done so far, whether all are, and a stop button's handler.
 */
export function useCalibrationRuns(request: CalibrationRequest): {
  readonly runs: readonly SeedRun[]
  readonly done: boolean
  readonly failed: string | undefined
  readonly stop: () => void
} {
  const [runs, setRuns] = useState<readonly SeedRun[]>([])
  const [done, setDone] = useState(false)
  const [failed, setFailed] = useState<string | undefined>(undefined)
  const worker = useRef<Worker | undefined>(undefined)
  const key = JSON.stringify(request)
  const stop = useCallback((): void => {
    worker.current?.terminate()
    worker.current = undefined
  }, [])
  useEffect(() => {
    const asked = JSON.parse(key) as CalibrationRequest
    const collected: SeedRun[] = []
    setRuns([])
    setDone(false)
    setFailed(undefined)
    const w = makeWorker()
    worker.current = w
    w.onmessage = (e: MessageEvent<SeedRun>): void => {
      collected.push(e.data)
      setRuns([...collected])
      if (collected.length >= asked.seeds) {
        setDone(true)
        w.terminate()
        if (worker.current === w) worker.current = undefined
      }
    }
    w.onerror = (e): void => setFailed(e.message)
    w.postMessage(asked)
    return () => {
      w.terminate()
      if (worker.current === w) worker.current = undefined
    }
  }, [key])
  return { runs, done, failed, stop }
}
