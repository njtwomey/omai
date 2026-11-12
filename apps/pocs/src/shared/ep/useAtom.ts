import {
  type AtomCheck,
  type AtomProblem,
  type AtomReference,
  atomCheck,
  atomReference,
  atomRun,
  atomVerdict,
  type EpAtom,
  type EpRun,
} from 'omai-engine/probprog/ep'
import type { Config } from 'omai-engine/trace'
import { useMemo } from 'react'
import { workerTask } from '../runWorker.ts'
import { type UseRun, useRun } from '../useRun.ts'
import type { AtomReferenceRequest } from './atomReference.worker.ts'

/** What a page of an atom shows: the problem, the library's verdict, EP's run (when it applies), the reference and the check. */
export interface AtomState {
  /** Undefined only when the knobs made no problem (`error` says why). */
  readonly problem: AtomProblem | undefined
  readonly verdict: ReturnType<typeof atomVerdict> | undefined
  readonly run: EpRun | undefined
  /** Milliseconds the EP run took. */
  readonly runMs: number
  /** The reference: undefined while it is being computed (the grid takes the longest). */
  readonly reference: AtomReference | undefined
  readonly check: AtomCheck | undefined
  /** An error from the problem, the run or the reference, in words. */
  readonly error: string | undefined
  /** The run of the reference when it is computed in a worker (a grid), for its bar; idle for a closed form. */
  readonly referenceRun: UseRun<AtomReference>
}

const makeWorker = (): Worker => new Worker(new URL('./atomReference.worker.ts', import.meta.url), { type: 'module' })

/**
 * The state of an atom's page for some knobs and some EP knobs: the problem and EP's run at once (they take a few
 * milliseconds), the reference after the first paint (a grid takes longer), and the check when both are there.
 *
 * @param atom - The atom.
 * @param knobs - The atom's knobs.
 * @param ep - The EP knobs.
 * @returns The state.
 */
export function useAtom(atom: EpAtom, knobs: Config, ep: Config): AtomState {
  const knobKey = JSON.stringify(knobs)
  const epKey = JSON.stringify(ep)
  const made = useMemo(() => {
    try {
      const problem = atom.problem(JSON.parse(knobKey) as Config)
      const verdict = atomVerdict(problem)
      const t0 = performance.now()
      const run = atomRun(problem, JSON.parse(epKey) as Config)
      return { problem, verdict, run, runMs: performance.now() - t0, error: undefined }
    } catch (e) {
      return {
        problem: undefined,
        verdict: undefined,
        run: undefined,
        runMs: 0,
        error: e instanceof Error ? e.message : String(e),
      }
    }
  }, [atom, knobKey, epKey])
  const problem = made.problem
  // A closed form or the truth is instant, on this thread. A grid takes up to seconds: it goes to a worker, with a bar.
  const inline = useMemo((): { reference?: AtomReference; error?: string } | undefined => {
    if (problem === undefined || atom.reference === 'grid') return undefined
    try {
      return { reference: atomReference(atom, problem) }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [atom, problem])
  const gridTask = useMemo(
    () =>
      problem === undefined || atom.reference !== 'grid'
        ? undefined
        : workerTask<AtomReferenceRequest, AtomReference>(
            makeWorker,
            { name: atom.name, knobs: JSON.parse(knobKey) as Config },
            0,
          ),
    [atom, problem, knobKey],
  )
  const referenceRun = useRun<AtomReference>(gridTask)
  const reference = inline?.reference ?? referenceRun.value
  const check = useMemo(
    () =>
      problem !== undefined && made.run?.status === 'ran' && reference !== undefined
        ? atomCheck(problem, made.run.beliefs, reference, atom.tolerance)
        : undefined,
    [atom, problem, made.run, reference],
  )
  return {
    problem,
    verdict: made.verdict,
    run: made.run,
    runMs: made.runMs,
    reference,
    check,
    error: made.error ?? inline?.error ?? (referenceRun.phase === 'failed' ? referenceRun.error : undefined),
    referenceRun,
  }
}
