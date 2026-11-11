import {
  type AtomCheck,
  type AtomProblem,
  type AtomReference,
  type AtomRun,
  atomCheck,
  atomReference,
  atomRun,
  atomVerdict,
  type EpAtom,
} from 'omai-engine/bayesian-examples'
import type { Config } from 'omai-engine/trace'
import { useCallback, useMemo } from 'react'
import { useDeferred } from '../useDeferred.ts'

/** What a page of an atom shows: the problem, the library's verdict, EP's run (when it applies), the reference and the check. */
export interface AtomState {
  /** Undefined only when the knobs made no problem (`error` says why). */
  readonly problem: AtomProblem | undefined
  readonly verdict: ReturnType<typeof atomVerdict> | undefined
  readonly run: AtomRun | undefined
  /** Milliseconds the EP run took. */
  readonly runMs: number
  /** The reference: undefined while it is being computed (the grid takes the longest). */
  readonly reference: AtomReference | undefined
  readonly check: AtomCheck | undefined
  /** An error from the problem, the run or the reference, in words. */
  readonly error: string | undefined
}

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
  const compute = useCallback((): { reference?: AtomReference; error?: string } => {
    if (problem === undefined) return {}
    try {
      return { reference: atomReference(atom, problem) }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [atom, problem])
  const refd = useDeferred(compute)
  const reference = refd?.reference
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
    error: made.error ?? refd?.error,
  }
}
