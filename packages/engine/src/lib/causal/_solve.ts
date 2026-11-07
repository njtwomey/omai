// Private: run a smooth problem to its optimum with a solver of the slot (`smoothSolvers`), for the causal fits that
// minimise a smooth objective (the entropy-balancing dual, the TMLE fluctuation). The solver is a configuration value,
// Newton by default; every solver of the table reaches the same optimum.
import { MdmlError } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { type SmoothProblem, type SolverReached, smoothSolverConfig, smoothSolvers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { runAlgorithm } from '../trace/index.ts'

/** The Newton solver at a gradient tolerance, as a value of the solver slot. */
export const newtonSolver = (gradTol = 1e-10): SpaceValue => smoothSolverConfig({ kind: 'newton', gradTol })

/** The point a solver reached. */
export interface Solved {
  readonly x: Float64Array
  readonly value: number
  readonly gradNorm: number
  readonly converged: boolean
  readonly steps: number
}

/** Minimise the problem with the configured solver of `smoothSolvers`. */
export function solveSmooth(
  problem: SmoothProblem,
  solver: SpaceValue,
  key: Key = makeKey(0),
  maxSteps = 500,
  who = 'solveSmooth',
): Solved {
  const alg = smoothSolvers.make(problem, solver, key)
  const run = runAlgorithm(alg, { key, maxSteps })
  const r = run.state as SolverReached
  if (!r.x.every(Number.isFinite)) throw new MdmlError('value', `${who}: the solver diverged`)
  return {
    x: r.x,
    value: r.value,
    gradNorm: dnrm2(r.grad),
    converged: run.status === 'converged',
    steps: run.steps,
  }
}
