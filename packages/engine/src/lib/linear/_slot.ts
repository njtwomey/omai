/**
 * The solver slot of the maximum-likelihood fits of this folder (multinomial, ordinal, count and mixed models): each
 * model states its objective as a `SmoothProblem` and this wraps a solver of `smoothSolvers` as the model's algorithm,
 * with the model's own state fields (the parameters, the objective, the largest gradient entry) read off the solver's
 * state, the solver's state kept inside so any solver (Newton, BFGS, L-BFGS, conjugate gradient, gradient descent,
 * stochastic gradient) can be stepped, serialised and resumed, and `solver/gradNorm` and `solver/step` beside the
 * model's channels, as `loggedSolver` logs them.
 */

import type { Json } from '../base/index.ts'
import { type SmoothProblem, type SolverReached, smoothSolverConfig, smoothSolvers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  channels,
  type Observation,
  stateFromJson,
  stateToJson,
  type TraceSchema,
} from '../trace/index.ts'

/** What a slotted model's state holds besides its own fields: the state of the solver. */
export interface SlotState {
  readonly solver: unknown
}

/** The Newton solver at a gradient tolerance: the default of the models whose goldens it reproduces. */
export const newtonAt = (gradTol: number): SpaceValue => smoothSolverConfig({ kind: 'newton', gradTol })

const reached = (s: SlotState): SolverReached => s.solver as SolverReached

/** The largest absolute entry. */
export const infNorm = (v: ArrayLike<number>): number => {
  let m = 0
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i] ?? 0))
  return m
}

/** What {@link slotAlgorithm} takes. */
export interface SlotSpec<S extends SlotState> {
  readonly name: string
  readonly config: Json
  /** The model's channels; `solver/gradNorm` and `solver/step` are added. */
  readonly schema: Pick<TraceSchema, 'metrics' | 'tensors'>
  readonly problem: SmoothProblem
  /** A value of `smoothSolvers.space`. */
  readonly solver: SpaceValue
  readonly key?: Key
  /** Stop with `converged` once the largest gradient entry is at most this, whichever the solver. */
  readonly tol: number
  /** The model's state fields from the solver's channels; `evaluations` counts the solver's function evaluations. */
  view(r: SolverReached, evaluations: number, stalled: boolean): Omit<S, 'solver'>
  /** The model's observation of its state. */
  observe(s: S): Observation
}

/**
 * A model's algorithm over a solver of the slot.
 *
 * @param spec - The model's objective, solver, state view and channels.
 * @returns The algorithm, whose state is the model's fields and the solver's state.
 */
export function slotAlgorithm<S extends SlotState>(spec: SlotSpec<S>): Algorithm<S> {
  const inner = smoothSolvers.make(spec.problem, spec.solver, spec.key ?? makeKey(0))
  const wrap = (solver: unknown): S => {
    const r = solver as SolverReached & { readonly nfev?: number; readonly stop?: { readonly status: string } }
    return { ...spec.view(r, r.nfev ?? 0, r.stop?.status === 'failed'), solver } as S
  }
  return {
    name: spec.name,
    config: spec.config,
    schema: channels({
      metrics: {
        ...spec.schema.metrics,
        gradNorm: { description: 'the norm of the gradient at the iterate', direction: 'lower' },
        step: { description: 'the length of the last step' },
      },
      tensors: spec.schema.tensors,
    }),
    init: (k) => wrap(inner.init(k)),
    step: (s, ctx) => wrap(inner.step(s.solver, ctx)),
    done: (s, t) => {
      const status = inner.done?.(s.solver, t)
      if (status !== undefined) return status
      return infNorm(reached(s).grad) <= spec.tol ? 'converged' : undefined
    },
    observe: (s) => {
      const own = spec.observe(s)
      const r = reached(s)
      return {
        ...own,
        metrics: { ...own.metrics, gradNorm: Math.hypot(...r.grad), step: r.step },
      }
    },
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as S,
  }
}
