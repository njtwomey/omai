// Private: the solver slot of the estimators of this folder whose fit minimises a smooth objective, with the closed
// form as one more way to reach the same optimum. A model states its objective as a `SmoothProblem`; this runs a solver
// of `smoothSolvers` on it (Newton by default), or, when the solver is `'closed-form'`, takes the one step that solves
// the normal equations directly. Either way the state holds the point reached in the solver's own shape, so the
// estimator reads the parameters the same way and `solver/gradNorm` and `solver/step` are logged for both.
import type { Json } from '../base/index.ts'
import { slotAlgorithm } from '../linear/_slot.ts'
import { analyticObjective, type SmoothProblem, type SolverReached } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import type { Algorithm, Observation, TraceSchema } from '../trace/index.ts'

export { infNorm, newtonAt, type SlotState } from '../linear/_slot.ts'

/** A solver of `smoothSolvers`, or the closed-form solution as one more mode of the slot. */
export type SolverChoice = SpaceValue | 'closed-form'

/** A smooth objective with its analytic derivatives. */
export interface Smooth {
  value(x: ArrayLike<number>): number
  gradient(x: ArrayLike<number>): ArrayLike<number>
  hessian?(x: ArrayLike<number>): ArrayLike<number>
}

/** The smooth problem of an analytic objective from a start. */
export function problemOf(f: Smooth, x0: ArrayLike<number>): SmoothProblem {
  return {
    objective: analyticObjective({
      value: (x) => f.value(x),
      gradient: (x) => f.gradient(x),
      ...(f.hessian === undefined ? {} : { hessian: (x: ArrayLike<number>) => f.hessian?.(x) as ArrayLike<number> }),
    }),
    x0: Array.from(x0),
  }
}

/** What {@link slotted} takes: a model's problem, solver choice and views of the point reached. */
export interface SlottedSpec<S extends { readonly solver: unknown }> {
  readonly name: string
  readonly config: Json
  readonly schema: Pick<TraceSchema, 'metrics' | 'tensors'>
  readonly problem: SmoothProblem
  readonly smooth: Smooth
  readonly solver: SolverChoice
  /** The closed-form point (used when the solver is `'closed-form'`). */
  readonly closed?: () => ArrayLike<number>
  readonly key?: Key
  readonly tol: number
  view(r: SolverReached, evaluations: number, stalled: boolean): Omit<S, 'solver'>
  observe(s: S): Observation
}

/**
 * A model's algorithm over the slot: a solver of `smoothSolvers`, or one closed-form step.
 *
 * @param spec - The problem, the solver choice and the model's views.
 * @returns The algorithm.
 */
export function slotted<S extends { readonly solver: unknown }>(spec: SlottedSpec<S>): Algorithm<S> {
  const common = {
    name: spec.name,
    config: spec.config,
    schema: spec.schema,
    problem: spec.problem,
    tol: spec.tol,
    ...(spec.key === undefined ? {} : { key: spec.key }),
    view: spec.view,
    observe: spec.observe,
  }
  if (spec.solver !== 'closed-form') {
    return slotAlgorithm({ ...common, solver: spec.solver } as never) as unknown as Algorithm<S>
  }
  if (spec.closed === undefined) throw new Error(`${spec.name}: no closed form for this configuration`)
  const closed = spec.closed
  const base = slotAlgorithm({ ...common, solver: { kind: 'newton' } } as never) as unknown as Algorithm<S>
  const reachedAt = (x: ArrayLike<number>, step: number): SolverReached => ({
    x: Float64Array.from(x),
    value: spec.smooth.value(x),
    grad: Float64Array.from(spec.smooth.gradient(x)),
    step,
  })
  const wrap = (r: SolverReached): S => ({ ...spec.view(r, 1, false), solver: r }) as S
  return {
    ...base,
    config: { ...(spec.config as object), solver: 'closed-form' } as Json,
    init: () => wrap(reachedAt(spec.problem.x0 as ArrayLike<number>, 0)),
    step: () => wrap(reachedAt(closed(), 1)),
    done: (_s, t) => (t >= 1 ? 'converged' : undefined),
  }
}
