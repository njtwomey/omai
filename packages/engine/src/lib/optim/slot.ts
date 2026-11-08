/**
 * The solver slot of a model that minimises a smooth objective, for models outside `linear`: a model states its
 * objective as a `SmoothProblem`; {@link familySolvers} makes the table of ways to minimise it, the model's own
 * algorithm (Johansen's eigen-solve, EM, minorise-maximise) as named modes beside every generic solver of
 * `smoothSolvers`; {@link slotAlgorithm} wraps any mode of such a table as the model's `Algorithm`, with the model's
 * own state fields read off the solver's state, `solver/gradNorm` and `solver/step` beside the model's channels, and
 * the state serialisable so any solver can be stepped, resumed and sought.
 */

import type { Json } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  channels,
  type ModeSpec,
  type ModeTable,
  modes,
  type Observation,
  stateFromJson,
  stateToJson,
  type TraceSchema,
} from '../trace/index.ts'
import { infNorm } from './minimize/objective.ts'
import {
  type SmoothProblem,
  type SolverReached,
  smoothSolverConfig,
  smoothSolvers,
  type VectorParams,
} from './solvers.ts'

/** What a slotted model's state holds besides its own fields: the state of the solver. */
export interface SlotState {
  readonly solver: unknown
}

/**
 * The Newton solver at a gradient tolerance: the default of the models whose goldens it reproduces.
 *
 * @param gradTol - The absolute gradient tolerance, on the infinity norm.
 * @returns A complete configuration of `smoothSolvers`: Newton with its other knobs at their defaults.
 * @example
 * ```ts
 * import { newtonAt } from 'omai-engine/optim'
 * newtonAt(1e-9) // => { modification: 'cholesky', beta: 0.001, gradTol: 1e-9, kind: 'newton' }
 * ```
 */
export const newtonAt = (gradTol: number): SpaceValue => smoothSolverConfig({ kind: 'newton', gradTol })

/**
 * The table of ways to minimise a model's objective: the model's own modes, then every generic solver of
 * `smoothSolvers` applied to the model's {@link SmoothProblem}. All of them yield the fitted vector.
 *
 * @param own - The model's own modes, by name (an eigen-solve, an EM or MM iteration).
 * @param toSmooth - The model's objective as a smooth problem.
 * @param options - Label, description and the default mode (the one that reproduces the reference).
 * @returns The table.
 * @example
 * ```ts
 * import { analyticObjective, familySolvers } from 'omai-engine/optim'
 * const objective = analyticObjective({ value: (x) => (x[0] ?? 0) ** 2, gradient: (x) => [2 * (x[0] ?? 0)] })
 * familySolvers({}, () => ({ objective, x0: [1] }), { default: 'newton' }).names.includes('lbfgs') // => true
 * ```
 */
export function familySolvers<Problem>(
  // biome-ignore lint/suspicious/noExplicitAny: each own mode has a state type of its own; the table erases them
  own: Readonly<Record<string, ModeSpec<Problem, any, VectorParams>>>,
  toSmooth: (problem: Problem) => SmoothProblem,
  options: { readonly label?: string; readonly description?: string; readonly default?: string } = {},
): ModeTable<Problem, VectorParams> {
  const generic: Record<string, ModeSpec<Problem, unknown, VectorParams>> = {}
  for (const name of smoothSolvers.names) {
    const base = smoothSolvers.modeOf({ kind: name })
    generic[name] = {
      description: `${base.description ?? name}, on the model's objective`,
      space: base.space,
      make: (p, k, key) => base.make(toSmooth(p), k, key),
      params: base.params,
    }
  }
  return modes<Problem, VectorParams>(
    { ...own, ...generic },
    {
      label: options.label ?? 'solver',
      description: options.description ?? "how the model's objective is minimised",
      ...(options.default === undefined ? {} : { default: options.default }),
    },
  )
}

/** What {@link slotAlgorithm} takes. */
export interface SlotSpec<S extends SlotState, Problem = SmoothProblem> {
  readonly name: string
  readonly config: Json
  /** The model's channels; `solver/gradNorm` and `solver/step` are added. */
  readonly schema: Pick<TraceSchema, 'metrics' | 'tensors'>
  readonly problem: Problem
  /** The table the solver is a mode of (default `smoothSolvers`, for a `SmoothProblem`). */
  readonly table?: ModeTable<Problem, VectorParams>
  /** A value of the table's space. */
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
 * @example
 * ```ts
 * import { type SlotState, slotAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * const algorithm = slotAlgorithm<SlotState>({
 *   name: 'bowl',
 *   config: {},
 *   schema: { metrics: {}, tensors: {} },
 *   problem: { objective: (x: Value) => sum(square(x)), x0: [3, 4] },
 *   solver: { kind: 'lbfgs' },
 *   tol: 1e-6,
 *   view: () => ({}),
 *   observe: () => ({ metrics: {} }),
 * })
 * [algorithm.name, runAlgorithm(algorithm, { seed: 0, maxSteps: 100 }).status] // => ['bowl', 'converged']
 * ```
 */
export function slotAlgorithm<S extends SlotState, Problem = SmoothProblem>(spec: SlotSpec<S, Problem>): Algorithm<S> {
  const table = (spec.table ?? smoothSolvers) as unknown as ModeTable<Problem, VectorParams>
  const inner = table.make(spec.problem, spec.solver, spec.key ?? makeKey(0))
  const reached = (s: SlotState): SolverReached => s.solver as SolverReached
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
        gradNorm: {
          description: 'the Euclidean norm of the gradient at the iterate (the stop is on its infinity norm, `tol`)',
          direction: 'lower',
        },
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
      return { ...own, metrics: { ...own.metrics, gradNorm: dnrm2(r.grad), step: r.step } }
    },
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as S,
  }
}
