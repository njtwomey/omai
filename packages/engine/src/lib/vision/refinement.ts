/**
 * The stepped form of the jointly refined geometry (calibration, bundle adjustment): `optim`'s Levenberg-Marquardt
 * algorithm wrapped so that each step also decodes the parameter vector into the geometry it describes and reports the
 * reprojection error, and the whole state serialises.
 */
import type { Json } from '../base/index.ts'
import { type LevenbergMarquardtState, levenbergMarquardtAlgorithm } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import type { Value } from '../tensor/index.ts'
import { type Algorithm, type StepContext, stateFromJson, stateToJson } from '../trace/index.ts'

/** The paths every stepped refinement logs. */
export type RefinementPath = 'refine/cost' | 'refine/reprojectionError' | 'refine/parameters'

/** A refinement's state: the solver's, the geometry decoded from its iterate, the error and the iteration. */
export interface RefinementState<D> {
  readonly solver: LevenbergMarquardtState
  readonly decoded: D
  /** The root-mean-square reprojection error per point at the iterate, in pixels. */
  readonly rms: number
  /** Levenberg-Marquardt iterations taken. */
  readonly iteration: number
}

/**
 * Levenberg-Marquardt over reprojection residuals as a refinement algorithm, one iteration per step. The residuals are
 * flat pairs of coordinates, so the RMS error per point is `sqrt(4 cost / m)` for `m` residuals.
 *
 * @param name - The algorithm's name.
 * @param config - Its configuration, for the trace.
 * @param residuals - The residual vector of a parameter vector (differentiable).
 * @param start - The starting parameters.
 * @param decode - The geometry a parameter vector describes.
 * @param maxIterations - The iterations after which the run completes.
 * @returns The algorithm.
 * @concept bundle-adjustment
 * @example
 * ```ts
 * import { steppedRefinement } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { tensor, sub } from 'mdml-engine/tensor'
 * // Fit a point to two observations: the RMS error per point settles at its residual.
 * const alg = steppedRefinement('point', {}, (p) => sub(p, tensor([1, 2])), [0, 0], (x) => Array.from(x), 10)
 * Array.from(runAlgorithm(alg, { maxSteps: 10 }).state.decoded).map(Math.round) // => [1, 2]
 * ```
 */
export function steppedRefinement<D>(
  name: string,
  config: Json,
  residuals: (p: Value) => Value,
  start: readonly number[],
  decode: (x: Float64Array) => D,
  maxIterations: number,
): Algorithm<RefinementState<D>, RefinementPath> {
  const inner = levenbergMarquardtAlgorithm(residuals, [...start], { ftol: 1e-15, xtol: 1e-15, gradTol: 1e-15 })
  const view = (solver: LevenbergMarquardtState, iteration: number): RefinementState<D> => ({
    solver,
    decoded: decode(solver.x),
    rms: Math.sqrt((4 * solver.value) / Math.max(1, solver.residuals.length)),
    iteration,
  })
  return {
    name,
    config,
    init: (key: Key) => view(inner.init(key), 0),
    step: (s, ctx) => {
      const solver = inner.step(s.solver, ctx as unknown as StepContext)
      const next = view(solver, s.iteration + 1)
      ctx.log('refine/cost', solver.value)
      ctx.log('refine/reprojectionError', next.rms)
      ctx.log('refine/parameters', solver.x)
      return next
    },
    done: (s, t) => (s.iteration >= maxIterations ? 'completed' : inner.done?.(s.solver, t)),
    observe: (s) => ({
      metrics: { cost: s.solver.value, reprojectionError: s.rms },
      tensors: { parameters: { role: 'param', value: s.solver.x } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RefinementState<D>,
  }
}
