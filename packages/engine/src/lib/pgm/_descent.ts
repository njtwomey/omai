// Minimising an objective with an analytic gradient by a gradient transformation from optim, as a traced algorithm
// (the training loop of the CRFs and the inner loop of NOTEARS; not re-exported).
import type { Json } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { applyUpdates, type GradientTransformation } from '../optim/index.ts'
import { fromStorage, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels } from '../trace/index.ts'

/** The state of a descent: the parameters, the optimiser state, the objective and its gradient. */
export interface DescentState {
  readonly parameters: Float64Array
  readonly optState: unknown
  readonly value: number
  readonly gradient: Float64Array
  readonly gradNorm: number
}

const vector = (w: Float64Array): Tensor => fromStorage(Float64Array.from(w), [w.length], 'float64')

/**
 * Minimise an objective over a parameter vector with a gradient transformation from `optim`, as an {@link Algorithm}
 * that records the objective and the gradient norm (the training loop shared by both kinds of CRF).
 */
export function gradientDescent(
  name: string,
  config: Json,
  start: Float64Array,
  objective: (w: Float64Array) => { value: number; gradient: Float64Array },
  method: GradientTransformation,
  gradTol: number,
): Algorithm<DescentState> {
  const evaluate = (w: Float64Array, optState: unknown): DescentState => {
    const { value, gradient } = objective(w)
    return { parameters: w, optState, value, gradient, gradNorm: dnrm2(gradient) }
  }
  return {
    name,
    config,
    schema: channels({
      metrics: {
        objective: { description: 'penalised negative conditional log-likelihood', direction: 'lower' },
        gradNorm: { description: 'gradient norm', direction: 'lower' },
      },
    }),
    init: () => evaluate(start, method.init(vector(start))),
    step: (s) => {
      const { updates, state } = method.update(vector(s.gradient), s.optState, vector(s.parameters))
      const next = Float64Array.from(toFlat(applyUpdates(vector(s.parameters), updates) as Value))
      return evaluate(next, state)
    },
    done: (s) => (s.gradNorm <= gradTol ? 'converged' : !Number.isFinite(s.value) ? 'diverged' : undefined),
    observe: (s) => ({ metrics: { objective: s.value, gradNorm: s.gradNorm } }),
  }
}
