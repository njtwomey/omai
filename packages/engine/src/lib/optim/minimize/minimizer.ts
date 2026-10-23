import { valueAndGrad } from '../../autodiff/index.ts'
import type { Json } from '../../base/index.ts'
import { item, treePaths, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { adam } from '../adaptive.ts'
import { applyUpdates, type GradientTransformation, globalNorm, type Tree } from '../transform.ts'

/** The state of a {@link minimizer} run. */
export interface MinimizerState {
  readonly params: Tree
  readonly optState: unknown
  readonly value: number
  readonly grad: Tree
  readonly gradNorm: number
}

/** Options of {@link minimizer} and {@link minimize}. */
export interface MinimizerOptions {
  /** Stop with status `converged` once the gradient norm falls below this. */
  readonly gradTol?: number
}

/**
 * A gradient transformation driving an objective, as an {@link Algorithm}: each step evaluates the objective and its
 * gradient (reverse mode) at the current parameters and applies the transformation's updates. Records the objective,
 * the gradient norm and (by policy) the parameters and gradients.
 *
 * @param f - The objective, a function of a parameter tree returning a scalar.
 * @param params - Initial parameters.
 * @param transformation - The optimiser.
 * @param options - Stopping tolerance.
 * @returns The algorithm.
 * @concept optimisers
 * @example
 * ```ts
 * import { minimizer, sgd } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * import { square, sum, tensor } from 'whitebox-engine/tensor'
 * const alg = minimizer((x) => sum(square(x as never)), tensor([3]), sgd({ learningRate: 0.25 }))
 * runAlgorithm(alg, { maxSteps: 100 }).state.value < 1e-20 // => true
 * ```
 */
export function minimizer(
  f: (params: Tree) => Value,
  params: Tree,
  transformation: GradientTransformation,
  options: MinimizerOptions = {},
): Algorithm<MinimizerState> {
  const vg = valueAndGrad(f)
  const evaluate = (p: Tree, optState: unknown): MinimizerState => {
    const { value, grad } = vg(p)
    return { params: p, optState, value: item(value), grad, gradNorm: item(globalNorm(grad)) }
  }
  const tol = options.gradTol ?? 0
  return {
    name: 'minimizer',
    config: { transformation: transformation.name, gradTol: tol } as Json,
    schema: channels({
      metrics: { loss: { description: 'objective value' }, gradNorm: { description: 'global gradient norm' } },
      tensors: {
        params: { role: 'param', description: 'parameters' },
        grad: { role: 'grad', description: 'gradient' },
      },
    }),
    init: () => evaluate(params, transformation.init(params)),
    step: (s) => {
      const { updates, state } = transformation.update(s.grad, s.optState, s.params)
      return evaluate(applyUpdates(s.params, updates), state)
    },
    done: (s) => (s.gradNorm <= tol ? 'converged' : !Number.isFinite(s.value) ? 'diverged' : undefined),
    observe: (s) => {
      const tensors: Record<string, { role: 'param' | 'grad'; value: Value }> = {}
      for (const [path, v] of Object.entries(treePaths(s.params)))
        tensors[`params.${path}`] = { role: 'param', value: v }
      for (const [path, v] of Object.entries(treePaths(s.grad))) tensors[`grad.${path}`] = { role: 'grad', value: v }
      return { metrics: { loss: s.value, gradNorm: s.gradNorm }, tensors }
    },
  }
}

/** Result of {@link minimize}. */
export interface MinimizeResult {
  readonly x: Tree
  readonly fun: number
  readonly gradNorm: number
  readonly steps: number
  readonly status: string
  readonly converged: boolean
}

/**
 * Minimise an objective with a gradient transformation (default Adam with learning rate 0.01), for up to `maxSteps`
 * steps or until the gradient norm falls below `gradTol`.
 *
 * @param f - The objective.
 * @param x0 - Initial parameters (any tree).
 * @param options - The method, steps, tolerance and the run options (tracer, policy, seed).
 * @returns The result, scipy-style.
 * @example
 * ```ts
 * import { minimize, sgd } from 'whitebox-engine/optim'
 * import { square, sub, sum, tensor, toArray } from 'whitebox-engine/tensor'
 * const r = minimize((x) => sum(square(sub(x as never, 2))), tensor([0, 5]), { method: sgd({ learningRate: 0.4 }), maxSteps: 200, gradTol: 1e-10 })
 * r.converged // => true
 * ```
 */
export function minimize(
  f: (params: Tree) => Value,
  x0: Tree,
  options: MinimizerOptions &
    Partial<Omit<RunOptions, 'maxSteps'>> & {
      readonly method?: GradientTransformation
      readonly maxSteps?: number
    } = {},
): MinimizeResult {
  const method = options.method ?? adam({ learningRate: 0.01 })
  const { method: _m, gradTol, maxSteps, ...run } = options
  const result = runAlgorithm(minimizer(f, x0, method, gradTol === undefined ? {} : { gradTol }), {
    ...run,
    maxSteps: maxSteps ?? 1000,
  })
  const s = result.state
  return {
    x: s.params,
    fun: s.value,
    gradNorm: s.gradNorm,
    steps: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}
