/**
 * The bridge between gradient transformations and minimisation: `minimizer` turns an optax-style transformation into an
 * `Algorithm` that evaluates an objective and its gradient at each step, and `minimize` is the scipy-style front end
 * that runs either a transformation or a named method (BFGS by default) on any parameter tree.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import type { Json } from '../../base/index.ts'
import { item, treePaths, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { applyUpdates, type GradientTransformation, globalNorm, type Tree } from '../transform.ts'
import type { MinimizeMethod, MinimizeResult, Objective, Parameters } from './objective.ts'
import { type BfgsOptions, bfgs } from './quasi-newton.ts'

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

/** Options of {@link minimize} with a gradient transformation. */
export interface TransformationMinimizeOptions
  extends MinimizerOptions,
    Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  readonly method?: GradientTransformation
  /** Steps to take (default 1000); `maxIterations` is accepted as a synonym. */
  readonly maxSteps?: number
  readonly maxIterations?: number
}

function isTransformation(m: unknown): m is GradientTransformation {
  return typeof m === 'object' && m !== null && 'update' in m && 'init' in m
}

/**
 * Minimise an objective, scipy-style. The method is a function value: a vector method such as `bfgs`, `lbfgs`,
 * `newtonCg` or `nelderMead` (called as `method(f, x0, options)` with the remaining options), or a gradient
 * transformation such as `adam({ learningRate: 0.01 })`, run for up to `maxSteps` steps or until the gradient norm
 * falls below `gradTol`. Without a method, BFGS runs (scipy's default for unconstrained problems).
 *
 * @param f - The objective.
 * @param x0 - Initial parameters (any tree, or anything tensor-like for vector methods).
 * @param options - The method and its options, and the run options (tracer, policy, seed).
 * @returns The result.
 * @example
 * ```ts
 * import { minimize, sgd } from 'whitebox-engine/optim'
 * import { square, sub, sum, tensor, toArray } from 'whitebox-engine/tensor'
 * const r = minimize((x) => sum(square(sub(x as never, 2))), tensor([0, 5]), { method: sgd({ learningRate: 0.4 }), maxSteps: 200, gradTol: 1e-10 })
 * r.converged // => true
 * ```
 */
export function minimize<O extends object>(
  f: Objective,
  x0: Parameters,
  options: O & { readonly method: MinimizeMethod<O> },
): MinimizeResult
export function minimize(f: Objective, x0: Tree, options?: TransformationMinimizeOptions): MinimizeResult
export function minimize(f: Objective, x0: Parameters, options: object = {}): MinimizeResult {
  const { method, ...rest } = options as { readonly method?: unknown }
  if (typeof method === 'function') return (method as MinimizeMethod<object>)(f, x0, rest)
  if (!isTransformation(method)) {
    const { maxSteps, ...others } = rest as TransformationMinimizeOptions
    return bfgs(f, x0, {
      ...(others as BfgsOptions),
      ...(maxSteps === undefined || others.maxIterations !== undefined ? {} : { maxIterations: maxSteps }),
    })
  }
  const transformation = method
  const { gradTol, maxSteps, maxIterations, ...run } = rest as TransformationMinimizeOptions
  const result = runAlgorithm(minimizer(f, x0, transformation, gradTol === undefined ? {} : { gradTol }), {
    ...run,
    maxSteps: maxSteps ?? maxIterations ?? 1000,
  })
  const s = result.state
  return {
    x: s.params,
    fun: s.value,
    grad: s.grad,
    gradNorm: s.gradNorm,
    steps: result.steps,
    nfev: result.steps + 1,
    ngev: result.steps + 1,
    nhev: 0,
    status: result.status,
    converged: result.status === 'converged',
    message:
      result.status === 'converged'
        ? 'gradient norm below tolerance'
        : result.status === 'completed'
          ? 'maximum number of steps reached'
          : `stopped: ${result.status}`,
  }
}
