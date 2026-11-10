/**
 * The Laplace approximation as an inference method in two stages: the optimiser's iterations to the MAP point of the
 * unconstrained log density (L-BFGS, one iteration per step, with the log density and the gradient norm on the tape),
 * then one step for the Hessian and the evidence. While it optimises, `posterior` is the point estimate at the current
 * iterate; after the Hessian it is the normal approximation of the one-call `laplaceApproximation` (unchanged, with its
 * planner entry). The state serialises, so a run resumes and forks.
 */

import type { Json } from '../../base/index.ts'
import { compose } from '../../dists/index.ts'
import { infNorm, type LbfgsState, lbfgsAlgorithm } from '../../optim/index.ts'
import { foldIn } from '../../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { fromStorage, neg, primalOf, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import {
  findInitialParams,
  type InitStrategy,
  initToMean,
  initToMedian,
  initToSample,
  initToUniform,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { type LaplaceFit, laplaceAt, laplaceEngine, laplacePosterior } from '../engines/laplace.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  checkOf,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'

/** The paths the Laplace method logs. */
export type LaplacePath =
  | 'laplace/stage'
  | 'laplace/logDensity'
  | 'laplace/gradNorm'
  | 'laplace/iteration'
  | 'laplace/logEvidence'
  | 'laplace/mode'
  | 'laplace/snapshot'

/** The state of {@link laplaceMethod}. */
export interface LaplaceRunState {
  /** `init` (find a starting point), `optimise` (L-BFGS iterations), `curvature` (the Hessian), `done`. */
  readonly stage: 'init' | 'optimise' | 'curvature' | 'done'
  readonly opt: LbfgsState | null
  readonly iteration: number
  readonly fit: LaplaceFit | null
  readonly failure: string | null
}

/** The knobs of {@link laplaceMethod} (the budget and snapshot knobs are added to them). */
export const laplaceKnobs: DictSpace = dict({
  maxIterations: int(1, 1_000_000, {
    default: 1000,
    label: 'iteration cap',
    description: 'L-BFGS iterations before the Hessian is taken',
  }),
  gradTol: real(1e-14, 1e-2, { log: true, default: 1e-10, label: 'gradient tolerance' }),
  init: choice(['median', 'mean', 'sample', 'uniform'], { default: 'median', label: 'starting point' }),
})

const initFor = (name: string): InitStrategy =>
  name === 'mean'
    ? initToMean()
    : name === 'sample'
      ? initToSample()
      : name === 'uniform'
        ? initToUniform()
        : initToMedian()

const failedPosterior = (reason: string): Posterior =>
  definePosterior({
    form: 'variational',
    engine: 'laplace',
    sites: [],
    status: 'failed',
    draws: {},
    diagnostics: { reason },
  })

/**
 * The Laplace approximation on a model as a method with a fine step: L-BFGS to the MAP point of the unconstrained density
 * one iteration per step, then one step for the Hessian. The run ends `converged` (or `completed` at the iteration cap) after
 * the Hessian and `failed` when no starting point is valid or minus the Hessian is not positive definite. `posterior(state)`
 * is the point estimate at the current iterate while optimising and the normal approximation, with the Laplace evidence, after.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method, with `check` (a likelihood, continuous latent sites and a differentiable density) and `posterior`.
 * @concept laplace-approximation
 * @index estimator The Laplace approximation, optimiser iterations then the Hessian, as a steppable method.
 * @example
 * ```ts
 * import { laplaceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const method = laplaceMethod(model)
 * const s = session(method, { steps: 200, mode: 'eager' })
 * Math.abs(item(method.posterior?.(s.state, s.config).variance('m') ?? 0) - 0.5) < 1e-8 // => true
 * ```
 */
export function laplaceMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<LaplaceRunState, Config, LaplacePath, ModelAnalysis> {
  const program = programOf(model)
  let umCache: UnconstrainedModel | undefined
  const unconstrained = (): UnconstrainedModel => {
    umCache ??= potentialEnergy(program, undefined)
    return umCache
  }
  const posterior = (state: LaplaceRunState): Posterior => {
    if (state.failure !== null) return failedPosterior(state.failure)
    if (state.fit !== null) return laplacePosterior(unconstrained(), state.fit)
    if (state.opt === null) return failedPosterior('no starting point yet')
    const um = unconstrained()
    const values = um.constrain(state.opt.x)
    return definePosterior({
      form: 'point',
      engine: 'laplace',
      sites: Object.keys(values),
      status: 'stopped',
      marginals: Object.fromEntries(
        Object.entries(values).map(([name, v]) => [name, compose.deterministic(primalOf(v))]),
      ),
      diagnostics: {
        partial: true,
        iteration: state.iteration,
        logDensity: -state.opt.value,
        gradNorm: infNorm(state.opt.grad),
      },
    })
  }
  const check = checkOf(laplaceEngine(), (a) => {
    if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
    if (a.continuous.length === 0) return no('no continuous latent nodes', ['continuous latents'], 'use enumeration')
    if (a.unboundedDiscrete.length > 0)
      return no(
        `unbounded discrete latents (${a.unboundedDiscrete.join(', ')})`,
        ['finite discrete latents'],
        'bound them',
      )
    return yes('continuous latents with a differentiable density (the Hessian must be negative definite at the mode)')
  })
  const ensure = guard(options.name ?? 'laplace', model, check)
  return defineInferenceMethod<LaplaceRunState, Config, LaplacePath, ModelAnalysis>({
    name: options.name ?? 'laplace',
    family: 'variational',
    description: 'the Laplace approximation: L-BFGS to the mode, one iteration per step, then the Hessian',
    config: laplaceKnobs,
    check,
    posterior,
    defaults: { 'laplace/logDensity': Log(), 'laplace/gradNorm': Log() },
    contract: (): Contract<LaplaceRunState, LaplacePath> =>
      defineContract({
        'laplace/stage': scalar({ description: '1 while optimising, 2 at the Hessian' }),
        'laplace/logDensity': scalar({ description: 'the log density at the iterate', direction: 'higher' }),
        'laplace/gradNorm': scalar({
          description: 'the infinity norm of the gradient at the iterate',
          direction: 'lower',
        }),
        'laplace/iteration': scalar({ description: 'optimiser iterations done' }),
        'laplace/logEvidence': scalar({
          description: 'the Laplace estimate of log p(data), once the Hessian is taken',
        }),
        'laplace/mode': tensor({
          role: 'param',
          shape: ['dim'],
          description: 'the iterate, on the unconstrained scale',
        }),
        'laplace/snapshot': derived<LaplaceRunState>({
          cost: 'dear',
          description:
            'the posterior of the state as JSON: the point estimate, or the normal approximation after the Hessian',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config, key): Algorithm<LaplaceRunState, LaplacePath> => {
      const um = unconstrained()
      const maxIterations = num(config, 'maxIterations')
      const gradTol = num(config, 'gradTol')
      const f = (u: Value): Value => neg(um.fn(u))
      const lbfgsOf = (start: Float64Array): ReturnType<typeof lbfgsAlgorithm> =>
        lbfgsAlgorithm(f, fromStorage(start, [um.dim], 'float64'), { gradTol, ftol: 0 })
      // Stepping and serialising an L-BFGS state do not depend on the start, so one algorithm serves the run.
      const shared = lbfgsOf(new Float64Array(um.dim))
      void key
      return {
        name: 'probprog/laplace-method',
        config: { maxIterations, gradTol },
        init: () => {
          ensure()
          return { stage: 'init', opt: null, iteration: 0, fit: null, failure: null }
        },
        step: (s, ctx) => {
          if (s.stage === 'init') {
            const init = findInitialParams(um, program, undefined, {
              key: foldIn(ctx.key, 1),
              strategy: initFor(pick(config, 'init')),
            })
            if (!init.valid)
              return { ...s, stage: 'done', failure: 'no starting point with a finite log density and gradient' }
            return { ...s, stage: 'optimise', opt: lbfgsOf(init.unconstrained).init(foldIn(ctx.key, 2)) }
          }
          if (s.stage === 'optimise') {
            const opt = shared.step(
              s.opt as LbfgsState,
              { ...ctx, log: () => undefined, wants: () => false, emit: () => undefined } as never,
            )
            const iteration = s.iteration + 1
            ctx.log('laplace/stage', 1)
            ctx.log('laplace/logDensity', -opt.value)
            ctx.log('laplace/gradNorm', infNorm(opt.grad))
            ctx.log('laplace/iteration', iteration)
            if (ctx.wants('laplace/mode')) ctx.log('laplace/mode', opt.x)
            if (!Number.isFinite(opt.value))
              return { ...s, opt, iteration, stage: 'done', failure: 'the log density is not finite' }
            const finished = opt.stop !== undefined || iteration >= maxIterations
            return { ...s, opt, iteration, stage: finished ? 'curvature' : 'optimise' }
          }
          // The Hessian at the point the optimiser reached.
          const opt = s.opt as LbfgsState
          ctx.log('laplace/stage', 2)
          try {
            const fit = laplaceAt(um, opt.x, {
              logDensity: -opt.value,
              gradNorm: infNorm(opt.grad),
              iterations: s.iteration,
              converged: opt.stop?.status === 'converged',
            })
            const evidence = laplacePosterior(um, fit).logEvidence?.value
            if (evidence !== undefined && Number.isFinite(evidence)) ctx.log('laplace/logEvidence', evidence)
            return { ...s, stage: 'done', fit }
          } catch (e) {
            return { ...s, stage: 'done', failure: e instanceof Error ? e.message : String(e) }
          }
        },
        done: (s) =>
          s.stage !== 'done'
            ? undefined
            : s.failure !== null
              ? 'failed'
              : s.fit?.converged === true
                ? 'converged'
                : 'completed',
        serialise: (s) => ({
          stage: s.stage,
          iteration: s.iteration,
          failure: s.failure,
          opt: s.opt === null ? null : stateToJson({ ...s.opt, lineSearch: undefined, curvature: undefined }),
          fit: s.fit === null ? null : stateToJson(s.fit),
        }),
        deserialise: (j: Json) => {
          const o = j as unknown as {
            stage: LaplaceRunState['stage']
            iteration: number
            failure: string | null
            opt: Json | null
            fit: Json | null
          }
          return {
            stage: o.stage,
            iteration: o.iteration,
            failure: o.failure,
            opt: o.opt === null ? null : (stateFromJson(o.opt) as LbfgsState),
            fit: o.fit === null ? null : (stateFromJson(o.fit) as LaplaceFit),
          }
        },
      }
    },
  })
}
