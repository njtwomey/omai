/**
 * Pathfinder as an inference method: one L-BFGS iteration of every path per step, the best ELBO and each path's latest on the
 * tape, a state that serialises (so a run resumes, seeks and forks) and `posterior(state)` reading the normal approximation of
 * the best ELBO so far (or, at the end, the importance-weighted draws), which is what an anytime snapshot shows while the
 * paths climb. Paths, iterations, memory, the draws of the ELBO and of the result and the weighting are knobs.
 */

import { foldIn } from '../../random/index.ts'
import { choice, dict, int } from '../../space/index.ts'
import { type Config, type Contract, defineContract, derived, Log, scalar } from '../../trace/index.ts'
import { type PathfinderState, pathfinderAlgorithm, pathfinderPosterior } from '../engines/pathfinder.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  guard,
  type MethodModel,
  type ModelAnalysis,
  num,
  pick,
  posteriorJson,
  programOf,
  restorable,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'
import { sviMethod } from './svi-method.ts'

/** The paths the Pathfinder method logs. */
export type PathfinderPath =
  | 'pathfinder/elbo'
  | 'pathfinder/iterate'
  | `pathfinder/path/${string}`
  | 'pathfinder/paretoK'
  | 'pathfinder/ess'
  | 'pathfinder/snapshot'

/**
 * Pathfinder (Zhang et al. 2022) as a steppable, resumable method: L-BFGS paths from random starts, a normal approximation
 * at every iterate with its Monte Carlo ELBO, the best one kept, and, when the paths end, importance draws from the mixture of
 * the paths' best normals with Pareto-smoothed weights and their k-hat. Knobs: `paths`, `maxIterations`, `memory`, `elboDraws`,
 * `numDraws`, `weights` (`psis`, `resample`, `none`) and `init`. The tape: `pathfinder/elbo` (best so far),
 * `pathfinder/iterate` (the furthest iteration), `pathfinder/path/<i>` (the latest iterate's ELBO per path), and at the end
 * `pathfinder/paretoK` and `pathfinder/ess`.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name.
 * @returns The method; `check` needs continuous latents and a differentiable density, as SVI does.
 * @concept variational-inference
 * @index estimator Pathfinder: L-BFGS path of normal approximations, best ELBO, PSIS importance draws, multi-path.
 * @example
 * ```ts
 * import { pathfinderMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const method = pathfinderMethod(model)
 * const s = session(method, { steps: 60, mode: 'eager', config: { maxIterations: 10 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 1) < 0.1 // => true
 * ```
 */
export function pathfinderMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<PathfinderState, Config, PathfinderPath, ModelAnalysis> {
  const program = programOf(model)
  const name = options.name ?? 'pathfinder'
  const check = sviMethod(model).check
  const ensure = guard(name, model, check)
  const posterior = (state: PathfinderState): Posterior => {
    try {
      return pathfinderPosterior(program, undefined, state)
    } catch (e) {
      return definePosterior({
        form: 'variational',
        engine: 'pathfinder',
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: e instanceof Error ? e.message : String(e) },
      })
    }
  }
  return defineInferenceMethod<PathfinderState, Config, PathfinderPath, ModelAnalysis>({
    name,
    family: 'variational',
    description: 'Pathfinder: normal approximations along an L-BFGS path, the best ELBO, importance draws',
    config: dict({
      paths: int(1, 64, { default: 1, label: 'paths', description: 'independent L-BFGS paths from different starts' }),
      maxIterations: int(1, 1000, { default: 50, label: 'iterations', description: 'L-BFGS iterations per path' }),
      memory: int(1, 100, {
        default: 10,
        label: 'memory',
        description: 'curvature pairs kept (the rank of the covariance correction)',
      }),
      elboDraws: int(2, 1000, {
        default: 50,
        label: 'ELBO draws',
        description: "draws of each iterate's ELBO estimate",
      }),
      numDraws: int(25, 100_000, {
        default: 400,
        label: 'draws',
        description: 'importance draws of the final approximation',
      }),
      weights: choice(['psis', 'resample', 'none'], {
        default: 'psis',
        label: 'weights',
        description: 'Pareto-smoothed importance weights, equal-weight resampling from them, or the unweighted draws',
      }),
      init: choice(['uniform', 'median'], {
        default: 'uniform',
        label: 'start',
        description: 'uniform on (-2, 2) or the median of the prior (first path)',
      }),
    }),
    ...(check === undefined ? {} : { check }),
    posterior,
    defaults: { 'pathfinder/elbo': Log(), 'pathfinder/iterate': Log() },
    contract: (): Contract<PathfinderState, PathfinderPath> =>
      defineContract({
        'pathfinder/elbo': scalar({
          description: 'the best ELBO over every path and iterate so far',
          direction: 'higher',
        }),
        'pathfinder/iterate': scalar({ description: 'the furthest L-BFGS iteration of any path' }),
        'pathfinder/path/*': scalar({
          description: 'the ELBO of the normal at the latest iterate of one path',
          direction: 'higher',
        }),
        'pathfinder/paretoK': scalar({
          description: 'the Pareto k-hat of the importance weights of the final approximation',
          direction: 'lower',
        }),
        'pathfinder/ess': scalar({ description: 'the effective sample size of the importance weights' }),
        'pathfinder/snapshot': derived<PathfinderState>({
          cost: 'dear',
          description: 'the approximation so far as JSON: means, sds, quantiles',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config, key) => {
      const alg = restorable(
        pathfinderAlgorithm(program, undefined, {
          paths: num(config, 'paths'),
          maxIterations: num(config, 'maxIterations'),
          memory: num(config, 'memory'),
          elboDraws: num(config, 'elboDraws'),
          numDraws: num(config, 'numDraws'),
          weights: pick(config, 'weights') as 'psis' | 'resample' | 'none',
          init: pick(config, 'init') as 'uniform' | 'median',
        }),
        foldIn(key, 0),
      )
      return {
        ...alg,
        init: (k) => {
          ensure()
          return alg.init(k)
        },
        step: (s, ctx) => {
          const next = alg.step(s, ctx)
          if (Number.isFinite(next.elbo)) ctx.log('pathfinder/elbo', next.elbo)
          ctx.log('pathfinder/iterate', Math.max(...next.paths.map((p) => p.iteration)))
          next.paths.forEach((p, i) => {
            const path = `pathfinder/path/${i}` as const
            if (Number.isFinite(p.latest) && ctx.wants(path)) ctx.log(path, p.latest)
          })
          if (Number.isFinite(next.paretoK)) ctx.log('pathfinder/paretoK', next.paretoK)
          if (Number.isFinite(next.ess)) ctx.log('pathfinder/ess', next.ess)
          return next
        },
      }
    },
  })
}
