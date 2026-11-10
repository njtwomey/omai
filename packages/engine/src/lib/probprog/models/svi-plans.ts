/**
 * The variational methods beyond `sviMethod` against the benchmark models' truths, as campaigns of the periodic validation
 * of their own (`svi/<model>`): Pathfinder, sticking the landing, the structured, flow and mixture guides, natural-gradient
 * SVI and conjugate natural-gradient SVI on the conjugate models (where the exact posterior is the target), the logistic
 * regression (a quadrature truth) and the two geometries that a Gaussian cannot follow (the banana ridge and the bimodal
 * density). Each plan says what it should do and why, and a method that must agree is held to the tolerances of the
 * validation kernel. Budgets scale; none of it runs in `make verify`.
 */

import type { Campaign } from '../validation/runner.ts'
import type { BenchmarkModel, MethodPlan } from './benchmark.ts'
import { benchmarkCampaigns } from './campaigns.ts'
import { gammaPoisson, normalNormal } from './conjugate.ts'
import { banana, bimodal } from './geometry.ts'
import { logisticRegression } from './glm.ts'
import { budget } from './plans.ts'

type Expect = MethodPlan['expect']

const plan = (
  kind: string,
  config: MethodPlan['config'],
  steps: number,
  expect: Expect,
  note: string,
  extra: Pick<MethodPlan, 'quantities' | 'tolerances'> | Record<string, never> = {},
): MethodPlan => ({ kind, config, steps, expect, note, ...extra })

const SVI = { learningRate: 0.05, numParticles: 4 }

/**
 * The variational plans of a benchmark model by name, with budgets multiplied by `scale`.
 *
 * @param name - A benchmark model's name.
 * @param scale - The budget multiplier (1 is the committed budget).
 * @returns The plans, or none for a model this file has no plans for.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { sviPlansFor } from 'omai-engine/probprog/models'
 * sviPlansFor('normal-normal', 1).map((p) => p.kind).includes('pathfinder') // => true
 * ```
 */
export function sviPlansFor(name: string, scale: number): MethodPlan[] {
  const steps = (base: number): number => budget(base, scale, 60)
  switch (name) {
    case 'normal-normal':
      return [
        plan(
          'svi-stl',
          SVI,
          steps(1500),
          'agree',
          'a mean-field normal is the right family; sticking the landing quiets the gradient near the optimum',
          { tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] } },
        ),
        plan(
          'svi-trace-graph',
          { learningRate: 0.02, numParticles: 8 },
          steps(1500),
          'agree',
          'no discrete or unreparameterised site: the estimator reduces to the plain ELBO (more particles and a smaller step than sticking the landing needs, for its noise)',
          { tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] } },
        ),
        plan(
          'svi-natural',
          { numParticles: 8, forgetting: 0.6 },
          steps(300),
          'agree',
          'natural-gradient steps in the mean parameters: the Gaussian guide reaches the conjugate posterior in tens of steps',
          { tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] } },
        ),
        plan('svi-conjugate-natural', {}, 1, 'agree', 'batch CAVI: one step of rate one is the exact posterior'),
        plan(
          'svi-structured',
          { learningRate: 0.05 },
          steps(1500),
          'agree',
          'a one-dimensional structured guide is a normal',
          { tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] } },
        ),
        plan(
          'pathfinder',
          { maxIterations: 20, numDraws: 2000 },
          60,
          'agree',
          'the posterior is normal and the first Newton step finds it; the importance weights are uniform',
          { tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] } },
        ),
      ]
    case 'gamma-poisson':
      return [
        plan('svi-conjugate-natural', {}, 1, 'agree', 'batch CAVI: the Gamma posterior in closed form'),
        plan(
          'pathfinder',
          { maxIterations: 30, numDraws: 2000 },
          80,
          'agree',
          'a normal on the log rate, corrected by Pareto-smoothed importance weights towards the Gamma posterior',
          { tolerances: { meanError: 0.2, sdRatio: [0.8, 1.25], ks: 0.15 } },
        ),
        plan(
          'svi-stl',
          SVI,
          steps(1500),
          'report',
          'a normal on the log rate: the right location and spread, the skew of the Gamma is not representable',
        ),
      ]
    case 'logistic-regression':
      return [
        plan(
          'pathfinder',
          { maxIterations: 40, numDraws: 2000, paths: 4 },
          200,
          'report',
          'four L-BFGS paths, importance draws from the mixture of their best normals with Pareto-smoothed weights',
        ),
        plan(
          'svi-structured',
          { learningRate: 0.03, dependencies: 'full', init: 'laplace' },
          steps(1000),
          'report',
          'a full-rank guide started at the Laplace approximation: the coefficients are correlated',
        ),
        plan(
          'svi-natural',
          { family: 'fullRank', numParticles: 8, forgetting: 0.6 },
          steps(400),
          'report',
          'natural-gradient steps for a full-rank normal',
        ),
        plan(
          'svi-stl',
          SVI,
          steps(1500),
          'report',
          'a mean-field normal with sticking the landing: close in the means, narrow',
        ),
      ]
    case 'banana':
      return [
        plan(
          'svi-structured',
          { guide: 'autoFlowNormal', flow: 'planar', numFlows: 6, learningRate: 0.02, numParticles: 4 },
          steps(2000),
          'report',
          'six planar flows bend the Gaussian along the ridge: wider than any Gaussian guide, still narrower than the truth (reverse KL is mode-seeking)',
        ),
        plan(
          'svi-structured',
          { guide: 'autoMultivariateNormal', learningRate: 0.04 },
          steps(1500),
          'differ',
          'a full-rank normal is a straight ellipse: wrong in the spread of x1 and in its skew',
          { quantities: ['x1'] },
        ),
        plan(
          'pathfinder',
          { maxIterations: 40, numDraws: 2000, paths: 4 },
          200,
          'report',
          'normal approximations along the path, importance weights from the mixture of four paths',
        ),
      ]
    case 'bimodal':
      return [
        plan(
          'svi-mixture',
          { components: 2, numParticles: 4, learningRate: 0.05 },
          steps(1500),
          'agree',
          'a two-component Gaussian mixture guide with its component enumerated starts its components on opposite sides of their centre and finds both modes',
          { tolerances: { meanError: 0.2, sdRatio: [0.8, 1.25], ks: 0.15 } },
        ),
        plan(
          'svi-stl',
          SVI,
          steps(1500),
          'differ',
          'one Gaussian is mode-seeking: it settles on one of the two modes (or straddles them) and misses the other',
          { quantities: ['x[0]'] },
        ),
      ]
    default:
      return []
  }
}

const MODELS: readonly BenchmarkModel[] = [normalNormal, gammaPoisson, logisticRegression, banana, bimodal]

/**
 * The campaigns of the variational methods: one `svi/<model>` per benchmark model that has plans in {@link sviPlansFor}, run
 * as the model's cross-engine agreement is, with these plans in place of the model's own.
 *
 * @param models - The models (default normal-normal, gamma-Poisson, logistic regression, the banana and the bimodal density).
 * @returns The campaigns.
 * @concept posterior-validation
 * @index function Periodic validation campaigns of the variational methods against the benchmark truths.
 * @example
 * ```ts
 * import { sviCampaigns } from 'omai-engine/probprog/models'
 * sviCampaigns()[0]?.name // => 'svi/normal-normal'
 * ```
 */
export function sviCampaigns(models: readonly BenchmarkModel[] = MODELS): Campaign[] {
  return models.flatMap((model) => {
    const wrapped: BenchmarkModel = { ...model, plans: (scale = 1) => sviPlansFor(model.name, scale) }
    return benchmarkCampaigns([wrapped]).map((c) => ({
      ...c,
      name: `svi/${model.name}`,
      description: `${model.title}: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.`,
    }))
  })
}
