// TODO: SAG and SAGA solvers (optim has them); ordinal (ordinal.ts) and mixed models (mixed.ts).
import { WhiteboxError } from '../../lib/base/index.ts'
import type { DispersionFamily, Link, LinkName } from '../../lib/dists/families/index.ts'
import type { Distribution } from '../../lib/dists/index.ts'
import { families } from '../../lib/dists/index.ts'
import { type CoefficientTable, coefficientTable, type Exposition } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
} from '../../lib/learn/index.ts'
import { bool, choice, dict, real, type Space } from '../../lib/space/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Design, designOf, designPredictor, linearPredictor, vectorTensor } from './_design.ts'
import { glmCovariance, waldTable } from './inference.ts'
import { chooseSolver, type GlmProblem, type GlmSolution, type LinearSolver, solveGlm } from './solvers.ts'

/** The response families of the GLM core. */
export type FamilyName =
  | 'gaussian'
  | 'binomial'
  | 'poisson'
  | 'gamma'
  | 'inverse-gaussian'
  | 'tweedie'
  | 'negative-binomial'

/** Hyperparameters of {@link linearModel}. */
export interface LinearModelHyper {
  readonly family: FamilyName
  /** The link (default the family's canonical or usual one). */
  readonly link?: LinkName
  /** The Tweedie power (default 1.5). */
  readonly power: number
  /** The negative binomial's ancillary alpha (default 1). */
  readonly nbAlpha: number
  /** The penalty strength (scikit-learn's per-sample scaling). */
  readonly alpha: number
  /** The L1 share of the penalty (0 ridge, 1 lasso). */
  readonly l1Ratio: number
  readonly fitIntercept: boolean
  readonly solver: LinearSolver
  readonly maxIter: number
  /** Solver tolerance (default per solver). */
  readonly tol?: number
}

/** Parameters of a fitted linear model. */
export type LinearModelParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The dispersion phi (sigma^2 for least squares). */
  readonly scale: number
  readonly solver: string
  readonly reason: string
  readonly iterations: number
  readonly converged: boolean
  readonly objective: number
}

/** Verbs of a fitted linear model. */
export interface LinearModelVerbs {
  /** The mean response g^{-1}(x beta + b). */
  readonly predict: (x: TensorLike) => Tensor
  /** The linear predictor x beta + b. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** The family's distribution of the response at each row. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted linear model. */
export type FittedLinearModel = Fitted<LinearModelParams> & LinearModelVerbs

/** The family and link a configuration names. */
export function familyOf(h: Pick<LinearModelHyper, 'family' | 'link' | 'power' | 'nbAlpha'>): {
  readonly family: DispersionFamily
  readonly link: Link
} {
  const link = h.link === undefined ? undefined : families.linkByName(h.link)
  const family = (() => {
    switch (h.family) {
      case 'gaussian':
        return families.gaussian()
      case 'binomial':
        return families.binomial()
      case 'poisson':
        return families.poisson()
      case 'gamma':
        return families.gamma(link ?? families.log())
      case 'inverse-gaussian':
        return families.inverseGaussian(link ?? families.log())
      case 'tweedie':
        return families.tweedie(h.power)
      case 'negative-binomial':
        return families.negativeBinomial(h.nbAlpha)
    }
  })()
  // scikit-learn's Gamma and inverse Gaussian regressors use the log link; others use the family's default.
  const fallback = h.family === 'gamma' || h.family === 'inverse-gaussian' ? families.log() : family.defaultLink
  return { family, link: link ?? fallback }
}

/** The fit of a linear model with what its explanation needs. */
export interface LinearFit {
  readonly params: LinearModelParams
  readonly solution: GlmSolution
  readonly design: Design
  readonly names: readonly string[]
  /** Coefficients with errors when the model is unpenalised, estimates only otherwise (the intercept last). */
  readonly table: CoefficientTable
}

/**
 * Fit the penalised GLM core and build its coefficient table: the shared fit behind every named linear model.
 *
 * @param data - Features, targets and weights.
 * @param h - The configuration.
 * @param options - Randomness and tracing, and a warm start.
 * @param who - The caller, for errors.
 * @returns The fit.
 */
export function fitLinear(
  data: LearnData,
  h: LinearModelHyper,
  options: FitOptions & { readonly start?: { readonly coef: ArrayLike<number>; readonly intercept: number } },
  who: string,
): LinearFit {
  const design = designOf(data, who)
  const { family, link } = familyOf(h)
  if (!(h.alpha >= 0) || !(h.l1Ratio >= 0 && h.l1Ratio <= 1))
    throw new WhiteboxError('value', `${who}: alpha must be >= 0 and l1Ratio in [0, 1]`)
  const problem: GlmProblem = { design, family, link, alpha: h.alpha, l1Ratio: h.l1Ratio, fitIntercept: h.fitIntercept }
  const chosen = h.solver === 'auto' ? chooseSolver(problem) : { solver: h.solver, reason: 'chosen by the caller' }
  const solution = solveGlm(problem, {
    solver: chosen.solver,
    maxIter: h.maxIter,
    ...(h.tol === undefined ? {} : { tol: h.tol }),
    ...(options.start === undefined ? {} : { start: options.start }),
    ...(options.key === undefined ? {} : { key: options.key }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const coef = Array.from(solution.coef)
  const eta = designPredictor(design, coef, solution.intercept)
  const mu = link.inverse(vectorTensor(eta))
  const muFlat = toFlat(primalOf(mu))
  const p = coef.length + (h.fitIntercept ? 1 : 0)
  const dfResid = Math.max(design.x.n - p, 1)
  const variance = toFlat(primalOf(family.variance(mu)))
  let pearson = 0
  for (let i = 0; i < design.x.n; i++)
    pearson += ((design.w[i] ?? 0) * ((design.y[i] ?? 0) - (muFlat[i] ?? 0)) ** 2) / (variance[i] ?? 1)
  const fixedScale = family.name === 'binomial' || family.name === 'poisson' || family.name.startsWith('negative')
  const scale = fixedScale ? 1 : pearson / dfResid
  const names = [...featureNamesOf(data, design.x.d), ...(h.fitIntercept ? ['intercept'] : [])]
  const estimates = [...coef, ...(h.fitIntercept ? [solution.intercept] : [])]
  let table: CoefficientTable
  if (h.alpha === 0 && design.x.n > p) {
    const working = toFlat(primalOf(family.irlsWeights(mu, link, vectorTensor(design.w))))
    const cov = glmCovariance(design, working, scale, h.fitIntercept)
    table = waldTable(names, estimates, cov, family.name === 'gaussian' ? { df: dfResid } : {})
  } else table = coefficientTable({ names, estimates })
  return {
    params: {
      coef,
      intercept: solution.intercept,
      scale,
      solver: solution.solver,
      reason: chosen.reason,
      iterations: solution.iterations,
      converged: solution.converged,
      objective: solution.objective,
    },
    solution,
    design,
    names,
    table,
  }
}

/** The verbs of a fitted linear model, from its parameters. */
export function linearVerbs(p: LinearModelParams, h: LinearModelHyper, who: string): LinearModelVerbs {
  const { family, link } = familyOf(h)
  const eta = (x: TensorLike): Tensor => vectorTensor(linearPredictor(x, p.coef, p.intercept, who))
  return {
    predict: (x) => vectorTensor(toFlat(primalOf(link.inverse(eta(x))))),
    decisionFunction: eta,
    predictDistribution: (x) => family.distribution(link.inverse(eta(x)), p.scale),
    nFeaturesIn: p.coef.length,
    featureImportances: p.coef.map(Math.abs),
  }
}

/** Defaults of {@link linearModel}. */
export function linearModelDefaults(hyper: Partial<LinearModelHyper>): LinearModelHyper {
  return {
    family: hyper.family ?? 'gaussian',
    ...(hyper.link === undefined ? {} : { link: hyper.link }),
    power: hyper.power ?? 1.5,
    nbAlpha: hyper.nbAlpha ?? 1,
    alpha: hyper.alpha ?? 0,
    l1Ratio: hyper.l1Ratio ?? 0,
    fitIntercept: hyper.fitIntercept ?? true,
    solver: hyper.solver ?? 'auto',
    maxIter: hyper.maxIter ?? 1000,
    ...(hyper.tol === undefined ? {} : { tol: hyper.tol }),
  }
}

/** The tunable penalty of a linear model. */
export function penaltySpace(defaults: { readonly alpha: number; readonly l1Ratio?: number }): Space {
  return dict({
    alpha: real(1e-4, 10, { log: true, default: defaults.alpha, label: 'penalty strength' }),
    ...(defaults.l1Ratio === undefined
      ? {}
      : { l1Ratio: real(0, 1, { default: defaults.l1Ratio, label: 'L1 share' }) }),
    fitIntercept: bool({ default: true }),
  })
}

/**
 * The one penalised GLM estimator every named linear model configures (docs/structure.md 5.1): a family (Gaussian,
 * binomial, Poisson, gamma, inverse Gaussian, Tweedie, negative binomial) with a link, an elastic-net penalty
 * alpha (l1Ratio |beta|_1 + (1 - l1Ratio) |beta|^2 / 2) on the per-sample deviance, and a solver (chosen
 * automatically with its reason, or named: Cholesky, QR, SVD, IRLS, coordinate descent, L-BFGS, OWL-QN).
 * `predictDistribution` is the family's distribution with the fitted dispersion. Exposition: the coefficient table
 * (with Wald standard errors, p-values and intervals when unpenalised).
 *
 * @param hyper - Family, link, penalty, intercept, solver and iteration limits.
 * @returns The estimator.
 */
export function linearModel(
  hyper: Partial<LinearModelHyper> = {},
): ExplainedEstimator<LinearModelHyper, FittedLinearModel> {
  return defineExplainedEstimator(
    {
      name: 'linear-model',
      kind: 'regressor',
      hyperSpace: (h): Space =>
        dict({
          family: choice(
            ['gaussian', 'binomial', 'poisson', 'gamma', 'inverse-gaussian', 'tweedie', 'negative-binomial'],
            {
              default: h.family,
            },
          ),
          alpha: real(0, 10, { default: h.alpha, label: 'penalty strength' }),
          l1Ratio: real(0, 1, { default: h.l1Ratio, label: 'L1 share' }),
          fitIntercept: bool({ default: h.fitIntercept }),
        }),
      fit: (data, h, options) => {
        const f = fitLinear(data, h, options, 'linearModel')
        const exposition: Exposition[] = [f.table]
        return { params: f.params, exposition }
      },
      methods: (p, h) => linearVerbs(p, h, 'linearModel'),
    },
    linearModelDefaults(hyper),
  )
}

/** {@link linearModel} as a class: `new LinearModel({ family: 'poisson' }).fit(x, y).predict(x)`. */
export class LinearModel extends ExplainedEstimatorClass<LinearModelHyper, FittedLinearModel> {
  /**
   * @param hyper - As {@link linearModel}.
   */
  constructor(hyper: Partial<LinearModelHyper> = {}) {
    super(linearModel(hyper))
  }
}
