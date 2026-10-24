import type { LinkName } from '../../lib/dists/families/index.ts'
import { defineExplainedEstimator, type ExplainedEstimator, ExplainedEstimatorClass } from '../../lib/learn/index.ts'
import { bool, dict, real, type Space } from '../../lib/space/index.ts'
import {
  type FamilyName,
  type FittedLinearModel,
  fitLinear,
  type LinearModelHyper,
  linearModelDefaults,
  linearVerbs,
} from './model.ts'
import type { LinearSolver } from './solvers.ts'

// TODO: zero-inflated and hurdle count models, quasi-likelihood families.

/** Hyperparameters of the GLM regressors. */
export interface GlmHyper {
  /** The L2 penalty on the per-sample deviance (scikit-learn's alpha, default 1). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly solver: LinearSolver
  readonly maxIter: number
  /** The link (default the family's: log for counts and positive responses). */
  readonly link?: LinkName
  /** The Tweedie power. */
  readonly power: number
  /** The negative binomial's ancillary alpha. */
  readonly nbAlpha: number
}

function glmRegressor(
  name: string,
  family: FamilyName,
  hyper: Partial<GlmHyper>,
): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  const h: GlmHyper = {
    alpha: hyper.alpha ?? 1,
    fitIntercept: hyper.fitIntercept ?? true,
    solver: hyper.solver ?? 'auto',
    maxIter: hyper.maxIter ?? 100,
    ...(hyper.link === undefined ? {} : { link: hyper.link }),
    power: hyper.power ?? 1.5,
    nbAlpha: hyper.nbAlpha ?? 1,
  }
  const config = (g: GlmHyper): LinearModelHyper =>
    linearModelDefaults({
      family,
      ...(g.link === undefined ? {} : { link: g.link }),
      power: g.power,
      nbAlpha: g.nbAlpha,
      alpha: g.alpha,
      fitIntercept: g.fitIntercept,
      solver: g.solver,
      maxIter: g.maxIter,
    })
  return defineExplainedEstimator(
    {
      name,
      kind: 'regressor',
      hyperSpace: (g: GlmHyper): Space =>
        dict({
          alpha: real(0, 10, { default: g.alpha, label: 'L2 penalty' }),
          fitIntercept: bool({ default: g.fitIntercept }),
          ...(family === 'tweedie' ? { power: real(1, 3, { default: g.power, label: 'Tweedie power' }) } : {}),
        }),
      fit: (data, g, options) => {
        const f = fitLinear(data, config(g), options, name)
        return { params: f.params, exposition: [f.table] }
      },
      methods: (p, g) => linearVerbs(p, config(g), name),
    },
    h,
  )
}

/**
 * Poisson regression with the log link (sklearn.linear_model.PoissonRegressor), by IRLS. Exposition: the coefficient
 * table (Wald errors when alpha = 0).
 *
 * @param hyper - Penalty (default 1), intercept, solver and iterations.
 * @returns The estimator; `predictDistribution` gives Poisson distributions.
 */
export function poissonRegressor(hyper: Partial<GlmHyper> = {}): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  return glmRegressor('poisson-regressor', 'poisson', hyper)
}

/**
 * Gamma regression with the log link (sklearn.linear_model.GammaRegressor), by IRLS; the dispersion is Pearson's
 * estimate. Exposition: the coefficient table.
 *
 * @param hyper - Penalty (default 1), intercept, link, solver and iterations.
 * @returns The estimator; `predictDistribution` gives gamma distributions.
 */
export function gammaRegressor(hyper: Partial<GlmHyper> = {}): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  return glmRegressor('gamma-regressor', 'gamma', hyper)
}

/**
 * Tweedie regression with the log link (sklearn.linear_model.TweedieRegressor): power 0 normal, 1 Poisson, between 1
 * and 2 compound Poisson-gamma, 2 gamma, 3 inverse Gaussian. Exposition: the coefficient table.
 *
 * @param hyper - Power (default 1.5), penalty (default 1), intercept, solver and iterations.
 * @returns The estimator.
 */
export function tweedieRegressor(hyper: Partial<GlmHyper> = {}): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  return glmRegressor('tweedie-regressor', 'tweedie', hyper)
}

/**
 * Inverse Gaussian regression with the log link (statsmodels' InverseGaussian family). Exposition: the coefficient
 * table.
 *
 * @param hyper - Penalty (default 1), intercept, link, solver and iterations.
 * @returns The estimator.
 */
export function inverseGaussianRegressor(
  hyper: Partial<GlmHyper> = {},
): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  return glmRegressor('inverse-gaussian-regressor', 'inverse-gaussian', hyper)
}

/**
 * Negative binomial regression with a fixed ancillary alpha and the log link (statsmodels' NegativeBinomial family).
 * Exposition: the coefficient table.
 *
 * @param hyper - The ancillary alpha (default 1), penalty (default 1), intercept, solver and iterations.
 * @returns The estimator.
 */
export function negativeBinomialRegressor(
  hyper: Partial<GlmHyper> = {},
): ExplainedEstimator<GlmHyper, FittedLinearModel> {
  return glmRegressor('negative-binomial-regressor', 'negative-binomial', hyper)
}

/** {@link poissonRegressor} as a class. */
export class PoissonRegressor extends ExplainedEstimatorClass<GlmHyper, FittedLinearModel> {
  /** @param hyper - As {@link poissonRegressor}. */
  constructor(hyper: Partial<GlmHyper> = {}) {
    super(poissonRegressor(hyper))
  }
}

/** {@link gammaRegressor} as a class. */
export class GammaRegressor extends ExplainedEstimatorClass<GlmHyper, FittedLinearModel> {
  /** @param hyper - As {@link gammaRegressor}. */
  constructor(hyper: Partial<GlmHyper> = {}) {
    super(gammaRegressor(hyper))
  }
}

/** {@link tweedieRegressor} as a class. */
export class TweedieRegressor extends ExplainedEstimatorClass<GlmHyper, FittedLinearModel> {
  /** @param hyper - As {@link tweedieRegressor}. */
  constructor(hyper: Partial<GlmHyper> = {}) {
    super(tweedieRegressor(hyper))
  }
}
