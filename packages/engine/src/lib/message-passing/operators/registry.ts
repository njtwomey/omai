/**
 * The registry of the library's factor operators, as one explicit call: `defaultOperators()` returns them by name, so a
 * page that imports only what it needs pulls only that, and a graph is compiled against the collection it is given
 * (the library's, a subset, or the library's with operators of your own added).
 */

import type { FactorOperator } from '../types.ts'
import { dirichletConcentration } from './concentration.ts'
import {
  bernoulliBeta,
  binomial,
  discreteDirichlet,
  gammaObserved,
  negativeBinomialBeta,
  poisson,
} from './conjugate.ts'
import { gaussianMixture, select, table } from './discrete.ts'
import { maximum, minimum } from './extremum.ts'
import { apply } from './functions.ts'
import { gammaProduct, gammaRate } from './gamma-scale.ts'
import { gateOperator } from './gate.ts'
import {
  equal,
  gaussianGammaPrecision,
  gaussianLogScale,
  gaussianNormalGamma,
  gaussianOperator,
  prior,
  product,
} from './gaussian.ts'
import { likelihoodRowOperator, likelihoodSiteOperator, negativeBinomialDispersion } from './likelihoods.ts'
import {
  difference,
  dotConstant,
  gaussianJoint,
  linear,
  linearRegressionRow,
  matmulConstant,
  plus,
  sum,
  timesConstant,
} from './linear.ts'
import { logistic } from './links.ts'
import { gaussianScale } from './log-scale.ts'
import { linearRegressionPlate, logisticPlate, probitPlate } from './plates.ts'
import { laplaceRow, laplaceSite, mixtureRow, mixtureSite, studentTRow, studentTSite } from './robust.ts'
import { logisticRow, probitRow, spinRow } from './rows.ts'
import { gaussianMvPrecisionScale } from './scale.ts'
import { spin } from './spin.ts'
import { greaterThan, isBetween, probit } from './truncation.ts'

/**
 * The library's factor operators by name: `prior`, `gaussian`, `gaussian-gamma-precision`, `gaussian-log-scale`, `gaussian-normal-gamma`,
 * `product`, `max`, `min`, `equal`, `linear`, `plus`, `difference`, `times-constant`, `sum`, `dot-constant`, `matmul-constant`,
 * `linear-regression-row`, `gaussian-joint`, `probit`, `greater-than`, `is-between`, `logistic`, `bernoulli`,
 * `discrete`, `poisson`, `binomial`, `gamma-observed`, `negative-binomial-beta`, `table`, `select`, `gaussian-mixture`, the regression rows `probit-row` and `logistic-row`, the plates of rows `linear-regression-plate`, `probit-plate` and `logistic-plate`, the
 * spin site `spin` and its row `spin-row`, and the robust sites `mixture-site`, `student-t-site` and `laplace-site` with their rows `mixture-row`, `student-t-row` and `laplace-row`, the shared smoothing parameter of a Gaussian vector `gaussian-mv-precision-scale`, the gamma scale hierarchies `gamma-rate` and `gamma-product`, the random concentration of a beta or Dirichlet `dirichlet-concentration`, the Gaussian with a random log scale and mean `gaussian-scale`, the negative binomial with a random log dispersion `negative-binomial-dispersion`, the unary function `apply` (`exp`, `square`, `tanh`, `relu`, `softplus`, `abs`, `sigmoid` of a Gaussian variable), and the generic numeric likelihood `likelihood-site` with its row `likelihood-row` (a registry of count, survival, ordinal, quantile and robust likelihoods), and the `gate` of Minka and Winn (a discrete variable selects among branches that are operators of this registry). A new object each call, so a caller may
 * add its own (`{ ...defaultOperators(), mine }`).
 *
 * @returns The operators, keyed by their names.
 * @concept factor-operators
 * @index table The library's factor operators by name.
 * @example
 * ```ts
 * import { defaultOperators } from 'omai-engine/message-passing'
 * Object.keys(defaultOperators()).includes('probit') // => true
 * ```
 */
export function defaultOperators(): Record<string, FactorOperator> {
  const all: FactorOperator[] = [
    prior,
    gaussianOperator,
    gaussianGammaPrecision,
    gaussianLogScale,
    gaussianNormalGamma,
    product,
    maximum,
    minimum,
    equal,
    linear,
    plus,
    difference,
    timesConstant,
    sum,
    dotConstant,
    matmulConstant,
    linearRegressionRow,
    gaussianJoint,
    probit,
    greaterThan,
    isBetween,
    logistic,
    bernoulliBeta,
    discreteDirichlet,
    poisson,
    binomial,
    gammaObserved,
    negativeBinomialBeta,
    table,
    select,
    gaussianMixture,
    probitRow,
    logisticRow,
    linearRegressionPlate,
    probitPlate,
    logisticPlate,
    spin,
    spinRow,
    mixtureSite,
    mixtureRow,
    studentTSite,
    studentTRow,
    laplaceSite,
    laplaceRow,
    likelihoodSiteOperator,
    likelihoodRowOperator,
    apply,
    gaussianScale,
    negativeBinomialDispersion,
    gaussianMvPrecisionScale,
    gammaRate,
    gammaProduct,
    dirichletConcentration,
  ]
  const registry: Record<string, FactorOperator> = Object.fromEntries(all.map((o) => [o.name, o]))
  // The gate runs any operator of the registry as a branch, itself included (gates nest).
  registry['gate'] = gateOperator(registry)
  return registry
}
