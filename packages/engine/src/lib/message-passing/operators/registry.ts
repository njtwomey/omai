/**
 * The registry of the library's factor operators, as one explicit call: `defaultOperators()` returns them by name, so a
 * page that imports only what it needs pulls only that, and a graph is compiled against the collection it is given
 * (the library's, a subset, or the library's with operators of your own added).
 */

import type { FactorOperator } from '../types.ts'
import { bernoulliBeta, binomial, discreteDirichlet, poisson } from './conjugate.ts'
import { gaussianMixture, select, table } from './discrete.ts'
import { equal, gaussianGammaPrecision, gaussianNormalGamma, gaussianOperator, prior, product } from './gaussian.ts'
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
import { logisticRow, probitRow } from './rows.ts'
import { greaterThan, isBetween, probit } from './truncation.ts'

/**
 * The library's factor operators by name: `prior`, `gaussian`, `gaussian-gamma-precision`, `gaussian-normal-gamma`,
 * `product`, `equal`, `linear`, `plus`, `difference`, `times-constant`, `sum`, `dot-constant`, `matmul-constant`,
 * `linear-regression-row`, `gaussian-joint`, `probit`, `greater-than`, `is-between`, `logistic`, `bernoulli`,
 * `discrete`, `poisson`, `binomial`, `table`, `select`, `gaussian-mixture`, and the regression rows `probit-row` and
 * `logistic-row`. A new object each call, so a caller may
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
    gaussianNormalGamma,
    product,
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
    table,
    select,
    gaussianMixture,
    probitRow,
    logisticRow,
  ]
  return Object.fromEntries(all.map((o) => [o.name, o]))
}
