/**
 * The registry of the library's factor operators, as one explicit call: `defaultOperators()` returns them by name, so a
 * page that imports only what it needs pulls only that, and a graph is compiled against the collection it is given
 * (the library's, a subset, or the library's with operators of your own added). Each folder lists its own operators
 * (`list.ts`); the registration order is `operatorNames` (`names.ts`), and the registry refuses a name that two operators
 * share, an operator the names do not list and a name that no operator has.
 */

import { ParameterError } from '../core/errors.ts'
import type { FactorOperator } from '../core/types.ts'
import { covarianceOperatorList } from './covariance/list.ts'
import { gateOperator } from './discrete/gate.ts'
import { discreteOperatorList } from './discrete/list.ts'
import { gaussianOperatorList } from './gaussian/list.ts'
import { liftOperatorList } from './lifts/list.ts'
import { likelihoodOperatorList } from './likelihoods/list.ts'
import { linkOperatorList } from './links/list.ts'
import { operatorNames } from './names.ts'
import { scaleOperatorList } from './scales/list.ts'

/**
 * Operators keyed by name in a given order, refusing a registry that disagrees with its own list.
 *
 * @param operators - The operators, in any order.
 * @param names - The names in the order of the result.
 * @returns The operators by name, in the order of `names`.
 * @throws ParameterError when two operators have one name, an operator is not in `names`, or a name has no operator.
 * @example
 * ```ts
 * import { collectOperators, prior, equal } from 'omai-engine/message-passing'
 * Object.keys(collectOperators([equal, prior], ['prior', 'equal'])) // => ['prior', 'equal']
 * ```
 */
export function collectOperators(
  operators: readonly FactorOperator[],
  names: readonly string[],
): Record<string, FactorOperator> {
  const byName = new Map<string, FactorOperator>()
  for (const op of operators) {
    if (byName.has(op.name)) throw new ParameterError(`registry: two operators are named '${op.name}'`)
    byName.set(op.name, op)
  }
  const registry: Record<string, FactorOperator> = {}
  for (const name of names) {
    const op = byName.get(name)
    if (op === undefined) throw new ParameterError(`registry: the names list '${name}' and no operator has it`)
    registry[name] = op
    byName.delete(name)
  }
  const [unlisted] = byName.keys()
  if (unlisted !== undefined)
    throw new ParameterError(`registry: the operator '${unlisted}' is not in the list of names`)
  return registry
}

/**
 * The library's factor operators by name, in the order of `operatorNames` (the gate, which runs the others as its branches, last).
 * A new object each call, so a caller may add its own (`{ ...defaultOperators(), mine }`). A registry that disagrees with its own list
 * (two operators with one name, an operator not in `operatorNames`, a name with no operator) throws: it is a bug, not a configuration.
 *
 * @returns The operators, keyed by their names.
 * @concept factor-operators
 * @index table The library's factor operators by name.
 * @example
 * ```ts
 * import { defaultOperators, operatorNames } from 'omai-engine/message-passing'
 * Object.keys(defaultOperators()).includes('probit') // => true
 * Object.keys(defaultOperators()).length === operatorNames.length // => true
 * ```
 */
export function defaultOperators(): Record<string, FactorOperator> {
  const registry = collectOperators(
    [
      ...gaussianOperatorList,
      ...linkOperatorList,
      ...discreteOperatorList,
      ...liftOperatorList,
      ...likelihoodOperatorList,
      ...scaleOperatorList,
      ...covarianceOperatorList,
    ],
    operatorNames.filter((name) => name !== 'gate'),
  )
  // The gate runs any operator of the registry as a branch, itself included (gates nest).
  registry['gate'] = gateOperator(registry)
  return registry
}
