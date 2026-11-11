/**
 * The registry of the expectation propagation atoms: every atom of the suite in the order the coverage page lists them,
 * and the lookup by name. The registry test is generated from this list: each atom is run (or refused) and its outcome
 * compared with what it declares, so a rule that lands in the library turns a `planned` atom's test red until the atom is
 * flipped to `runs`.
 */

import { OmaiError } from '../base/index.ts'
import type { EpAtom } from './ep-atoms.ts'
import {
  callsPerHourAtom,
  coinBiasAtom,
  gaussianMeanAtom,
  germinationAtom,
  potholesAtom,
} from './ep-atoms-conjugate.ts'

/**
 * Every EP atom, grouped by family in the order of the coverage page.
 *
 * @concept expectation-propagation
 * @index table The EP atoms: tiny concrete problems that each prove one factor of the message passing suite.
 * @example
 * ```ts
 * import { epAtoms } from 'omai-engine/bayesian-examples'
 * epAtoms.map((a) => a.name).slice(0, 2) // => ['gaussian-mean', 'coin-bias']
 * ```
 */
export const epAtoms: readonly EpAtom[] = [
  gaussianMeanAtom,
  coinBiasAtom,
  callsPerHourAtom,
  potholesAtom,
  germinationAtom,
]

/**
 * The atom of a name.
 *
 * @param name - The slug.
 * @returns The atom.
 * @throws OmaiError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { epAtom } from 'omai-engine/bayesian-examples'
 * epAtom('coin-bias').title // => 'A coin flipped a dozen times'
 * ```
 */
export function epAtom(name: string): EpAtom {
  const found = epAtoms.find((a) => a.name === name)
  if (found === undefined) throw new OmaiError('value', `epAtom: no atom named '${name}'`)
  return found
}
