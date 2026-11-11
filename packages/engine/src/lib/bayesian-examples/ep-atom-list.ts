/**
 * The registry of the expectation propagation atoms: every atom of the suite in the order the coverage page lists them,
 * and the lookup by name. The registry test is generated from this list: each atom is run (or refused) and its outcome
 * compared with what it declares, so a rule that lands in the library turns a `planned` atom's test red until the atom is
 * flipped to `runs`.
 */

import { OmaiError } from '../base/index.ts'
import type { EpAtom } from './ep-atoms.ts'
import { betaBernoulliAtom, gammaPoissonAtom, gaussianAtom } from './ep-atoms-conjugate.ts'
import { expAtom, logScaleAtom, maxAtom, productAtom, tanhAtom } from './ep-atoms-functions.ts'
import { orderedMeansAtom, orderedPairAtom } from './ep-atoms-gates.ts'
import {
  gpLogisticAtom,
  gpProbitAtom,
  laplaceAtom,
  logisticAtom,
  poissonLogAtom,
  probitAtom,
  skillDifferenceAtom,
  studentTAtom,
} from './ep-atoms-likelihoods.ts'
import {
  clutterAtom,
  clutterMixtureAtom,
  gateKnownAtom,
  gateRandomAtom,
  mixtureComponentsAtom,
  spikeSlabAtom,
  spikeSlabMarginalAtom,
} from './ep-atoms-mixtures.ts'
import {
  priorBetaGeometricAtom,
  priorBetaStrengthGammaAtom,
  priorBetaStrengthLogNormalAtom,
  priorBetaTwoShapesAtom,
  priorDirichletCategoricalAtom,
  priorDirichletStrengthAtom,
  priorGammaExponentialAtom,
  priorGammaPoissonHyperAtom,
  priorInverseGammaHyperAtom,
  priorLogitNormalAtom,
  priorLogNormalPoissonAtom,
  priorLogNormalPoissonHyperAtom,
  priorLogNormalScaleAtom,
  priorNormalHyperAtom,
  priorNormalInverseGammaAtom,
  priorNormalWeibullAtom,
  priorUniformAtom,
} from './ep-atoms-priors.ts'
import {
  gammaPrecisionAtom,
  halfCauchyAtom,
  halfNormalAtom,
  scaledPrecisionAtom,
  sharedPrecisionAtom,
} from './ep-atoms-scales.ts'

/**
 * Every EP atom, in the order of the coverage page: the gates and mixtures first, then the rest as they are added.
 *
 * @concept expectation-propagation
 * @index table The EP atoms: bare setups that each exercise one factor of the message passing suite.
 * @example
 * ```ts
 * import { epAtoms } from 'omai-engine/bayesian-examples'
 * epAtoms.map((a) => a.name).slice(0, 2) // => ['gate', 'gate-random']
 * ```
 */
export const epAtoms: readonly EpAtom[] = [
  gateKnownAtom,
  gateRandomAtom,
  mixtureComponentsAtom,
  spikeSlabAtom,
  spikeSlabMarginalAtom,
  clutterAtom,
  clutterMixtureAtom,
  orderedPairAtom,
  orderedMeansAtom,
  expAtom,
  tanhAtom,
  logScaleAtom,
  productAtom,
  maxAtom,
  gammaPrecisionAtom,
  sharedPrecisionAtom,
  scaledPrecisionAtom,
  halfCauchyAtom,
  halfNormalAtom,
  studentTAtom,
  laplaceAtom,
  probitAtom,
  logisticAtom,
  skillDifferenceAtom,
  gpProbitAtom,
  gpLogisticAtom,
  poissonLogAtom,
  gaussianAtom,
  betaBernoulliAtom,
  gammaPoissonAtom,
  priorNormalInverseGammaAtom,
  priorNormalHyperAtom,
  priorInverseGammaHyperAtom,
  priorGammaPoissonHyperAtom,
  priorGammaExponentialAtom,
  priorLogNormalPoissonAtom,
  priorLogNormalPoissonHyperAtom,
  priorNormalWeibullAtom,
  priorBetaGeometricAtom,
  priorBetaStrengthGammaAtom,
  priorBetaStrengthLogNormalAtom,
  priorDirichletCategoricalAtom,
  priorDirichletStrengthAtom,
  priorLogitNormalAtom,
  priorUniformAtom,
  priorLogNormalScaleAtom,
  priorBetaTwoShapesAtom,
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
 * epAtom('gate').title // => 'Gate'
 * ```
 */
export function epAtom(name: string): EpAtom {
  const found = epAtoms.find((a) => a.name === name)
  if (found === undefined) throw new OmaiError('value', `epAtom: no atom named '${name}'`)
  return found
}
