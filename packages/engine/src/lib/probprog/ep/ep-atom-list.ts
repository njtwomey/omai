/**
 * The registry of the expectation propagation atoms: every atom of the suite and the lookup by name. The registry test is
 * generated from this list: each atom is run and its outcome compared with what it declares (`runs` unless it says `planned`),
 * so a rule that breaks turns its atom's test red.
 */

import { OmaiError } from '../../base/index.ts'
import type { EpAtom } from './ep-atoms.ts'
import { betaBernoulliAtom, gammaPoissonAtom, gaussianAtom } from './ep-atoms-conjugate.ts'
import { hurdlePoissonAtom } from './ep-atoms-counts.ts'
import {
  bivariateCorrelationAtom,
  bivariateCorrelationLearnedAtom,
  mvnWishartAtom,
  mvnWishartLearnedAtom,
  randomInterceptsSlopesAtom,
  randomInterceptsSlopesLearnedAtom,
} from './ep-atoms-covariance.ts'
import {
  betaBernoulliLearnedAtom,
  gammaPoissonLearnedAtom,
  gaussianLearnedAtom,
} from './ep-atoms-extended-conjugate.ts'
import {
  clutterLearnedAtom,
  clutterMixtureLearnedAtom,
  gateRandomLearnedAtom,
  mixtureComponentsLearnedAtom,
  spikeSlabLearnedAtom,
  spikeSlabMarginalLearnedAtom,
} from './ep-atoms-extended-gates.ts'
import {
  laplaceLearnedAtom,
  logisticLearnedAtom,
  poissonLogDispersionLearnedAtom,
  poissonLogLearnedAtom,
  probitLearnedAtom,
  skillDifferenceLearnedAtom,
  studentTDfLearnedAtom,
  studentTLearnedAtom,
  weibullLearnedAtom,
  weibullShapeLearnedAtom,
} from './ep-atoms-extended-likelihoods.ts'
import { gammaPrecisionLearnedAtom, sharedPrecisionLearnedAtom } from './ep-atoms-extended-scales.ts'
import { expAtom, logScaleAtom, maxAtom, productAtom, tanhAtom } from './ep-atoms-functions.ts'
import { orderedMeansAtom, orderedPairAtom } from './ep-atoms-gates.ts'
import { gpLengthScaleAtom, gpProbitLearnedAtom } from './ep-atoms-kernels.ts'
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
import { softmaxRegressionAtom, softmaxRegressionLearnedAtom } from './ep-atoms-multiclass.ts'
import { stickBreakingAtom, topicModelAtom } from './ep-atoms-nonparametric.ts'
import {
  ordinalProbitAtom,
  ordinalProbitLearnedAtom,
  quantileRegressionAtom,
  quantileRegressionLearnedAtom,
  skillDrawsAtom,
  skillDrawsLearnedAtom,
} from './ep-atoms-ordinal.ts'
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
import { mixtureRegressionAtom } from './ep-atoms-regression-gates.ts'
import {
  gammaPrecisionAtom,
  halfCauchyAtom,
  halfNormalAtom,
  scaledPrecisionAtom,
  sharedPrecisionAtom,
} from './ep-atoms-scales.ts'
import {
  gateWeightAtom,
  gateWeightDirichletAtom,
  gateWeightHyperAtom,
  gateWeightSummedOutAtom,
} from './ep-atoms-weights.ts'

/**
 * Every EP atom, in the order they were added (the gates and mixtures first).
 *
 * @concept expectation-propagation
 * @index table The EP atoms: bare setups that each exercise one factor of the message passing suite.
 * @example
 * ```ts
 * import { epAtoms } from 'omai-engine/probprog/ep'
 * epAtoms.map((a) => a.name).slice(0, 2) // => ['gate', 'gate-random']
 * ```
 */
export const epAtoms: readonly EpAtom[] = [
  gateKnownAtom,
  gateRandomAtom,
  gateRandomLearnedAtom,
  mixtureComponentsAtom,
  mixtureComponentsLearnedAtom,
  spikeSlabAtom,
  spikeSlabLearnedAtom,
  spikeSlabMarginalAtom,
  spikeSlabMarginalLearnedAtom,
  clutterAtom,
  clutterLearnedAtom,
  clutterMixtureAtom,
  clutterMixtureLearnedAtom,
  orderedPairAtom,
  orderedMeansAtom,
  expAtom,
  tanhAtom,
  logScaleAtom,
  productAtom,
  maxAtom,
  gammaPrecisionAtom,
  gammaPrecisionLearnedAtom,
  sharedPrecisionAtom,
  sharedPrecisionLearnedAtom,
  scaledPrecisionAtom,
  halfCauchyAtom,
  halfNormalAtom,
  studentTAtom,
  studentTLearnedAtom,
  laplaceAtom,
  laplaceLearnedAtom,
  probitAtom,
  probitLearnedAtom,
  logisticAtom,
  logisticLearnedAtom,
  skillDifferenceAtom,
  skillDifferenceLearnedAtom,
  gpProbitAtom,
  gpLogisticAtom,
  softmaxRegressionAtom,
  softmaxRegressionLearnedAtom,
  ordinalProbitAtom,
  ordinalProbitLearnedAtom,
  quantileRegressionAtom,
  quantileRegressionLearnedAtom,
  skillDrawsAtom,
  skillDrawsLearnedAtom,
  gpLengthScaleAtom,
  gpProbitLearnedAtom,
  poissonLogAtom,
  poissonLogLearnedAtom,
  poissonLogDispersionLearnedAtom,
  gaussianAtom,
  gaussianLearnedAtom,
  betaBernoulliAtom,
  betaBernoulliLearnedAtom,
  gammaPoissonAtom,
  gammaPoissonLearnedAtom,
  priorNormalInverseGammaAtom,
  priorNormalHyperAtom,
  priorInverseGammaHyperAtom,
  priorGammaPoissonHyperAtom,
  priorGammaExponentialAtom,
  priorLogNormalPoissonAtom,
  priorLogNormalPoissonHyperAtom,
  priorNormalWeibullAtom,
  weibullLearnedAtom,
  weibullShapeLearnedAtom,
  studentTDfLearnedAtom,
  priorBetaGeometricAtom,
  priorBetaStrengthGammaAtom,
  priorBetaStrengthLogNormalAtom,
  priorDirichletCategoricalAtom,
  priorDirichletStrengthAtom,
  priorLogitNormalAtom,
  priorUniformAtom,
  priorLogNormalScaleAtom,
  priorBetaTwoShapesAtom,
  gateWeightAtom,
  gateWeightSummedOutAtom,
  gateWeightDirichletAtom,
  gateWeightHyperAtom,
  stickBreakingAtom,
  topicModelAtom,
  mixtureRegressionAtom,
  hurdlePoissonAtom,
  mvnWishartAtom,
  mvnWishartLearnedAtom,
  bivariateCorrelationAtom,
  bivariateCorrelationLearnedAtom,
  randomInterceptsSlopesAtom,
  randomInterceptsSlopesLearnedAtom,
]

/**
 * The atom of a name.
 *
 * @param name - The slug.
 * @returns The atom.
 * @throws OmaiError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { epAtom } from 'omai-engine/probprog/ep'
 * epAtom('gate').title // => 'Gate'
 * ```
 */
export function epAtom(name: string): EpAtom {
  const found = epAtoms.find((a) => a.name === name)
  if (found === undefined) throw new OmaiError('value', `epAtom: no atom named '${name}'`)
  return found
}
