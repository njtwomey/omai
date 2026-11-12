/**
 * The registry of the expectation propagation atoms: the arrays of the family files (`_atoms/`, one file per family with the
 * basic atoms and their learned twins together), concatenated, and the lookup by name. The registry test is generated from
 * this list: each atom is run and its outcome compared with what it declares (`runs` unless it says `planned`), so a rule that
 * breaks turns its atom's test red. An atom is added to the array of its family file and nowhere else; the atoms are not
 * exported one by one, a page asks for `epAtom('name')`.
 */

import { OmaiError } from '../../base/index.ts'
import { composedChainAtoms } from './_atoms/composed-chain.ts'
import { composedGatesAtoms } from './_atoms/composed-gates.ts'
import { composedAtoms } from './_atoms/composed.ts'
import { conjugateAtoms } from './_atoms/conjugate.ts'
import { countsAtoms } from './_atoms/counts.ts'
import { covarianceAtoms } from './_atoms/covariance.ts'
import { functionsAtoms } from './_atoms/functions.ts'
import { gatesAtoms } from './_atoms/gates.ts'
import { kernelsAtoms } from './_atoms/kernels.ts'
import { likelihoodsAtoms } from './_atoms/likelihoods.ts'
import { nonparametricAtoms } from './_atoms/nonparametric.ts'
import { ordinalAtoms } from './_atoms/ordinal.ts'
import { priorsAtoms } from './_atoms/priors.ts'
import { scalesAtoms } from './_atoms/scales.ts'
import { weightsAtoms } from './_atoms/weights.ts'
import type { AtomFamily, EpAtom } from './atom-types.ts'

/** The atoms by file, in the order of the coverage table's rows. */
const BY_FILE: readonly (readonly EpAtom[])[] = [
  conjugateAtoms,
  priorsAtoms,
  scalesAtoms,
  likelihoodsAtoms,
  ordinalAtoms,
  gatesAtoms,
  weightsAtoms,
  functionsAtoms,
  kernelsAtoms,
  covarianceAtoms,
  nonparametricAtoms,
  countsAtoms,
  composedAtoms,
  composedGatesAtoms,
  composedChainAtoms,
]

/** The order the atoms were added in, which the coverage table and the pages keep until a family order replaces it; an atom not listed follows, in its file's order. */
const ADDED_ORDER: readonly string[] = [
  'gate',
  'gate-random',
  'gate-random-learned',
  'mixture-components',
  'mixture-components-learned',
  'spike-and-slab',
  'spike-and-slab-learned',
  'spike-and-slab-marginal',
  'spike-and-slab-marginal-learned',
  'clutter',
  'clutter-learned',
  'clutter-mixture',
  'clutter-mixture-learned',
  'ordered-pair',
  'ordered-means',
  'exp',
  'tanh',
  'log-scale',
  'product',
  'max',
  'gamma-precision',
  'gamma-precision-learned',
  'shared-precision',
  'shared-precision-learned',
  'scaled-precision',
  'half-cauchy',
  'half-normal',
  'student-t',
  'student-t-learned',
  'laplace',
  'laplace-learned',
  'probit',
  'probit-learned',
  'logistic',
  'logistic-learned',
  'skill-difference',
  'skill-difference-learned',
  'gp-probit',
  'gp-logistic',
  'softmax-regression',
  'softmax-regression-learned',
  'ordinal-probit',
  'ordinal-probit-learned',
  'quantile-regression',
  'quantile-regression-learned',
  'skill-draws',
  'skill-draws-learned',
  'gp-length-scale',
  'gp-probit-learned',
  'gp-hyperparameters',
  'poisson-log',
  'poisson-log-learned',
  'poisson-log-dispersion-learned',
  'gaussian',
  'gaussian-learned',
  'beta-bernoulli',
  'beta-bernoulli-learned',
  'gamma-poisson',
  'gamma-poisson-learned',
  'prior-normal-inverse-gamma',
  'prior-normal-hyperprior',
  'prior-inverse-gamma-hyperprior',
  'prior-gamma-poisson-hyperprior',
  'prior-gamma-exponential',
  'prior-lognormal-poisson',
  'prior-lognormal-poisson-hyperprior',
  'prior-normal-weibull-scale',
  'weibull-learned',
  'weibull-shape-learned',
  'student-t-df-learned',
  'prior-beta-geometric',
  'prior-beta-strength-gamma',
  'prior-beta-strength-lognormal',
  'prior-dirichlet-categorical',
  'prior-dirichlet-strength-gamma',
  'prior-logit-normal-bernoulli',
  'prior-uniform-mean',
  'prior-lognormal-scale-hyperprior',
  'prior-beta-two-shapes',
  'gate-weight',
  'gate-weight-summed-out',
  'gate-weight-dirichlet',
  'gate-weight-hyperprior',
  'stick-breaking',
  'topic-model',
  'mixture-regression',
  'hurdle-poisson',
  'mvn-wishart',
  'mvn-wishart-learned',
  'bivariate-correlation',
  'bivariate-correlation-learned',
  'random-intercepts-slopes',
  'random-intercepts-slopes-learned',
]

const rank = (name: string): number => {
  const i = ADDED_ORDER.indexOf(name)
  return i < 0 ? ADDED_ORDER.length : i
}

/**
 * Every EP atom: the bare setups that each exercise one factor of the message passing suite, with the reference they are checked
 * against, the tolerance and the page that shows them. The registry of the suite: one array per family file, concatenated.
 *
 * @concept expectation-propagation
 * @index table The EP atoms: bare setups of one factor each, with a reference and a tolerance (look one up by name).
 * @example
 * ```ts
 * import { epAtoms } from 'omai-engine/probprog/ep'
 * epAtoms.map((a) => a.name).slice(0, 2) // => ['gate', 'gate-random']
 * ```
 */
export const epAtoms: readonly EpAtom[] = BY_FILE.flat().sort((a, b) => rank(a.name) - rank(b.name))

/**
 * The atoms of a family, in registry order.
 *
 * @param family - The family (`conjugate`, `priors`, `gates`, ...).
 * @returns The atoms.
 * @example
 * ```ts
 * import { epAtomsOf } from 'omai-engine/probprog/ep'
 * epAtomsOf('conjugate').map((a) => a.name).slice(0, 3) // => ['gaussian', 'gaussian-learned', 'beta-bernoulli']
 * ```
 */
export function epAtomsOf(family: AtomFamily): readonly EpAtom[] {
  return epAtoms.filter((a) => a.family === family)
}

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
