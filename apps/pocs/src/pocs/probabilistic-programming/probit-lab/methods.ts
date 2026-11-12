/**
 * The inference methods the lab offers, from the library's `methodKinds`: those that run on the probit model as the
 * lab binds it (checked by the lab's smoke test, which runs every one). Each has the knobs the lab sets, how often to take a
 * snapshot beyond the clock, and the path of the tape its curve is read from.
 */

import type { Organisation } from './types.ts'

/** A method on offer. */
export interface LabMethod {
  readonly id: string
  readonly label: string
  /** A name of `methodKinds` in `omai-engine/probprog/inference`. */
  readonly kind: string
  readonly group: 'sampler' | 'approximation'
  /** How the method's draws are organised: grow (chains) or are redrawn at each snapshot. */
  readonly organisation: Organisation
  readonly config: Readonly<Record<string, string | number | boolean>>
  /** A snapshot every this many steps as well as by time (for methods with few, slow steps). */
  readonly everySteps?: number
  /** The tape path of the curve shown for the method, and what it is. */
  readonly curve?: { readonly path: string; readonly label: string; readonly direction: 'higher' | 'lower' }
  /** One line on what the method is, for the control. */
  readonly note: string
}

const chains = { numChains: 2, numWarmup: 150, numDraws: 3000 }

/** The methods, in the order the queue starts them (the quick ones first, so they free their workers). */
export const METHODS: readonly LabMethod[] = [
  {
    id: 'laplace',
    label: 'Laplace',
    kind: 'laplace',
    group: 'approximation',
    organisation: 'iterate',
    config: {},
    everySteps: 1,
    curve: { path: 'laplace/logDensity', label: 'log density at the iterate', direction: 'higher' },
    note: 'a normal at the mode, with the curvature there',
  },
  {
    id: 'ep',
    label: 'EP',
    kind: 'ep',
    group: 'approximation',
    organisation: 'iterate',
    config: { tol: 1e-8 },
    everySteps: 1,
    note: 'expectation propagation: one normal message per factor, swept to a fixed point',
  },
  {
    id: 'pathfinder',
    label: 'Pathfinder',
    kind: 'pathfinder',
    group: 'approximation',
    organisation: 'iterate',
    config: {},
    everySteps: 1,
    note: 'a normal along the L-BFGS path with the best ELBO',
  },
  {
    id: 'svi-mf',
    label: 'SVI mean-field',
    kind: 'svi',
    group: 'approximation',
    organisation: 'iterate',
    config: { guide: 'autoNormal', learningRate: 0.02 },
    everySteps: 25,
    curve: { path: 'svi/elbo', label: 'ELBO', direction: 'higher' },
    note: 'stochastic variational inference with an independent normal per coefficient',
  },
  {
    id: 'svi-fr',
    label: 'SVI full-rank',
    kind: 'svi',
    group: 'approximation',
    organisation: 'iterate',
    config: { guide: 'autoMultivariateNormal', learningRate: 0.02 },
    everySteps: 25,
    curve: { path: 'svi/elbo', label: 'ELBO', direction: 'higher' },
    note: 'the same with a full covariance, so the correlations are kept',
  },
  {
    id: 'smc',
    label: 'Tempered SMC',
    kind: 'tempered-smc',
    group: 'approximation',
    organisation: 'iterate',
    config: { numParticles: 300, mcmcSteps: 3 },
    everySteps: 1,
    note: 'particles carried from the prior to the posterior through tempered likelihoods',
  },
  {
    id: 'nuts',
    label: 'NUTS',
    kind: 'nuts',
    group: 'sampler',
    organisation: 'chain',
    config: { ...chains, algorithm: 'nuts', maxTreeDepth: 8 },
    note: 'the no-U-turn sampler: the reference when it is run',
  },
  {
    id: 'hmc',
    label: 'HMC',
    kind: 'hmc',
    group: 'sampler',
    organisation: 'chain',
    config: { ...chains, numSteps: 10 },
    note: 'Hamiltonian Monte Carlo with a fixed path length',
  },
  {
    id: 'mala',
    label: 'MALA',
    kind: 'mala',
    group: 'sampler',
    organisation: 'chain',
    config: chains,
    note: 'Metropolis-adjusted Langevin',
  },
  {
    id: 'slice',
    label: 'Slice',
    kind: 'slice',
    group: 'sampler',
    organisation: 'chain',
    config: { ...chains, numWarmup: 40 },
    note: 'slice sampling, no gradients',
  },
  {
    id: 'ess',
    label: 'Elliptical slice',
    kind: 'elliptical-slice',
    group: 'sampler',
    organisation: 'chain',
    config: chains,
    note: 'elliptical slice sampling for a Gaussian prior',
  },
]

/** The method with an id. */
export function methodById(id: string): LabMethod {
  const m = METHODS.find((x) => x.id === id)
  if (m === undefined) throw new Error(`the lab has no method '${id}'`)
  return m
}

/** The ids chosen at first. */
export const DEFAULT_METHODS: readonly string[] = ['laplace', 'ep', 'svi-mf', 'svi-fr', 'nuts']
