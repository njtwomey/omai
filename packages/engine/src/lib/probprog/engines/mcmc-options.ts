/**
 * The slots of the MCMC driver that vary as option tables: the kernel (NUTS, or HMC with a fixed number of leapfrog
 * steps) with the warm-up, draws, chains and target acceptance as knobs, and how chains start (uniformly in a box on the
 * unconstrained scale, or from the prior draw as it is). Each table is one `union` space for controls and grids.
 */

import { bool, dict, int, type OptionTable, optionTable, plainOption, real } from '../../space/index.ts'
import { type InitStrategy, initToFeasible, initToUniform } from '../density.ts'
import type { McmcOptions } from './nuts.ts'

const commonSpaces = () => ({
  numChains: int(1, 16, { default: 2, label: 'chains' }),
  numWarmup: int(0, 5000, { default: 500, label: 'warm-up' }),
  numDraws: int(1, 10000, { default: 500, label: 'draws' }),
  targetAcceptance: real(0.5, 0.99, { default: 0.8, label: 'target acceptance' }),
  dense: bool({ default: false, label: 'dense mass matrix' }),
})

const shared = (k: (name: string) => number | boolean): McmcOptions => ({
  numChains: k('numChains') as number,
  numWarmup: k('numWarmup') as number,
  numDraws: k('numDraws') as number,
  targetAcceptance: k('targetAcceptance') as number,
  dense: k('dense') as boolean,
})

/**
 * The MCMC kernels as one slot: `nuts` (the No-U-Turn sampler with a tree depth cap) and `hmc` (a fixed number of leapfrog
 * steps), each with the chains, warm-up, draws, target acceptance and mass-matrix form as knobs. A table's `make` gives the
 * `McmcOptions` of `mcmc` and `mcmcAlgorithm`.
 *
 * @concept nuts
 * @index table The MCMC kernels as one slot.
 * @example
 * ```ts
 * import { mcmcKernels } from 'omai-engine/probprog'
 * mcmcKernels.make({ kind: 'hmc', numSteps: 5 }).numSteps // => 5
 * ```
 */
export const mcmcKernels: OptionTable<McmcOptions, Record<string, never>> = optionTable(
  {
    nuts: plainOption<McmcOptions, number | boolean>(
      'the No-U-Turn sampler',
      dict({ ...commonSpaces(), maxTreeDepth: int(1, 15, { default: 10, label: 'tree depth cap' }) }),
      (k) => ({ ...shared(k), algorithm: 'nuts', maxTreeDepth: k('maxTreeDepth') as number }),
    ),
    hmc: plainOption<McmcOptions, number | boolean>(
      'Hamiltonian Monte Carlo with a fixed number of leapfrog steps',
      dict({ ...commonSpaces(), numSteps: int(1, 200, { default: 10, label: 'leapfrog steps' }) }),
      (k) => ({ ...shared(k), algorithm: 'hmc', numSteps: k('numSteps') as number }),
    ),
  },
  { default: 'nuts', label: 'MCMC kernel', description: 'The Hamiltonian kernel that draws the chains.' },
)

/**
 * How chains start, as one slot: `uniform` (a draw in a box of the given radius on the unconstrained scale, as Stan; retried
 * until the density and its gradient are finite) and `prior` (each site's own prior draw, radius 0).
 *
 * @concept log-density
 * @example
 * ```ts
 * import { initStrategies } from 'omai-engine/probprog'
 * typeof initStrategies.make({ kind: 'uniform', radius: 1 }) // => 'function'
 * ```
 */
export const initStrategies: OptionTable<InitStrategy, Record<string, never>> = optionTable(
  {
    uniform: plainOption<InitStrategy, number | boolean>(
      'uniform in a box on the unconstrained scale',
      dict({ radius: real(0.1, 10, { default: 2, label: 'radius' }) }),
      (k) => initToUniform({ radius: k('radius') as number }),
    ),
    prior: plainOption<InitStrategy, number | boolean>('the prior draw', dict({}), () => initToFeasible()),
  },
  { default: 'uniform', label: 'chain start', description: 'Where each chain begins.' },
)
