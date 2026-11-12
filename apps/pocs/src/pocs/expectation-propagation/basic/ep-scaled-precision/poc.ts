import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Scaled precision (spline penalty)',
  summary:
    'Six latent values theta_j ~ N(0, 1/(tau c_j)) with known multipliers c_j and one gamma-distributed tau, each observed once: the penalised-spline coefficient prior, one random smoothing parameter on a vector.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'scales',
    'prior',
    'gaussian-gamma-precision with a known multiplier',
    'gaussian likelihood',
  ],
  added: '2026-11-11',
}
