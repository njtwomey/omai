import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Shared precision',
  summary:
    'Six latent values theta_j ~ N(0, 1/tau) with one gamma-distributed precision tau, each observed once with unit noise: the gamma-precision factor on latent values, six factors on one gamma variable (the hierarchy).',
  tags: ['expectation propagation', 'ep', 'atom', 'scales', 'prior', 'gaussian-gamma-precision', 'gaussian likelihood'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
