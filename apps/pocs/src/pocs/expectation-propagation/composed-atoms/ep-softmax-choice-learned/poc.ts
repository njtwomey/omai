import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Softmax over eight alternatives: parameters learned',
  summary:
    'The eight-alternative choice model of the basic atom with the two coefficients unknown, the choices simulated from stated coefficients.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'prior (gaussian) x2',
    'scale (known attribute times a coefficient)',
    'multiclass-site (softmax, eight classes)',
  ],
  added: '2026-11-12',
}
