import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian: parameters learned',
  summary:
    'The Gaussian of the basic atom with its mean and its variance both unknown (a normal-inverse-gamma prior), observed through data simulated from a stated mean and sd: does EP recover them, and how often does its 90% interval hold the truth?',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'conjugate',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'gaussian-normal-gamma',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
