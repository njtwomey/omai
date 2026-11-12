import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Student-t likelihood: parameters learned (the scale and the degrees of freedom too)',
  summary:
    'The Student-t noise of the basic atom with the location, the scale and the degrees of freedom all unknown, the data simulated from stated values of the three.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'prior (gaussian)',
    'prior (log-normal)',
    'prior (gamma)',
    'likelihood-triple: student-t with random location, scale and degrees of freedom',
  ],
  added: '2026-11-12',
}
