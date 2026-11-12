import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Program: sparse regression with outliers and unknown noise',
  summary:
    'Simulated sparse regression (a few non-zero coefficients among many, outliers in the response, an unknown noise level) written as a probabilistic program: spike and slab gates on the coefficients, a gamma precision for the noise or a Student-t likelihood, run by EP. It recovers which coefficients are active (P(slab) against the truth and against the exact enumeration of all subsets) and their values.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'sparse regression',
    'spike and slab',
    'gates',
    'student-t',
    'outliers',
    'gamma precision',
    'variable selection',
    'truth recovery',
  ],
  added: '2026-11-11',
}
